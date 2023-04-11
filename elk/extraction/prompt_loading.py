from collections import Counter
from dataclasses import dataclass
from itertools import cycle
from random import Random
from typing import Any, Iterator, Literal, Optional

from datasets import (
    Dataset,
    load_dataset,
)
from datasets.distributed import split_dataset_by_node
from simple_parsing.helpers import Serializable, field

from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    infer_label_column,
    infer_num_classes,
    select_train_val_splits,
)
from .balanced_sampler import BalancedSampler


@dataclass
class PromptConfig(Serializable):
    """
    Args:
        dataset: Space-delimited name of the HuggingFace dataset to use, e.g.
            `"super_glue boolq"` or `"imdb"`.
        balance: Whether to force class balance in the dataset using undersampling.
        data_dir: The directory to use for caching the dataset. Defaults to
            `~/.cache/huggingface/datasets`.
        label_column: The column containing the labels. By default, we infer this from
            the datatypes of the columns in the dataset; if there is only one column
            with a `ClassLabel` datatype, we use that.
        max_examples: The maximum number of examples to use from the val dataset.
            If a single number, use at most that many examples for each split. If a list
            of length 2, use the first element for the train split and the second for
            the val split. If empty, use all examples. Defaults to empty.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to 1.
        seed: The seed to use for prompt randomization. Defaults to 42.
        stream: Whether to stream the dataset from the Internet. Defaults to False.
    """

    datasets: list[str] = field(positional=True)
    balance: bool = False
    data_dir: Optional[str] = None
    label_column: Optional[str] = None
    max_examples: list[int] = field(default_factory=lambda: [750, 250])
    num_classes: int = 0
    num_shots: int = 0
    num_variants: int = -1
    seed: int = 42
    stream: bool = False

    def __post_init__(self):
        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = [int(1e100)]

        # Broadcast the limit to all splits
        if len(self.max_examples) == 1:
            self.max_examples *= 2


def load_prompts(
    *dataset_strings: str,
    label_column: Optional[str] = None,
    num_classes: int = 0,
    num_shots: int = 0,
    num_variants: int = -1,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    stream: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified datasets.

    Args:
        dataset_strings: Space-delimited names of the HuggingFace datasets to use,
            e.g. `"super_glue boolq"` or `"imdb"`.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        stream: Whether to stream the dataset from the Internet. Defaults to False.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.

    Returns:
        An iterable dataset of prompts.
    """
    class_counts = []
    prompters = []
    datasets = []
    label_cols = []
    train_datasets = []
    rng = Random(seed)
    assert num_shots == 0

    # First load the datasets and prompters. We need to know the minimum number of
    # templates for any dataset in order to make sure we don't run out of prompts.
    for ds_string in dataset_strings:
        ds_name, _, config_name = ds_string.partition(" ")
        prompters.append(DatasetTemplates(ds_name, config_name))

        ds_dict = assert_type(
            dict, load_dataset(ds_name, config_name or None, streaming=stream)
        )
        train_name, val_name = select_train_val_splits(ds_dict)
        split_name = val_name if split_type == "val" else train_name

        ds = ds_dict[split_name].shuffle(seed=seed)
        train_ds = ds_dict[train_name].shuffle(seed=seed)

        if not stream:
            ds = assert_type(Dataset, ds)
            if world_size > 1:
                ds = ds.shard(world_size, rank)

            ds = ds.to_iterable_dataset().cast(ds.features)

        elif world_size > 1:
            # This prints to stdout which is slightly annoying
            ds = split_dataset_by_node(dataset=ds, rank=rank, world_size=world_size)

        ds_label_col = label_column or infer_label_column(ds.features)
        num_classes = num_classes or infer_num_classes(ds.features[ds_label_col])
        class_counts.append(num_classes)
        datasets.append(ds)
        label_cols.append(ds_label_col)
        train_datasets.append(train_ds)

    # Number of classes should be the same for all datasets
    num_classes, *rest = class_counts
    if not all(num_classes == x for x in rest):
        raise ValueError(
            f"# classes should be the same for all datasets, but got {class_counts}"
        )

    min_num_templates = min(len(prompter.templates) for prompter in prompters)
    num_variants = (
        min_num_templates
        if num_variants == -1
        else min(num_variants, min_num_templates)
    )
    assert num_variants > 0
    if rank == 0:
        print(f"Using {num_variants} variants of each prompt")

    ds_iters = [
        iter(BalancedSampler(ds, num_classes, label_col=label_col))
        for ds, label_col in zip(datasets, label_cols)
    ]
    for ds_iter, ds, label_col, prompter in cycle(
        zip(ds_iters, datasets, label_cols, prompters)
    ):
        try:
            example = next(ds_iter)
        except StopIteration:
            return

        example = _convert_to_prompts(
            example,
            label_column=label_col,
            num_classes=num_classes,
            num_variants=num_variants,
            prompter=prompter,
            rng=rng,
            fewshot_iter=None,
        )

        # Add the builder and config name to the records directly to make
        # sure we don't forget what dataset they came from.
        example["builder_name"] = ds.info.builder_name
        example["config_name"] = ds.info.config_name

        yield example


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    label_column: str,
    num_classes: int,
    num_variants: int,
    rng: Random,
    fewshot_iter: Optional[Iterator[list[dict]]] = None,
) -> dict[str, Any]:
    """Prompt-generating function to pass to `IterableDataset.map`."""
    labels_are_strings = isinstance(example[label_column], str)
    prompts = []
    templates = list(prompter.templates.values())
    if num_variants < len(templates):
        templates = rng.sample(templates, num_variants)

    def qa_cat(q: str, a: str) -> str:
        # if the jinja template already adds whitespace, don't add more
        sep = "" if not q or q[-1].isspace() or not a or a[0].isspace() else " "
        return f"{q}{sep}{a}" if a and not a.isspace() else q

    # For sanity checking that prompts are unique
    prompt_counter = Counter()
    label_indices = set()

    for template in templates:
        choices = []
        string_choices = template.get_answer_choices_list(example)

        label = example[label_column]
        label_indices.add(string_choices.index(label) if labels_are_strings else label)

        for answer_idx in range(num_classes):
            fake_example = example.copy()
            if labels_are_strings:
                fake_example[label_column] = string_choices[answer_idx]
            else:
                fake_example[label_column] = answer_idx

            q, a = template.apply(fake_example)
            text = qa_cat(q, a or string_choices[answer_idx])
            prompt_counter[text] += 1

            if fewshot_iter is not None:
                # Infinite iterator so we don't need to worry about StopIteration
                fewshot_examples = next(fewshot_iter)
                fewshot_texts = [
                    qa_cat(q, a) for q, a in map(template.apply, fewshot_examples)
                ]
                text = "\n\n".join(fewshot_texts) + "\n\n" + text

            choices.append(
                dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    text=text,
                )
            )

        prompts.append(choices)

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    # Sanity check: label should be the same across all variants
    if len(label_indices) > 1:
        raise ValueError(
            f"Label index should be the same all variants, but got {label_indices}"
        )

    return dict(
        label=label_indices.pop(),
        prompts=prompts,
        template_names=[template.name for template in templates],
    )
