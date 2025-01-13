"""Functions for extracting the hidden states of a model."""

import os
from collections import defaultdict
from dataclasses import InitVar, dataclass, replace
from itertools import zip_longest
from typing import Any, Iterable, Literal

import torch
from datasets import (
    Array2D,
    Dataset,
    DatasetDict,
    DatasetInfo,
    DownloadMode,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
)
from simple_parsing import Serializable, field
from torch import Tensor
from transformers import AutoConfig

from ..utils import (
    Color,
    assert_type,
    colorize,
    float_to_int16,
    instantiate_tokenizer,
    prevent_name_conflicts,
    select_split,
    select_train_val_splits,
)
from ..utils.hf_utils import is_autoregressive
from .dataset_name import (
    DatasetDictWithName,
    parse_dataset_string,
)
from .generator import _GeneratorBuilder
from .inference_server import InferenceServer
from .prompt_loading import get_prompter, load_prompts


@dataclass
class Extract(Serializable):
    """Config for extracting hidden states from a language model."""

    model: str = field(positional=True)
    """HF model string identifying the language model to extract hidden states from."""

    datasets: tuple[str, ...] = field(positional=True)
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"`"""

    data_dirs: tuple[str, ...] = ()
    """Directory to use for caching the hiddens. Defaults to `HF_DATASETS_CACHE`."""

    get_lm_preds: bool = True
    """Whether to extract the LM predictions."""

    int8: bool = False
    """Whether to perform inference in mixed int8 precision with `bitsandbytes`."""

    fsdp: bool = False
    """Whether to use FullyShardedDataParallel for inference."""

    max_examples: tuple[int, int] = (1000, 1000)
    """Maximum number of examples to use from each split of the dataset."""

    num_shots: int = 0
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    num_variants: int = -1
    """The number of prompt templates to use for each example. If -1, all available
    templates are used."""

    balance: bool = True
    """Whether to balance the number of examples per class."""

    statement_column: str | None = None
    """Name of the column containing the model input strings when using a built-in
    prompt template. If None, we use the "statement" column."""

    layers: tuple[int, ...] = ()
    """Indices of layers to extract hidden states from. We follow the HF convention, so
    0 is the embedding, and 1 is the output of the first transformer layer."""

    layer_stride: InitVar[int] = 1
    """Shortcut for `layers = (0,) + tuple(range(1, num_layers + 1, stride))`."""

    seed: int = 42
    """Seed to use for prompt randomization. Defaults to 42."""

    template_path: str | None = None
    """Path to pass into `DatasetTemplates`. By default we use the dataset name."""

    token_loc: Literal["first", "last", "penultimate", "mean"] = "last"
    """The location of the token to extract hidden states from."""

    def __post_init__(self, layer_stride: int):
        if self.num_variants != -1:
            print("WARNING: num_variants is deprecated; use prompt_indices instead.")
        if len(self.datasets) == 0:
            raise ValueError(
                "Must specify at least one dataset to extract hiddens from."
            )

        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = (int(1e100), int(1e100))

        # Broadcast the dataset name to all data_dirs
        if len(self.data_dirs) == 1:
            self.data_dirs *= len(self.datasets)
        elif self.data_dirs and len(self.data_dirs) != len(self.datasets):
            raise ValueError(
                "data_dirs should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.data_dirs)}"
            )

        if self.layers and layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif layer_stride > 1:
            from transformers import AutoConfig, PretrainedConfig

            # Look up the model config to get the number of layers
            config = assert_type(
                PretrainedConfig, AutoConfig.from_pretrained(self.model)
            )
            # Note that we always include 0 which is the embedding layer
            layer_range = range(1, config.num_hidden_layers + 1, layer_stride)
            self.layers = (0,) + tuple(layer_range)

    def explode(self) -> list["Extract"]:
        """Explode this config into a list of configs, one for each layer."""
        return [
            replace(self, datasets=(ds,), data_dirs=(data_dir,) if data_dir else ())
            for ds, data_dir in zip_longest(self.datasets, self.data_dirs)
        ]


def tokenize_dataset(
    cfg: "Extract",
    split_type: Literal["train", "val"] = "train",
) -> Dataset:
    """Apply the prompt templates to the dataset and return the tokenized LM inputs.
    Each dict contains the keys `input_ids`, `variant_id`,
    `row_id`, `text`, and `label`. If lm_preds is True, we also include `answer_ids`
    and `num_suffix_tokens`.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ds_names = cfg.datasets
    assert len(ds_names) == 1, "Can only extract hiddens from one dataset at a time."

    tokenizer = instantiate_tokenizer(cfg.model, truncation_side="left")

    # TODO: support using the encoder only of an encoder-decoder model

    prompt_ds = load_prompts(
        ds_names[0],
        num_shots=cfg.num_shots,
        split_type=split_type,
        template_path=cfg.template_path,
        include_answers=cfg.get_lm_preds,
        balance=cfg.balance,
        seed=cfg.seed,
        statement_column=cfg.statement_column,
    )

    max_examples = cfg.max_examples[0 if split_type == "train" else 1]

    max_length = assert_type(int, tokenizer.model_max_length)

    out_records = []
    for example in prompt_ds:
        num_variants = len(example["template_names"])

        # Check if we've appended enough examples
        if len(out_records) >= max_examples * num_variants:
            break

        # Throw out all variants if any of them are too long
        any_too_long = False
        record_variants = []

        # Iterate over variants
        for i, statement in enumerate(example["statements"]):
            if cfg.get_lm_preds:
                suffix = example["suffixes"][i]
                answer_choices = example["answer_choices"][i]
                assert len(answer_choices) == 2
                answer_ids = []
                for choice in answer_choices:
                    a_id = tokenizer.encode(" " + choice, add_special_tokens=False)

                    # the Llama tokenizer splits off leading spaces
                    if tokenizer.decode(a_id[0]).strip() == "":
                        a_id_without_space = tokenizer.encode(
                            choice, add_special_tokens=False
                        )
                        assert a_id_without_space == a_id[1:]
                        a_id = a_id_without_space

                    if len(a_id) > 1:
                        print(
                            f"WARNING: answer choice '{choice}' is more than one "
                            "token, LM probabilities will be calculated using the "
                            f"first token only ({tokenizer.decode(a_id[0])})"
                        )
                    answer_ids.append(a_id[0])
            else:
                suffix = ""

            suffix_tokens = torch.tensor(
                tokenizer.encode(suffix, add_special_tokens=False),
                dtype=torch.long,
            )

            encoding = tokenizer(
                statement,
                # Keep [CLS] and [SEP] for BERT-style models
                add_special_tokens=True,
                return_tensors="pt",
            )

            # suffix comes right after the last statement token, before the answer
            ids = torch.cat([encoding.input_ids, suffix_tokens.unsqueeze(0)], dim=-1)

            # If this input is too long, skip it
            if ids.shape[-1] > max_length:
                any_too_long = True
                break

            out_record: dict[str, Any] = dict(
                row_id=example["row_id"],
                variant_id=example["template_names"][i],
                label=example["label"],
                text=statement + suffix,
                input_ids=ids.long(),
            )
            if cfg.get_lm_preds:
                out_record["answer_ids"] = answer_ids  # type: ignore
                # keep track of where to extract hiddens from
                out_record["num_suffix_tokens"] = len(suffix_tokens)
            record_variants.append(out_record)

        if any_too_long:
            continue

        # print an example text to stdout
        if len(out_records) == 0:
            print(f"Example text: {record_variants[0]['text']}")
            neg_id, pos_id = record_variants[0]["answer_ids"]
            print(f'\tneg choice token: "{tokenizer.decode(neg_id)}"')
            print(f'\tpos choice token: "{tokenizer.decode(pos_id)}"')
        out_records.extend(record_variants)
    else:
        print(
            f"WARNING: reached end of dataset {ds_names[0]} before collecting "
            f"{max_examples} examples (only got {len(out_records)})."
        )

    # transpose the list of dicts into a dict of lists
    out_records = {k: [d[k] for d in out_records] for k in out_records[0]}
    return Dataset.from_dict(out_records)


def hidden_features(cfg: Extract) -> tuple[DatasetInfo, Features]:
    """Return the HuggingFace `Features` corresponding to an `Extract` config."""
    with prevent_name_conflicts():
        model_cfg = AutoConfig.from_pretrained(cfg.model)

    ds_name, config_name = parse_dataset_string(dataset_config_str=cfg.datasets[0])
    info = get_dataset_config_info(ds_name, config_name or None)

    assert_type(Features, info.features)

    prompter, _ = get_prompter(ds_name, config_name, cfg.template_path)

    # num_dropped = prompter.drop_non_mc_templates()
    num_variants = len(prompter.templates)
    # if num_dropped:
    # print(f"Dropping {num_dropped} non-multiple choice templates")

    layer_cols = {
        f"hidden_{layer}": Array2D(
            dtype="int16",
            shape=(num_variants, model_cfg.hidden_size),
        )
        # Add 1 to include the embedding layer
        for layer in cfg.layers or range(model_cfg.num_hidden_layers + 1)
    }
    other_cols = {
        "row_id": Value(dtype="int64"),
        "variant_ids": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
        "label": Value(dtype="int64"),
        "texts": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
    }
    if cfg.get_lm_preds:
        other_cols["lm_log_odds"] = Sequence(
            Value(dtype="float32"),
            length=num_variants,
        )

    return info, Features({**layer_cols, **other_cols})


def extract(
    cfg: "Extract",
    *,
    disable_cache: bool = False,
    highlight_color: Color = "cyan",
    num_gpus: int = -1,
    split_type: Literal["train", "val", None] = None,
) -> DatasetDictWithName:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""

    info, features = hidden_features(cfg)

    model_config = AutoConfig.from_pretrained(cfg.model)
    if not is_autoregressive(model_config, include_enc_dec=True) and cfg.get_lm_preds:
        raise ValueError("Can only extract LM predictions from autoregressive models.")
    limits = cfg.max_examples
    splits = assert_type(SplitDict, info.splits)

    pretty_name = colorize(assert_type(str, cfg.datasets[0]), highlight_color)
    if split_type is None:
        train, val = select_train_val_splits(splits)

        print(f"{pretty_name} using '{train}' for training and '{val}' for validation")
        splits = SplitDict({train: splits[train], val: splits[val]})
        split_types = ["train", "val"]
    else:
        # Remove the split we're not using
        limits = [limits[0]] if split_type == "train" else limits
        split_name = select_split(splits, split_type)
        splits = SplitDict({split_name: splits[split_name]})
        split_types = [split_type]

        if split_type == "train":
            print(f"{pretty_name} using '{split_name}' for training")
        else:
            print(f"{pretty_name} using '{split_name}' for validation")

    def select_hiddens(outputs: Any, **kwargs: Any) -> tuple[dict[str, Tensor], Tensor]:
        tok_loc_offset = kwargs.get("num_suffix_tokens", 0)
        # Add one to the number of layers to account for the embedding layer
        layer_indices = cfg.layers or tuple(range(model_config.num_hidden_layers + 1))

        hiddens = outputs.get("decoder_hidden_states") or outputs["hidden_states"]
        # Throw out layers we don't care about
        hiddens = [hiddens[i] for i in layer_indices]

        # Current shape of each element: (batch_size, seq_len, hidden_size)
        if cfg.token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif cfg.token_loc == "last":
            hiddens = [h[..., h.shape[-2] - tok_loc_offset - 1, :] for h in hiddens]
        elif cfg.token_loc == "penultimate":
            hiddens = [h[..., h.shape[-2] - tok_loc_offset - 2, :] for h in hiddens]
        elif cfg.token_loc == "mean":
            hiddens = [h[..., :-tok_loc_offset, :].mean(dim=-2) for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {cfg.token_loc}")

        hidden_dict = dict()
        for layer_idx, hidden in zip(layer_indices, hiddens):
            hidden_dict[f"hidden_{layer_idx}"] = float_to_int16(hidden.flatten()).cpu()

        if (answer_ids := kwargs.get("answer_ids")) is not None:
            # log_odds = log(p(yes)/(p(no)) = log(p(yes)) - log(p(no))
            logits = outputs["logits"][0, -1, answer_ids]
            logprobs = logits.log_softmax(dim=-1)
            lm_log_odds = logprobs[1] - logprobs[0]
        else:
            lm_log_odds = torch.Tensor([torch.nan])

        return hidden_dict, lm_log_odds

    def extract_hiddens(
        cfg: Extract,
        split_type: Literal["train", "val"],
        server: InferenceServer,
    ) -> Iterable[dict]:
        encodings = tokenize_dataset(cfg, split_type=split_type)
        num_variants = len(encodings.unique("variant_id"))

        if not server.running:
            server.start()
        encodings = encodings.add_column("id", range(len(encodings)))  # type: ignore

        buffer = defaultdict(list)  # row_id -> list of dicts
        for idx, (hidden_dict, lm_log_odds) in server.imap(
            select_hiddens,
            encodings,
            use_tqdm=False,
            model_kwargs=dict(output_hidden_states=True),
        ):
            encoding = encodings[idx]
            row_id = encoding["row_id"]
            buffer[row_id].append(
                dict(lm_log_odds=lm_log_odds, **encoding, **hidden_dict)
            )
            if len(buffer[row_id]) == num_variants:
                # we have a complete example
                ex = buffer[row_id]
                ex = sorted(ex, key=lambda d: d["variant_id"])
                assert all(d["label"] == ex[0]["label"] for d in ex)
                assert len(set(d["variant_id"] for d in ex)) == num_variants
                out_record: dict[str, Any] = dict(
                    variant_ids=[d["variant_id"] for d in ex],
                    label=ex[0]["label"],
                    row_id=ex[0]["row_id"],
                    texts=[d["text"] for d in ex],
                    **{k: torch.stack([d[k] for d in ex]) for k in hidden_dict},
                )
                if cfg.get_lm_preds:
                    out_record["lm_log_odds"] = torch.stack(
                        [d["lm_log_odds"] for d in ex]  # type: ignore
                    )
                del buffer[row_id]
                yield out_record

    # hf wraps everything in a list here, so we unpack them here
    def _extraction_worker(**kwargs):
        yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})

    # TODO: support int8
    server = InferenceServer(
        model_str=cfg.model, num_workers=num_gpus, cpu_offload=True, fsdp=cfg.fsdp
    )

    builders = {
        split_name: _GeneratorBuilder(
            cache_dir=None,
            features=features,
            generator=_extraction_worker,
            split_name=split_name,
            split_info=SplitInfo(
                name=split_name,
                num_examples=min(limit, v.num_examples) * len(cfg.datasets),
                dataset_name=v.dataset_name,
            ),
            gen_kwargs=dict(
                cfg=[cfg],
                split_type=[ty],
                server=[server],
            ),
        )
        for limit, (split_name, v), ty in zip(limits, splits.items(), split_types)
    }

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(
            download_mode=DownloadMode.FORCE_REDOWNLOAD if disable_cache else None,
            num_proc=None,
        )
        ds[split] = builder.as_dataset(split=split)  # type: ignore[assignment]

    if server.running:
        server.shutdown()

    dataset_dict = DatasetDict(ds)

    return DatasetDictWithName(
        name=cfg.datasets[0],
        dataset=dataset_dict,
    )
