import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml
from jinja2 import BaseLoader, Environment, meta

# Truncation of jinja template variables
# 1710 = 300 words x 4.7 avg characters per word + 300 spaces
TEXT_VAR_LENGTH = 2048

# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = Path(__file__).parent / "templates"

env = Environment(loader=BaseLoader)  # type: ignore

# Allow the python function zip()
env.globals.update(enumerate=enumerate, zip=zip)

# These are users whose datasets should be included in the results returned by
# filter_english_datasets (regardless of their metadata)
INCLUDED_USERS = {"Zaid", "craffel", "lauritowal", "christykoh"}


def highlight(input):
    return "<span style='color: #F08080'>" + input + "</span>"


def permutation(n):
    return random.sample(range(n), n)


def reorder(arr, permutation):
    return [arr[i] for i in permutation]


def to_letter(n):
    return chr(n + ord("A"))


def most_frequent(items):
    """Returns the set of items which appear most frequently in the input"""
    if not items:
        return
    item_counts = Counter(items).most_common()
    max_freq = item_counts[0][1]
    most_frequent_items = [c[0] for c in item_counts if c[1] == max_freq]
    return most_frequent_items


env.filters["highlight"] = highlight
env.filters["choice"] = random.choice
env.filters["most_frequent"] = most_frequent
env.filters["permutation"] = permutation
env.filters["reorder"] = reorder
env.filters["to_letter"] = to_letter


class Template(yaml.YAMLObject):
    """
    A prompt template.
    """

    yaml_tag = "!Template"

    def __init__(
        self, name, jinja, reference, metadata=None, answer_choices=None, suffix=""
    ):
        """
        Creates a prompt template.

        A prompt template is expressed in Jinja. It is rendered using an example
        from the corresponding Hugging Face datasets library (a dictionary). The
        separator ||| should appear once to divide the template into prompt and
        output. Generally, the prompt should provide information on the desired
        behavior, e.g., text passage and instructions, and the output should be
        a desired response.

        :param name: unique name (per dataset) for template
        :param jinja: template expressed in Jinja
        :param reference: string describing author or paper reference for template
        :param metadata: a Metadata object with template annotations
        :param answer_choices: Jinja expression for answer choices. Should produce
                               a ||| delimited string of choices that enumerates
                               the possible completions for templates that should
                               be evaluated as ranked completions. If None, then
                               the template is open-ended. This list is accessible
                               from within Jinja as the variable `answer_choices`.
        :param suffix: string to append to the end of the statement before the answer
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.jinja = jinja
        self.reference = reference
        self.metadata = metadata or Template.Metadata()
        self.answer_choices = answer_choices
        self.suffix = suffix

    def get_answer_choices_list(self, example):
        """
        Returns a list of answer choices for a given example

        :return: list of strings, or None if get_answer_choices_expr is None
        """
        jinja = self.answer_choices
        if jinja is None:
            return None

        rtemplate = env.from_string(jinja)
        protected_example = self._escape_pipe(example)
        rendered_choices = rtemplate.render(**protected_example)
        return [
            self._unescape_pipe(answer_choice.strip())
            for answer_choice in rendered_choices.split("|||")
        ]

    def get_fixed_answer_choices_list(self):
        """
        Returns a list of answer choices that is static across examples, if possible
        :return: list of strings, or None if no static list exists
        """
        jinja = self.answer_choices
        if jinja is None:
            return None

        parse = env.parse(jinja)
        variables = meta.find_undeclared_variables(parse)
        if len(variables) == 0:
            rtemplate = env.from_string(jinja)
            rendered_choices = rtemplate.render()
            return [
                answer_choice.strip() for answer_choice in rendered_choices.split("|||")
            ]
        else:
            return None

    def apply(self, example, truncate=False, highlight_variables=False):
        """
        Creates a prompt by applying this template to an example

        :param example: the dataset example to create a prompt for
        :param truncate: if True, fields will be truncated to TEXT_VAR_LENGTH chars
        :param highlight_variables: highlight the added variables
        :return: tuple of 2 strings, for prompt and output
        """
        jinja = self.jinja

        # Truncates the prompt if needed
        if truncate:
            # Escaping curly braces requires doubling them
            trunc_command = f" | string | truncate({TEXT_VAR_LENGTH}) }}}}"
            jinja = jinja.replace("}}", trunc_command)

        # Highlights text that was substituted for variables, if requested
        if highlight_variables:
            jinja = jinja.replace("}}", " | highlight }}")

        rtemplate = env.from_string(jinja)

        protected_example = self._escape_pipe(example)

        # Adds in answer_choices variable
        if "answer_choices" in protected_example:
            raise ValueError("Example contains the restricted key 'answer_choices'.")

        try:
            protected_example["answer_choices"] = self.get_answer_choices_list(example)
        except AttributeError:
            # there's no answer_choices field
            pass

        # Renders the Jinja template
        rendered_example = rtemplate.render(**protected_example)

        # Splits on the separator, and then replaces back any occurrences of the
        # separator in the original example
        statement_text, *_ = rendered_example.split("|||")
        return Template._strip_spaces(self._unescape_pipe(statement_text))

    @staticmethod
    def _strip_spaces(string):
        """Same functionality as str.strip(), but ignores newlines"""

        if string.isspace():
            return "\n" * string.count("\n")

        num_newlines = 0
        # Remove leading whitespace
        while string and string[0].isspace():
            if string[0] == "\n":
                num_newlines += 1
            string = string[1:]

        string = "\n" * num_newlines + string

        num_newlines = 0
        # Remove trailing whitespace
        while string and string[-1].isspace():
            if string[-1] == "\n":
                num_newlines += 1
            string = string[:-1]

        string = string + "\n" * num_newlines

        return string

    pipe_protector = "3ed2dface8203c4c9dfb1a5dc58e41e0"

    @classmethod
    def _escape_pipe(cls, example):
        # Replaces any occurrences of the "|||" separator in the example, which
        # which will be replaced back after splitting
        protected_example = {
            key: (
                value.replace("|||", cls.pipe_protector)
                if isinstance(value, str)
                else value
            )
            for key, value in example.items()
        }
        return protected_example

    @classmethod
    def _unescape_pipe(cls, string):
        # replaces back any occurrences of the separator in a string
        return string.replace(cls.pipe_protector, "|||")

    @dataclass
    class Metadata(yaml.YAMLObject):
        """Metadata for a prompt template."""

        yaml_tag: ClassVar[str] = "!TemplateMetadata"

        original_task: bool | None = None
        """If True, this prompt asks a model to perform the original task designed for
        this dataset."""

        choices_in_prompt: bool | None = None
        """If True, the answer choices are included in the templates such that models
        see those choices in the input. Only applicable to classification tasks."""

        metrics: list[str] | None = None
        """Strings denoting metrics to use for evaluation"""

        languages: list[str] | None = None
        """Strings denoting languages used in the prompt"""


class DatasetTemplates:
    """
    Class that wraps all templates for a specific dataset/subset and implements all the
    helper functions necessary to read/write to the yaml file
    """

    binarize: bool = False
    label_column: str | None
    templates: dict[str, Template]

    def __init__(self, dataset_name: str, subset_name: str | None = None):
        self.dataset_name = dataset_name
        self.subset_name = subset_name

        with open(self.yaml_path, "r") as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

            # Required field; contains all the templates keyed by ID
            self.templates = yaml_dict["templates"]
            for template in self.templates.values():
                if not hasattr(template, "suffix"):
                    template.suffix = ""
            self.binarize = yaml_dict.get("binarize", False)
            self.label_column = yaml_dict.get("label_column")

    def drop_non_mc_templates(self) -> int:
        """Drop all templates that aren't multiple choice, return the number dropped"""
        mc_templates = {
            k: v for k, v in self.templates.items() if v.answer_choices is not None
        }
        if not mc_templates:
            raise ValueError("No multiple choice templates found")

        num_dropped = len(self.templates) - len(mc_templates)
        self.templates = mc_templates

        return num_dropped

    @property
    def all_template_names(self) -> list[str]:
        """
        Sorted list of all templates names for this dataset
        """
        return sorted([template.name for template in self.templates.values()])

    @property
    def folder_path(self) -> str:
        if self.subset_name:
            return os.path.join(
                TEMPLATES_FOLDER_PATH, self.dataset_name, self.subset_name
            )
        else:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)

    @property
    def yaml_path(self) -> str:
        path = os.path.join(self.folder_path, "templates.yaml")
        if not os.path.exists(path):
            raise ValueError(f"Expected prompt templates to exist at {path}")

        return path
