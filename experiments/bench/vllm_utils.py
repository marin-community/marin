# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal utilities extracted from vLLM for benchmark compatibility.
"""

import re
import sys
import textwrap
from argparse import (
    Action,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
)
from typing import Any


class PlaceholderModule:
    """
    A placeholder object that raises an ImportError when accessed.
    Used for optional dependencies that aren't available.
    """

    def __init__(self, name: str):
        self._name = name
        self._error_message = (
            f"Module '{name}' is not installed. "
            f"Please install it to use this functionality."
        )

    def __getattr__(self, name: str) -> "PlaceholderModule":
        return PlaceholderModule(f"{self._name}.{name}")

    def placeholder_attr(self, name: str) -> "PlaceholderModule":
        """Return a placeholder for an attribute."""
        return PlaceholderModule(f"{self._name}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(self._error_message)

    def __bool__(self) -> bool:
        return False


class SortedHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        """
        1. Sentences split across lines have their single newlines removed.
        2. Paragraphs and explicit newlines are split into separate lines.
        3. Each line is wrapped to the specified width (width of terminal).
        """
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(" ", text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()
    _json_tip: str = (
        "When passing JSON CLI arguments, the following sets of arguments "
        "are equivalent:\n"
        '   --json-arg \'{"key1": "value1", "key2": {"key3": "value2"}}\'\n'
        "   --json-arg.key1 value1 --json-arg.key2.key3 value2\n\n"
        "Additionally, list elements can be passed individually using +:\n"
        '   --json-arg \'{"key4": ["value3", "value4", "value5"]}\'\n'
        "   --json-arg.key4+ value3 --json-arg.key4+='value4,value5'\n\n"
    )
    _search_keyword: str | None = None

    def __init__(self, *args, **kwargs):
        # Set the default "formatter_class" to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        # Pop kwarg "add_json_tip" to control whether to add the JSON tip
        self.add_json_tip = kwargs.pop("add_json_tip", True)
        super().__init__(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        """Override to convert dashes to underscores in argument names."""
        if args is None:
            args = sys.argv[1:]

        # Convert dashes to underscores in long option names
        converted_args = []
        for arg in args:
            if arg.startswith("--") and "=" in arg:
                key, value = arg.split("=", 1)
                key = key.replace("-", "_")
                converted_args.append(f"{key}={value}")
            elif arg.startswith("--"):
                converted_args.append(arg.replace("-", "_"))
            else:
                converted_args.append(arg)

        return super().parse_args(converted_args, namespace)

    def format_help(self):
        """Override to add JSON tip if enabled."""
        help_text = super().format_help()
        if self.add_json_tip:
            help_text = self._json_tip + help_text
        return help_text
