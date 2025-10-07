"""
Data filtering module for processing datasets to remove or replace problematic content.

This module provides a flexible framework for filtering dataset examples based on
configurable patterns. It supports different strategies like removing examples entirely
or replacing problematic content with alternative text.

Example usage:
    from marin.processing.data_filter import DataFilter, FilterPattern

    # Create filter configuration
    filter_config = DataFilter(
        strategy="replace",
        patterns=[
            FilterPattern(r"My name is Tulu", "I am an AI assistant"),
            FilterPattern(r"Ai2", "my developers"),
        ]
    )

    # Apply to dataset examples
    filtered_examples = apply_data_filter(examples, filter_config)
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from rich.progress import track


@dataclass
class FilterPattern:
    """A single filter pattern with its replacement strategy.

    Args:
        pattern: Regex pattern to match against content
        replacement: Text to replace matches with
        field: Field name to apply pattern to (currently only 'content' supported)
    """

    pattern: str
    replacement: str
    field: str = "content"


@dataclass
class DataFilter:
    """Configuration for dataset filtering.

    Args:
        strategy: How to handle examples with problematic content:
            - "remove": Delete the entire example
            - "replace": Replace problematic content with alternative text
        patterns: List of FilterPattern objects defining what to filter
    """

    strategy: Literal["remove", "replace"] = "replace"
    patterns: list[FilterPattern] = field(default_factory=list)

    def __post_init__(self):
        """Compile regex patterns for better performance."""
        self._compiled_patterns = [(re.compile(p.pattern, re.IGNORECASE), p.replacement, p.field) for p in self.patterns]

    def should_filter_example(self, example: dict) -> bool:
        """Check if this example contains any problematic patterns.

        Args:
            example: Example dict with 'messages' field containing conversation

        Returns:
            True if example contains problematic patterns, False otherwise
        """
        messages = example.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                for pattern, _, field in self._compiled_patterns:
                    if pattern.search(content):
                        return True
        return False

    def apply_filter(self, example: dict) -> dict | None:
        """Apply filtering to an example.

        Args:
            example: Example dict to filter

        Returns:
            Filtered example dict, or None if example should be removed
        """
        if not self.should_filter_example(example):
            return example

        if self.strategy == "remove":
            return None

        filtered_example = example.copy()
        filtered_messages = []

        for message in example.get("messages", []):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                for pattern, replacement, field in self._compiled_patterns:
                    content = pattern.sub(replacement, content)

                filtered_message = message.copy()
                filtered_message["content"] = content
                filtered_messages.append(filtered_message)
            else:
                filtered_messages.append(message)

        filtered_example["messages"] = filtered_messages
        return filtered_example


def apply_data_filter(examples: list[dict], filter_config: DataFilter | None) -> list[dict]:
    """Apply data filter to a list of examples.

    Args:
        examples: List of example dicts to filter
        filter_config: Filter configuration, or None to skip filtering

    Returns:
        List of filtered examples (may be shorter if examples were removed)
    """
    if filter_config is None:
        return examples

    filtered_examples = []
    removed_count = 0
    modified_count = 0

    for example in track(examples, description="Filtering examples"):
        original_example = example
        filtered_example = filter_config.apply_filter(example)

        if filtered_example is None:
            removed_count += 1
        else:
            filtered_examples.append(filtered_example)
            if filtered_example != original_example:
                modified_count += 1

    if removed_count > 0 or modified_count > 0:
        print(
            f"Data filtering results: {len(examples)} -> {len(filtered_examples)} examples "
            f"({removed_count} removed, {modified_count} modified)"
        )

    return filtered_examples
