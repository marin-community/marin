# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight task configuration for the LM Evaluation Harness."""

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for one LM Evaluation Harness task.

    Fields left as ``None`` are omitted from the harness task dictionary so the
    harness can apply its own defaults.
    """

    task: str
    task_alias: str | None = None
    num_fewshot: int | None = None

    use_prompt: str | None = None
    description: str | None = None
    target_delimiter: str | None = None
    fewshot_delimiter: str | None = None
    doc_to_text: str | None = None
    doc_to_target: str | None = None
    doc_to_choice: str | None = None

    # Inline task-spec fields bypass registered-task override semantics when a
    # task definition is not present in lm-eval's registry.
    dataset_path: str | None = None
    dataset_name: str | None = None
    output_type: str | None = None
    test_split: str | None = None
    training_split: str | None = None
    validation_split: str | None = None
    fewshot_split: str | None = None
    metric_list: list[dict] | None = None
    tag: list[str] | None = None
    metadata: dict | None = None

    additional_stop_strings: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the task dictionary expected by LM Evaluation Harness."""
        return {key: value for key, value in dataclasses.asdict(self).items() if value is not None}
