# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The output of a template converter."""

from collections.abc import Callable
from dataclasses import dataclass, field

from experiments.post_training.tasktrove.taskbinary import TaskFiles
from experiments.post_training.tasktrove.verifier_spec import ImageTier, VerifierSpec


@dataclass(frozen=True)
class ConvertedTask:
    instruction: str
    verifier: VerifierSpec
    tier: ImageTier
    data_files: dict[str, bytes] = field(default_factory=dict)
    """Files the verifier spec references, relative to the task root (e.g. ``tests/cases/...``)."""
    solution_files: dict[str, bytes] = field(default_factory=dict)
    """Oracle solution, kept for validation gates and never mounted for the agent."""
    repo_setup: str | None = None
    """SWE tier only: Dockerfile RUN block that prepares the repository."""
    agent_timeout: float = 900.0
    verifier_timeout: float = 120.0
    metadata: dict = field(default_factory=dict)


Converter = Callable[[TaskFiles], ConvertedTask]
