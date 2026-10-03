# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Ultra source selections; family modules own metadata and rubrics."""

from taskcompendium.pipeline.datasets.nemotron_ultra import (
    agentic_safety,
    arc_agi,
    chemistry,
    competitive_programming,
    instruction_following,
    math_answer,
    math_proof,
    preference,
    qa_abstention,
    qa_multiple_choice,
    reasoning_gym,
    safety,
    swe_repo,
    tool_use,
)
from taskcompendium.pipeline.datasets.nemotron_ultra.source import NemotronSource

_FAMILY_SOURCES = (
    *agentic_safety.SOURCES,
    *arc_agi.SOURCES,
    *chemistry.SOURCES,
    *competitive_programming.SOURCES,
    *instruction_following.SOURCES,
    *math_answer.SOURCES,
    *math_proof.SOURCES,
    *preference.SOURCES,
    *qa_abstention.SOURCES,
    *qa_multiple_choice.SOURCES,
    *reasoning_gym.SOURCES,
    *safety.SOURCES,
    *swe_repo.SOURCES,
    *tool_use.SOURCES,
)

NEMOTRON_SOURCES: dict[str, NemotronSource] = dict(sorted((source.name, source) for source in _FAMILY_SOURCES))
