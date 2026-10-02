# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Ultra selection metadata shared by family recipes and acquisition."""

import re
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

ACTION_COMPARISON_CRITERION = (
    "Only when agent_ref.name is single_step_tool_use_with_argument_comparison_agent, "
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent, or "
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent at verifier revision "
    "d8b6e8c163def3660e9d3072c1c174226a1709fa, expected_action.type=message accepts any nonempty "
    "assistant text with no tool calls; the stored message is not a literal answer key. For "
    "expected_action.type=function_call, the scorer requires one call with the expected name and "
    "recursively matching argument keys and values, allowing floating-point tolerance 1e-6. These "
    "documented rules do not certify a bound runtime."
)


@dataclass(frozen=True)
class NemotronSource:
    name: str
    blend: str
    selector: str
    component: str
    family: str
    atlas_id: str
    upstream: str
    rubric: ReviewRubric
    family_module: str


def recipe_for_source(source: NemotronSource, snapshot: Path) -> DatasetRecipe:
    """Bind a retained snapshot to its original source and reward contract."""
    return recipe(
        source.name,
        snapshot,
        source.blend,
        source.selector,
        source.family,
        source.rubric,
        component=source.component,
    )


def quality_source(
    blend: str,
    selector: str,
    upstream: str,
    family: str,
    criteria: tuple[str, ...],
    family_module: str,
    *,
    component: str | None = None,
) -> NemotronSource:
    """Describe a blend selection with its family and provenance review criteria."""
    component = selector if component is None else component
    name = "nemotron_ultra_" + blend + "_" + re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
    provenance = (
        f"This is the {component} selection from the {blend} training blend, originating at {upstream}. "
        "Judge its actual retained source fields and agent reward contract; do not infer byte "
        "equivalence with another blend."
    )
    return NemotronSource(
        name=name,
        blend=blend,
        selector=selector,
        component=component,
        family=family,
        atlas_id=f"MarinSkyRL:nemotron_ultra_{blend}/{component}",
        upstream=upstream,
        rubric=ReviewRubric(id=name + "-quality", version="1", criteria=(*criteria, provenance)),
        family_module=family_module,
    )


def preference_source(
    blend: str, selector: str, upstream: str, criteria: tuple[str, ...], family_module: str
) -> NemotronSource:
    """Describe the generation-based GenRM contract without inventing preference pairs."""
    name = f"nemotron_ultra_{blend}_{selector}"
    return NemotronSource(
        name=name,
        blend=blend,
        selector=selector,
        component=selector,
        family="preference",
        atlas_id=f"MarinSkyRL:nemotron_ultra_{blend}/{selector}",
        upstream=upstream,
        rubric=ReviewRubric(id=name + "-answerability", version="1", criteria=criteria),
        family_module=family_module,
    )
