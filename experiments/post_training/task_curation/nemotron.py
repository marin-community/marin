# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron Ultra selections and their experiment-owned input bindings."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rigging.filesystem.storage_path import StoragePath
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
from taskcompendium.pipeline.datasets.nemotron_ultra.source import placeholder_record, swe_gym_ids
from taskcompendium.pipeline.inputs import HubDownload, RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, IntendedUse, TaskPipeline

DATASET = "nvidia/Nemotron-RL-Ultra-Training-Blends"
REVISION = "482392c14c6418e26804ea2e5d10359df9877df4"
SWE_GYM_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SWE_GYM_FILE = "swe-gym-membership/data/train-00000-of-00001.parquet"
PLACEHOLDER_PINS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "Skywork/Skywork-OR1-RL-Data": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
}
PLACEHOLDER_FILES = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "placeholder-dapo/data/dapo-math-17k.parquet",
    "Skywork/Skywork-OR1-RL-Data": "placeholder-skywork/data/math-00000-of-00001.parquet",
}
PLACEHOLDER_SPLITS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "train",
    "Skywork/Skywork-OR1-RL-Data": "math",
}


@dataclass(frozen=True)
class ComponentSelection:
    blends: tuple[str, ...]
    selector: str
    component: str
    upstream: str


@dataclass(frozen=True)
class UltraComponentSelector:
    selector: str

    def __call__(self, row: dict[str, Any], staged_root: StoragePath) -> bool:
        identity = row.get("dataset") or "agent:" + row["agent_ref"]["name"]
        return identity == self.selector


@dataclass(frozen=True)
class SWEComponentSelector:
    selector: str
    component: str

    def __call__(self, row: dict[str, Any], staged_root: StoragePath) -> bool:
        if not UltraComponentSelector(self.selector)(row, staged_root):
            return False
        gym_member = row["metadata"]["instance_id"] in swe_gym_ids(staged_root / SWE_GYM_FILE)
        return gym_member == self.component.endswith("/SWE-Gym/SWE-Gym")


def resolve_ultra_placeholder(row: dict[str, Any], staged_root: StoragePath) -> dict[str, Any]:
    """Attach pinned upstream evidence needed to reconstruct Ultra math placeholders."""
    placeholder = row.get("_hf_question_placeholder")
    if placeholder is None:
        return row
    dataset = placeholder["dataset"]
    split = placeholder["split"]
    index = int(placeholder["row"])
    if split != PLACEHOLDER_SPLITS[dataset]:
        raise ValueError(f"Unsupported placeholder split {dataset}/{split}")
    record = placeholder_record(staged_root / PLACEHOLDER_FILES[dataset], index)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    source = {
        "dataset": dataset,
        "revision": PLACEHOLDER_PINS[dataset],
        "split": split,
        "row_index": index,
        "record_sha256": digest,
        "record": record,
    }
    return {**row, "placeholder_source": source}


def _blend_inputs(blend: str, selector: str, _component: str) -> RecipeInputs:
    blend_file = f"{blend}.jsonl"
    return RecipeInputs(
        files=SourceFiles((blend_file,), SourceFormat.JSONL, selector=UltraComponentSelector(selector)),
        downloads=(HubDownload(DATASET, REVISION, (blend_file,)),),
    )


def _math_inputs(blend: str, selector: str, component: str) -> RecipeInputs:
    base = _blend_inputs(blend, selector, component)
    downloads = list(base.downloads)
    for dataset, revision in PLACEHOLDER_PINS.items():
        subdirectory, path = PLACEHOLDER_FILES[dataset].split("/", 1)
        downloads.append(HubDownload(dataset, revision, (path,), subdirectory))
    return RecipeInputs(
        files=SourceFiles(
            base.files.patterns, base.files.format, selector=base.files.selector, decoder=resolve_ultra_placeholder
        ),
        downloads=tuple(downloads),
    )


def _swe_inputs(blend: str, selector: str, component: str) -> RecipeInputs:
    base = _blend_inputs(blend, selector, component)
    return RecipeInputs(
        files=SourceFiles(base.files.patterns, base.files.format, selector=SWEComponentSelector(selector, component)),
        downloads=(
            *base.downloads,
            HubDownload(
                "SWE-Gym/SWE-Gym", SWE_GYM_REVISION, ("data/train-00000-of-00001.parquet",), "swe-gym-membership"
            ),
        ),
    )


InputBuilder = Callable[[str, str, str], RecipeInputs]
QualityPolicy = Callable[[str, str, str], TaskPipeline]


def _quality_sources(
    policy: QualityPolicy,
    selections: tuple[ComponentSelection, ...],
    inputs: InputBuilder,
) -> tuple[DatasetRecipe, ...]:
    sources = []
    for selection in selections:
        for blend in selection.blends:
            name = "nemotron_ultra_" + blend + "_" + re.sub(r"[^a-z0-9]+", "_", selection.component.lower()).strip("_")
            provenance = (
                f"This is the {selection.component} selection from the {blend} training blend, "
                f"originating at {selection.upstream}. Judge its actual retained source fields and agent reward "
                "contract; do not infer byte equivalence with another blend."
            )
            sources.append(
                DatasetRecipe(
                    name=name,
                    version=name + "-v1",
                    source=HFSource(DATASET, REVISION, f"{blend}/{selection.component}", "train"),
                    intended_use=IntendedUse.TRAIN,
                    pipeline=policy(selection.selector, name + "-quality", provenance),
                    inputs=inputs(blend, selection.selector, selection.component),
                )
            )
    return tuple(sources)


def _preference_sources(selections: tuple[ComponentSelection, ...]) -> tuple[DatasetRecipe, ...]:
    sources = []
    for selection in selections:
        for blend in selection.blends:
            name = f"nemotron_ultra_{blend}_{selection.selector}"
            sources.append(
                DatasetRecipe(
                    name=name,
                    version=name + "-v1",
                    source=HFSource(DATASET, REVISION, f"{blend}/{selection.component}", "train"),
                    intended_use=IntendedUse.TRAIN,
                    pipeline=preference.pipeline(selection.selector, name + "-answerability"),
                    inputs=_blend_inputs(blend, selection.selector, selection.component),
                )
            )
    return tuple(sources)


AGENTIC_SAFETY_SOURCES = _quality_sources(
    agentic_safety.pipeline,
    (
        ComponentSelection(
            ("mopd",),
            "makeshn_ultra_v3_ipi_train",
            "makeshn_ultra_v3_ipi_train",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        ),
    ),
    _blend_inputs,
)

ARC_AGI_SOURCES = _quality_sources(
    arc_agi.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_nvarc_inductive",
            "ultra_sft_step3200_nvarc_inductive",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_nvarc_transductive",
            "ultra_sft_step3200_nvarc_transductive",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
        ),
    ),
    _blend_inputs,
)

CHEMISTRY_SOURCES = _quality_sources(
    chemistry.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr2"),
            "ultra_sft_step3200_rdkit",
            "ultra_sft_step3200_rdkit",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Litmus-Bench-v0.1",
        ),
    ),
    _blend_inputs,
)

COMPETITIVE_PROGRAMMING_SOURCES = _quality_sources(
    competitive_programming.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_comp_coding",
            "ultra_sft_step3200_comp_coding",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding",
        ),
    ),
    _blend_inputs,
)

INSTRUCTION_FOLLOWING_SOURCES = _quality_sources(
    instruction_following.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_calendar_v2",
            "ultra_sft_step3200_calendar_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
        ),
        ComponentSelection(
            ("rlvr2",),
            "ultra_sft_step3200_ds2_freeform",
            "ultra_sft_step3200_ds2_freeform",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        ),
        ComponentSelection(
            ("rlvr2",),
            "ultra_sft_step3200_ds3_citation",
            "ultra_sft_step3200_ds3_citation",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_instruction_following",
            "ultra_sft_step3200_instruction_following",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_multichallenge_len40k",
            "ultra_sft_step3200_multichallenge_len40k",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_structured_outputs_v2",
            "ultra_sft_step3200_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        ComponentSelection(
            ("rlvr2",),
            "ultra_sft_step3200_structured_outputs_v3",
            "ultra_sft_step3200_structured_outputs_v3",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        ComponentSelection(
            ("mopd",),
            "ultra_v3_agentic_rl_step73_citation_format_v2",
            "ultra_v3_agentic_rl_step73_citation_format_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        ),
        ComponentSelection(
            ("mopd",),
            "ultra_v3_agentic_rl_step73_freeform_text_v2",
            "ultra_v3_agentic_rl_step73_freeform_text_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        ),
        ComponentSelection(
            ("mopd",),
            "ultra_v3_agentic_rl_step73_structured_outputs_v2",
            "ultra_v3_agentic_rl_step73_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
    ),
    _blend_inputs,
)

MATH_ANSWER_SOURCES = _quality_sources(
    math_answer.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_math_cot",
            "ultra_sft_step3200_math_cot",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_math_tir",
            "ultra_sft_step3200_math_tir",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
        ),
    ),
    _math_inputs,
)

MATH_PROOF_SOURCES = _quality_sources(
    math_proof.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_lean",
            "ultra_sft_step3200_lean",
            "https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1",
        ),
    ),
    _blend_inputs,
)

PREFERENCE_SOURCES = _preference_sources(
    (
        ComponentSelection(
            ("mopd",), "hs3_en", "hs3_en", "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
        ),
        ComponentSelection(
            ("mopd",), "hs3_multi", "hs3_multi", "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
        ),
        ComponentSelection(
            ("mopd",), "hs3_multiturn", "hs3_multiturn", "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
        ),
        ComponentSelection(
            ("rlvr1", "rlvr2"),
            "language_mixing_hs3_ultra_genrm_fmt",
            "language_mixing_hs3_ultra_genrm_fmt",
            "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
        ),
        ComponentSelection(
            ("mopd",), "safety_en", "safety_en", "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
        ),
    ),
)

QA_ABSTENTION_SOURCES = _quality_sources(
    qa_abstention.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_abstention",
            "ultra_sft_step3200_abstention",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1",
        ),
    ),
    _blend_inputs,
)

QA_MULTIPLE_CHOICE_SOURCES = _quality_sources(
    qa_multiple_choice.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_stem_mcqa",
            "ultra_sft_step3200_stem_mcqa",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_stem_mcqa_cot_rima_new",
            "ultra_sft_step3200_stem_mcqa_cot_rima_new",
            "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
        ),
    ),
    _blend_inputs,
)

REASONING_GYM_SOURCES = _quality_sources(
    reasoning_gym.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_reasoning_gym",
            "ultra_sft_step3200_reasoning_gym",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1",
        ),
    ),
    _blend_inputs,
)

SAFETY_SOURCES = _quality_sources(
    safety.pipeline,
    (
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_jailbreak",
            "ultra_sft_step3200_jailbreak",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1",
        ),
    ),
    _blend_inputs,
)

SWE_REPO_SOURCES = _quality_sources(
    swe_repo.pipeline,
    (
        ComponentSelection(
            ("mopd",),
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        ),
        ComponentSelection(
            ("mopd",),
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        ),
        ComponentSelection(
            ("mopd",),
            "swe_pivot_len40k",
            "swe_pivot_len40k/SWE-Gym/SWE-Gym",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        ),
        ComponentSelection(
            ("mopd",),
            "swe_pivot_len40k",
            "swe_pivot_len40k/nebius/SWE-rebench-V2",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        ),
        ComponentSelection(
            ("rlvr1", "rlvr2"),
            "ultra_sft_step3200_swe_pivot_len40k",
            "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        ),
        ComponentSelection(
            ("rlvr1", "rlvr2"),
            "ultra_sft_step3200_swe_pivot_len40k",
            "ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        ),
        ComponentSelection(
            ("mopd",),
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
            "https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
        ),
        ComponentSelection(
            ("mopd",),
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
            "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        ),
    ),
    _swe_inputs,
)

TOOL_USE_SOURCES = _quality_sources(
    tool_use.pipeline,
    (
        ComponentSelection(
            ("rlvr1", "rlvr2"),
            "ultra_sft_step3200_tau_pivot",
            "ultra_sft_step3200_tau_pivot",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
        ),
        ComponentSelection(
            ("mopd", "rlvr1", "rlvr2"),
            "ultra_sft_step3200_toolcall_schema",
            "ultra_sft_step3200_toolcall_schema",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        ),
    ),
    _blend_inputs,
)


RECIPES: dict[str, DatasetRecipe] = dict(
    sorted(
        (source.name, source)
        for source in (
            *AGENTIC_SAFETY_SOURCES,
            *ARC_AGI_SOURCES,
            *CHEMISTRY_SOURCES,
            *COMPETITIVE_PROGRAMMING_SOURCES,
            *INSTRUCTION_FOLLOWING_SOURCES,
            *MATH_ANSWER_SOURCES,
            *MATH_PROOF_SOURCES,
            *PREFERENCE_SOURCES,
            *QA_ABSTENTION_SOURCES,
            *QA_MULTIPLE_CHOICE_SOURCES,
            *REASONING_GYM_SOURCES,
            *SAFETY_SOURCES,
            *SWE_REPO_SOURCES,
            *TOOL_USE_SOURCES,
        )
    )
)
