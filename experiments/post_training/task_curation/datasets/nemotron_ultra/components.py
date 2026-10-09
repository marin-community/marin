# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Ultra RL training blends: one declaration per blend component.

Each blend (``mopd``, ``rlvr1``, ``rlvr2``) is one JSONL file mixing components; a row's component
is its ``dataset`` field, or ``agent:<name>`` for rows keyed by their NeMo Gym agent. SWE components
are further split by whether the row's instance belongs to SWE-Gym. ``COMPONENTS`` maps each
component path to its converter, rubric, controls, grader environment and shipped scorer directories;
``BLENDS`` lists the paths each blend carries. Converters live in ``graders.py``, except NVARC
(``datasets/arc``), which shares its scorer with the TaskTrove ARC sources.

Selection and decoding are identified by name and parameters, not by this module's bytes; bump the
affected versions when they change.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.nemotron_ultra import PLACEHOLDER_FIELD, blend_component
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Controls, Converter, IntendedUse
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.datasets.arc.arc import ARC_SHIPS, ULTRA_ARC_CONTROLS, convert_ultra_arc
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.nemotron_ultra.graders import (
    CODE_CONTROLS,
    CODE_SHIPS,
    DAPO,
    MATH_CONTROLS,
    MCQA_CONTROLS,
    PLACEHOLDER_SOURCE_FIELD,
    RDKIT_CONTROLS,
    REASONING_GYM_CONTROLS,
    REPLY_CONTROLS,
    SHIPS,
    TOOL_ACTION_CONTROLS,
    VERIFIER_REVISION,
    convert_calendar,
    convert_code,
    convert_format,
    convert_instruction_following,
    convert_math,
    convert_mcqa,
    convert_rdkit,
    convert_reasoning_gym,
    convert_structured_output,
    convert_tool_action,
    convert_ungraded,
    convert_ungraded_agent,
)
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, RowDecoder, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

SKYRL_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="MarinSkyRL",
    dataset_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
    revision="e44c4bfcb62c489286a1264094e6d9c883aaf0d2",
    revised_at="2026-10-08T05:59:16Z",
    dataset_revision="482392c14c6418e26804ea2e5d10359df9877df4",
    environment="nemotron_ultra",
    count_precision="exact",
    notes="Complete-file count of this component selection; not the original component repository size.",
    benchmark_basis="SkyRL test-only designation or HF benchmark:official tag; false means no designation found",
    family_basis="Actual record dataset/agent contract and component card, audited 2026-09-28",
    family_url=(
        "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/79f8eda"
        "15ea12e1adf7bb14dcb338a29d391b80e/README.md"
    ),
    classification_basis="Actual selected record population; Agentic tasks use Multi-turn",
    canonical_url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends",
    provenance_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c8"
        "83aaf0d2/infra/rl_data/sources.py"
    ),
    license=("cc-by-4.0",),
    verification="row_selected",
    snapshot_safe=True,
    gym_alias="gym/nemotron_ultra",
    gym_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c883a"
        "af0d2/skyrl-gym/skyrl_gym/envs/__init__.py"
    ),
    gym_entrypoint="skyrl_gym.envs.nemotron_ultra.env:NemotronUltraEnv",
    dataset_revised_at="2026-09-29T05:46:42.000Z",
    registry_revised_at="2026-10-01T14:18:17Z",
    verifier_revised_at="2026-10-08T05:59:16Z",
    revision_basis="Latest upstream dataset repository or MarinSkyRL verifier change",
    recorded_at="2026-10-08",
)

ULTRA_REPO = "nvidia/Nemotron-RL-Ultra-Training-Blends"
ULTRA_REVISION = "482392c14c6418e26804ea2e5d10359df9877df4"
NAME_PREFIX = "nemotron_ultra_"


SWE_GYM = HfSource(
    "SWE-Gym/SWE-Gym",
    "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
    ("data/train-00000-of-00001.parquet",),
    SourceFormat.PARQUET,
)
SKYWORK = "Skywork/Skywork-OR1-RL-Data"
PLACEHOLDER_INPUTS = {
    DAPO: HfSource(
        DAPO, "65877096c24ffa7abc4e4fa5edb95cf3413a5674", ("data/dapo-math-17k.parquet",), SourceFormat.PARQUET
    ),
    SKYWORK: HfSource(
        SKYWORK, "1cdedc52e0e2db85fdf252f9be682e63a5a38c33", ("data/math-00000-of-00001.parquet",), SourceFormat.PARQUET
    ),
}
PLACEHOLDER_SPLITS = {DAPO: "train", SKYWORK: "math"}


@lru_cache(maxsize=1)
def swe_gym_ids(path: StoragePath) -> frozenset[str]:
    return frozenset(row["instance_id"] for row in load_parquet(str(path)))


@lru_cache(maxsize=1024)
def placeholder_record(path: StoragePath, index: int) -> dict[str, Any]:
    records = list(load_parquet(InputFileSpec(path=str(path), row_start=index, row_end=index + 1)))
    if len(records) != 1:
        raise ValueError(f"Placeholder file {path} omitted row {index}")
    return records[0]


@dataclass(frozen=True)
class ComponentRows:
    """Select the rows of one blend component."""

    component: str

    def __call__(self, row: dict[str, Any], _context: ConversionContext) -> bool:
        return blend_component(row) == self.component


class SweSplit(StrEnum):
    SWE_GYM = "SWE-Gym/SWE-Gym"
    SWE_REBENCH = "nebius/SWE-rebench-V2"


@dataclass(frozen=True)
class SweRows:
    """Select a SWE component's rows from SWE-Gym, or from SWE-rebench (every other instance)."""

    component: str
    split: SweSplit

    def __call__(self, row: dict[str, Any], context: ConversionContext) -> bool:
        if blend_component(row) != self.component:
            return False
        members = swe_gym_ids(context.inputs[SWE_GYM.repo] / SWE_GYM.files[0])
        return (row["metadata"]["instance_id"] in members) == (self.split == SweSplit.SWE_GYM)


def attach_placeholder_source(row: dict[str, Any], context: ConversionContext) -> dict[str, Any]:
    """Attach the pinned upstream record that a question placeholder points at."""
    placeholder = row.get(PLACEHOLDER_FIELD)
    if placeholder is None:
        return row
    dataset = placeholder["dataset"]
    split = placeholder["split"]
    index = int(placeholder["row"])
    if split != PLACEHOLDER_SPLITS[dataset]:
        raise ValueError(f"Unsupported placeholder split {dataset}/{split}")
    source = PLACEHOLDER_INPUTS[dataset]
    record = placeholder_record(context.inputs[dataset] / source.files[0], index)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        **row,
        PLACEHOLDER_SOURCE_FIELD: {
            "dataset": dataset,
            "revision": source.revision,
            "split": split,
            "row_index": index,
            "record_sha256": digest,
            "record": record,
        },
    }


ACTION_COMPARISON_CRITERION = f"""
Only when agent_ref.name is single_step_tool_use_with_argument_comparison_agent,
swe_pivot_single_step_tool_use_with_argument_comparison_agent, or
toolcall_schema_single_step_tool_use_with_argument_comparison_agent at verifier revision {VERIFIER_REVISION},
expected_action.type=message accepts any nonempty assistant text with no tool calls; the stored message is not a
literal answer key. For expected_action.type=function_call, the scorer requires one call with the expected name and
recursively matching argument keys and values, allowing floating-point tolerance 1e-6.
"""

AGENTIC_SAFETY_RUBRIC = """
Preserve trusted instructions, tool schemas, full tool observations, initial environment, and attacker injection
boundaries.

Injected instructions are intentional untrusted observations; compare the requested legitimate objective with
verifier_config without treating the injection as authoritative or exposing hidden evaluator goals.
"""

ARC_RUBRIC = """
Inspect the source scorer's final-stdout acceptance: a valid grid can receive reward even after a nonzero candidate
exit. Report this source grading weakness when relevant; do not silently replace the source reward
with a stricter process-exit rule.

Check that all training grids, public test inputs, and hidden expected_output match the stated grid transformation
and dimensions.

Inductive variants require producing a reusable transformation program; transductive variants request the output
grid. Do not replace one grading contract with the other or expose hidden outputs.
"""

CHEMISTRY_RUBRIC = """
Check public molecular inputs and requested properties against retained target/validator fields and their units or
format.

The pinned rdkit_chemistry_agent grades the stored numeric property target; it does not recompute molecular
properties or test stereochemical equivalence. It requires the row-selected boxed or double-parentheses answer wrapper
and compares Python-rounded predicted and expected values. Check whether the public request agrees with these
extraction and scoring rules.
"""

COMPETITIVE_PROGRAMMING_RUBRIC = """
Check complete input/output definitions, boundaries, examples, and consistency with retained unit_tests.

Special judges, alternative valid constructions, and function versus stdio delivery must retain their source
contracts. No reference solution or unavailable execution alone is a quality defect.
"""

INSTRUCTION_FOLLOWING_RUBRIC = """
Identify the underlying content request and verify that all supplied formal constraints and semantic rubric
requirements can hold together.

Preserve every conversation turn. Public schemas/examples are legitimate context. Distinguish factual extraction from
authorized arbitrary schema generation, and compare hidden constraints with public instructions.
"""

MATH_ANSWER_RUBRIC = """
Verify complete mathematical inputs and agreement of expected_answer with the actual public problem; difficulty alone
is not a defect.

A symbolic comparator grades the reply's last boxed expression, or its last line, against expected_answer and
accepts equivalent forms; the source's LLM-judge fallback does not run. Flag an expected_answer that only a judge
could compare, such as prose or a choice among alternatives. Unresolved external question placeholders are
acquisition gaps.
"""

MATH_PROOF_RUBRIC = """
Check that the complete Lean header, formal_statement, imports, and holes to be filled are present or available
through the stated environment.

Hard proofs and absent reference proofs alone are not defects. Verify the formal target agrees with informal text and
preserve exact Lean/toolchain requirements.
"""

PREFERENCE_CRITERIA = """
These are generation prompts with a GenRM principle, not stored chosen/rejected pairs; do not invent pair labels.

The original principle and agent settings remain hidden grader evidence; no exact-answer key is supplied.

Reject missing context or contradictory requirements, separating those defects from the unavailable GenRM evaluator.

Do not assume records in another blend with the same selector are byte-identical or a verified alias.
"""

QA_ABSTENTION_RUBRIC = """
Check whether the actual question is answerable, and whether the hidden answer is correct.

Abstention rules and any [IDK] output requirements belong to the source contract; do not substitute exact-only
matching for its semantic evaluator.
"""

QA_MULTIPLE_CHOICE_RUBRIC = """
Check option labels and answer encoding against the complete public choices and expected_answer.

Knowledge questions can use ordinary external knowledge. Missing referenced passages, images, or material
contradictions are defects; source labels must not be exposed as public hints.
"""

REASONING_GYM_RUBRIC = """
Compare the complete question and hidden answer/metadata, checking cheap contradictions and missing puzzle context.

The source_dataset can determine scoring, aliases and partial credit. A hard puzzle or several surface forms of the
same answer is not automatically a defect.
"""

SAFETY_RUBRIC = """
Judge whether the actual public request and source response_policy_mapped define a coherent response objective.

Adversarial or jailbreak text is intentional task input; assess reference contradictions with the safety rules and
impossible instructions rather than treating adversarial wording itself as corruption.
"""

SWE_REPO_RUBRIC = f"""
The public repository/environment reference supplies code context; a pinned checkout and supplied issue can be
coherent without inline repository files.

Compare expected_action, ref_patch, issue, historical observations and environment to detect unrelated hidden repair
requirements. Preserve SWE-Gym versus SWE-rebench attribution; a shared agent selector is not proof of source
equivalence.
{ACTION_COMPARISON_CRITERION}"""

TOOL_USE_RUBRIC = f"""
Check the complete role/tool sequence and advertised schemas against expected_action, scenario, and source
environment state.

Historical observations are public context. Expected future tool calls and reward state are hidden evidence; multiple
valid actions require the original comparison rules rather than invented exact matching.
{ACTION_COMPARISON_CRITERION}"""


def preference_rubric(scope: str) -> str:
    return f"{scope}\n{PREFERENCE_CRITERIA}"


@dataclass(frozen=True)
class Component:
    """How one component's rows become tasks; ``decode`` and ``inputs`` restore placeholder questions.

    ``grader`` is what the component's grade script needs and ``ships`` holds the scorer directories it
    packages; components graded in process or kept as ``NoGrader`` contracts name neither.
    """

    convert: Converter
    rubric: str
    controls: Controls | None = None
    decode: RowDecoder | None = None
    inputs: Mapping[str, HfSource] = field(default_factory=dict)
    grader: Environment | None = None
    ships: tuple[Path, ...] = ()


def scored(
    convert: Converter,
    rubric: str,
    controls: Controls = REPLY_CONTROLS,
    ships: tuple[Path, ...] = SHIPS,
) -> Component:
    """A component whose grade script needs the grader packages."""
    return Component(convert, rubric, controls, grader=GRADER_PACKAGES, ships=ships)


ABSTENTION = Component(convert_ungraded, QA_ABSTENTION_RUBRIC)
AGENTIC_SAFETY = Component(convert_ungraded_agent, AGENTIC_SAFETY_RUBRIC)
CALENDAR = scored(convert_calendar, INSTRUCTION_FOLLOWING_RUBRIC)
COMPETITIVE_CODE = scored(convert_code, COMPETITIVE_PROGRAMMING_RUBRIC, CODE_CONTROLS, CODE_SHIPS)
FORMAT = scored(convert_format, INSTRUCTION_FOLLOWING_RUBRIC)
INSTRUCTION_FOLLOWING = scored(convert_instruction_following, INSTRUCTION_FOLLOWING_RUBRIC)
MATH = Component(
    convert_math, MATH_ANSWER_RUBRIC, MATH_CONTROLS, decode=attach_placeholder_source, inputs=PLACEHOLDER_INPUTS
)
MATH_PROOF = Component(convert_ungraded, MATH_PROOF_RUBRIC)
MCQA = scored(convert_mcqa, QA_MULTIPLE_CHOICE_RUBRIC, MCQA_CONTROLS)
MULTICHALLENGE = Component(convert_ungraded, INSTRUCTION_FOLLOWING_RUBRIC)
NEXT_ACTION = scored(convert_tool_action, SWE_REPO_RUBRIC, TOOL_ACTION_CONTROLS)
NVARC = scored(convert_ultra_arc, ARC_RUBRIC, ULTRA_ARC_CONTROLS, ARC_SHIPS)
RDKIT = scored(convert_rdkit, CHEMISTRY_RUBRIC, RDKIT_CONTROLS)
REASONING_GYM = scored(convert_reasoning_gym, REASONING_GYM_RUBRIC, REASONING_GYM_CONTROLS)
SAFETY = Component(convert_ungraded, SAFETY_RUBRIC)
STRUCTURED_OUTPUT = scored(convert_structured_output, INSTRUCTION_FOLLOWING_RUBRIC)
SWE_REPO = Component(convert_ungraded_agent, SWE_REPO_RUBRIC)
TAU_PIVOT = Component(convert_ungraded_agent, TOOL_USE_RUBRIC)
TOOLCALL_SCHEMA = scored(convert_tool_action, TOOL_USE_RUBRIC, TOOL_ACTION_CONTROLS)

HS3_EN = Component(
    convert_ungraded,
    preference_rubric(
        "Assess the full English conversation and hidden GenRM principle; earlier assistant errors are context, not a "
        "new gold answer."
    ),
)
HS3_MULTI = Component(
    convert_ungraded,
    preference_rubric(
        "Preserve every multilingual turn and assess the final request in its actual language; multilingual context "
        "alone is not incoherent."
    ),
)
HS3_MULTITURN = Component(
    convert_ungraded,
    preference_rubric(
        "Follow the complete conversation and earlier requirements; do not judge only the final short request without "
        "its history."
    ),
)
LANGUAGE_MIXING = Component(
    convert_ungraded,
    preference_rubric(
        "Check the actual language instructions against the full multilingual history and hidden evaluation principle."
    ),
)
SAFETY_PREFERENCE = Component(
    convert_ungraded,
    preference_rubric(
        "Compare the request with its safety principle: a benign craft request mentioning a gun may call for a glue "
        "gun and helpful guidance."
    ),
)

NEXT_ACTION_COMPONENT = "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent"

COMPONENTS: dict[str, Component] = {
    "hs3_en": HS3_EN,
    "hs3_multi": HS3_MULTI,
    "hs3_multiturn": HS3_MULTITURN,
    "language_mixing_hs3_ultra_genrm_fmt": LANGUAGE_MIXING,
    "makeshn_ultra_v3_ipi_train": AGENTIC_SAFETY,
    "safety_en": SAFETY_PREFERENCE,
    f"{NEXT_ACTION_COMPONENT}/{SweSplit.SWE_GYM}": NEXT_ACTION,
    f"{NEXT_ACTION_COMPONENT}/{SweSplit.SWE_REBENCH}": NEXT_ACTION,
    f"swe_pivot_len40k/{SweSplit.SWE_GYM}": SWE_REPO,
    f"swe_pivot_len40k/{SweSplit.SWE_REBENCH}": SWE_REPO,
    "ultra_sft_step3200_abstention": ABSTENTION,
    "ultra_sft_step3200_calendar_v2": CALENDAR,
    "ultra_sft_step3200_comp_coding": COMPETITIVE_CODE,
    "ultra_sft_step3200_ds2_freeform": FORMAT,
    "ultra_sft_step3200_ds3_citation": FORMAT,
    "ultra_sft_step3200_instruction_following": INSTRUCTION_FOLLOWING,
    "ultra_sft_step3200_jailbreak": SAFETY,
    "ultra_sft_step3200_lean": MATH_PROOF,
    "ultra_sft_step3200_math_cot": MATH,
    "ultra_sft_step3200_math_tir": MATH,
    "ultra_sft_step3200_multichallenge_len40k": MULTICHALLENGE,
    "ultra_sft_step3200_nvarc_inductive": NVARC,
    "ultra_sft_step3200_nvarc_transductive": NVARC,
    "ultra_sft_step3200_rdkit": RDKIT,
    "ultra_sft_step3200_reasoning_gym": REASONING_GYM,
    "ultra_sft_step3200_stem_mcqa": MCQA,
    "ultra_sft_step3200_stem_mcqa_cot_rima_new": MCQA,
    "ultra_sft_step3200_structured_outputs_v2": STRUCTURED_OUTPUT,
    "ultra_sft_step3200_structured_outputs_v3": STRUCTURED_OUTPUT,
    f"ultra_sft_step3200_swe_pivot_len40k/{SweSplit.SWE_GYM}": SWE_REPO,
    f"ultra_sft_step3200_swe_pivot_len40k/{SweSplit.SWE_REBENCH}": SWE_REPO,
    "ultra_sft_step3200_tau_pivot": TAU_PIVOT,
    "ultra_sft_step3200_toolcall_schema": TOOLCALL_SCHEMA,
    "ultra_v3_agentic_rl_step73_citation_format_v2": FORMAT,
    "ultra_v3_agentic_rl_step73_freeform_text_v2": FORMAT,
    "ultra_v3_agentic_rl_step73_structured_outputs_v2": STRUCTURED_OUTPUT,
    f"ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/{SweSplit.SWE_GYM}": SWE_REPO,
    f"ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/{SweSplit.SWE_REBENCH}": SWE_REPO,
}


MOPD_FILE_SHA256 = "c6c94da041c6ef2e2e7b057f10817418f134ed46a33a274eede892fc089f0f28"
RLVR1_FILE_SHA256 = "b73d72788ecf13ab15457ea475f4ee619eccb1a6c0e211be999ffcf835c10ccf"
RLVR2_FILE_SHA256 = "fc3987d28bc8cd942c9012388dd25808582f5d41b3d79e9514f90c574b23dfd3"
BLENDS: dict[str, dict[str, DataSourceMetadata]] = {
    "mopd": {
        "hs3_en": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/hs3_en",
            name="nemotron_ultra_mopd/hs3_en",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · hs3_en · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="4fa7989c106455b584397fb18b72ca84f44f10b3cf122c6d24c4d325a9c2f308",
            family="preference",
            type="Alignment",
            turns="Single-turn",
            task_count=1281,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="hs3_en",
            component_selector="hs3_en",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.4899%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "hs3_multi": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/hs3_multi",
            name="nemotron_ultra_mopd/hs3_multi",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · hs3_multi · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="110a1af331809e56e1268bcdd9f524a11dd0f69b5a0e7adca218c422e23d7e2e",
            family="preference",
            type="Alignment",
            turns="Single-turn",
            task_count=962,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="hs3_multi",
            component_selector="hs3_multi",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.1189%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "hs3_multiturn": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/hs3_multiturn",
            name="nemotron_ultra_mopd/hs3_multiturn",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · hs3_multiturn · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="0f8e77dcd20b0046901d41e55bb31278c7e50da693ec5a1a3c36e485da86b0d0",
            family="preference",
            type="Alignment",
            turns="Multi-turn",
            task_count=1243,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="hs3_multiturn",
            component_selector="hs3_multiturn",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.4457%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "makeshn_ultra_v3_ipi_train": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
            name="nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
            display_name="nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
            verifier_revision="3c22687383fb3360c57685b0f92a2123f94f4e1e",
            family="agentic-safety",
            type="Agentic",
            turns="Multi-turn",
            task_count=2000,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="makeshn_ultra_v3_ipi_train",
            component_selector="makeshn_ultra_v3_ipi_train",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.3261%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "safety_en": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/safety_en",
            name="nemotron_ultra_mopd/safety_en",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · safety_en · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="b2b02d239d99ad683992c24f962736ba8b85cc2a04084e3174657820d0751e13",
            family="preference",
            type="Alignment",
            turns="Single-turn",
            task_count=629,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="safety_en",
            component_selector="safety_en",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="0.7316%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym": replace(
            SKYRL_METADATA,
            id=(
                "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_com"
                "parison_agent/SWE-Gym/SWE-Gym"
            ),
            name=(
                "nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_ag"
                "ent/SWE-Gym/SWE-Gym"
            ),
            display_name="SWE-Gym/SWE-Gym · unlabeled dataset records · mopd",
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            verifier_revision="3b04884dd2139714321f8e06416596dd5e60dd79d6cb4c3aab2a86b52e96bcdb",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=1884,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
            component_selector="agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.1912%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2": replace(
            SKYRL_METADATA,
            id=(
                "MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_com"
                "parison_agent/nebius/SWE-rebench-V2"
            ),
            name=(
                "nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_ag"
                "ent/nebius/SWE-rebench-V2"
            ),
            display_name="nebius/SWE-rebench-V2 · unlabeled dataset records · mopd",
            url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            verifier_revision="5f19be9f7240c4b6cc597922b4037fb08648738cd6593c4ff18a98ccc94e792b",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=5307,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/nebius/SWE-rebench-V2",
            component_selector="agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="6.1724%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "swe_pivot_len40k/SWE-Gym/SWE-Gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
            name="nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym",
            display_name="SWE-Gym/SWE-Gym · pivot records · mopd",
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            verifier_revision="8d531a83a74e477b9adae6dd7cf2af3af1e17800a051ebdabe37f468acee61ac",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=860,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="swe_pivot_len40k/SWE-Gym/SWE-Gym",
            component_selector="swe_pivot_len40k",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.0002%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "swe_pivot_len40k/nebius/SWE-rebench-V2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
            name="nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
            display_name="nebius/SWE-rebench-V2 · pivot records · mopd",
            url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            verifier_revision="011ce022c8a4ec20f483945ae34e42169f0034da9d65e980e509392f6ad913c3",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=3903,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="swe_pivot_len40k/nebius/SWE-rebench-V2",
            component_selector="swe_pivot_len40k",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="4.5394%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_sft_step3200_abstention": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_abstention",
            name="nemotron_ultra_mopd/ultra_sft_step3200_abstention",
            display_name="nvidia/Nemotron-RL-QA-Abstention-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1",
            verifier_revision="e3e85a847527d0ace97e1a6459ce92b3ce54d5278ac6f5a4eae7e6da9a564623",
            family="qa-abstention",
            type="RLVR",
            turns="Single-turn",
            task_count=4025,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_abstention",
            component_selector="ultra_sft_step3200_abstention",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="4.6813%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_calendar_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_calendar_v2",
            name="nemotron_ultra_mopd/ultra_sft_step3200_calendar_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Calendar-v2 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
            verifier_revision="5c9656671baf79a9711cd5c7ecbacffc6d1a169129b81ab098c00a4be8838833",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=910,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_calendar_v2",
            component_selector="ultra_sft_step3200_calendar_v2",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.0584%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_comp_coding": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_comp_coding",
            name="nemotron_ultra_mopd/ultra_sft_step3200_comp_coding",
            display_name="nvidia/Nemotron-RL-coding-competitive_coding · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding",
            verifier_revision="0878a10ef2dd4c89daf18e325b48720553c3f5f0ab60d99d233b4520db3e7684",
            family="competitive-programming",
            type="RLVR",
            turns="Single-turn",
            task_count=7929,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_comp_coding",
            component_selector="ultra_sft_step3200_comp_coding",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="9.2219%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_instruction_following": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_instruction_following",
            name="nemotron_ultra_mopd/ultra_sft_step3200_instruction_following",
            display_name="nvidia/Nemotron-RL-instruction_following · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
            verifier_revision="f0571dbf560eaceae2609645364cb6df639318e701a31e0cb55303adfa6539bc",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=10409,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_instruction_following",
            component_selector="ultra_sft_step3200_instruction_following",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="12.1063%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_jailbreak": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_jailbreak",
            name="nemotron_ultra_mopd/ultra_sft_step3200_jailbreak",
            display_name="nvidia/Nemotron-RL-Safety-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1",
            verifier_revision="76b4bdebc9c8a2db5ee4b9b1bd6d1703b556e8ab00a9a8426aa942e3d2778022",
            family="safety",
            type="Alignment",
            turns="Single-turn",
            task_count=2861,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_jailbreak",
            component_selector="ultra_sft_step3200_jailbreak",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="3.3275%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_lean": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean",
            name="nemotron_ultra_mopd/ultra_sft_step3200_lean",
            display_name="nvidia/Nemotron-Math-Proofs-v1 · Lean refinement · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1",
            verifier_revision="44e963bdf18c6b3e95f85aee3ab59c03fc5c9e8fa1c598bc633ae117dd3d0315",
            family="math-proof",
            type="Agentic",
            turns="Multi-turn",
            task_count=902,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_lean",
            component_selector="ultra_sft_step3200_lean",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.0491%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_cot": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_cot",
            name="nemotron_ultra_mopd/ultra_sft_step3200_math_cot",
            display_name="nvidia/Nemotron-RL-Math-v2 · chain of thought · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="7fff13efc379f4bfd200ecd2d225dad60c79be801d09ddc27439f10b5fd87abb",
            family="math-answer",
            type="RLVR",
            turns="Single-turn",
            task_count=4776,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_math_cot",
            component_selector="ultra_sft_step3200_math_cot",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="5.5548%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_tir": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
            name="nemotron_ultra_mopd/ultra_sft_step3200_math_tir",
            display_name="nvidia/Nemotron-RL-Math-v2 · tool-assisted reasoning · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="2ab0dc76ee4bab074aec8879a6a6aa44405868d2ced449fbb2d64e3e9f9ab2d5",
            family="math-answer",
            type="Agentic",
            turns="Multi-turn",
            task_count=5917,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_math_tir",
            component_selector="ultra_sft_step3200_math_tir",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="6.8818%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_multichallenge_len40k": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_multichallenge_len40k",
            name="nemotron_ultra_mopd/ultra_sft_step3200_multichallenge_len40k",
            display_name="nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
            verifier_revision="abeefa8df27e3ab6decdfbfa33e7eca3abf6e7c7f37019d29419663ca7424b8e",
            family="instruction-following",
            type="RLVR",
            turns="Multi-turn",
            task_count=4028,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_multichallenge_len40k",
            component_selector="ultra_sft_step3200_multichallenge_len40k",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="4.6848%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_inductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive",
            name="nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · inductive · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="7593b96ad47dc731d023f6d4e8d45df8d456b309fad6374698f815287470aa10",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_nvarc_inductive",
            component_selector="ultra_sft_step3200_nvarc_inductive",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.4389%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_transductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_transductive",
            name="nemotron_ultra_mopd/ultra_sft_step3200_nvarc_transductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · transductive · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="43c2d7cd6b0cb7365492cef009595a9687f2671ed545d7d209b354c358eda40f",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2069,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_nvarc_transductive",
            component_selector="ultra_sft_step3200_nvarc_transductive",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.4064%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_rdkit": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
            name="nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
            display_name="nvidia/Nemotron-RL-Litmus-Bench-v0.1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Litmus-Bench-v0.1",
            verifier_revision="cf4029df469964dabedfa347aa974e590744e9d4807d132042641e024d5d83f1",
            family="chemistry",
            type="Agentic",
            turns="Multi-turn",
            task_count=1403,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_rdkit",
            component_selector="ultra_sft_step3200_rdkit",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.6318%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_reasoning_gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_reasoning_gym",
            name="nemotron_ultra_mopd/ultra_sft_step3200_reasoning_gym",
            display_name="nvidia/Nemotron-RL-ReasoningGym-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1",
            verifier_revision="9d4d3a356f99a79533ac7d25cf25481f6c975a75532c9e3ddbc0e9f4dbc4dea9",
            family="reasoning-gym",
            type="RLVR",
            turns="Single-turn",
            task_count=1385,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_reasoning_gym",
            component_selector="ultra_sft_step3200_reasoning_gym",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.6108%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa",
            name="nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa",
            display_name="nvidia/Nemotron-RL-knowledge-mcqa · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa",
            verifier_revision="4e21612c6acfff9c614d0c5e0e6868e0b137609aa2dca3c5f70aadf38dfa8aa0",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_stem_mcqa",
            component_selector="ultra_sft_step3200_stem_mcqa",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.4389%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa_cot_rima_new": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            name="nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            display_name="nvidia/Nemotron-SFT-Science-v2 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
            verifier_revision="c745334d6e868cfa4388f84f39a11c4985fe3bc28996c306d3db9eb5a16800c5",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2077,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_selector="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="2.4157%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_structured_outputs_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_structured_outputs_v2",
            name="nemotron_ultra_mopd/ultra_sft_step3200_structured_outputs_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 · v2 records · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
            verifier_revision="b84e98dd9bfc9daff96f66dc2f7fa9779d6dd31ed04e01b5a7a1b437a6b31ad8",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=682,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_structured_outputs_v2",
            component_selector="ultra_sft_step3200_structured_outputs_v2",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="0.7932%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_toolcall_schema": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
            name="nemotron_ultra_mopd/ultra_sft_step3200_toolcall_schema",
            display_name="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
            verifier_revision="752d664be2f90fd7b5b87f042b5e710734a9c927cc44e2d77a5729261390c34f",
            family="tool-use",
            type="Agentic",
            turns="Multi-turn",
            task_count=2666,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_sft_step3200_toolcall_schema",
            component_selector="ultra_sft_step3200_toolcall_schema",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="3.1007%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_v3_agentic_rl_step73_citation_format_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_citation_format_v2",
            name="nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_citation_format_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1 · step73 v2 records · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
            verifier_revision="bb2fa3ad44683ff0912d4d9b354ec259af50ffaf4d83d0d7e85c0ae4abf7460d",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=250,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_v3_agentic_rl_step73_citation_format_v2",
            component_selector="ultra_v3_agentic_rl_step73_citation_format_v2",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="0.2908%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_v3_agentic_rl_step73_freeform_text_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_freeform_text_v2",
            name="nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_freeform_text_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1 · step73 v2 records · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
            verifier_revision="31f2227faf10acc6c12d2072e0b9210cd965edf9e8afe31309f5de24af007741",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=250,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_v3_agentic_rl_step73_freeform_text_v2",
            component_selector="ultra_v3_agentic_rl_step73_freeform_text_v2",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="0.2908%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_v3_agentic_rl_step73_structured_outputs_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_structured_outputs_v2",
            name="nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_structured_outputs_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 · step73 v2 records · mopd",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
            verifier_revision="a0bf6ec5d0ef1f5a2ed62b485210c371209c85a84c01e33e9e668e8356bfab57",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=10000,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_v3_agentic_rl_step73_structured_outputs_v2",
            component_selector="ultra_v3_agentic_rl_step73_structured_outputs_v2",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="11.6306%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
            name="nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
            display_name="SWE-Gym/SWE-Gym · step73 pivot records · mopd",
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            verifier_revision="1b8b156a91f4647ed898c1fa0899447b2b08b2419acf292ca05110510fc2e028",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=246,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/SWE-Gym/SWE-Gym",
            component_selector="ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="0.2861%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
            name="nemotron_ultra_mopd/ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
            display_name="nebius/SWE-rebench-V2 · step73 pivot records · mopd",
            url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            verifier_revision="dd3310c994d63f365d41bba3ca91deda96c8c472e13a68b751f26610e99ca83d",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=932,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/mopd.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · mopd",
            canonical_id="MarinSkyRL:nemotron_ultra_mopd",
            registry_name="nemotron_ultra_mopd",
            component_name="ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k/nebius/SWE-rebench-V2",
            component_selector="ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k",
            component_file_sha256=MOPD_FILE_SHA256,
            canonical_task_count=85980,
            component_ratio="1.0840%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
    },
    "rlvr1": {
        "language_mixing_hs3_ultra_genrm_fmt": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt",
            name="nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="b280136ced512ece80e95ba06ccd4a72b89dbd83009e91204dc537296e1ee379",
            family="preference",
            type="Alignment",
            turns="Single-turn",
            task_count=4795,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="language_mixing_hs3_ultra_genrm_fmt",
            component_selector="language_mixing_hs3_ultra_genrm_fmt",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="4.8718%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_abstention": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_abstention",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_abstention",
            display_name="nvidia/Nemotron-RL-QA-Abstention-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1",
            verifier_revision="e3e85a847527d0ace97e1a6459ce92b3ce54d5278ac6f5a4eae7e6da9a564623",
            family="qa-abstention",
            type="RLVR",
            turns="Single-turn",
            task_count=4025,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_abstention",
            component_selector="ultra_sft_step3200_abstention",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="4.0894%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_calendar_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Calendar-v2 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
            verifier_revision="5c9656671baf79a9711cd5c7ecbacffc6d1a169129b81ab098c00a4be8838833",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=910,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_calendar_v2",
            component_selector="ultra_sft_step3200_calendar_v2",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="0.9246%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_comp_coding": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_comp_coding",
            display_name="nvidia/Nemotron-RL-coding-competitive_coding · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding",
            verifier_revision="0878a10ef2dd4c89daf18e325b48720553c3f5f0ab60d99d233b4520db3e7684",
            family="competitive-programming",
            type="RLVR",
            turns="Single-turn",
            task_count=7929,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_comp_coding",
            component_selector="ultra_sft_step3200_comp_coding",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="8.0560%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_instruction_following": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_instruction_following",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_instruction_following",
            display_name="nvidia/Nemotron-RL-instruction_following · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
            verifier_revision="f0571dbf560eaceae2609645364cb6df639318e701a31e0cb55303adfa6539bc",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=11828,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_instruction_following",
            component_selector="ultra_sft_step3200_instruction_following",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="12.0174%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_jailbreak": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_jailbreak",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_jailbreak",
            display_name="nvidia/Nemotron-RL-Safety-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1",
            verifier_revision="76b4bdebc9c8a2db5ee4b9b1bd6d1703b556e8ab00a9a8426aa942e3d2778022",
            family="safety",
            type="Alignment",
            turns="Single-turn",
            task_count=2861,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_jailbreak",
            component_selector="ultra_sft_step3200_jailbreak",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.9068%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_lean": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
            display_name="nvidia/Nemotron-Math-Proofs-v1 · Lean refinement · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1",
            verifier_revision="44e963bdf18c6b3e95f85aee3ab59c03fc5c9e8fa1c598bc633ae117dd3d0315",
            family="math-proof",
            type="Agentic",
            turns="Multi-turn",
            task_count=902,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_lean",
            component_selector="ultra_sft_step3200_lean",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="0.9164%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_cot": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_cot",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_math_cot",
            display_name="nvidia/Nemotron-RL-Math-v2 · chain of thought · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="7fff13efc379f4bfd200ecd2d225dad60c79be801d09ddc27439f10b5fd87abb",
            family="math-answer",
            type="RLVR",
            turns="Single-turn",
            task_count=4776,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_math_cot",
            component_selector="ultra_sft_step3200_math_cot",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="4.8525%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_tir": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_math_tir",
            display_name="nvidia/Nemotron-RL-Math-v2 · tool-assisted reasoning · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="2ab0dc76ee4bab074aec8879a6a6aa44405868d2ced449fbb2d64e3e9f9ab2d5",
            family="math-answer",
            type="Agentic",
            turns="Multi-turn",
            task_count=5917,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_math_tir",
            component_selector="ultra_sft_step3200_math_tir",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="6.0117%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_multichallenge_len40k": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k",
            display_name="nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
            verifier_revision="abeefa8df27e3ab6decdfbfa33e7eca3abf6e7c7f37019d29419663ca7424b8e",
            family="instruction-following",
            type="RLVR",
            turns="Multi-turn",
            task_count=4028,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_multichallenge_len40k",
            component_selector="ultra_sft_step3200_multichallenge_len40k",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="4.0925%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_inductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_inductive",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_inductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · inductive · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="7593b96ad47dc731d023f6d4e8d45df8d456b309fad6374698f815287470aa10",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_nvarc_inductive",
            component_selector="ultra_sft_step3200_nvarc_inductive",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1306%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_transductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_transductive",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_nvarc_transductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · transductive · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="43c2d7cd6b0cb7365492cef009595a9687f2671ed545d7d209b354c358eda40f",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2069,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_nvarc_transductive",
            component_selector="ultra_sft_step3200_nvarc_transductive",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1021%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_reasoning_gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_reasoning_gym",
            display_name="nvidia/Nemotron-RL-ReasoningGym-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1",
            verifier_revision="9d4d3a356f99a79533ac7d25cf25481f6c975a75532c9e3ddbc0e9f4dbc4dea9",
            family="reasoning-gym",
            type="RLVR",
            turns="Single-turn",
            task_count=2089,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_reasoning_gym",
            component_selector="ultra_sft_step3200_reasoning_gym",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1224%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa",
            display_name="nvidia/Nemotron-RL-knowledge-mcqa · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa",
            verifier_revision="4e21612c6acfff9c614d0c5e0e6868e0b137609aa2dca3c5f70aadf38dfa8aa0",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_stem_mcqa",
            component_selector="ultra_sft_step3200_stem_mcqa",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1306%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa_cot_rima_new": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            display_name="nvidia/Nemotron-SFT-Science-v2 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
            verifier_revision="c745334d6e868cfa4388f84f39a11c4985fe3bc28996c306d3db9eb5a16800c5",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2077,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_selector="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1103%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_structured_outputs_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_structured_outputs_v2",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_structured_outputs_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 · v2 records · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
            verifier_revision="b84e98dd9bfc9daff96f66dc2f7fa9779d6dd31ed04e01b5a7a1b437a6b31ad8",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=2080,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_structured_outputs_v2",
            component_selector="ultra_sft_step3200_structured_outputs_v2",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.1133%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            display_name="SWE-Gym/SWE-Gym · rlvr1",
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            verifier_revision="0cc68adcf55f5a2579adc3accf900968e2ca456ff721cf68ec3dd9d935ef68e4",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=2838,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            component_selector="ultra_sft_step3200_swe_pivot_len40k",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="2.8834%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            display_name="nebius/SWE-rebench-V2 · rlvr1",
            url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            verifier_revision="5cc0a0ac3da28111ddaecaeaa9cd4d03d39214dd5c12df22f64a2c87a08a0287",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=11060,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            component_selector="ultra_sft_step3200_swe_pivot_len40k",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="11.2371%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_sft_step3200_tau_pivot": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_tau_pivot",
            display_name="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
            verifier_revision="997b4d12a352345341d33861763b3680dee9d4039156cf9d0a21e7a18102620e",
            family="tool-use",
            type="Agentic",
            turns="Multi-turn",
            task_count=20035,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_tau_pivot",
            component_selector="ultra_sft_step3200_tau_pivot",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="20.3558%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_toolcall_schema": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
            name="nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema",
            display_name="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 · rlvr1",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
            verifier_revision="752d664be2f90fd7b5b87f042b5e710734a9c927cc44e2d77a5729261390c34f",
            family="tool-use",
            type="Agentic",
            turns="Multi-turn",
            task_count=4011,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr1.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr1",
            registry_name="nemotron_ultra_rlvr1",
            component_name="ultra_sft_step3200_toolcall_schema",
            component_selector="ultra_sft_step3200_toolcall_schema",
            component_file_sha256=RLVR1_FILE_SHA256,
            canonical_task_count=98424,
            component_ratio="4.0752%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
    },
    "rlvr2": {
        "language_mixing_hs3_ultra_genrm_fmt": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/language_mixing_hs3_ultra_genrm_fmt",
            name="nemotron_ultra_rlvr2/language_mixing_hs3_ultra_genrm_fmt",
            display_name="nvidia/Nemotron-RLHF-GenRM-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1",
            verifier_revision="b280136ced512ece80e95ba06ccd4a72b89dbd83009e91204dc537296e1ee379",
            family="preference",
            type="Alignment",
            turns="Single-turn",
            task_count=4757,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="language_mixing_hs3_ultra_genrm_fmt",
            component_selector="language_mixing_hs3_ultra_genrm_fmt",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="4.7994%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_abstention": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_abstention",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_abstention",
            display_name="nvidia/Nemotron-RL-QA-Abstention-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1",
            verifier_revision="e3e85a847527d0ace97e1a6459ce92b3ce54d5278ac6f5a4eae7e6da9a564623",
            family="qa-abstention",
            type="RLVR",
            turns="Single-turn",
            task_count=4025,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_abstention",
            component_selector="ultra_sft_step3200_abstention",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="4.0609%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_calendar_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_calendar_v2",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_calendar_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Calendar-v2 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
            verifier_revision="5c9656671baf79a9711cd5c7ecbacffc6d1a169129b81ab098c00a4be8838833",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=910,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_calendar_v2",
            component_selector="ultra_sft_step3200_calendar_v2",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="0.9181%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_comp_coding": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding",
            display_name="nvidia/Nemotron-RL-coding-competitive_coding · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding",
            verifier_revision="0878a10ef2dd4c89daf18e325b48720553c3f5f0ab60d99d233b4520db3e7684",
            family="competitive-programming",
            type="RLVR",
            turns="Single-turn",
            task_count=7929,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_comp_coding",
            component_selector="ultra_sft_step3200_comp_coding",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="7.9997%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_instruction_following": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_instruction_following",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_instruction_following",
            display_name="nvidia/Nemotron-RL-instruction_following · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
            verifier_revision="f0571dbf560eaceae2609645364cb6df639318e701a31e0cb55303adfa6539bc",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=10409,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_instruction_following",
            component_selector="ultra_sft_step3200_instruction_following",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="10.5018%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_jailbreak": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_jailbreak",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_jailbreak",
            display_name="nvidia/Nemotron-RL-Safety-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1",
            verifier_revision="76b4bdebc9c8a2db5ee4b9b1bd6d1703b556e8ab00a9a8426aa942e3d2778022",
            family="safety",
            type="Alignment",
            turns="Single-turn",
            task_count=2861,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_jailbreak",
            component_selector="ultra_sft_step3200_jailbreak",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.8865%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_lean": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
            display_name="nvidia/Nemotron-Math-Proofs-v1 · Lean refinement · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1",
            verifier_revision="44e963bdf18c6b3e95f85aee3ab59c03fc5c9e8fa1c598bc633ae117dd3d0315",
            family="math-proof",
            type="Agentic",
            turns="Multi-turn",
            task_count=902,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_lean",
            component_selector="ultra_sft_step3200_lean",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="0.9100%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_cot": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_cot",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_math_cot",
            display_name="nvidia/Nemotron-RL-Math-v2 · chain of thought · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="7fff13efc379f4bfd200ecd2d225dad60c79be801d09ddc27439f10b5fd87abb",
            family="math-answer",
            type="RLVR",
            turns="Single-turn",
            task_count=4776,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_math_cot",
            component_selector="ultra_sft_step3200_math_cot",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="4.8186%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_math_tir": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
            display_name="nvidia/Nemotron-RL-Math-v2 · tool-assisted reasoning · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2",
            verifier_revision="2ab0dc76ee4bab074aec8879a6a6aa44405868d2ced449fbb2d64e3e9f9ab2d5",
            family="math-answer",
            type="Agentic",
            turns="Multi-turn",
            task_count=5917,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_math_tir",
            component_selector="ultra_sft_step3200_math_tir",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="5.9698%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_multichallenge_len40k": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k",
            display_name="nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
            verifier_revision="abeefa8df27e3ab6decdfbfa33e7eca3abf6e7c7f37019d29419663ca7424b8e",
            family="instruction-following",
            type="RLVR",
            turns="Multi-turn",
            task_count=4028,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_multichallenge_len40k",
            component_selector="ultra_sft_step3200_multichallenge_len40k",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="4.0639%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_inductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_inductive",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_inductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · inductive · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="7593b96ad47dc731d023f6d4e8d45df8d456b309fad6374698f815287470aa10",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_nvarc_inductive",
            component_selector="ultra_sft_step3200_nvarc_inductive",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.1157%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_nvarc_transductive": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_transductive",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_nvarc_transductive",
            display_name="nvidia/Nemotron-RL-ARC-AGI-v1 · transductive · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1",
            verifier_revision="43c2d7cd6b0cb7365492cef009595a9687f2671ed545d7d209b354c358eda40f",
            family="arc-agi",
            type="RLVR",
            turns="Single-turn",
            task_count=2069,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_nvarc_transductive",
            component_selector="ultra_sft_step3200_nvarc_transductive",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.0875%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_reasoning_gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym",
            display_name="nvidia/Nemotron-RL-ReasoningGym-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1",
            verifier_revision="9d4d3a356f99a79533ac7d25cf25481f6c975a75532c9e3ddbc0e9f4dbc4dea9",
            family="reasoning-gym",
            type="RLVR",
            turns="Single-turn",
            task_count=1385,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_reasoning_gym",
            component_selector="ultra_sft_step3200_reasoning_gym",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="1.3974%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa",
            display_name="nvidia/Nemotron-RL-knowledge-mcqa · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa",
            verifier_revision="4e21612c6acfff9c614d0c5e0e6868e0b137609aa2dca3c5f70aadf38dfa8aa0",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2097,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_stem_mcqa",
            component_selector="ultra_sft_step3200_stem_mcqa",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.1157%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_stem_mcqa_cot_rima_new": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_stem_mcqa_cot_rima_new",
            display_name="nvidia/Nemotron-SFT-Science-v2 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
            verifier_revision="c745334d6e868cfa4388f84f39a11c4985fe3bc28996c306d3db9eb5a16800c5",
            family="qa-multiple-choice",
            type="RLVR",
            turns="Single-turn",
            task_count=2077,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_selector="ultra_sft_step3200_stem_mcqa_cot_rima_new",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.0955%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_structured_outputs_v2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v2",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v2",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 · v2 records · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
            verifier_revision="b84e98dd9bfc9daff96f66dc2f7fa9779d6dd31ed04e01b5a7a1b437a6b31ad8",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=682,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_structured_outputs_v2",
            component_selector="ultra_sft_step3200_structured_outputs_v2",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="0.6881%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            display_name="SWE-Gym/SWE-Gym · rlvr2",
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym",
            verifier_revision="0cc68adcf55f5a2579adc3accf900968e2ca456ff721cf68ec3dd9d935ef68e4",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=2838,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
            component_selector="ultra_sft_step3200_swe_pivot_len40k",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.8633%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            display_name="nebius/SWE-rebench-V2 · rlvr2",
            url="https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            verifier_revision="5cc0a0ac3da28111ddaecaeaa9cd4d03d39214dd5c12df22f64a2c87a08a0287",
            family="swe-repo",
            type="Agentic",
            turns="Multi-turn",
            task_count=11060,
            count_basis=(
                "Every record in the pinned original blend JSONL was counted once by dataset "
                "selection. SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
                "remaining SWE records are attributed to SWE-rebench-V2 by the blend card "
                "exhaustive two-source composition."
            ),
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_swe_pivot_len40k/nebius/SWE-rebench-V2",
            component_selector="ultra_sft_step3200_swe_pivot_len40k",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="11.1586%",
            verifier_url=(
                "https://github.com/marin-community/harbor/tree/8abc63e3bdb37af1d345fcac123ef7d21"
                "22598f3/src/harbor/verifier"
            ),
        ),
        "ultra_sft_step3200_tau_pivot": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
            display_name="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
            verifier_revision="997b4d12a352345341d33861763b3680dee9d4039156cf9d0a21e7a18102620e",
            family="tool-use",
            type="Agentic",
            turns="Multi-turn",
            task_count=20035,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_tau_pivot",
            component_selector="ultra_sft_step3200_tau_pivot",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="20.2137%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_toolcall_schema": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
            display_name="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
            verifier_revision="752d664be2f90fd7b5b87f042b5e710734a9c927cc44e2d77a5729261390c34f",
            family="tool-use",
            type="Agentic",
            turns="Multi-turn",
            task_count=2666,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_toolcall_schema",
            component_selector="ultra_sft_step3200_toolcall_schema",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="2.6898%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_ds2_freeform": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds2_freeform",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_ds2_freeform",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
            verifier_revision="78997a2391c51890ea9ef737c3c894a4307b90561bde4cfd3f730e283e9ad352",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=1392,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_ds2_freeform",
            component_selector="ultra_sft_step3200_ds2_freeform",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="1.4044%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_ds3_citation": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_ds3_citation",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_ds3_citation",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
            verifier_revision="11de81bec2bbf26a6480ca9a43065fafe7638181cefd13926869f3a5d54ae6c0",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=1403,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_ds3_citation",
            component_selector="ultra_sft_step3200_ds3_citation",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="1.4155%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_rdkit": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
            display_name="nvidia/Nemotron-RL-Litmus-Bench-v0.1 · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Litmus-Bench-v0.1",
            verifier_revision="cf4029df469964dabedfa347aa974e590744e9d4807d132042641e024d5d83f1",
            family="chemistry",
            type="Agentic",
            turns="Multi-turn",
            task_count=1403,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_rdkit",
            component_selector="ultra_sft_step3200_rdkit",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="1.4155%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
        "ultra_sft_step3200_structured_outputs_v3": replace(
            SKYRL_METADATA,
            id="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v3",
            name="nemotron_ultra_rlvr2/ultra_sft_step3200_structured_outputs_v3",
            display_name="nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 · v3 records · rlvr2",
            url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
            verifier_revision="1d36a9a2a0d93c9acb61bfb3eccc498d2c359e5ab2bfc76ccc28aadc9de224b1",
            family="instruction-following",
            type="RLVR",
            turns="Single-turn",
            task_count=1398,
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. ",
            count_url=(
                "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/482"
                "392c14c6418e26804ea2e5d10359df9877df4/rlvr2.jsonl"
            ),
            canonical_source="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr2",
            canonical_id="MarinSkyRL:nemotron_ultra_rlvr2",
            registry_name="nemotron_ultra_rlvr2",
            component_name="ultra_sft_step3200_structured_outputs_v3",
            component_selector="ultra_sft_step3200_structured_outputs_v3",
            component_file_sha256=RLVR2_FILE_SHA256,
            canonical_task_count=99116,
            component_ratio="1.4105%",
            verifier_url=(
                "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d"
                "9c883aaf0d2/skyrl-gym/skyrl_gym/envs/nemotron_ultra"
            ),
        ),
    },
}


def pipeline_name(blend: str, path: str) -> str:
    return NAME_PREFIX + blend + "_" + re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")


def _source(blend: str, path: str) -> RlDataSource:
    component = COMPONENTS[path]
    name, _, split = path.partition("/")
    select = SweRows(name, SweSplit(split)) if split else ComponentRows(name)
    inputs = {SWE_GYM.repo: SWE_GYM} if split else dict(component.inputs)
    return RlDataSource(
        metadata=BLENDS[blend][path],
        pipeline=RlDataPipeline(
            name=pipeline_name(blend, path),
            source=HfSource(
                ULTRA_REPO,
                ULTRA_REVISION,
                (f"{blend}.jsonl",),
                SourceFormat.JSONL,
                select=select,
                decode=component.decode,
            ),
            convert=component.convert,
            version="2",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=component.rubric,
            controls=component.controls,
            inputs=inputs,
            grader=component.grader,
            ships=component.ships,
        ),
    )


def sources() -> list[RlDataSource]:
    return [_source(blend, path) for blend, paths in BLENDS.items() for path in paths]
