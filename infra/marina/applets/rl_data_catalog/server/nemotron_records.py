# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count Nemotron source selections from complete local JSONL records."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordSource:
    repository: str
    selection: str
    task_type: str
    interaction: str
    family: str


RECORD_SOURCES = {
    "comp_coding": RecordSource(
        "Nemotron-RL-coding-competitive_coding", "", "RLVR", "Single-turn", "competitive-programming"
    ),
    "tau_pivot": RecordSource("Nemotron-RL-Agentic-Conversational-Tool-Use-v1", "", "Agentic", "Multi-turn", "tool-use"),
    "math_cot": RecordSource("Nemotron-RL-Math-v2", "chain of thought", "RLVR", "Single-turn", "math-answer"),
    "math_tir": RecordSource("Nemotron-RL-Math-v2", "tool-assisted reasoning", "Agentic", "Multi-turn", "math-answer"),
    "instruction_following": RecordSource(
        "Nemotron-RL-instruction_following", "", "RLVR", "Single-turn", "instruction-following"
    ),
    "calendar_v2": RecordSource(
        "Nemotron-RL-Instruction-Following-Calendar-v2", "", "RLVR", "Single-turn", "instruction-following"
    ),
    "jailbreak": RecordSource("Nemotron-RL-Safety-v1", "", "Alignment", "Single-turn", "safety"),
    "toolcall_schema": RecordSource(
        "Nemotron-RL-Agentic-Function-Calling-Pivot-v1", "", "Agentic", "Multi-turn", "tool-use"
    ),
    "nvarc_transductive": RecordSource("Nemotron-RL-ARC-AGI-v1", "transductive", "RLVR", "Single-turn", "arc-agi"),
    "nvarc_inductive": RecordSource("Nemotron-RL-ARC-AGI-v1", "inductive", "RLVR", "Single-turn", "arc-agi"),
    "abstention": RecordSource("Nemotron-RL-QA-Abstention-v1", "", "RLVR", "Single-turn", "qa-abstention"),
    "stem_mcqa_cot_rima_new": RecordSource("Nemotron-SFT-Science-v2", "", "RLVR", "Single-turn", "qa-multiple-choice"),
    "stem_mcqa": RecordSource("Nemotron-RL-knowledge-mcqa", "", "RLVR", "Single-turn", "qa-multiple-choice"),
    "structured_outputs_v2": RecordSource(
        "Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "v2 records",
        "RLVR",
        "Single-turn",
        "instruction-following",
    ),
    "structured_outputs_v3": RecordSource(
        "Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "v3 records",
        "RLVR",
        "Single-turn",
        "instruction-following",
    ),
    "multichallenge_len40k": RecordSource(
        "Nemotron-RL-Instruction-Following-MultiTurnChat-v1", "", "RLVR", "Multi-turn", "instruction-following"
    ),
    "reasoning_gym": RecordSource("Nemotron-RL-ReasoningGym-v1", "", "RLVR", "Single-turn", "reasoning-gym"),
    "lean": RecordSource("Nemotron-Math-Proofs-v1", "Lean refinement", "Agentic", "Multi-turn", "math-proof"),
    "rdkit": RecordSource("Nemotron-RL-Litmus-Bench-v0.1", "", "Agentic", "Multi-turn", "chemistry"),
    "ds3_citation": RecordSource(
        "Nemotron-RL-Instruction-Following-Citation-Formatting-v1", "", "RLVR", "Single-turn", "instruction-following"
    ),
    "ds2_freeform": RecordSource(
        "Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1", "", "RLVR", "Single-turn", "instruction-following"
    ),
    "language_mixing_hs3_ultra_genrm_fmt": RecordSource(
        "Nemotron-RLHF-GenRM-v1", "", "Alignment", "Single-turn", "preference"
    ),
    "hs3_en": RecordSource("Nemotron-RLHF-GenRM-v1", "hs3_en", "Alignment", "Single-turn", "preference"),
    "hs3_multi": RecordSource("Nemotron-RLHF-GenRM-v1", "hs3_multi", "Alignment", "Single-turn", "preference"),
    "hs3_multiturn": RecordSource("Nemotron-RLHF-GenRM-v1", "hs3_multiturn", "Alignment", "Multi-turn", "preference"),
    "safety_en": RecordSource("Nemotron-RLHF-GenRM-v1", "safety_en", "Alignment", "Single-turn", "preference"),
    "ultra_v3_agentic_rl_step73_structured_outputs_v2": RecordSource(
        "Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        "step73 v2 records",
        "RLVR",
        "Single-turn",
        "instruction-following",
    ),
    "ultra_v3_agentic_rl_step73_citation_format_v2": RecordSource(
        "Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        "step73 v2 records",
        "RLVR",
        "Single-turn",
        "instruction-following",
    ),
    "ultra_v3_agentic_rl_step73_freeform_text_v2": RecordSource(
        "Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        "step73 v2 records",
        "RLVR",
        "Single-turn",
        "instruction-following",
    ),
    "makeshn_ultra_v3_ipi_train": RecordSource(
        "Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1", "", "Agentic", "Multi-turn", "agentic-safety"
    ),
}
SWE_RECORD_SOURCE = "ultra_sft_step3200_swe_pivot_len40k"
SWE_UNLABELED = "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent"
SWE_SELECTIONS = {
    SWE_RECORD_SOURCE: "",
    SWE_UNLABELED: "unlabeled dataset records",
    "swe_pivot_len40k": "pivot records",
    "ultra_v3_agentic_rl_step73_swe_pivot_v1_len40k": "step73 pivot records",
}


def record_selection(record: dict[str, Any]) -> str:
    dataset = record.get("dataset")
    if dataset:
        return dataset
    agent = record["agent_ref"]["name"]
    if agent == "swe_pivot_single_step_tool_use_with_argument_comparison_agent":
        return SWE_UNLABELED
    raise ValueError(f"Cannot identify dataset population for agent {agent}")


def record_source(dataset: str, swe_source: str = "") -> RecordSource:
    if dataset in SWE_SELECTIONS:
        if swe_source not in {"nebius/SWE-rebench-V2", "SWE-Gym/SWE-Gym"}:
            raise ValueError(f"SWE record requires a resolved source: {swe_source}")
        return RecordSource(swe_source, SWE_SELECTIONS[dataset], "Agentic", "Multi-turn", "swe-repo")
    source = RECORD_SOURCES[dataset.removeprefix("ultra_sft_step3200_")]
    return RecordSource(
        "nvidia/" + source.repository, source.selection, source.task_type, source.interaction, source.family
    )


def count_records(records: Iterable[dict[str, Any]], swe_instances: dict[str, str]) -> list[dict[str, Any]]:
    """Count each record once; resolve SWE origins by audited instance membership."""
    counts = Counter()
    agents: dict[tuple[str, str], set[str]] = {}
    for record in records:
        dataset = record_selection(record)
        swe_source = swe_instances[record["metadata"]["instance_id"]] if dataset in SWE_SELECTIONS else ""
        record_source(dataset, swe_source)
        key = (dataset, swe_source)
        counts[key] += 1
        agents.setdefault(key, set()).add(record["agent_ref"]["name"])
    return [
        {"dataset": dataset, "swe_source": swe_source, "count": count, "agents": sorted(agents[(dataset, swe_source)])}
        for (dataset, swe_source), count in sorted(counts.items())
    ]
