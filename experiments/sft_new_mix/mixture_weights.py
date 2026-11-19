# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate candidate families using retained tokens and the frozen replay topology."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from math import fsum

WEIGHT_TOLERANCE = 1e-12


class SourceFamily(StrEnum):
    REPLAY = "replay"
    MATH_V2_V3 = "math_v2_v3"
    OTHER_MATH = "other_math"
    CODE = "competitive_code"
    VERIFIED_AGENTS = "recorded_pass_agents"
    RELAY = "relay"
    CHAT = "instruction_chat"
    SCIENCE = "science"
    MULTILINGUAL = "multilingual_v2"
    SAFETY = "safety"
    FINANCE = "finance"
    OPENCODE = "opencode_without_outcome"
    IDENTITY = "existing_identity"
    NEMOTRON_SWE = "nemotron_swe"
    MISCELLANEOUS = "miscellaneous"


@dataclass(frozen=True)
class CandidateWeights:
    weights: dict[str, float]
    families: dict[str, str]
    control_average: dict[str, float]
    replay_multiplicities: dict[str, dict[str, int]]


def source_family(source: str) -> SourceFamily:
    if source.startswith("replay/"):
        return SourceFamily.REPLAY
    if source.startswith(("nemotron_sft_v3/math_v2/", "nemotron_sft_v3/math_v3/")):
        return SourceFamily.MATH_V2_V3
    if source.startswith(("nemotron_sft_v3/math_proofs_", "nemotron_sft_v3/math_v4/", "nemotron_sft_v3/arc_agi_")):
        return SourceFamily.OTHER_MATH
    if source.startswith(
        (
            "nemotron_sft_v3/competitive_programming_",
            "nemotron_sft_v3/cuda_",
            "openthoughts4-code-",
            "glm-5.2-kernelgym-",
        )
    ):
        return SourceFamily.CODE
    if source.startswith(("open_swe_traces/", "penfever-traces/")) or source == "agenttrove":
        return SourceFamily.VERIFIED_AGENTS
    if source.startswith("calibforge-relay/"):
        return SourceFamily.RELAY
    if source.startswith(("nemotron_sft_v3/instruction_following_chat_", "ultrachat-", "wildchat-")):
        return SourceFamily.CHAT
    if source.startswith("nemotron_sft_v3/science_v2/") or source == "science-tool-use-conversations":
        return SourceFamily.SCIENCE
    if source.startswith("nemotron_sft_v3/multilingual_v2/"):
        return SourceFamily.MULTILINGUAL
    if source.startswith("nemotron_sft_v3/safety_"):
        return SourceFamily.SAFETY
    if source.startswith("nemotron_sft_v3/finance_v1/"):
        return SourceFamily.FINANCE
    if source.startswith("nemotron_sft_v3/opencode_v1/"):
        return SourceFamily.OPENCODE
    if source == "identity-data":
        return SourceFamily.IDENTITY
    if source.startswith(("nemotron_sft_v3/swe_v1/", "nemotron_sft_v3/swe_v2/")):
        return SourceFamily.NEMOTRON_SWE
    return SourceFamily.MISCELLANEOUS


def schedule_average(schedule: list, updates: int) -> dict[str, float]:
    totals = Counter()
    for index, (start, weights) in enumerate(schedule):
        if start >= updates:
            break
        stop = min(schedule[index + 1][0], updates) if index + 1 < len(schedule) else updates
        for name, weight in weights.items():
            totals[name] += weight * (stop - start) / updates
    return dict(totals)


def candidate_weights(
    inventory: dict,
    scripted_plan: dict,
    components: dict,
    control_schedule: list,
    family_shares: dict[str, float],
    updates: int,
) -> CandidateWeights:
    """Preserve replay child repetition before adjusting its mass for cleanup."""
    if inventory["complete"] is not True or abs(fsum(family_shares.values()) - 1) > WEIGHT_TOLERANCE:
        raise ValueError("Weights require a complete population and family shares summing to one")
    control = schedule_average(control_schedule, updates)
    replay_tokens: dict[str, int] = defaultdict(int)
    replay_source: dict[str, str] = {}
    for job in scripted_plan["jobs"]:
        if job["loss_mode"] != "full_token":
            continue
        # ranged_remainder splits unfinished caches into children that share the
        # parent input path; the replay basis needs the whole cache's tokens.
        replay_tokens[job["input_path"]] += job["expected_tokens"]
        replay_source[job["input_path"]] = job["source"]
    retained = {source["source"]: source["counts"]["output_tokens"] for source in inventory["sources"]}
    replay_basis = Counter()
    multiplicities = {}
    for name, component in components.items():
        if not name.startswith("pretrain/"):
            continue
        if component["type"] != "concat":
            raise ValueError("Frozen replay topology must explicitly declare its concatenated children")
        repeated = Counter(child["cache_dir"] for child in component["children"].values())
        denominator = sum(repeats * replay_tokens[path] for path, repeats in repeated.items())
        multiplicities[name] = dict(repeated)
        for path, repeats in repeated.items():
            source = replay_source[path]
            replay_basis[source] += control[name] * repeats * retained[source] / denominator
    grouped = defaultdict(dict)
    families = {}
    for source in inventory["sources"]:
        if not source["shards"]:
            continue
        family = source_family(source["source"])
        families[source["component"]] = family.value
        basis = replay_basis[source["source"]] if family == SourceFamily.REPLAY else source["counts"]["output_tokens"]
        grouped[family.value][source["component"]] = basis
    if set(grouped) != {name for name, share in family_shares.items() if share > 0}:
        raise ValueError("Family targets and retained candidate families differ; revise the recorded proposal")
    weights = {}
    for family, members in grouped.items():
        total = fsum(members.values())
        if total <= 0:
            raise ValueError("A selected family has no retained sampling mass")
        weights.update({name: family_shares[family] * basis / total for name, basis in members.items()})
    return CandidateWeights(weights, families, control, multiplicities)
