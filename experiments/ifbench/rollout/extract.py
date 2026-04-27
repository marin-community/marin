# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-4: build DPO pairs + SFT cache from verified rollouts.

Inputs:
  - per-prompt rollouts (one or more `Rollout` per prompt, across the model pool)
  - the row's parsed `(instruction_id_list, kwargs)` from `parse_ground_truth`

For each prompt we run every rollout's response through the verifier registry
(strict mode), then split the rollouts into pass-all and fail-≥1 buckets.

Outputs (two parallel jsonl artefacts):
  - DPO pairs:  one (chosen, rejected) per prompt where both buckets are non-empty.
  - SFT cache:  one example per passing rollout. Auto-quality-filtered by the verifier.

Tracks the skip rate per cause (no-passes, no-fails) and per-num_constraints.

See `.agents/logbooks/dpo_sft.md` § Stage 4a/4b.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
from collections.abc import Iterable
from typing import Any

from experiments.ifbench.data.prepare import PreparedRow
from experiments.ifbench.rollout.backend import Rollout
from experiments.ifbench.verifiers import INSTRUCTION_DICT_ALL
from experiments.ifbench.verifiers.parse import parse_ground_truth

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class VerifiedRollout:
    """A rollout plus per-constraint pass/fail flags."""

    rollout: Rollout
    per_constraint_passes: list[bool]
    passes_all: bool

    @property
    def fails_at_least_one(self) -> bool:
        return not self.passes_all


@dataclasses.dataclass(frozen=True)
class DpoPair:
    """One preference pair: chosen passes all constraints; rejected fails ≥1."""

    prompt_id: str
    messages: list[dict[str, str]]
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any] | None]
    chosen_response: str
    chosen_model: str
    rejected_response: str
    rejected_model: str
    rejected_failed_constraints: list[str]  # which instruction_ids it failed
    num_constraints: int


@dataclasses.dataclass(frozen=True)
class SftExample:
    """One supervised example: prompt + a passing response (verifier-quality-filtered)."""

    prompt_id: str
    messages: list[dict[str, str]]
    response: str
    model: str
    instruction_id_list: list[str]
    num_constraints: int


@dataclasses.dataclass
class ExtractionStats:
    """Skip-rate accounting for the smoke gate."""

    n_prompts_seen: int = 0
    n_prompts_yielding_pair: int = 0
    n_prompts_no_passers: int = 0
    n_prompts_no_failers: int = 0
    n_sft_examples: int = 0
    skip_by_num_constraints: dict[int, int] = dataclasses.field(default_factory=dict)
    yield_by_num_constraints: dict[int, int] = dataclasses.field(default_factory=dict)

    @property
    def pair_yield(self) -> float:
        if self.n_prompts_seen == 0:
            return 0.0
        return self.n_prompts_yielding_pair / self.n_prompts_seen


def verify_rollouts(
    prepared: PreparedRow,
    rollouts: Iterable[Rollout],
    registry: dict[str, type] = INSTRUCTION_DICT_ALL,
) -> list[VerifiedRollout]:
    """Score every rollout for `prepared` against its constraint set.

    Strict-only: the IFBench paper uses strict for pair construction. Loose
    scoring is reserved for the final eval where we want a softer signal.
    """
    parsed = parse_ground_truth(prepared.ground_truth)
    out: list[VerifiedRollout] = []
    for rollout in rollouts:
        per_constraint: list[bool] = []
        for index, instruction_id in enumerate(parsed.instruction_id_list):
            if instruction_id not in registry:
                # Unknown id — count as fail. Stage-2 should have caught this in dev.
                logger.warning(
                    "Unknown instruction_id %s in prompt %s — counting as fail",
                    instruction_id,
                    prepared.prompt_id,
                )
                per_constraint.append(False)
                continue
            instruction = registry[instruction_id](instruction_id)
            kw = parsed.kwargs[index] or {}
            clean_kw = {k: v for k, v in kw.items() if v is not None}
            instruction.build_description(**clean_kw)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(
                    prompt=prepared.messages[0].get("content", ""),
                )
            response = rollout.response_text or ""
            if response.strip() and instruction.check_following(response):
                per_constraint.append(True)
            else:
                per_constraint.append(False)
        out.append(
            VerifiedRollout(
                rollout=rollout,
                per_constraint_passes=per_constraint,
                passes_all=all(per_constraint),
            )
        )
    return out


def _failed_constraint_ids(parsed_ids: list[str], passes: list[bool]) -> list[str]:
    return [iid for iid, ok in zip(parsed_ids, passes, strict=True) if not ok]


def extract_pairs_and_sft(
    prepared_rows: Iterable[PreparedRow],
    rollouts_by_prompt: dict[str, list[Rollout]],
    registry: dict[str, type] = INSTRUCTION_DICT_ALL,
    seed: int = 0,
) -> tuple[list[DpoPair], list[SftExample], ExtractionStats]:
    """For each prompt: verify rollouts, then build at most one DPO pair + zero+
    SFT examples. Skips prompts with empty pass-set or empty fail-set; tracks why.
    """
    rng = random.Random(seed)
    pairs: list[DpoPair] = []
    sft: list[SftExample] = []
    stats = ExtractionStats()

    for prepared in prepared_rows:
        stats.n_prompts_seen += 1
        rollouts = rollouts_by_prompt.get(prepared.prompt_id, [])
        if not rollouts:
            stats.n_prompts_no_passers += 1
            stats.skip_by_num_constraints[prepared.num_constraints] = (
                stats.skip_by_num_constraints.get(prepared.num_constraints, 0) + 1
            )
            continue

        verified = verify_rollouts(prepared, rollouts, registry=registry)
        passers = [v for v in verified if v.passes_all]
        failers = [v for v in verified if v.fails_at_least_one]

        # SFT cache: every passing rollout becomes one example, regardless of
        # whether the prompt also yields a pair. Auto-quality-filtered.
        for v in passers:
            sft.append(
                SftExample(
                    prompt_id=prepared.prompt_id,
                    messages=prepared.messages,
                    response=v.rollout.response_text,
                    model=v.rollout.model_id,
                    instruction_id_list=parse_ground_truth(prepared.ground_truth).instruction_id_list,
                    num_constraints=prepared.num_constraints,
                )
            )
            stats.n_sft_examples += 1

        # DPO pair: skip if either bucket is empty.
        if not passers:
            stats.n_prompts_no_passers += 1
            stats.skip_by_num_constraints[prepared.num_constraints] = (
                stats.skip_by_num_constraints.get(prepared.num_constraints, 0) + 1
            )
            continue
        if not failers:
            stats.n_prompts_no_failers += 1
            stats.skip_by_num_constraints[prepared.num_constraints] = (
                stats.skip_by_num_constraints.get(prepared.num_constraints, 0) + 1
            )
            continue

        chosen = rng.choice(passers)
        rejected = rng.choice(failers)
        parsed = parse_ground_truth(prepared.ground_truth)
        pairs.append(
            DpoPair(
                prompt_id=prepared.prompt_id,
                messages=prepared.messages,
                instruction_id_list=parsed.instruction_id_list,
                kwargs=parsed.kwargs,
                chosen_response=chosen.rollout.response_text,
                chosen_model=chosen.rollout.model_id,
                rejected_response=rejected.rollout.response_text,
                rejected_model=rejected.rollout.model_id,
                rejected_failed_constraints=_failed_constraint_ids(
                    parsed.instruction_id_list,
                    rejected.per_constraint_passes,
                ),
                num_constraints=prepared.num_constraints,
            )
        )
        stats.n_prompts_yielding_pair += 1
        stats.yield_by_num_constraints[prepared.num_constraints] = (
            stats.yield_by_num_constraints.get(prepared.num_constraints, 0) + 1
        )

    return pairs, sft, stats


def write_pairs_jsonl(pairs: Iterable[DpoPair], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(dataclasses.asdict(p), ensure_ascii=False) + "\n")


def write_sft_jsonl(sft: Iterable[SftExample], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in sft:
            f.write(json.dumps(dataclasses.asdict(s), ensure_ascii=False) + "\n")
