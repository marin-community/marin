# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent per-position log-prob parity against a stored vLLM reference.

The reference holds, for every representative prompt, vLLM's prompt log-probs at every position: the log-prob of
the actual next token and vLLM's top-k tokens. Entry ``j`` of a case is the distribution over token ``j + 1``
given tokens ``0..j``, so a prompt of ``T`` tokens has ``T - 1`` entries.

A scored backend reports, per entry, the log-prob of the actual next token and its log-prob at exactly vLLM's
top-k ids. Every vLLM top-k token with probability above ``MIN_PROB`` is compared, in nats, wherever it falls in the
backend's own ranking, and so is every target token. Groups are the prompt categories (the case id's first word)
and ``past_window``, every position past the sliding window; each group records the attention implementation its
positions ran under. The gate bounds each group's p99 and max absolute difference over top-k tokens and over
target tokens. Top-1 agreement (the backend's best of vLLM's top-k against vLLM's top-1) is reported, not gated.
"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

REFERENCE_SCHEMA_VERSION = 1
REFERENCE_TOP_K = 20
MIN_PROB = 1e-3
"""Reference tokens below this probability are neither stored (beyond the top two) nor compared."""
MIN_STORED_TOKENS = 2
"""Always keep vLLM's top two, so a top-1 near tie can be told apart from a disagreement."""
NEAR_TIE = 0.1
SLIDING_WINDOW = 2048
PAST_WINDOW_GROUP = "past_window"
# Provisional, in nats: the identity recipe's t310 precision probe, measured on Snowball v0.1 with its own prompts.
# They must be re-set from an ablation on this check: the all-on settings run twice for the noise floor, then one
# run per precision fix turned off (snowball_training_parity.TrainingForwardSettings). Every fix-off run must fail
# the new limits. The target-token limits reuse the top-k ones until then.
MAX_P99_LOGPROB_ERROR = 0.70
MAX_LOGPROB_ERROR = 3.5
MAX_P99_TARGET_LOGPROB_ERROR = MAX_P99_LOGPROB_ERROR
MAX_TARGET_LOGPROB_ERROR = MAX_LOGPROB_ERROR


@dataclass(frozen=True)
class ReferenceMetadata:
    """The vLLM run a reference came from, so a stale reference can be told apart from a regression."""

    vllm_version: str
    vllm_commit: str
    vllm_release_tag: str
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel: bool
    attention_backend: str
    model_uri: str
    model_sha256: str
    prompt_fixture_sha256: str
    top_k: int
    min_prob: float


@dataclass(frozen=True)
class CasePositionReference:
    """vLLM's prompt log-probs for one prompt; every sequence has one entry per token after the first."""

    id: str
    token_ids_sha256: str
    target_logprobs: tuple[float, ...]
    top_token_ids: tuple[tuple[int, ...], ...]
    top_logprobs: tuple[tuple[float, ...], ...]

    @property
    def num_entries(self) -> int:
        return len(self.target_logprobs)


@dataclass(frozen=True)
class PositionReference:
    metadata: ReferenceMetadata
    cases: tuple[CasePositionReference, ...]


@dataclass(frozen=True)
class ScoredPositions:
    """One backend's scores for one prompt, in the reference layout.

    ``reference_logprobs[j, r]`` is the backend's log-prob of ``top_token_ids[j][r]`` of the reference; NaN where
    the reference stored no id.
    """

    case_id: str
    target_logprobs: np.ndarray
    reference_logprobs: np.ndarray


@dataclass(frozen=True)
class GroupStats:
    positions: int
    tokens_compared: int
    missing: int
    max_abs_logprob_diff: float
    p99_abs_logprob_diff: float
    p50_abs_logprob_diff: float
    max_abs_target_logprob_diff: float
    p99_abs_target_logprob_diff: float
    top1_agree: int
    top1_near_tie: int
    top1_disagree: int
    target_not_finite: int
    attention_positions: dict[str, int]
    """Positions per attention implementation."""


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(token) for token in token_ids).encode()).hexdigest()


def case_group(case_id: str) -> str:
    """The prompt category: ``long-longbench-03`` -> ``long``."""
    return case_id.split("-", 1)[0]


def _prompt_logprob_entry(entry: Mapping) -> dict[int, float]:
    """One vLLM ``prompt_logprobs`` entry: ``{token id: {"logprob": ...}}``; keys may be ``token_id:N``."""
    found = {}
    for key, value in entry.items():
        token = int(str(key).removeprefix("token_id:"))
        found[token] = float(value["logprob"] if isinstance(value, Mapping) else value)
    return found


def case_reference_from_prompt_logprobs(
    case_id: str,
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Sequence[Mapping | None],
    *,
    top_k: int = REFERENCE_TOP_K,
    min_prob: float = MIN_PROB,
) -> CasePositionReference:
    """Convert vLLM's completions ``prompt_logprobs`` (``None`` at position 0) into a stored reference case.

    vLLM returns the top-k tokens plus the actual prompt token at each position. The actual token's log-prob is
    kept as the target; top-k tokens below ``min_prob`` are dropped beyond the top ``MIN_STORED_TOKENS``.
    """
    if len(prompt_logprobs) != len(prompt_token_ids):
        raise ValueError(f"{case_id}: {len(prompt_logprobs)} prompt log-prob entries for {len(prompt_token_ids)} tokens")
    if prompt_logprobs[0] is not None:
        raise ValueError(f"{case_id}: vLLM returned a log-prob for the first prompt token")
    floor = math.log(min_prob)
    targets, top_ids, top_lps = [], [], []
    for position in range(1, len(prompt_token_ids)):
        entry = prompt_logprobs[position]
        if entry is None:
            raise ValueError(f"{case_id}: no prompt log-probs at position {position}")
        scores = _prompt_logprob_entry(entry)
        target = int(prompt_token_ids[position])
        if target not in scores:
            raise ValueError(f"{case_id}: vLLM omitted the prompt token at position {position}")
        targets.append(scores[target])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        kept = [
            (token, value) for rank, (token, value) in enumerate(ranked) if rank < MIN_STORED_TOKENS or value >= floor
        ]
        top_ids.append(tuple(token for token, _ in kept))
        top_lps.append(tuple(value for _, value in kept))
    return CasePositionReference(
        id=case_id,
        token_ids_sha256=token_ids_sha256(prompt_token_ids),
        target_logprobs=tuple(targets),
        top_token_ids=tuple(top_ids),
        top_logprobs=tuple(top_lps),
    )


def position_reference_payload(reference: PositionReference) -> dict:
    """The JSON object stored for ``reference``; ``parse_position_reference`` reads it back."""
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "metadata": asdict(reference.metadata),
        "cases": [
            {
                "id": case.id,
                "token_ids_sha256": case.token_ids_sha256,
                "target_logprobs": list(case.target_logprobs),
                "top_token_ids": [list(ids) for ids in case.top_token_ids],
                "top_logprobs": [list(values) for values in case.top_logprobs],
            }
            for case in reference.cases
        ],
    }


def parse_position_reference(payload: Mapping) -> PositionReference:
    """Validate and read a stored reference; fail closed on any schema or shape mismatch."""
    if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Reference schema {payload.get('schema_version')!r}, expected {REFERENCE_SCHEMA_VERSION}")
    metadata = ReferenceMetadata(**payload["metadata"])
    cases = []
    for raw in payload["cases"]:
        case = CasePositionReference(
            id=raw["id"],
            token_ids_sha256=raw["token_ids_sha256"],
            target_logprobs=tuple(float(value) for value in raw["target_logprobs"]),
            top_token_ids=tuple(tuple(int(token) for token in ids) for ids in raw["top_token_ids"]),
            top_logprobs=tuple(tuple(float(value) for value in values) for values in raw["top_logprobs"]),
        )
        if not len(case.top_token_ids) == len(case.top_logprobs) == case.num_entries:
            raise ValueError(f"{case.id}: reference entry counts differ")
        for position, (ids, values) in enumerate(zip(case.top_token_ids, case.top_logprobs, strict=True)):
            if not ids or len(ids) != len(values) or len(ids) > metadata.top_k:
                raise ValueError(f"{case.id}: malformed top-k at entry {position}")
        cases.append(case)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Reference repeats a case id")
    return PositionReference(metadata=metadata, cases=tuple(cases))


def check_reference_matches_prompts(reference: PositionReference, prompts: Mapping[str, Sequence[int]]) -> None:
    """The reference must cover exactly these prompts with these exact token ids."""
    by_id = {case.id: case for case in reference.cases}
    if by_id.keys() != prompts.keys():
        raise ValueError(f"Reference cases differ from the prompts: {sorted(by_id.keys() ^ prompts.keys())}")
    for case_id, token_ids in prompts.items():
        case = by_id[case_id]
        if case.token_ids_sha256 != token_ids_sha256(token_ids) or case.num_entries != len(token_ids) - 1:
            raise ValueError(f"{case_id}: reference token ids differ from the prompt")


def reference_id_grid(case: CasePositionReference, width: int = REFERENCE_TOP_K) -> np.ndarray:
    """vLLM's stored top-k ids as ``[entries, width]`` int32, -1 where none: the ids a backend gathers at."""
    grid = np.full((case.num_entries, width), -1, dtype=np.int32)
    for position, ids in enumerate(case.top_token_ids):
        grid[position, : len(ids)] = ids
    return grid


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def compare_positions(
    reference: PositionReference,
    scored: Mapping[str, ScoredPositions],
    attention_by_case: Mapping[str, str],
    *,
    sliding_window: int = SLIDING_WINDOW,
) -> dict[str, GroupStats]:
    """Per group: absolute log-prob differences over vLLM's top-k tokens above ``MIN_PROB``, target-token
    differences, and top-1 agreement. A NaN backend score at a reference id counts as missing; a non-finite
    target score is counted separately. ``attention_by_case`` names the attention each case ran under."""
    case_ids = {case.id for case in reference.cases}
    if scored.keys() != case_ids or attention_by_case.keys() != case_ids:
        raise ValueError("The backend scored a different case set than the reference")
    floor = math.log(reference.metadata.min_prob)
    diffs: dict[str, list[float]] = defaultdict(list)
    target_diffs: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in reference.cases:
        ours = scored[case.id]
        expected_shape = (case.num_entries,)
        if ours.target_logprobs.shape != expected_shape or ours.reference_logprobs.shape[0] != case.num_entries:
            raise ValueError(f"{case.id}: backend scores have the wrong number of entries")
        for position in range(case.num_entries):
            # Entry j predicts token j + 1.
            groups = [case_group(case.id)]
            if position + 1 >= sliding_window:
                groups.append(PAST_WINDOW_GROUP)
            values = case.top_logprobs[position]
            measured_row = ours.reference_logprobs[position, : len(values)]
            if np.all(np.isnan(measured_row)):
                outcome = "top1_disagree"
            elif int(np.nanargmax(measured_row)) == 0:
                outcome = "top1_agree"
            elif len(values) > 1 and values[0] - values[1] < NEAR_TIE:
                outcome = "top1_near_tie"
            else:
                outcome = "top1_disagree"
            position_diffs, missing = [], 0
            for rank, value in enumerate(values):
                if value < floor:
                    continue
                measured = float(ours.reference_logprobs[position, rank])
                if math.isnan(measured):
                    missing += 1
                    continue
                position_diffs.append(abs(measured - value))
            target = float(ours.target_logprobs[position])
            for group in groups:
                counts[group]["positions"] += 1
                counts[group][outcome] += 1
                counts[group]["missing"] += missing
                counts[group][f"attention:{attention_by_case[case.id]}"] += 1
                diffs[group].extend(position_diffs)
                if math.isfinite(target):
                    target_diffs[group].append(abs(target - case.target_logprobs[position]))
                else:
                    counts[group]["target_not_finite"] += 1
    return {
        group: GroupStats(
            positions=counts[group]["positions"],
            tokens_compared=len(diffs[group]),
            missing=counts[group]["missing"],
            max_abs_logprob_diff=max(diffs[group], default=float("nan")),
            p99_abs_logprob_diff=quantile(diffs[group], 0.99),
            p50_abs_logprob_diff=quantile(diffs[group], 0.50),
            max_abs_target_logprob_diff=max(target_diffs[group], default=float("nan")),
            p99_abs_target_logprob_diff=quantile(target_diffs[group], 0.99),
            top1_agree=counts[group]["top1_agree"],
            top1_near_tie=counts[group]["top1_near_tie"],
            top1_disagree=counts[group]["top1_disagree"],
            target_not_finite=counts[group]["target_not_finite"],
            attention_positions={
                key.removeprefix("attention:"): count
                for key, count in sorted(counts[group].items())
                if key.startswith("attention:")
            },
        )
        for group in sorted(counts)
    }


def logprob_failures(
    stats: Mapping[str, GroupStats],
    *,
    expected_groups: Iterable[str],
    max_p99: float = MAX_P99_LOGPROB_ERROR,
    max_abs: float = MAX_LOGPROB_ERROR,
    max_target_p99: float = MAX_P99_TARGET_LOGPROB_ERROR,
    max_target_abs: float = MAX_TARGET_LOGPROB_ERROR,
) -> list[str]:
    """Every gate failure; empty is a pass. Fail closed: a missing group, an unmeasured token, a non-finite target
    score or a NaN statistic fails."""

    def within(value: float, limit: float) -> bool:
        return value <= limit  # NaN fails

    failures = []
    for group in sorted(set(expected_groups)):
        group_stats = stats.get(group)
        if group_stats is None or group_stats.tokens_compared == 0:
            failures.append(f"{group}: no compared tokens")
            continue
        if group_stats.missing:
            failures.append(f"{group}: {group_stats.missing} reference tokens not measured")
        if not within(group_stats.p99_abs_logprob_diff, max_p99):
            failures.append(f"{group}: p99 abs log-prob diff {group_stats.p99_abs_logprob_diff:.4f} > {max_p99}")
        if not within(group_stats.max_abs_logprob_diff, max_abs):
            failures.append(f"{group}: max abs log-prob diff {group_stats.max_abs_logprob_diff:.4f} > {max_abs}")
        if group_stats.target_not_finite:
            failures.append(f"{group}: {group_stats.target_not_finite} target log-probs not finite")
        if not within(group_stats.p99_abs_target_logprob_diff, max_target_p99):
            failures.append(
                f"{group}: p99 abs target log-prob diff {group_stats.p99_abs_target_logprob_diff:.4f} > {max_target_p99}"
            )
        if not within(group_stats.max_abs_target_logprob_diff, max_target_abs):
            failures.append(
                f"{group}: max abs target log-prob diff {group_stats.max_abs_target_logprob_diff:.4f} > {max_target_abs}"
            )
    return failures


def expected_groups(reference: PositionReference, *, sliding_window: int = SLIDING_WINDOW) -> set[str]:
    groups = {case_group(case.id) for case in reference.cases}
    if any(case.num_entries >= sliding_window for case in reference.cases):
        groups.add(PAST_WINDOW_GROUP)
    return groups


def dispatch_failures(sender_dropped: Sequence[int], receiver_dropped: Sequence[int]) -> list[str]:
    """The forward must be dropless, as vLLM is: any dropped valid assignment in any layer fails."""
    return [
        f"layer {layer}: {sender + receiver} valid assignments dropped ({sender} sender, {receiver} receiver)"
        for layer, (sender, receiver) in enumerate(zip(sender_dropped, receiver_dropped, strict=True))
        if sender or receiver
    ]
