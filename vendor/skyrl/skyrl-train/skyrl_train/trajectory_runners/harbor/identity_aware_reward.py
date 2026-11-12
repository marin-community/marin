from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from numbers import Real

from skyrl_train.trajectory_runners.types import VerifierTestCollection, VerifierTestRecord
from skyrl_train.utils.reward_shaping import VerifierTestOutcome


IDENTITY_AWARE_SHAPER = "identity_aware_pass_ratio"

TestWeightFilter = Callable[[VerifierTestRecord], float]


class IdentityAwareFallback(StrEnum):
    NO_ELIGIBLE_TRIALS = "no_eligible_trials"
    INCOMPLETE_TESTS = "incomplete_tests"
    MISMATCHED_TESTS = "mismatched_tests"


@dataclass(frozen=True)
class IdentityAwareRewardResult:
    rewards: tuple[float, ...]
    informative_test_count: int
    fallback_reason: IdentityAwareFallback | None = None


def _weight(value: object, *, source: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{source} must be a finite non-negative number, got {value!r}")
    return float(value)


def _record_weight(
    record: VerifierTestRecord,
    exact_weights: Mapping[str, float],
    weight_filter: TestWeightFilter | None,
) -> float:
    weight = _weight(exact_weights.get(record["record_id"], 1.0), source=f"weight for {record['record_id']}")
    if weight_filter is None:
        return weight
    return weight * _weight(weight_filter(record), source=f"filter weight for {record['record_id']}")


def identity_aware_pass_ratios(
    collections: Sequence[VerifierTestCollection | None],
    aggregate_rewards: Sequence[float],
    baseline_eligible: Sequence[bool],
    *,
    exact_weights: Mapping[str, float] | None = None,
    weight_filter: TestWeightFilter | None = None,
) -> IdentityAwareRewardResult:
    """Score non-uniform test outcomes across one rollout group.

    Incomplete or mismatched test evidence returns the aggregate rewards unchanged.
    Exact weights use the deterministic test-and-trial ``record_id``. A post-hoc
    filter returns a multiplier; a zero multiplier excludes the record.
    """
    if not (len(collections) == len(aggregate_rewards) == len(baseline_eligible)):
        raise ValueError("collections, aggregate_rewards, and baseline_eligible must have equal length")

    eligible_indices = [index for index, eligible in enumerate(baseline_eligible) if eligible]
    if not eligible_indices:
        return IdentityAwareRewardResult(
            rewards=tuple(aggregate_rewards),
            informative_test_count=0,
            fallback_reason=IdentityAwareFallback.NO_ELIGIBLE_TRIALS,
        )

    eligible_collections = [collections[index] for index in eligible_indices]
    if any(collection is None or not collection["complete"] for collection in eligible_collections):
        return IdentityAwareRewardResult(
            rewards=tuple(aggregate_rewards),
            informative_test_count=0,
            fallback_reason=IdentityAwareFallback.INCOMPLETE_TESTS,
        )

    records_by_id: list[dict[str, VerifierTestRecord]] = []
    for collection in eligible_collections:
        assert collection is not None
        records = {record["test_id"]: record for record in collection["tests"]}
        if len(records) != len(collection["tests"]):
            return IdentityAwareRewardResult(
                rewards=tuple(aggregate_rewards),
                informative_test_count=0,
                fallback_reason=IdentityAwareFallback.MISMATCHED_TESTS,
            )
        records_by_id.append(records)

    expected_ids = set(records_by_id[0])
    if not expected_ids or any(set(records) != expected_ids for records in records_by_id[1:]):
        return IdentityAwareRewardResult(
            rewards=tuple(aggregate_rewards),
            informative_test_count=0,
            fallback_reason=IdentityAwareFallback.MISMATCHED_TESTS,
        )

    informative_ids = tuple(
        sorted(
            test_id
            for test_id in expected_ids
            if len({records[test_id]["outcome"] == VerifierTestOutcome.PASSED.value for records in records_by_id}) > 1
        )
    )
    rewards = list(aggregate_rewards)
    weights = exact_weights or {}
    for index, records in zip(eligible_indices, records_by_id, strict=True):
        passed_weight = 0.0
        total_weight = 0.0
        for test_id in informative_ids:
            record = records[test_id]
            record_weight = _record_weight(record, weights, weight_filter)
            total_weight += record_weight
            if record["outcome"] == VerifierTestOutcome.PASSED.value:
                passed_weight += record_weight
        rewards[index] = passed_weight / total_weight if total_weight else 0.0

    return IdentityAwareRewardResult(
        rewards=tuple(rewards),
        informative_test_count=len(informative_ids),
    )
