"""Algorithm-aware admission rules for generated rollout groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol, Sequence

import numpy as np


# The fewest eligible responses that define a group baseline.
MIN_BASELINE_GROUP_SIZE = 2
# Eligible reward ranges at or below this tolerance contribute no outcome advantage.
FLAT_GROUP_REWARD_TOLERANCE = 1e-6

_INITIAL_ADMISSION_STALL_TIMEOUT = 1800.0
_MINIMUM_ADMISSION_STALL_TIMEOUT = 600.0
_STEP_TIME_MULTIPLIER = 5.0


class GroupAdmissionStalledError(RuntimeError):
    """Admission made no progress before its shared deadline."""


def admission_stall_timeout(*, recent_step_times: Sequence[float], timeout_override: float | None) -> float:
    """Seconds admission may go without progress before training fails.

    An explicit override is returned unchanged. Otherwise the deadline is a multiple of the recent median step
    time, at least 10 minutes; before any step has been timed, it is 30 minutes.
    """
    if timeout_override is not None:
        if timeout_override <= 0:
            raise ValueError(f"group admission stall_timeout must be positive, got {timeout_override}")
        return float(timeout_override)
    if not recent_step_times:
        return _INITIAL_ADMISSION_STALL_TIMEOUT
    sorted_times = sorted(recent_step_times)
    median = sorted_times[len(sorted_times) // 2]
    return max(median * _STEP_TIME_MULTIPLIER, _MINIMUM_ADMISSION_STALL_TIMEOUT)


class GroupAdvantageKind(StrEnum):
    """Supported group-relative advantage contracts."""

    EXACT_PHYSICAL = "exact_physical"
    MINIMUM_BASELINE_ELIGIBLE = "minimum_baseline_eligible"
    NONE = "none"


@dataclass(frozen=True)
class GroupAdvantageInvariant:
    """Resolved per-run group contract shared by admission and advantage math."""

    kind: GroupAdvantageKind
    physical_group_size: int
    minimum_group_size: int | None

    def __post_init__(self) -> None:
        if self.physical_group_size < 1:
            raise ValueError(f"physical_group_size must be positive, got {self.physical_group_size}")
        if self.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
            if self.minimum_group_size is None:
                raise ValueError("minimum_baseline_eligible requires minimum_group_size")
            if not MIN_BASELINE_GROUP_SIZE <= self.minimum_group_size <= self.physical_group_size:
                raise ValueError(
                    f"minimum_group_size must be between {MIN_BASELINE_GROUP_SIZE} and physical_group_size, got "
                    f"{self.minimum_group_size} and {self.physical_group_size}"
                )
        elif self.minimum_group_size is not None:
            raise ValueError(f"{self.kind} does not accept minimum_group_size")

    @classmethod
    def exact_physical(cls, *, physical_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.EXACT_PHYSICAL, physical_group_size, None)

    @classmethod
    def minimum_baseline_eligible(cls, *, physical_group_size: int, minimum_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE, physical_group_size, minimum_group_size)

    @classmethod
    def no_group_advantage(cls, *, physical_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.NONE, physical_group_size, None)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> GroupAdvantageInvariant:
        return cls(
            kind=GroupAdvantageKind(str(config["kind"])),
            physical_group_size=int(config["physical_group_size"]),
            minimum_group_size=(
                int(config["minimum_group_size"]) if config.get("minimum_group_size") is not None else None
            ),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "physical_group_size": self.physical_group_size,
            "minimum_group_size": self.minimum_group_size,
        }


def validate_group_baseline(invariant: GroupAdvantageInvariant) -> None:
    """Require enough samples per prompt to train a group-relative estimator."""
    if invariant.kind is not GroupAdvantageKind.NONE and invariant.physical_group_size < MIN_BASELINE_GROUP_SIZE:
        raise ValueError(
            f"a group-relative advantage estimator needs at least {MIN_BASELINE_GROUP_SIZE} samples per prompt, "
            f"got {invariant.physical_group_size}; raise generator.n_samples_per_prompt or use advantage_estimator=reward"
        )


class AdmissionRejection(StrEnum):
    """Reasons that prevent a completed group from entering training."""

    STALE = "stale"
    FULLY_MASKED = "fully_masked"
    PHYSICAL_GROUP_SIZE = "physical_group_size"
    BELOW_MINIMUM_GROUP_SIZE = "below_minimum_group_size"
    MISSING_ROLLOUT_LOGPROBS = "missing_rollout_logprobs"
    MISSING_BEHAVIOR_TOPK = "missing_behavior_topk"
    DUPLICATE_UID = "duplicate_uid"


# Rejections that mean the harness broke the run's structural contract; they fail training.
FATAL_REJECTIONS = frozenset(
    {
        AdmissionRejection.PHYSICAL_GROUP_SIZE,
        AdmissionRejection.MISSING_ROLLOUT_LOGPROBS,
        AdmissionRejection.MISSING_BEHAVIOR_TOPK,
    }
)


@dataclass(frozen=True)
class AdmissionDecision:
    """Pure admission result; queue routing is owned by the caller."""

    rejections: tuple[AdmissionRejection, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.rejections

    @property
    def primary_rejection(self) -> AdmissionRejection | None:
        return self.rejections[0] if self.rejections else None

    @property
    def fatal(self) -> bool:
        return any(rejection in FATAL_REJECTIONS for rejection in self.rejections)


@dataclass(frozen=True)
class BatchGroupAdmission:
    """Admission result for one UID group in a concatenated batch."""

    uid: str
    row_indices: tuple[int, ...]
    decision: AdmissionDecision
    physical_count: int


class TrainingGroupInvariantError(ValueError):
    """A concatenated training group violates its resolved run contract."""

    def __init__(
        self,
        *,
        uid: str,
        decision: AdmissionDecision,
        physical_count: int,
        expected_physical_count: int,
        row_indices: Sequence[int],
    ) -> None:
        self.uid = uid
        self.rejections = decision.rejections
        self.physical_count = physical_count
        self.expected_physical_count = expected_physical_count
        self.row_indices = tuple(row_indices)
        reasons = ", ".join(rejection.value for rejection in decision.rejections)
        super().__init__(
            f"training group {uid!r} violates the resolved group invariant: {reasons} "
            f"(physical_count={physical_count}, expected={expected_physical_count}, "
            f"row_count={len(row_indices)}, row_indices={list(row_indices)})"
        )

    @classmethod
    def from_generated_group(
        cls,
        *,
        uid: str,
        group: GeneratedGroup,
        decision: AdmissionDecision,
        invariant: GroupAdvantageInvariant,
    ) -> TrainingGroupInvariantError:
        """Build the same typed invariant error for an asynchronous group."""
        facts = _inspect_group(group)
        response_ids = group.trajectory_batch["response_ids"]
        assert isinstance(response_ids, Sequence)
        return cls(
            uid=uid,
            decision=decision,
            physical_count=facts.physical_count,
            expected_physical_count=invariant.physical_group_size,
            row_indices=range(len(response_ids)),
        )


class GeneratedGroup(Protocol):
    trajectory_batch: Mapping[str, object]


@dataclass(frozen=True)
class _BatchGroup:
    trajectory_batch: Mapping[str, object]


def _aligned_sequence(batch: Mapping[str, object], key: str, row_count: int) -> Sequence[object] | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence when present")
    if len(value) != row_count:
        raise ValueError(f"{key} must have one entry per response row, got {len(value)} and {row_count}")
    return value


def final_row_mask(trajectory_batch: Mapping[str, object]) -> np.ndarray:
    """Identify each trial's final response row."""
    responses = trajectory_batch.get("response_ids")
    if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    final = _aligned_sequence(trajectory_batch, "is_last_step", len(responses))
    return np.ones(len(responses), dtype=bool) if final is None else np.asarray(final, dtype=bool)


def baseline_eligible_mask(trajectory_batch: Mapping[str, object]) -> np.ndarray:
    """Identify final rows included in their prompt group's baseline."""
    final = final_row_mask(trajectory_batch)
    exclusions = _aligned_sequence(trajectory_batch, "exclude_from_baseline", len(final))
    return final if exclusions is None else final & ~np.asarray(exclusions, dtype=bool)


def rewards_are_flat(rewards: Sequence[float]) -> bool:
    """Return whether the eligible reward range is within the flat-group tolerance."""
    return max(rewards) - min(rewards) <= FLAT_GROUP_REWARD_TOLERANCE


@dataclass(frozen=True)
class _GroupFacts:
    physical_count: int
    trainable_count: int
    baseline_contributor_count: int
    has_rollout_logprobs: bool


def _inspect_group(group: GeneratedGroup) -> _GroupFacts:
    batch = group.trajectory_batch
    response_ids_value = batch.get("response_ids")
    if not isinstance(response_ids_value, Sequence) or isinstance(response_ids_value, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    if len(response_ids_value) == 0:
        raise ValueError("response_ids must contain at least one response row")
    response_ids = response_ids_value
    row_count = len(response_ids)
    loss_masks = _aligned_sequence(batch, "loss_masks", row_count)
    assert loss_masks is not None

    for row_index, (response, loss_mask) in enumerate(zip(response_ids, loss_masks, strict=True)):
        if not isinstance(response, Sequence) or not isinstance(loss_mask, Sequence):
            raise ValueError(f"response_ids and loss_masks row {row_index} must be sequences")
        if len(response) != len(loss_mask):
            raise ValueError(
                f"loss_masks row {row_index} must align with response_ids, got {len(loss_mask)} and {len(response)}"
            )

    final = final_row_mask(batch)
    if not final.any():
        raise ValueError("is_last_step must identify at least one final trial row")
    baseline_contributor_count = int(baseline_eligible_mask(batch).sum())

    rollout_logprobs = _aligned_sequence(batch, "rollout_logprobs", row_count)
    has_trainable_rollout_logprobs = rollout_logprobs is not None
    if rollout_logprobs is not None:
        for row_index, (response, loss_mask, logprobs) in enumerate(
            zip(response_ids, loss_masks, rollout_logprobs, strict=True)
        ):
            if not isinstance(logprobs, np.ndarray) or logprobs.ndim != 1:
                raise ValueError(f"rollout_logprobs row {row_index} must be a one-dimensional array")
            if len(response) != len(logprobs):
                raise ValueError(
                    f"rollout_logprobs row {row_index} must align with response_ids, "
                    f"got {len(logprobs)} and {len(response)}"
                )
            trained_logprobs = logprobs[np.asarray(loss_mask, dtype=bool)]
            if not np.isfinite(trained_logprobs).all() or (trained_logprobs > 0).any():
                has_trainable_rollout_logprobs = False

    return _GroupFacts(
        physical_count=int(final.sum()),
        trainable_count=sum(any(bool(token) for token in loss_mask) for loss_mask in loss_masks),
        baseline_contributor_count=baseline_contributor_count,
        has_rollout_logprobs=has_trainable_rollout_logprobs,
    )


def group_is_fully_excluded_from_training(trajectory_batch: Mapping[str, object]) -> bool:
    """Return whether a group contributes neither loss tokens nor a group baseline."""
    facts = _inspect_group(_BatchGroup(trajectory_batch))
    return facts.trainable_count == 0 and facts.baseline_contributor_count == 0


def _has_behavior_topk(batch: Mapping[str, object], width: int) -> bool:
    masks = batch["loss_masks"]
    indices = batch.get("student_topk_indices")
    logprobs = batch.get("behavior_topk_logprobs")
    for values in (indices, logprobs):
        if not isinstance(values, Sequence) or len(values) != len(masks):
            return False
    for mask, row_indices, row_logprobs in zip(masks, indices, logprobs, strict=True):
        if not any(mask):
            continue
        for values in (row_indices, row_logprobs):
            if not isinstance(values, np.ndarray) or values.shape != (len(mask), width):
                return False
        if not np.issubdtype(row_indices.dtype, np.integer) or row_logprobs.dtype.kind not in "iuf":
            return False
        eligible = np.asarray(mask, dtype=bool)
        if (row_indices[eligible] < 0).any() or not np.isfinite(row_logprobs[eligible]).all():
            return False
    return True


class GroupAdmissionPolicy:
    """Check a completed group's content against the run's training contract.

    Staleness depends on when a group is read, so the rollout buffer checks it separately.
    """

    def __init__(
        self,
        invariant: GroupAdvantageInvariant,
        *,
        rollout_logprobs_required: bool,
        student_topk_width: int | None = None,
    ) -> None:
        self.invariant = invariant
        self.rollout_logprobs_required = rollout_logprobs_required
        self.student_topk_width = student_topk_width

    def evaluate(self, group: GeneratedGroup) -> AdmissionDecision:
        facts = _inspect_group(group)
        rejections = []
        if facts.trainable_count == 0:
            rejections.append(AdmissionRejection.FULLY_MASKED)
        if (
            self.invariant.kind is not GroupAdvantageKind.NONE
            and facts.physical_count != self.invariant.physical_group_size
        ):
            rejections.append(AdmissionRejection.PHYSICAL_GROUP_SIZE)
        if (
            self.invariant.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE
            and facts.baseline_contributor_count < self.invariant.minimum_group_size
        ):
            rejections.append(AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE)
        if self.rollout_logprobs_required and facts.trainable_count > 0 and not facts.has_rollout_logprobs:
            rejections.append(AdmissionRejection.MISSING_ROLLOUT_LOGPROBS)
        if (
            self.student_topk_width is not None
            and facts.trainable_count > 0
            and not _has_behavior_topk(group.trajectory_batch, self.student_topk_width)
        ):
            rejections.append(AdmissionRejection.MISSING_BEHAVIOR_TOPK)
        return AdmissionDecision(tuple(rejections))

    def evaluate_batch(
        self,
        trajectory_batch: Mapping[str, object],
        uids: Sequence[str],
    ) -> tuple[BatchGroupAdmission, ...]:
        """Evaluate each UID group in a concatenated trajectory batch."""
        response_ids = trajectory_batch.get("response_ids")
        if not isinstance(response_ids, Sequence) or isinstance(response_ids, (str, bytes)):
            raise ValueError("response_ids must be a sequence")
        if len(response_ids) != len(uids):
            raise ValueError(f"uids must have one entry per response row, got {len(uids)} and {len(response_ids)}")

        grouped_indices: dict[str, list[int]] = {}
        for row_index, uid in enumerate(uids):
            grouped_indices.setdefault(uid, []).append(row_index)

        aligned_keys = (
            "response_ids",
            "loss_masks",
            "exclude_from_baseline",
            "rollout_logprobs",
            "is_last_step",
            "student_topk_indices",
            "behavior_topk_logprobs",
        )
        admissions = []
        for uid, indices in grouped_indices.items():
            group_batch = {}
            for key in aligned_keys:
                value = _aligned_sequence(trajectory_batch, key, len(response_ids))
                if value is not None:
                    group_batch[key] = [value[index] for index in indices]
            batch_group = _BatchGroup(group_batch)
            facts = _inspect_group(batch_group)
            admissions.append(
                BatchGroupAdmission(
                    uid=uid,
                    row_indices=tuple(indices),
                    decision=self.evaluate(batch_group),
                    physical_count=facts.physical_count,
                )
            )
        return tuple(admissions)


def resolve_group_advantage_invariant(
    *, advantage_estimator: str, physical_group_size: int, minimum_group_size: int | None
) -> GroupAdvantageInvariant:
    """Resolve registry metadata and user input into one validated run invariant."""
    # Local import breaks the package cycle through skyrl_train.utils.__init__.
    from skyrl_train.utils.algorithm_registry import (  # noqa: PLC0415
        AdvantageEstimatorRegistry,
        ExactPhysicalGroup,
        MinimumBaselineEligibleGroup,
        NoGroupAdvantage,
    )

    contract = AdvantageEstimatorRegistry.group_contract(advantage_estimator)
    if isinstance(contract, MinimumBaselineEligibleGroup):
        if isinstance(minimum_group_size, bool) or not isinstance(minimum_group_size, int):
            raise ValueError(f"advantage estimator '{advantage_estimator}' requires integer group_advantage_min_size")
        return GroupAdvantageInvariant.minimum_baseline_eligible(
            physical_group_size=physical_group_size,
            minimum_group_size=minimum_group_size,
        )
    if minimum_group_size is not None:
        raise ValueError(
            f"advantage estimator '{advantage_estimator}' does not use group_advantage_min_size; set it to null"
        )
    if isinstance(contract, ExactPhysicalGroup):
        return GroupAdvantageInvariant.exact_physical(physical_group_size=physical_group_size)
    if isinstance(contract, NoGroupAdvantage):
        return GroupAdvantageInvariant.no_group_advantage(physical_group_size=physical_group_size)
    raise TypeError(f"unsupported group advantage contract: {type(contract).__name__}")


def assert_training_groups_eligible(
    trajectory_batch: Mapping[str, object],
    uids: Sequence[str],
    invariant: GroupAdvantageInvariant,
) -> None:
    """Fail when a synchronous or async training batch violates its group contract."""
    policy = GroupAdmissionPolicy(invariant, rollout_logprobs_required=False)
    for admission in policy.evaluate_batch(trajectory_batch, uids):
        if not admission.decision.accepted:
            raise TrainingGroupInvariantError(
                uid=admission.uid,
                decision=admission.decision,
                physical_count=admission.physical_count,
                expected_physical_count=invariant.physical_group_size,
                row_indices=admission.row_indices,
            )
