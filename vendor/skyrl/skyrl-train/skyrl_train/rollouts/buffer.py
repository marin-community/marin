"""Rollout buffer: leases generation capacity to rollout workers and selects training batches.

A rollout worker writes each completed prompt group through a ``RolloutWriter``. The writer checks the
group's content where the payload already is, stores the payload in the run's payload store
(``skyrl_train.rollouts.payloads``), and commits a small verdict together with a reference to the payload, so
the buffer, a Ray actor, selects batches without reading payloads. The trainer fetches only the payloads it
trains on.
"""

from __future__ import annotations

import asyncio
import collections
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdmissionStalledError,
    TrainingGroupInvariantError,
)
from skyrl_train.rollout_observability import rollout_wait
from skyrl_train.rollouts.loader import JudgedGroup
from skyrl_train.telemetry import GeneratedWork
from skyrl_train.trajectory_runners.trajectory_processing import get_outcome_rewards, get_trajectory_passes
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


class BatchPolicy(StrEnum):
    """How committed groups are assigned to training batches; the two coincide at ``max_staleness_steps=0``."""

    FULL_BATCH = "full_batch"
    ROLLING = "rolling"


@dataclass(frozen=True)
class RolloutBufferConfig:
    """Batch shape, generation concurrency, and selection rules of one training run.

    ``max_in_flight`` caps concurrent rollouts and ``max_candidate_groups`` bounds the dynamic-sampling candidates
    one batch may inspect; None leaves either unbounded.
    """

    batch_size: int
    max_in_flight: int | None
    max_staleness_steps: int
    batch_policy: BatchPolicy
    dynamic_sampling: DynamicSamplingType | None
    max_candidate_groups: int | None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"rollout batch size must be positive, got {self.batch_size}")
        if self.max_in_flight is not None and self.max_in_flight < 1:
            raise ValueError(f"max in-flight rollouts must be positive, got {self.max_in_flight}")
        if self.max_staleness_steps < 0:
            raise ValueError(f"max_staleness_steps must be non-negative, got {self.max_staleness_steps}")

    @property
    def max_untrained_groups(self) -> int:
        """Groups leased, committed, or batched that the next ``max_staleness_steps + 1`` steps can train on."""
        return (self.max_staleness_steps + 1) * self.batch_size

    @property
    def max_concurrent_rollouts(self) -> int:
        """The most groups that can generate at once."""
        if self.max_in_flight is None:
            return self.max_untrained_groups
        return min(self.max_in_flight, self.max_untrained_groups)


@dataclass(frozen=True)
class RolloutLease:
    """Permission to generate one prompt group for batch ``batch_id`` with the policy published at ``policy_step``.

    A batch id is the training step expected to train the group; the batch policy decides whether it must.
    """

    lease_id: str
    policy_step: int
    batch_id: int


@dataclass(frozen=True)
class RolloutTask:
    """One prompt group to generate under a buffer lease."""

    lease: RolloutLease
    prompt: dict
    request: TrajectoryRequestBatch


@dataclass
class RolloutGroup:
    """One prompt's samples, generated under one lease."""

    trajectory_batch: TrajectoryBatch
    uid: str
    policy_step: int
    prompt: dict


@dataclass(frozen=True)
class GroupRewards:
    """Per-sample rewards of one group: each sample's optimization reward total and its outcome reward.

    ``passed`` is whether any sample succeeded, by its verifier's verdict when it has one.
    """

    optimization: tuple[float, ...]
    outcome: tuple[float, ...]
    passed: bool

    @classmethod
    def from_batch(cls, batch: TrajectoryBatch) -> GroupRewards:
        return cls(
            optimization=tuple(NormalizedReward.from_output(reward).total for reward in batch["rewards"]),
            outcome=tuple(get_outcome_rewards(batch)),
            passed=any(get_trajectory_passes(batch)),
        )


@dataclass(frozen=True)
class RolloutVerdict:
    """What the buffer needs to know about a group's content to select it.

    ``selection`` and ``rewards`` are None when a content rejection already excludes the group.
    """

    uid: str
    rejections: tuple[AdmissionRejection, ...]
    selection: GroupSelectionResult | None
    rewards: GroupRewards | None
    work: GeneratedWork

    @property
    def trainable(self) -> bool:
        return not self.rejections and self.selection is GroupSelectionResult.KEEP


@dataclass(frozen=True)
class RolloutContentPolicy:
    """Content checks that do not depend on when a group is read."""

    admission: GroupAdmissionPolicy
    selection: GroupSelectionPolicy

    def verdict(self, group: RolloutGroup) -> RolloutVerdict:
        """Judge one group, raising when it violates the run's structural contract."""
        decision = self.admission.evaluate(group)
        if decision.fatal:
            raise TrainingGroupInvariantError.from_generated_group(
                uid=group.uid, group=group, decision=decision, invariant=self.admission.invariant
            )
        batch = group.trajectory_batch
        work = GeneratedWork.from_batch(batch["response_ids"], batch.get("is_last_step"))
        if not decision.accepted:
            return RolloutVerdict(group.uid, decision.rejections, None, None, work)
        return RolloutVerdict(group.uid, (), self.selection.evaluate(group), GroupRewards.from_batch(batch), work)


class RolloutWriter(Protocol):
    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None: ...


async def write_trajectory_batch(task: RolloutTask, writer: RolloutWriter, batch: TrajectoryBatch) -> int:
    """Write a completed batch under its prompt group's lease."""
    with rollout_wait("enqueue"):
        await writer.write_rollout(
            task.lease, RolloutGroup(batch, task.prompt["uid"], task.lease.policy_step, task.prompt)
        )
    return sum(len(response) for response in batch["response_ids"])


@dataclass(frozen=True)
class PayloadReference:
    """Nest one reference so Ray passes an ObjectRef to ``commit`` without resolving it."""

    value: object | None


@dataclass
class ReadyRollout:
    """A committed group that no batch has taken yet.

    ``payload`` holds one reference to the ``RolloutGroup``; it is None when the verdict
    excludes the group.
    ``committed_at`` is the buffer process's monotonic time at commit, and None for a group restored from a
    checkpoint.
    """

    lease_id: str
    policy_step: int
    batch_id: int
    prompt: dict
    verdict: RolloutVerdict
    payload: object | None
    committed_at: float | None


@dataclass(frozen=True)
class GroupDisposition:
    """How one committed group left the buffer: ``consumed`` by a batch or an admission or selection outcome."""

    uid: str
    disposition: str
    tokens: int
    dwell_seconds: float | None


@dataclass(frozen=True)
class BufferSnapshot:
    """Committed groups, prompts awaiting regeneration, and leases not yet committed, for a checkpoint."""

    ready: list[ReadyRollout]
    retries: list[dict]
    leases: frozenset[str]


@dataclass(frozen=True)
class BatchSelection:
    """How a completed batch was selected: its metrics and every group judged for it, kept or discarded."""

    metrics: dict[str, float]
    judged: list[JudgedGroup]


@dataclass(frozen=True)
class AdmittedRollout:
    """One selected group's stable batch index, identity, and size."""

    index: int
    uid: str
    policy_step: int
    sample_count: int
    response_tokens: int


@dataclass(frozen=True)
class Admission:
    """Progress toward the current batch since the previous ``admit`` call.

    ``dispositions`` holds every committed group that left the buffer, so its uid may be generated again.
    ``selection`` is set only on the call that completes the batch.
    """

    batch_id: int
    admitted: list[AdmittedRollout]
    retries: list[dict]
    generated: list[tuple[int, GeneratedWork]]
    dispositions: list[GroupDisposition]
    ready_count: int
    selection: BatchSelection | None


@dataclass
class _SelectionStats:
    """Admission and dynamic-sampling outcomes since the previous batch."""

    inspected: int = 0
    rejections: collections.Counter[AdmissionRejection] = field(default_factory=collections.Counter)
    candidates: int = 0
    candidate_samples: int = 0
    samples_per_group: int | None = None
    optimization_reward_sum: float = 0.0
    outcome_reward_sum: float = 0.0
    passed: int = 0
    dynamic_discarded: int = 0
    judged: list[JudgedGroup] = field(default_factory=list)

    def reject(self, rejection: AdmissionRejection) -> None:
        self.inspected += 1
        self.rejections[rejection] += 1

    def observe_candidate(self, rewards: GroupRewards) -> None:
        sample_count = len(rewards.optimization)
        if self.samples_per_group is None:
            self.samples_per_group = sample_count
        elif self.samples_per_group != sample_count:
            raise ValueError(
                "dynamic sampling candidates must have a consistent physical group size: "
                f"got {self.samples_per_group} and {sample_count}"
            )
        self.candidates += 1
        self.candidate_samples += sample_count
        self.optimization_reward_sum += sum(rewards.optimization)
        self.outcome_reward_sum += sum(rewards.outcome)
        self.passed += int(rewards.passed)

    def metrics(self, dynamic_sampling: DynamicSamplingType | None) -> dict[str, float]:
        rejected = sum(self.rejections.values())
        metrics = {
            "async/rejected_count": rejected,
            "async/rejected_rate": rejected / self.inspected,
            **{f"async/rejected_count/{reason.value}": self.rejections[reason] for reason in AdmissionRejection},
        }
        if dynamic_sampling is DynamicSamplingType.FILTER:
            metrics.update(
                {
                    "async/dynamic_sampling/candidate_count": self.candidates,
                    "async/dynamic_sampling/discarded_count": self.dynamic_discarded,
                    "async/dynamic_sampling/discarded_rate": self.dynamic_discarded / self.candidates,
                    "async/dynamic_sampling/candidate_trajectory_count": self.candidate_samples,
                    "async/dynamic_sampling/candidate_optimization_reward_mean": (
                        self.optimization_reward_sum / self.candidate_samples
                    ),
                    "async/dynamic_sampling/candidate_outcome_reward_mean": (
                        self.outcome_reward_sum / self.candidate_samples
                    ),
                    f"async/dynamic_sampling/candidate_pass_at_{self.samples_per_group}": self.passed / self.candidates,
                }
            )
        return metrics


class AsyncRolloutPolicy(Protocol):
    """Which batch a rollout generates for, and which batch it trains in once committed.

    Batch ids are training steps: after ``publish(step)`` the trainer assembles batch ``step``.
    """

    def lease_batch(self, step: int, occupancy: collections.Counter[int]) -> int | None:
        """The batch a lease granted now generates for, or None when generation may not run further ahead.

        ``occupancy`` counts the leased, committed, and admitted groups of each batch id, including the batch the
        trainer has taken until the next publish.
        """
        ...

    def train_batch(self, rollout: ReadyRollout, open_batch: int, admitted: collections.Counter[int]) -> int | None:
        """The batch a committed group joins, or None when it is too stale to join any.

        ``open_batch`` is the earliest batch still taking groups and ``admitted`` counts each batch's admitted groups.
        """
        ...


@dataclass(frozen=True)
class FullBatchRolloutPolicy:
    """Train each batch on exactly the groups leased for it.

    Generation for the next ``max_staleness_steps`` batches runs while a batch completes, but a batch waits for its
    slowest group, and groups leased for later batches never join it. A lease opens only within the staleness window
    of its batch, so no group is ever too stale, and neither whether nor when a group trains depends on how long it
    took to generate.
    """

    config: RolloutBufferConfig

    def lease_batch(self, step: int, occupancy: collections.Counter[int]) -> int | None:
        for batch_id in range(step, step + self.config.max_staleness_steps + 1):
            if occupancy[batch_id] < self.config.batch_size:
                return batch_id
        return None

    def train_batch(self, rollout: ReadyRollout, open_batch: int, admitted: collections.Counter[int]) -> int | None:
        if rollout.batch_id < open_batch:
            # Only a checkpoint written under the rolling policy holds such a group.
            raise ValueError(
                f"group {rollout.verdict.uid} was generated for batch {rollout.batch_id}, but batch {open_batch} is "
                f"the earliest still open; resume a {BatchPolicy.ROLLING} checkpoint with that batch policy"
            )
        return rollout.batch_id


@dataclass(frozen=True)
class RollingBatchPolicy:
    """Train groups in the order they commit.

    A batch never waits for a slow group: it takes the first groups to commit after the previous batch filled. Groups
    that generate quickly therefore train sooner and at lower staleness than slow ones, and a group that would train
    more than ``max_staleness_steps`` after its lease's policy step is discarded and its prompt regenerated. A prompt
    whose rollouts outlast that window never trains, and for others the regenerated groups that do train are the ones
    that happened to finish quickly.
    """

    config: RolloutBufferConfig

    def lease_batch(self, step: int, occupancy: collections.Counter[int]) -> int | None:
        untrained = sum(occupancy.values())
        if untrained >= self.config.max_untrained_groups:
            return None
        # The batch the group joins if groups commit in lease order.
        return step + untrained // self.config.batch_size

    def train_batch(self, rollout: ReadyRollout, open_batch: int, admitted: collections.Counter[int]) -> int | None:
        batch_id = open_batch
        while admitted[batch_id] >= self.config.batch_size:
            batch_id += 1
        if batch_id - rollout.policy_step > self.config.max_staleness_steps:
            return None
        return batch_id


def async_rollout_policy(config: RolloutBufferConfig) -> AsyncRolloutPolicy:
    if config.batch_policy is BatchPolicy.FULL_BATCH:
        return FullBatchRolloutPolicy(config)
    return RollingBatchPolicy(config)


class RolloutBuffer:
    """Lease accounting, committed groups, and batch selection for one training run.

    The trainer publishes each policy step after syncing its weights to inference. Leases open only after
    the first publish. Besides any ``max_in_flight`` cap, leases are bounded so that groups not yet trained (leased,
    committed, or in the current batch) never exceed ``max_staleness_steps + 1`` batches: a group leased now
    can only be trained within that many steps, so generating more would only produce stale work.

    Every commit is assigned to a batch by the batch policy and judged against it immediately: a group too stale for
    any batch returns its prompt for regeneration, a content rejection or duplicate UID is dropped, and dynamic
    sampling decides the rest. Groups admitted to a later batch wait for its step.
    """

    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self._policy = async_rollout_policy(config)
        self._policy_step = 0
        self._leases: dict[str, RolloutLease] = {}
        # Committed groups not yet assigned to a batch; only groups restored before the first publish wait here.
        self._ready: list[ReadyRollout] = []
        self._admitted: collections.defaultdict[int, list[ReadyRollout]] = collections.defaultdict(list)
        self._unreported: list[ReadyRollout] = []
        self._batch_taken = False
        self._retries: list[dict] = []
        self._generated: list[tuple[int, GeneratedWork]] = []
        self._dispositions: list[GroupDisposition] = []
        self._stats: collections.defaultdict[int, _SelectionStats] = collections.defaultdict(_SelectionStats)
        self._changed = asyncio.Condition()

    def _lease_batch(self) -> int | None:
        if self._policy_step == 0:
            return None
        if self.config.max_in_flight is not None and len(self._leases) >= self.config.max_in_flight:
            return None
        occupancy = collections.Counter(lease.batch_id for lease in self._leases.values())
        occupancy.update(rollout.batch_id for rollout in self._ready)
        occupancy.update({batch_id: len(groups) for batch_id, groups in self._admitted.items()})
        return self._policy.lease_batch(self._policy_step, occupancy)

    async def acquire_lease(self) -> RolloutLease:
        """Wait for generation capacity and lease it at the current policy step."""
        async with self._changed:
            await self._changed.wait_for(lambda: self._lease_batch() is not None)
            lease = RolloutLease(uuid.uuid4().hex, self._policy_step, self._lease_batch())
            self._leases[lease.lease_id] = lease
            return lease

    async def commit(self, lease_id: str, prompt: dict, verdict: RolloutVerdict, payload: PayloadReference) -> None:
        """Record a written group and release its lease."""
        async with self._changed:
            lease = self._leases.pop(lease_id)
            self._generated.append((lease.policy_step, verdict.work))
            self._ready.append(
                ReadyRollout(
                    lease_id, lease.policy_step, lease.batch_id, prompt, verdict, payload.value, time.monotonic()
                )
            )
            self._select()
            self._changed.notify_all()

    async def publish(self, policy_step: int) -> None:
        """Acknowledge the taken batch, if any, and lease at the newly synced policy step."""
        async with self._changed:
            if policy_step <= self._policy_step:
                raise ValueError(f"policy step must advance past {self._policy_step}, got {policy_step}")
            if self._policy_step and not self._batch_taken:
                raise RuntimeError("cannot publish a new policy step before taking the current batch")
            self._admitted.pop(self._policy_step, None)
            self._policy_step = policy_step
            self._unreported = list(self._admitted[policy_step])
            self._batch_taken = False
            self._select()
            self._changed.notify_all()

    def payload_refs(self, batch_id: int, indices: Sequence[int]) -> list[object]:
        """Resolve selected group indices while their batch is the current training step."""
        if batch_id != self._policy_step:
            raise ValueError(f"batch {batch_id} is no longer available; current batch is {self._policy_step}")
        batch = self._admitted[batch_id]
        if any(index < 0 or index >= len(batch) for index in indices):
            raise IndexError(f"group index outside batch {batch_id} with {len(batch)} admitted groups")
        refs: list[object] = []
        for index in indices:
            ref = batch[index].payload
            if ref is None:
                raise RuntimeError(f"batch {batch_id} contains an admitted group without a payload")
            refs.append(ref)
        return refs

    async def admit(self, timeout: float) -> Admission:
        """Wait up to ``timeout`` seconds for the current batch to progress.

        Returns newly admitted group metadata, prompts to regenerate, and groups that left the buffer. The call that
        completes the batch also returns how the batch was selected; the batch then counts as taken until the next
        ``publish``.

        Raises:
            GroupAdmissionStalledError: No group was admitted, returned for regeneration, or left the buffer within
                ``timeout``.
            RuntimeError: Dynamic sampling inspected a batch's candidate budget without filling it.
        """
        async with self._changed:
            if self._batch_taken:
                raise RuntimeError("the current batch was already taken; publish the next policy step first")
            try:
                await asyncio.wait_for(
                    self._changed.wait_for(
                        lambda: self._unreported
                        or self._retries
                        or self._dispositions
                        or self._batch_complete()
                        or self._over_budget_batch() is not None
                    ),
                    timeout,
                )
            except TimeoutError as error:
                raise GroupAdmissionStalledError(
                    f"no rollout group admitted for {timeout:.0f}s: policy_step={self._policy_step} "
                    f"admitted={len(self._admitted[self._policy_step])}/{self.config.batch_size} "
                    f"leases={len(self._leases)} rejections={dict(self._stats[self._policy_step].rejections)}"
                ) from error
            selection = None
            if self._batch_complete():
                for rollout in self._admitted[self._policy_step]:
                    self._dispose(rollout, "consumed")
                stats = self._stats.pop(self._policy_step)
                selection = BatchSelection(stats.metrics(self.config.dynamic_sampling), stats.judged)
                self._batch_taken = True
            elif (batch_id := self._over_budget_batch()) is not None:
                raise RuntimeError(
                    "dynamic sampling inspected its limit of "
                    f"{self.config.max_candidate_groups} candidate groups for batch {batch_id} with "
                    f"{len(self._admitted[batch_id])} of {self.config.batch_size} admitted"
                )
            admission = Admission(
                batch_id=self._policy_step,
                admitted=[
                    AdmittedRollout(
                        index=index,
                        uid=rollout.verdict.uid,
                        policy_step=rollout.policy_step,
                        sample_count=rollout.verdict.work.sample_count,
                        response_tokens=rollout.verdict.work.generated_token_count,
                    )
                    for index, rollout in enumerate(
                        self._unreported, start=len(self._admitted[self._policy_step]) - len(self._unreported)
                    )
                ],
                retries=self._retries,
                generated=self._generated,
                dispositions=self._dispositions,
                ready_count=sum(
                    len(groups) for batch_id, groups in self._admitted.items() if batch_id > self._policy_step
                ),
                selection=selection,
            )
            self._unreported, self._retries, self._generated, self._dispositions = [], [], [], []
            self._changed.notify_all()
            return admission

    def snapshot(self) -> BufferSnapshot:
        """Copy committed groups not yet in a taken batch, and prompts awaiting regeneration."""
        admitted = [
            rollout
            for batch_id, groups in sorted(self._admitted.items())
            if not (self._batch_taken and batch_id == self._policy_step)
            for rollout in groups
        ]
        return BufferSnapshot(
            ready=[*admitted, *self._ready], retries=list(self._retries), leases=frozenset(self._leases)
        )

    async def restore(self, snapshot: BufferSnapshot) -> None:
        """Load a checkpoint's groups into an unpublished, empty buffer."""
        async with self._changed:
            if self._policy_step or self._leases or self._ready or self._retries or any(self._admitted.values()):
                raise RuntimeError("a rollout buffer can only be restored before training starts")
            self._ready.extend(snapshot.ready)
            self._retries.extend(snapshot.retries)

    def _reject(self, stats: _SelectionStats, rollout: ReadyRollout, rejection: AdmissionRejection) -> None:
        stats.reject(rejection)
        self._dispose(rollout, rejection.value)

    def _dispose(self, rollout: ReadyRollout, disposition: str) -> None:
        dwell = None if rollout.committed_at is None else time.monotonic() - rollout.committed_at
        self._dispositions.append(
            GroupDisposition(rollout.verdict.uid, disposition, rollout.verdict.work.generated_token_count, dwell)
        )

    def _batch_complete(self) -> bool:
        return not self._batch_taken and len(self._admitted[self._policy_step]) == self.config.batch_size

    def _over_budget_batch(self) -> int | None:
        """A batch whose dynamic-sampling candidates reached the limit before it filled, if any."""
        limit = self.config.max_candidate_groups
        if limit is None:
            return None
        for batch_id, stats in self._stats.items():
            if stats.candidates >= limit and len(self._admitted[batch_id]) < self.config.batch_size:
                return batch_id
        return None

    def _select(self) -> None:
        """Assign committed groups to batches in arrival order and judge each against its batch."""
        if self._policy_step == 0:
            return
        open_batch = self._policy_step + int(self._batch_taken)
        admitted = collections.Counter({batch_id: len(groups) for batch_id, groups in self._admitted.items()})
        for rollout in self._ready:
            verdict = rollout.verdict
            batch_id = self._policy.train_batch(rollout, open_batch, admitted)
            if batch_id is None:
                self._reject(self._stats[open_batch], rollout, AdmissionRejection.STALE)
                self._retries.append(rollout.prompt)
                continue
            stats = self._stats[batch_id]
            batch = self._admitted[batch_id]
            if verdict.rejections:
                self._reject(stats, rollout, verdict.rejections[0])
            elif any(admitted_rollout.verdict.uid == verdict.uid for admitted_rollout in batch):
                self._reject(stats, rollout, AdmissionRejection.DUPLICATE_UID)
            else:
                stats.inspected += 1
                stats.judged.append(JudgedGroup(verdict.uid, verdict.rewards.optimization))
                if self.config.dynamic_sampling is DynamicSamplingType.FILTER:
                    stats.observe_candidate(verdict.rewards)
                if verdict.selection is not GroupSelectionResult.KEEP:
                    stats.dynamic_discarded += 1
                    self._dispose(rollout, verdict.selection.value)
                    continue
                batch.append(rollout)
                admitted[batch_id] += 1
                if batch_id == self._policy_step:
                    self._unreported.append(rollout)
        self._ready = []
