"""Rollout state of one training run and the coordinator loop that feeds its buffer."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import os
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marinskyrl.environment_contract import EXECUTION_UID_ENV, TrainingType
from marinskyrl.runtime_options import PolicyLossType, static_preference_pairs_requested
from marinskyrl.distillation import DistillationObjectiveKind, compile_distillation_plan_from_config
from skyrl_train.config.rollout_validation import (
    FAMILY_COLUMN,
    GROUPS_PER_FAMILY_KEY,
    SPARE_ENTRIES_PER_POOL,
    BatchPolicy,
    minimum_family_pool_sizes,
    validate_family_pools,
    validate_groups_per_family,
)
from skyrl_train.curriculum import CurriculumConfig, CurriculumOrder, SamplingKind
from skyrl_train.dataset import PromptDataset
from skyrl_train.domain_sampling import DomainWeightedOrder
from skyrl_train.dynamic_sampling import (
    DynamicSamplingType,
    GroupSelectionPolicy,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.grader_reporting import write_grader_report
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import (
    Admission,
    AdmittedGroupMetadata,
    BufferSnapshot,
    SubmittedGroup,
    RolloutBuffer,
    RolloutBufferConfig,
    RowFacts,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutGroupJob,
)
from skyrl_train.rollouts.loader import (
    DatasetEntryOrder,
    DatasetEntrySelector,
    DatasetEntrySelectorState,
    RolloutDataset,
    SeededPasses,
)
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads, PayloadStore
from skyrl_train.rollouts.payloads import RolloutReader, PAYLOAD_IO_THREADS
from skyrl_train.rollout_observability import dispatch_wait, observe_rollout_call, record_group_disposition
from skyrl_train.rollouts.workers import RolloutWorkers
from skyrl_train.telemetry import record_generated_work, record_rollout_buffer
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.config.objective_spec import rollout_logprobs_required
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

_T = TypeVar("_T")


@dataclass(frozen=True)
class RolloutRequestSpec:
    """How the coordinator turns one dataset entry into a trajectory request."""

    repetitions: int
    sampling_params: dict

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutRequestSpec:
        return cls(
            repetitions=config.generator.n_samples_per_prompt,
            sampling_params=get_sampling_params_for_backend(config.generator.backend, config.generator.sampling_params),
        )

    def request(self, entry: dict, policy_step: int) -> TrajectoryRequestBatch:
        request, _ = prepare_trajectory_request([entry], self.repetitions, self.sampling_params, "train", policy_step)
        return request


@dataclass(frozen=True)
class TrainingContextState:
    """Checkpointed rollout state: the selector, entries to regenerate, and submitted groups.

    Each submitted group's ``payload`` holds the durable form its payload store checkpoints: the ``RolloutGroup``
    for payloads in memory, or the URI of its object under ``object_store_root``.
    """

    selector: DatasetEntrySelectorState
    submitted: list[SubmittedGroup]
    object_store_root: str | None

    def __setstate__(self, state: dict) -> None:
        # Migrate the names stored in earlier versions of the rollout checkpoint.
        if "loader" in state:
            state["selector"] = state.pop("loader")
            state["submitted"] = state.pop("ready")
        self.__dict__.update(state)


@dataclass(frozen=True)
class RolloutBatchMetadata:
    """Ordered selected groups and metrics, without reading their trajectory payloads."""

    batch_id: int
    groups: tuple[AdmittedGroupMetadata, ...]
    metrics: dict[str, float]


def entry_order_from_config(config: DictConfig, dataset: RolloutDataset) -> DatasetEntryOrder:
    """Build the selector's entry order: seeded passes by default, or the ``data.sampling`` order."""
    sampling = config.data.sampling
    if sampling.kind is None:
        return SeededPasses(len(dataset), seed=config.trainer.seed, shuffle=config.data.shuffle)
    if config.trainer.step_wise_training:
        raise ValueError("data.sampling.kind requires one group per entry; step_wise_training is not supported")
    if not isinstance(dataset, PromptDataset):
        raise ValueError(f"data.sampling.kind requires a dataset with a table, got {type(dataset).__name__}")
    seed = sampling.seed if sampling.seed is not None else config.trainer.seed
    batch_size = config.trainer.train_batch_size
    if SamplingKind(sampling.kind) is SamplingKind.DOMAIN_WEIGHTED:
        return DomainWeightedOrder(dataset.dataframe, sampling.domain_weights, seed=seed, window_size=batch_size)
    curriculum = CurriculumConfig.from_dict_config(sampling, group_size=config.generator.n_samples_per_prompt)
    return CurriculumOrder(dataset, curriculum, seed=seed, window_size=batch_size)


def start_rollout_buffer(config: RolloutBufferConfig) -> ActorHandle:
    """Start the rollout buffer actor on this node, beside the trainer that reads every payload through it."""
    node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    return ray.remote(RolloutBuffer).options(num_cpus=0, scheduling_strategy=node).remote(config)


class TrainingContext:
    """The dataset entry selector, rollout buffer, and group jobs in flight between them.

    ``buffer`` is a ``RolloutBuffer`` actor, usually from ``start_rollout_buffer``; the context kills it on ``close``.
    ``start`` runs the coordinator loop: for every lease the buffer grants, it selects the next entry
    and hands one group job to a rollout worker, which writes the result to the buffer. A failed
    group job fails training: the next ``next_batch`` or ``publish`` raises its error. With ``rollout_spans`` it
    records each rollout call, the loop's waits, and every group's disposition.

    A uid is live from dispatch until the buffer reports that its group left: trained, rejected, discarded by
    dynamic sampling, or returned for regeneration. The selector offers only uids that are not live, and a lease
    waits for one to free up rather than generate a copy the batch would reject.
    """

    def __init__(
        self,
        selector: DatasetEntrySelector,
        config: RolloutBufferConfig,
        buffer: ActorHandle,
        content_policy: RolloutContentPolicy,
        request_spec: RolloutRequestSpec,
        workers: RolloutWorkers,
        payloads: PayloadStore,
        *,
        rollout_spans: bool,
        wait_labels: tuple[str, str],
        grader_report_path: str | None = None,
    ):
        self.selector = selector
        self.config = config
        self._request_spec = request_spec
        self._workers = workers
        self._rollout_spans = rollout_spans
        self._lease_wait, self._entry_wait = wait_labels
        self._grader_report_path = grader_report_path
        self._grader_report_attempt = os.environ.get(EXECUTION_UID_ENV) or uuid4().hex
        self.mode = TrainingType.SYNC if config.max_staleness_steps == 0 else TrainingType.ASYNC
        self._policy_step = 0
        self._buffer = buffer
        self._payloads = payloads
        self._writer = payloads.writer(self._buffer, content_policy)
        self._in_flight: dict[str, RolloutGroupJob] = {}
        self._live_uids: collections.Counter[str] = collections.Counter()
        self._uids_released = asyncio.Condition()
        self._running: set[asyncio.Task] = set()
        self._dispatcher: asyncio.Task | None = None
        self._failure: asyncio.Future | None = None
        self._submission_allowed = asyncio.Event()
        self._submission_allowed.set()

    @classmethod
    def from_config(cls, config: DictConfig, dataset: RolloutDataset, workers: RolloutWorkers) -> TrainingContext:
        algorithm = config.trainer.algorithm
        dynamic_sampling = algorithm.dynamic_sampling
        selection = GroupSelectionPolicy(
            DynamicSamplingType(dynamic_sampling.type) if dynamic_sampling.type is not None else None,
            criteria=resolve_dynamic_sampling_criteria(
                dynamic_sampling.informative_on,
                float(dynamic_sampling.min_reward_std),
                dynamic_sampling.max_mean_reward,
            ),
        )
        batch_size = config.trainer.train_batch_size
        max_sample_batches = int(dynamic_sampling.max_sample_batches)
        groups_per_family = validate_groups_per_family(config)
        buffer_config = RolloutBufferConfig(
            batch_size=batch_size,
            max_outstanding_leases=config.trainer.rollout_buffer.max_outstanding_leases,
            max_staleness_steps=config.trainer.rollout_buffer.max_staleness_steps,
            batch_policy=BatchPolicy(config.trainer.rollout_buffer.batch_policy),
            dynamic_sampling=selection.sampling_type,
            max_candidate_groups=max_sample_batches * batch_size if max_sample_batches > 0 else None,
            groups_per_family=groups_per_family,
        )
        families: dict[str, str] = {}
        if groups_per_family:
            if not isinstance(dataset, PromptDataset) or FAMILY_COLUMN not in dataset.dataframe.column_names:
                raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires a {FAMILY_COLUMN} column in the training data")
            column = dataset.dataframe.with_format(None)[FAMILY_COLUMN]
            families = {dataset.uid(index): family for index, family in enumerate(column)}
            validate_family_pools(
                groups_per_family, collections.Counter(families.values()), buffer_config.max_staleness_steps
            )
        plan = compile_distillation_plan_from_config(config)
        admission = GroupAdmissionPolicy(
            GroupAdvantageInvariant.from_config(algorithm.resolved_group_advantage),
            rollout_logprobs_required=rollout_logprobs_required(
                algorithm, loss_spec=PolicyLossRegistry.spec(algorithm.policy_loss_type)
            ),
            student_topk_width=(
                config.generator.sampling_params.logprobs
                if algorithm.policy_loss_type == PolicyLossType.FTPO
                else plan.teachers[0].top_k
                if plan is not None and plan.objective is DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE
                else None
            ),
        )
        object_store_root = config.trainer.rollout_buffer.object_store_root
        report_graders = bool(config.trainer.get("dump_grader_report", False))
        return cls(
            DatasetEntrySelector(
                dataset,
                entry_order_from_config(config, dataset),
                batch_size=batch_size,
                minimum_pool_size=buffer_config.max_untrained_groups + SPARE_ENTRIES_PER_POOL,
                max_stale_retries=config.trainer.rollout_buffer.max_stale_retries,
                max_unusable_groups=config.trainer.rollout_buffer.max_unusable_groups,
                families=families,
                family_minimums=minimum_family_pool_sizes(groups_per_family, buffer_config.max_staleness_steps),
            ),
            buffer_config,
            start_rollout_buffer(buffer_config),
            RolloutContentPolicy(admission, selection, report_graders=report_graders),
            RolloutRequestSpec.from_config(config),
            workers,
            ObjectStorePayloads(object_store_root) if object_store_root is not None else MemoryPayloads(),
            rollout_spans=config.trainer.rollout_spans,
            wait_labels=("slot", "prompt") if static_preference_pairs_requested(config) else ("lease", "task"),
            grader_report_path=str(config.trainer.grader_report_dir) if report_graders else None,
        )

    def start(self) -> None:
        """Start dispatching; the buffer grants no lease before the first ``publish``."""
        self._failure = asyncio.get_running_loop().create_future()
        self._dispatcher = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        tasks = [task for task in (self._dispatcher, *self._running) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._write_grader_report()
        finally:
            ray.kill(self._buffer)

    async def _write_grader_report(self) -> None:
        if self._grader_report_path is None:
            return
        try:
            summary = await self._buffer.grader_summary.remote()
        except ray.exceptions.RayError:
            logger.exception("Failed to collect grader stats for {}", self._grader_report_path)
            return
        if summary["subsets"]:
            write_grader_report(
                self._grader_report_path,
                self._grader_report_attempt,
                {"training": summary["subsets"], "training_tasks": summary["tasks"]},
            )

    async def pause_and_drain(self) -> None:
        """Stop dispatching new generation and finish already submitted groups."""
        self._submission_allowed.clear()
        if self._running:
            await self.wait(asyncio.gather(*self._running))

    def resume(self) -> None:
        self._submission_allowed.set()

    async def publish(self, policy_step: int) -> None:
        """Acknowledge the previous batch and lease rollouts at ``policy_step``, whose weights are now live."""
        await self.wait(self._buffer.publish.remote(policy_step))
        self._policy_step = policy_step

    async def next_batch(
        self,
        *,
        stall_timeout: float,
        on_admitted: Callable[[list[RolloutGroup]], Awaitable[None]],
    ) -> tuple[list[RolloutGroup], RolloutBatchMetadata]:
        """Wait for the current step's batch and return its groups with selection and entry-order metrics.

        Groups reach ``on_admitted`` as they are admitted, before the batch is complete. Every group judged for
        the batch, kept or discarded, then updates the selector's entry order.

        Raises:
            GroupAdmissionStalledError: No group was admitted, or admitted payloads did not arrive, for
                ``stall_timeout`` seconds.
        """
        groups: list[RolloutGroup] = []

        async def fetch_admitted(admitted: tuple[AdmittedGroupMetadata, ...]) -> None:
            reader = RolloutReader(self._buffer, self._payloads, admitted, stall_timeout)
            fetched = await self.wait(reader.read(self._policy_step, tuple(group.index for group in admitted)))
            groups.extend(fetched)
            await on_admitted(fetched)

        metadata = await self.next_batch_metadata(stall_timeout=stall_timeout, on_admitted=fetch_admitted)
        return groups, metadata

    def reader(self, metadata: RolloutBatchMetadata, *, stall_timeout: float) -> RolloutReader:
        """Construct the bounded reader for an admitted batch's ordered groups."""
        return RolloutReader(self._buffer, self._payloads, metadata.groups, stall_timeout)

    async def next_batch_metadata(
        self,
        *,
        stall_timeout: float,
        on_admitted: Callable[[tuple[AdmittedGroupMetadata, ...]], Awaitable[None]] | None = None,
    ) -> RolloutBatchMetadata:
        """Wait for a selected batch without reading its trajectory payloads."""
        loop = asyncio.get_running_loop()
        groups: list[AdmittedGroupMetadata] = []
        deadline = loop.time() + stall_timeout
        while True:
            timeout = max(deadline - loop.time(), 0.0)
            admission = await self.wait(self._buffer.admit.remote(timeout))
            await self._release(admission)
            for policy_step, work in admission.generated:
                record_generated_work(work, policy_step)
            if self._rollout_spans:
                for outcome in admission.dispositions:
                    record_group_disposition(
                        disposition=outcome.disposition,
                        tokens=outcome.tokens,
                        step=self._policy_step,
                        dwell_seconds=outcome.dwell_seconds,
                    )
            record_rollout_buffer(admission.ready_count, self.config.max_untrained_groups)
            if admission.admitted:
                admitted = tuple(admission.admitted)
                groups.extend(admitted)
                if on_admitted is not None:
                    await on_admitted(admitted)
                deadline = loop.time() + stall_timeout
            if admission.selection is not None:
                if len(groups) != self.config.batch_size:
                    raise ValueError(f"selected batch has {len(groups)} groups, expected {self.config.batch_size}")
                if any(group.index != index for index, group in enumerate(groups)):
                    raise ValueError("selected batch group indices are not ordered from zero")
                await self._write_grader_report()
                return RolloutBatchMetadata(
                    self._policy_step,
                    tuple(groups),
                    {**admission.selection.metrics, **self.selector.observe(admission.selection.judged)},
                )

    async def state_dict(self) -> TrainingContextState:
        """Capture every dispatched group that no trained batch has consumed.

        The selector position and outstanding jobs are read together, before the buffer snapshot, so a job
        that submits its group meanwhile appears in the buffer and not also among the entries to regenerate.
        """
        in_flight = dict(self._in_flight)
        selector = self.selector.state_dict()
        snapshot: BufferSnapshot = await self._buffer.snapshot.remote()
        unsubmitted = [job.entry for lease_id, job in in_flight.items() if lease_id in snapshot.leases]
        payloads = await asyncio.gather(*(self._payloads.checkpoint(rollout.payload) for rollout in snapshot.submitted))
        submitted = [
            # The buffer's clock does not carry across processes.
            dataclasses.replace(
                rollout,
                payload=payload,
                submitted_at=None,
                verdict=dataclasses.replace(rollout.verdict, row_facts=None),
            )
            for rollout, payload in zip(snapshot.submitted, payloads, strict=True)
        ]
        retries = [*selector.retries, *snapshot.retries, *unsubmitted]
        return TrainingContextState(
            dataclasses.replace(selector, retries=retries), submitted, self._payloads.object_store_root
        )

    async def load_state_dict(self, state: TrainingContextState) -> None:
        """Restore a checkpoint's rollout state before ``start``."""
        if self._dispatcher is not None:
            raise RuntimeError("rollout state must be restored before dispatching starts")
        # Object URIs are absolute, so a resumed run may write new payloads under a different root, but it must
        # read the checkpoint's payloads the way they were stored.
        if state.submitted and (state.object_store_root is None) != (self._payloads.object_store_root is None):
            raise ValueError(
                f"the checkpoint's rollout payloads are in {state.object_store_root or 'memory'}, but this run "
                f"keeps them in {self._payloads.object_store_root or 'memory'}; set "
                "trainer.rollout_buffer.object_store_root to match"
            )
        self.selector.load_state_dict(state.selector)
        submitted = [
            dataclasses.replace(group, payload=self._payloads.restore(group.payload)) for group in state.submitted
        ]
        for start in range(0, len(submitted), PAYLOAD_IO_THREADS):
            indices = [
                index
                for index in range(start, min(start + PAYLOAD_IO_THREADS, len(submitted)))
                if submitted[index].verdict.trainable
            ]
            groups = await self._payloads.fetch([submitted[index].payload for index in indices])
            for index, group in zip(indices, groups, strict=True):
                verdict = dataclasses.replace(
                    submitted[index].verdict, row_facts=RowFacts.from_batch(group.trajectory_batch)
                )
                submitted[index] = dataclasses.replace(submitted[index], verdict=verdict)
        await self._buffer.restore.remote(BufferSnapshot(submitted=submitted, retries=[], leases=frozenset()))
        self._live_uids.update(rollout.verdict.uid for rollout in submitted)
        logger.info(
            "Restored rollout state: {} submitted groups and {} entries to regenerate",
            len(submitted),
            len(state.selector.retries),
        )

    async def _dispatch(self) -> None:
        try:
            while True:
                await self._submission_allowed.wait()
                with self._wait(self._lease_wait):
                    lease = await self._buffer.acquire_lease.remote()
                with self._wait(self._entry_wait):
                    async with self._uids_released:
                        entry = await self._uids_released.wait_for(
                            lambda: self.selector.next_entry(self._live_uids, lease.family)
                        )
                        self._live_uids[entry["uid"]] += 1
                # A lease or entry wait may span an evaluation boundary. Keep the
                # single acquired lease, but do not submit its generation yet.
                await self._submission_allowed.wait()
                job = RolloutGroupJob(lease, entry, self._request_spec.request(entry, lease.policy_step))
                self._in_flight[lease.lease_id] = job
                running = asyncio.create_task(self._run(job))
                self._running.add(running)
                running.add_done_callback(self._running.discard)
        except Exception as error:
            self._fail(error)

    async def _release(self, admission: Admission) -> None:
        """Free the uids of groups that left the buffer, queue entries to regenerate, and wake dispatch."""
        if not admission.dispositions and not admission.retries:
            return
        async with self._uids_released:
            self._live_uids -= collections.Counter(outcome.uid for outcome in admission.dispositions)
            for outcome in admission.dispositions:
                self.selector.record_disposition(outcome.uid, outcome.disposition)
            for entry in admission.retries:
                self.selector.retry(entry)
            self._uids_released.notify_all()

    def _wait(self, name: str) -> AbstractContextManager[None]:
        return dispatch_wait(name, step=self._policy_step, mode=self.mode, enabled=self._rollout_spans)

    async def _run(self, job: RolloutGroupJob) -> None:
        try:
            with observe_rollout_call(
                step=job.lease.policy_step, mode=self.mode, enabled=self._rollout_spans
            ) as observation:
                response_tokens = await self._workers.run_task(job, self._writer)
                if observation is not None:
                    observation.response_tokens = response_tokens
        except Exception as error:
            self._fail(error)
        finally:
            del self._in_flight[job.lease.lease_id]

    def _fail(self, error: Exception) -> None:
        logger.opt(exception=error).error("Rollout dispatch failed; training will stop")
        assert self._failure is not None
        if not self._failure.done():
            self._failure.set_exception(error)

    async def wait(self, awaitable: Awaitable[_T]) -> _T:
        """Await ``awaitable`` unless a rollout task fails first, then raise that failure."""
        assert self._failure is not None, "start the training context before reading from it"
        work: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        await asyncio.wait({work, self._failure}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return work.result()
        work.cancel()
        return self._failure.result()
