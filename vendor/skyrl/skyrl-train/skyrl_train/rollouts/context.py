"""Rollout state of one training run and the coordinator loop that feeds its buffer."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

import numpy as np
import ray
from loguru import logger
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marinskyrl.environment_contract import EXECUTION_UID_ENV, TrainingType
from marinskyrl.runtime_options import PolicyLossType
from marinskyrl.distillation import DistillationObjectiveKind, compile_distillation_plan_from_config
from skyrl_train.adaptive_sampling import AdaptiveSourceOrder
from skyrl_train.config.rollout_validation import (
    FAMILY_COLUMN,
    GROUPS_PER_FAMILY_KEY,
    BatchPolicy,
    minimum_family_pool_sizes,
    validate_family_pools,
    validate_groups_per_family,
)
from skyrl_train.contextual_sampling import (
    PromptAcquisition,
    PromptFeedback,
    PromptThompsonOrder,
    PublicPrompt,
    public_prompt_digest,
)
from skyrl_train.curriculum import (
    ADAPTIVE_SOURCE_KINDS,
    CurriculumConfig,
    CurriculumOrder,
    SamplingKind,
    validate_sampling_reversion,
)
from skyrl_train.dataset import PromptDataset
from skyrl_train.dataset.tasks import TaskDataset
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
    SAMPLING_SOURCE,
    Admission,
    AdmittedRollout,
    BufferSnapshot,
    DiscardedCandidate,
    GroupDisposition,
    ReadyRollout,
    RolloutBuffer,
    RolloutBufferConfig,
    RowFacts,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
    SourceRejections,
)
from skyrl_train.rollouts.loader import PromptLoader, PromptLoaderState, PromptGroupDataset, PromptOrder, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads, PayloadStore
from skyrl_train.rollouts.payloads import RolloutReader, PAYLOAD_IO_THREADS
from skyrl_train.rollout_observability import dispatch_wait, observe_rollout_call, record_group_disposition
from skyrl_train.rollouts.workers import RolloutWorkers
from skyrl_train.telemetry import record_event, record_generated_work, record_rollout_buffer
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.config.objective_spec import rollout_logprobs_required
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

_T = TypeVar("_T")


@dataclass(frozen=True)
class RolloutRequestSpec:
    """How the coordinator turns one prompt into a trajectory request."""

    samples_per_prompt: int
    sampling_params: dict

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutRequestSpec:
        return cls(
            samples_per_prompt=config.generator.n_samples_per_prompt,
            sampling_params=get_sampling_params_for_backend(config.generator.backend, config.generator.sampling_params),
        )

    def request(self, prompt: dict, policy_step: int) -> TrajectoryRequestBatch:
        request, _ = prepare_trajectory_request(
            [prompt], self.samples_per_prompt, self.sampling_params, "train", policy_step
        )
        return request


@dataclass(frozen=True)
class TrainingContextState:
    """Checkpointed rollout state: the loader, including prompts to regenerate, and committed groups.

    Each ready rollout's ``payload`` holds the durable form its payload store checkpoints: the ``RolloutGroup``
    for payloads in memory, or the URI of its object under ``object_store_root``.
    """

    loader: PromptLoaderState
    ready: list[ReadyRollout]
    object_store_root: str | None
    discarded: tuple[DiscardedCandidate, ...] = ()
    source_rejections: SourceRejections | None = None
    sampling_dispositions: tuple[GroupDisposition, ...] = ()


@dataclass(frozen=True)
class RolloutBatchMetadata:
    """Ordered selected groups and metrics, without reading their trajectory payloads."""

    batch_id: int
    groups: tuple[AdmittedRollout, ...]
    metrics: dict[str, float]
    sampling_receipts_enabled: bool = False
    sampling_dispositions: tuple[GroupDisposition, ...] = ()


def _public_prompt_texts(dataset: PromptDataset) -> list[str]:
    # Read only the model-visible prompt column. Canonical prepared rows store messages as JSON.
    texts = []
    for prompt in dataset.dataframe.with_format(None)[dataset.prompt_key]:
        if isinstance(dataset, TaskDataset) and isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(prompt, str):
            texts.append(prompt)
            continue
        if any(not isinstance(message.get("content"), str) for message in prompt):
            raise ValueError("Prompt Thompson sampling requires text-only public messages")
        texts.append("\n".join(message["content"] for message in prompt))
    return texts


def prompt_order_from_config(
    config: DictConfig, dataset: PromptGroupDataset, *, held_out: Sequence[PromptDataset] = ()
) -> PromptOrder:
    """Build the loader's prompt order: seeded passes by default, or the ``data.sampling`` order."""
    sampling = config.data.sampling
    if sampling.kind is None:
        return SeededPasses(len(dataset), seed=config.trainer.seed, shuffle=config.data.shuffle)
    if config.trainer.step_wise_training:
        raise ValueError("data.sampling.kind requires one group per prompt; step_wise_training is not supported")
    if not isinstance(dataset, PromptDataset):
        raise ValueError(f"data.sampling.kind requires a prompt dataset, got {type(dataset).__name__}")
    seed = sampling.seed if sampling.seed is not None else config.trainer.seed
    batch_size = config.trainer.train_batch_size
    kind = SamplingKind(sampling.kind)
    validate_sampling_reversion(sampling)
    if kind is SamplingKind.DOMAIN_WEIGHTED:
        return DomainWeightedOrder(dataset.dataframe, sampling.domain_weights, seed=seed, window_size=batch_size)
    if kind in PromptAcquisition:
        samples = config.generator.n_samples_per_prompt
        if samples < 2:
            raise ValueError("prompt-thompson requires at least two samples per prompt")
        sources = dataset.dataframe.with_format(None)[sampling.source_column]
        prompts = [
            PublicPrompt(dataset.uid(index), source, text)
            for index, (source, text) in enumerate(zip(sources, _public_prompt_texts(dataset), strict=True))
        ]
        held_out_digests = {public_prompt_digest(text) for panel in held_out for text in _public_prompt_texts(panel)}
        return PromptThompsonOrder(
            prompts,
            held_out_digests,
            sampling.domain_weights,
            acquisition=PromptAcquisition(kind),
            seed=seed,
            window_size=batch_size,
            decay=sampling.decay,
            epsilon=sampling.epsilon,
            reference_tokens=max(1, samples * config.generator.sampling_params.max_generate_length // 2),
        )
    if kind in ADAPTIVE_SOURCE_KINDS:
        if config.generator.n_samples_per_prompt < 2:
            raise ValueError("adaptive-source requires at least two samples per prompt")
        return AdaptiveSourceOrder(
            dataset.dataframe.with_format(None)[sampling.source_column],
            sampling.domain_weights,
            seed=seed,
            window_size=batch_size,
            decay=sampling.decay,
            epsilon=sampling.epsilon,
            prior_alpha=sampling.prior_alpha,
            prior_beta=sampling.prior_beta,
            posterior_rng=np.random.default_rng(seed) if kind is SamplingKind.ADAPTIVE_SOURCE_THOMPSON else None,
        )
    curriculum = CurriculumConfig.from_dict_config(sampling, group_size=config.generator.n_samples_per_prompt)
    return CurriculumOrder(dataset, curriculum, seed=seed, window_size=batch_size)


def start_rollout_buffer(config: RolloutBufferConfig) -> ActorHandle:
    """Start the rollout buffer actor on this node, beside the trainer that reads every payload through it."""
    node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    return ray.remote(RolloutBuffer).options(num_cpus=0, scheduling_strategy=node).remote(config)


def record_sampling_result(outcome: GroupDisposition, admission_batch: int) -> None:
    """Emit verifier coverage and generated cost for a completed sampled lease."""
    result = outcome.sampling_result
    if result is None:
        return
    scores = [score for score in result.verified_scores or () if score is not None]
    record_event(
        "sampling_group_result",
        {
            "lease_id": result.lease_id,
            "group_uid": outcome.uid,
            "committed_unix_ns": result.committed_unix_ns,
            "feedback_batch": result.feedback_batch,
            "admission_batch": admission_batch,
            "dispatch_step": result.dispatch_step,
            "disposition": outcome.disposition,
            "generated_tokens": outcome.tokens,
            "sample_count": result.sample_count,
            "verified_zero": scores.count(0.0),
            "verified_one": scores.count(1.0),
            "verified_partial": sum(0 < score < 1 for score in scores),
            "verified_missing": result.sample_count - len(scores),
            "verified_sum": sum(scores),
            "verified_min": min(scores, default=None),
            "verified_max": max(scores, default=None),
        },
    )


def _record_admission(admission: Admission, trace_rollouts: bool, step: int, max_untrained_groups: int) -> None:
    for outcome in admission.dispositions:
        record_sampling_result(outcome, admission.batch_id)
        if trace_rollouts:
            record_group_disposition(
                disposition=outcome.disposition,
                tokens=outcome.tokens,
                step=step,
                dwell_seconds=outcome.dwell_seconds,
            )
    for policy_step, work in admission.generated:
        record_generated_work(work, policy_step)
    record_rollout_buffer(admission.ready_count, max_untrained_groups)


def _record_sampling_feedback(step: int) -> None:
    record_event("sampling_feedback_observed", {"feedback_batch": step, "observed_unix_ns": time.time_ns()})


def record_sampling_training(batch: RolloutBatchMetadata, nonzero_advantage_tokens: collections.Counter[str]) -> None:
    """Emit one optimizer-input receipt for each selected group, including groups with zero signal."""
    for group in batch.groups:
        record_event(
            "sampling_group_trained",
            {
                "group_uid": group.uid,
                "training_step": batch.batch_id,
                "nonzero_advantage_tokens": nonzero_advantage_tokens[group.uid],
            },
        )


class TrainingContext:
    """The prompt loader, the rollout buffer, and the rollout tasks in flight between them.

    ``buffer`` is a ``RolloutBuffer`` actor, usually from ``start_rollout_buffer``; the context kills it on ``close``.
    ``start`` runs the coordinator loop: for every lease the buffer grants, it takes the next prompt
    from the loader and hands one task to a rollout worker, which writes the result to the buffer. A failed
    task fails training: the next ``next_batch`` or ``publish`` raises its error. With ``rollout_spans`` it
    records each rollout call, the loop's waits, and every group's disposition.

    A uid is live from dispatch until the buffer reports that its group left: trained, rejected, discarded by
    dynamic sampling, or returned for regeneration. The loader offers only uids that are not live, and a lease
    waits for one to free up rather than generate a copy the batch would reject.
    """

    def __init__(
        self,
        loader: PromptLoader,
        config: RolloutBufferConfig,
        buffer: ActorHandle,
        content_policy: RolloutContentPolicy,
        request_spec: RolloutRequestSpec,
        workers: RolloutWorkers,
        payloads: PayloadStore,
        *,
        rollout_spans: bool,
        grader_report_path: str | None = None,
        assign_sampling_source: Callable[[str], str] | None = None,
        observe_after_optimizer: Callable[[int, Sequence[PromptFeedback]], dict[str, float]] | None = None,
    ):
        self.loader = loader
        self.config = config
        self._request_spec = request_spec
        self._workers = workers
        self._rollout_spans = rollout_spans
        self._grader_report_path = grader_report_path
        self._assign_sampling_source = assign_sampling_source
        self._observe_after_optimizer = observe_after_optimizer
        self._grader_report_attempt = os.environ.get(EXECUTION_UID_ENV) or uuid4().hex
        self.mode = TrainingType.SYNC if config.max_staleness_steps == 0 else TrainingType.ASYNC
        self._policy_step = 0
        self._buffer = buffer
        self._payloads = payloads
        self._writer = payloads.writer(self._buffer, content_policy)
        self._in_flight: dict[str, RolloutTask] = {}
        self._live_uids: collections.Counter[str] = collections.Counter()
        self._uids_released = asyncio.Condition()
        self._running: set[asyncio.Task] = set()
        self._dispatcher: asyncio.Task | None = None
        self._failure: asyncio.Future | None = None
        self._submission_allowed = asyncio.Event()
        self._submission_allowed.set()

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        dataset: PromptGroupDataset,
        workers: RolloutWorkers,
        *,
        held_out: Sequence[PromptDataset] = (),
    ) -> TrainingContext:
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
            max_in_flight=config.trainer.rollout_buffer.max_in_flight,
            max_staleness_steps=config.trainer.rollout_buffer.max_staleness_steps,
            batch_policy=BatchPolicy(config.trainer.rollout_buffer.batch_policy),
            dynamic_sampling=selection.sampling_type,
            max_candidate_groups=max_sample_batches * batch_size if max_sample_batches > 0 else None,
            groups_per_family=groups_per_family,
        )
        families: dict[str, str] = {}
        if groups_per_family:
            if not isinstance(dataset, PromptDataset) or FAMILY_COLUMN not in dataset.dataframe.column_names:
                raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires a {FAMILY_COLUMN} column in the training rows")
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
        order = prompt_order_from_config(config, dataset, held_out=held_out)
        assign_source = order.assign if isinstance(order, (AdaptiveSourceOrder, PromptThompsonOrder)) else None
        return cls(
            PromptLoader(
                dataset,
                order,
                batch_size=batch_size,
                max_stale_retries=config.trainer.rollout_buffer.max_stale_retries,
                max_unusable_groups=config.trainer.rollout_buffer.max_unusable_groups,
                families=families,
                family_minimums=minimum_family_pool_sizes(groups_per_family, buffer_config.max_staleness_steps),
                observe_judged=order.observe if isinstance(order, (AdaptiveSourceOrder, CurriculumOrder)) else None,
            ),
            buffer_config,
            start_rollout_buffer(buffer_config),
            RolloutContentPolicy(admission, selection, report_graders=report_graders),
            RolloutRequestSpec.from_config(config),
            workers,
            ObjectStorePayloads(object_store_root) if object_store_root is not None else MemoryPayloads(),
            rollout_spans=config.trainer.rollout_spans,
            grader_report_path=str(config.trainer.grader_report_dir) if report_graders else None,
            assign_sampling_source=assign_source,
            observe_after_optimizer=order.finish_step if isinstance(order, PromptThompsonOrder) else None,
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
        """Wait for the current step's batch and return its groups with selection and prompt-order metrics.

        Groups reach ``on_admitted`` as they are admitted, before the batch is complete. Every group judged for
        the batch, kept or discarded, then updates the loader's prompt order.

        Raises:
            GroupAdmissionStalledError: No group was admitted, or admitted payloads did not arrive, for
                ``stall_timeout`` seconds.
        """
        groups: list[RolloutGroup] = []

        async def fetch_admitted(admitted: tuple[AdmittedRollout, ...]) -> None:
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
        on_admitted: Callable[[tuple[AdmittedRollout, ...]], Awaitable[None]] | None = None,
    ) -> RolloutBatchMetadata:
        """Wait for a selected batch without reading its trajectory payloads."""
        loop = asyncio.get_running_loop()
        groups: list[AdmittedRollout] = []
        sampling_dispositions: list[GroupDisposition] = []
        sampling_receipts_enabled = self._assign_sampling_source is not None
        tracks_prompt_feedback = self._observe_after_optimizer is not None
        deadline = loop.time() + stall_timeout
        while True:
            timeout = max(deadline - loop.time(), 0.0)
            admission = await self.wait(self._buffer.admit.remote(timeout))
            await self._release(admission)
            _record_admission(admission, self._rollout_spans, self._policy_step, self.config.max_untrained_groups)
            if tracks_prompt_feedback:
                sampling_dispositions.extend(admission.dispositions)
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
                sampling_metrics = self.loader.observe(admission.selection.judged)
                if sampling_receipts_enabled and not tracks_prompt_feedback:
                    _record_sampling_feedback(admission.batch_id)
                return RolloutBatchMetadata(
                    self._policy_step,
                    tuple(groups),
                    {**admission.selection.metrics, **sampling_metrics},
                    sampling_receipts_enabled=sampling_receipts_enabled,
                    sampling_dispositions=tuple(sampling_dispositions),
                )

    def finish_sampling_step(
        self, batch: RolloutBatchMetadata, counts: collections.Counter[str] | None
    ) -> dict[str, float]:
        """Update the prompt policy only once optimizer participation is known."""
        if counts is not None:
            record_sampling_training(batch, counts)
        if self._observe_after_optimizer is None:
            return {}
        feedback = []
        for outcome in batch.sampling_dispositions:
            result = outcome.sampling_result
            if result is None:
                raise ValueError(f"Prompt Thompson outcome has no dispatch receipt: {outcome.uid}")
            feedback.append(
                PromptFeedback(
                    lease_id=result.lease_id,
                    uid=outcome.uid,
                    dispatch_step=result.dispatch_step,
                    sample_count=result.sample_count,
                    verified_scores=result.verified_scores,
                    generated_tokens=outcome.tokens,
                    nonzero_advantage_tokens=(
                        counts[outcome.uid] if counts is not None and outcome.disposition == "consumed" else None
                    ),
                )
            )
        metrics = self._observe_after_optimizer(batch.batch_id, feedback)
        _record_sampling_feedback(batch.batch_id)
        return metrics

    async def state_dict(self) -> TrainingContextState:
        """Capture every dispatched group that no trained batch has consumed.

        The loader position and in-flight tasks are read together, before the buffer snapshot, so a task
        that commits meanwhile appears in the buffer and not also among the prompts to regenerate.
        """
        in_flight = dict(self._in_flight)
        loader = self.loader.state_dict()
        snapshot: BufferSnapshot = await self._buffer.snapshot.remote()
        uncommitted = [task.prompt for lease_id, task in in_flight.items() if lease_id in snapshot.leases]
        payloads = await asyncio.gather(*(self._payloads.checkpoint(rollout.payload) for rollout in snapshot.ready))
        ready = [
            # The buffer's clock does not carry across processes.
            dataclasses.replace(
                rollout,
                payload=payload,
                committed_at=None,
                verdict=dataclasses.replace(rollout.verdict, row_facts=None),
            )
            for rollout, payload in zip(snapshot.ready, payloads, strict=True)
        ]
        retries = [*loader.retries, *snapshot.retries, *uncommitted]
        return TrainingContextState(
            dataclasses.replace(loader, retries=retries),
            ready,
            self._payloads.object_store_root,
            snapshot.discarded,
            snapshot.source_rejections,
            snapshot.sampling_dispositions,
        )

    async def load_state_dict(self, state: TrainingContextState) -> None:
        """Restore a checkpoint's rollout state before ``start``."""
        if self._dispatcher is not None:
            raise RuntimeError("rollout state must be restored before dispatching starts")
        # Object URIs are absolute, so a resumed run may write new payloads under a different root, but it must
        # read the checkpoint's payloads the way they were stored.
        if state.ready and (state.object_store_root is None) != (self._payloads.object_store_root is None):
            raise ValueError(
                f"the checkpoint's rollout payloads are in {state.object_store_root or 'memory'}, but this run "
                f"keeps them in {self._payloads.object_store_root or 'memory'}; set "
                "trainer.rollout_buffer.object_store_root to match"
            )
        self.loader.load_state_dict(state.loader)
        ready = [
            dataclasses.replace(rollout, payload=self._payloads.restore(rollout.payload)) for rollout in state.ready
        ]
        for start in range(0, len(ready), PAYLOAD_IO_THREADS):
            indices = [
                index
                for index in range(start, min(start + PAYLOAD_IO_THREADS, len(ready)))
                if ready[index].verdict.trainable
            ]
            groups = await self._payloads.fetch([ready[index].payload for index in indices])
            for index, group in zip(indices, groups, strict=True):
                verdict = dataclasses.replace(
                    ready[index].verdict, row_facts=RowFacts.from_batch(group.trajectory_batch)
                )
                ready[index] = dataclasses.replace(ready[index], verdict=verdict)
        await self._buffer.restore.remote(
            BufferSnapshot(
                ready=ready,
                retries=[],
                leases=frozenset(),
                discarded=state.discarded,
                source_rejections=state.source_rejections,
                sampling_dispositions=state.sampling_dispositions,
            )
        )
        self._live_uids.update(rollout.verdict.uid for rollout in ready)
        logger.info(
            "Restored rollout state: {} committed groups and {} prompts to regenerate",
            len(ready),
            len(state.loader.retries),
        )

    async def _dispatch(self) -> None:
        try:
            while True:
                await self._submission_allowed.wait()
                with self._wait("slot"):
                    lease = await self._buffer.acquire_lease.remote()
                with self._wait("prompt"):
                    async with self._uids_released:
                        prompt = await self._uids_released.wait_for(
                            lambda: self.loader.next_prompt(self._live_uids, lease.family)
                        )
                        self._live_uids[prompt["uid"]] += 1
                # A lease or prompt wait may span an evaluation boundary. Keep the
                # single acquired lease, but do not submit its generation yet.
                await self._submission_allowed.wait()
                request = self._request_spec.request(prompt, lease.policy_step)
                if self._assign_sampling_source is not None:
                    prompt = {**prompt, SAMPLING_SOURCE: self._assign_sampling_source(prompt["uid"])}
                    metadata = prompt["input"].metadata
                    identity = metadata.evaluation_identity
                    record_event(
                        "sampling_group_assigned",
                        {
                            "lease_id": lease.lease_id,
                            "group_uid": prompt["uid"],
                            "assigned_unix_ns": time.time_ns(),
                            "policy_step": lease.policy_step,
                            "target_batch": lease.batch_id,
                            "task_id": metadata.task_id,
                            "task_digest": identity.task_digest if identity is not None else None,
                            "source": prompt[SAMPLING_SOURCE],
                            "family": identity.strata.family if identity is not None else lease.family,
                        },
                    )
                task = RolloutTask(lease, prompt, request)
                self._in_flight[lease.lease_id] = task
                running = asyncio.create_task(self._run(task))
                self._running.add(running)
                running.add_done_callback(self._running.discard)
        except Exception as error:
            self._fail(error)

    async def _release(self, admission: Admission) -> None:
        """Free the uids of groups that left the buffer, queue prompts to regenerate, and wake dispatch."""
        if not admission.dispositions and not admission.retries:
            return
        async with self._uids_released:
            self._live_uids -= collections.Counter(outcome.uid for outcome in admission.dispositions)
            for outcome in admission.dispositions:
                self.loader.record_disposition(outcome.uid, outcome.disposition)
            for prompt in admission.retries:
                self.loader.retry(prompt)
            self._uids_released.notify_all()

    def _wait(self, name: str) -> AbstractContextManager[None]:
        return dispatch_wait(name, step=self._policy_step, mode=self.mode, enabled=self._rollout_spans)

    async def _run(self, task: RolloutTask) -> None:
        try:
            with observe_rollout_call(
                step=task.lease.policy_step, mode=self.mode, enabled=self._rollout_spans
            ) as observation:
                response_tokens = await self._workers.run_task(task, self._writer)
                if observation is not None:
                    observation.response_tokens = response_tokens
        except Exception as error:
            self._fail(error)
        finally:
            del self._in_flight[task.lease.lease_id]

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
