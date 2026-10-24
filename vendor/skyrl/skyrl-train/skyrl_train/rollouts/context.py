"""Rollout state of one training run and the coordinator loop that feeds its buffer."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marinskyrl.environment_contract import TrainingType
from marinskyrl.runtime_options import PolicyLossType, SamplingKind
from marinskyrl.distillation import DistillationObjectiveKind, compile_distillation_plan_from_config
from skyrl_train.curriculum import CurriculumConfig, CurriculumOrder
from skyrl_train.dataset import FixedCompletionsDataset, PromptDataset
from skyrl_train.domain_sampling import DomainWeightedOrder
from skyrl_train.dynamic_sampling import (
    DynamicSamplingType,
    GroupSelectionPolicy,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdmissionStalledError, GroupAdvantageInvariant
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    Admission,
    AdmittedRollout,
    BufferSnapshot,
    ReadyRollout,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
)
from skyrl_train.rollouts.loader import PromptLoader, PromptLoaderState, PromptGroupDataset, PromptOrder, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, ObjectStorePayloads, PayloadStore
from skyrl_train.rollout_observability import dispatch_wait, observe_rollout_call, record_group_disposition
from skyrl_train.rollouts.workers import RolloutWorkers
from skyrl_train.telemetry import GeneratedWork, record_generated_work, record_rollout_buffer
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.config.objective_spec import rollout_logprobs_required
from skyrl_train.config.training_budget import ResponseCaps, response_caps
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

_T = TypeVar("_T")


# The backend sampling key that caps one generation, as vLLM and SGLang spell it.
_OUTPUT_LIMIT_KEYS = ("max_tokens", "max_new_tokens")


@dataclass(frozen=True)
class RolloutRequestSpec:
    """How the coordinator turns one prompt into a trajectory request.

    With per-task response caps, each request caps generation at its prompt's ``data_source`` cap.
    """

    samples_per_prompt: int
    sampling_params: dict
    response_caps: ResponseCaps

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutRequestSpec:
        return cls(
            samples_per_prompt=config.generator.n_samples_per_prompt,
            sampling_params=get_sampling_params_for_backend(config.generator.backend, config.generator.sampling_params),
            response_caps=response_caps(config),
        )

    def request(self, prompt: dict, policy_step: int) -> TrajectoryRequestBatch:
        request, _ = prepare_trajectory_request(
            [prompt], self.samples_per_prompt, self._sampling_params(prompt), "train", policy_step
        )
        return request

    def _sampling_params(self, prompt: dict) -> dict:
        if not self.response_caps.by_data_source:
            return self.sampling_params
        (key,) = (key for key in _OUTPUT_LIMIT_KEYS if key in self.sampling_params)
        cap = self.response_caps.for_source(prompt["input"].metadata.data_source)
        return {**self.sampling_params, key: cap}


@dataclass(frozen=True)
class TrainingContextState:
    """Checkpointed rollout state: the loader, including prompts to regenerate, and committed groups.

    Each ready rollout's ``payload`` holds the durable form its payload store checkpoints: the ``RolloutGroup``
    for payloads in memory, or the URI of its object under ``object_store_root``.
    """

    loader: PromptLoaderState
    ready: list[ReadyRollout]
    object_store_root: str | None


@dataclass(frozen=True)
class RolloutBatchMetadata:
    """Ordered selected groups and metrics, without reading their trajectory payloads."""

    batch_id: int
    groups: tuple[AdmittedRollout, ...]
    metrics: dict[str, float]


def prompt_order_from_config(config: DictConfig, dataset: PromptGroupDataset) -> PromptOrder:
    """Build the loader's prompt order: seeded passes by default, or the ``data.sampling`` order."""
    sampling = config.data.sampling
    if sampling.kind is None:
        routes = dataset.teacher_routes() if isinstance(dataset, FixedCompletionsDataset) else set()
        if len(routes) > 1 and config.data.shuffle:
            raise ValueError(
                f"fixed-completion rows span teacher routes {sorted(routes)}; shuffled passes leave each batch's "
                "route mix to chance, so set data.sampling.kind=domain-weighted with data.sampling.domain_weights, "
                "or data.shuffle=false to read the file's order"
            )
        return SeededPasses(len(dataset), seed=config.trainer.seed, shuffle=config.data.shuffle)
    if config.trainer.step_wise_training:
        raise ValueError("data.sampling.kind requires one group per prompt; step_wise_training is not supported")
    if not isinstance(dataset, PromptDataset):
        raise ValueError(f"data.sampling.kind requires a prompt dataset, got {type(dataset).__name__}")
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
    ):
        self.loader = loader
        self.config = config
        self._request_spec = request_spec
        self._workers = workers
        self._rollout_spans = rollout_spans
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
    def from_config(cls, config: DictConfig, dataset: PromptGroupDataset, workers: RolloutWorkers) -> TrainingContext:
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
        buffer_config = RolloutBufferConfig(
            batch_size=batch_size,
            max_in_flight=config.trainer.rollout_buffer.max_in_flight,
            max_staleness_steps=config.trainer.rollout_buffer.max_staleness_steps,
            batch_policy=BatchPolicy(config.trainer.rollout_buffer.batch_policy),
            dynamic_sampling=selection.sampling_type,
            max_candidate_groups=max_sample_batches * batch_size if max_sample_batches > 0 else None,
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
        return cls(
            PromptLoader(dataset, prompt_order_from_config(config, dataset), batch_size=batch_size),
            buffer_config,
            start_rollout_buffer(buffer_config),
            RolloutContentPolicy(admission, selection),
            RolloutRequestSpec.from_config(config),
            workers,
            ObjectStorePayloads(object_store_root) if object_store_root is not None else MemoryPayloads(),
            rollout_spans=config.trainer.rollout_spans,
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
        ray.kill(self._buffer)

    async def pause_and_drain(self) -> None:
        """Stop dispatching new generation and finish already submitted groups."""
        self._submission_allowed.clear()
        if self._running:
            await self._until_failure(asyncio.gather(*self._running))

    def resume(self) -> None:
        self._submission_allowed.set()

    async def publish(self, policy_step: int) -> None:
        """Acknowledge the previous batch and lease rollouts at ``policy_step``, whose weights are now live."""
        await self._until_failure(self._buffer.publish.remote(policy_step))
        self._policy_step = policy_step

    async def next_batch(
        self,
        *,
        stall_timeout: float,
        on_admitted: Callable[[list[RolloutGroup]], Awaitable[None]],
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Wait for the current step's batch and return its groups with selection and prompt-order metrics.

        Groups reach ``on_admitted`` as they are admitted, before the batch is complete. Every group judged for
        the batch, kept or discarded, then updates the loader's prompt order.

        Raises:
            GroupAdmissionStalledError: No group was admitted, or admitted payloads did not arrive, for
                ``stall_timeout`` seconds.
        """
        groups: list[RolloutGroup] = []

        async def fetch_admitted(admitted: tuple[AdmittedRollout, ...]) -> None:
            fetched = await self._fetch_groups(self._policy_step, admitted, stall_timeout=stall_timeout)
            groups.extend(fetched)
            await on_admitted(fetched)

        metadata = await self.next_batch_metadata(stall_timeout=stall_timeout, on_admitted=fetch_admitted)
        return groups, metadata.metrics

    async def _fetch_groups(
        self, batch_id: int, selected: tuple[AdmittedRollout, ...], *, stall_timeout: float
    ) -> list[RolloutGroup]:
        try:
            async with asyncio.timeout(stall_timeout):
                refs = await self._until_failure(
                    self._buffer.payload_refs.remote(batch_id, tuple(group.index for group in selected))
                )
                groups = await self._until_failure(self._payloads.fetch(refs))
        except TimeoutError as error:
            raise GroupAdmissionStalledError(
                f"{len(selected)} selected rollout payloads did not arrive within "
                f"{stall_timeout:.0f}s: batch_id={batch_id} indices={[group.index for group in selected]}"
            ) from error
        if len(groups) != len(selected):
            raise ValueError("payload store returned the wrong number of selected rollout groups")
        for expected, group in zip(selected, groups, strict=True):
            if group.uid != expected.uid or group.policy_step != expected.policy_step:
                raise ValueError(f"selected rollout payload does not match batch metadata for {expected.uid}")
            work = GeneratedWork.from_batch(
                group.trajectory_batch["response_ids"], group.trajectory_batch.get("is_last_step")
            )
            if work.sample_count != expected.sample_count or work.generated_token_count != expected.response_tokens:
                raise ValueError(f"selected rollout payload size does not match batch metadata for {expected.uid}")
        return groups

    async def next_batch_metadata(
        self,
        *,
        stall_timeout: float,
        on_admitted: Callable[[tuple[AdmittedRollout, ...]], Awaitable[None]] | None = None,
    ) -> RolloutBatchMetadata:
        """Wait for a selected batch without reading its trajectory payloads."""
        loop = asyncio.get_running_loop()
        groups: list[AdmittedRollout] = []
        deadline = loop.time() + stall_timeout
        while True:
            timeout = max(deadline - loop.time(), 0.0)
            admission = await self._until_failure(self._buffer.admit.remote(timeout))
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
                return RolloutBatchMetadata(
                    self._policy_step,
                    tuple(groups),
                    {**admission.selection.metrics, **self.loader.observe(admission.selection.judged)},
                )

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
            dataclasses.replace(rollout, payload=payload, committed_at=None)
            for rollout, payload in zip(snapshot.ready, payloads, strict=True)
        ]
        retries = [*loader.retries, *snapshot.retries, *uncommitted]
        return TrainingContextState(
            dataclasses.replace(loader, retries=retries), ready, self._payloads.object_store_root
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
        await self._buffer.restore.remote(BufferSnapshot(ready=ready, retries=[], leases=frozenset()))
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
                        prompt = await self._uids_released.wait_for(lambda: self.loader.next_prompt(self._live_uids))
                        self._live_uids[prompt["uid"]] += 1
                # A lease or prompt wait may span an evaluation boundary. Keep the
                # single acquired lease, but do not submit its generation yet.
                await self._submission_allowed.wait()
                task = RolloutTask(lease, prompt, self._request_spec.request(prompt, lease.policy_step))
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

    async def _until_failure(self, awaitable: Awaitable[_T]) -> _T:
        """Await ``awaitable`` unless a rollout task fails first, then raise that failure."""
        assert self._failure is not None, "start the training context before reading from it"
        work: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        await asyncio.wait({work, self._failure}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return work.result()
        work.cancel()
        return self._failure.result()
