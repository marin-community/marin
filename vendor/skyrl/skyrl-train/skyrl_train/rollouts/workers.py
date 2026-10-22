"""Rollout workers: processes that each host one trajectory runner, and the pool the trainer uses as its runner."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypeVar

import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.actor import ActorHandle
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import PreTrainedTokenizerBase

from skyrl_train.rollout_observability import RolloutTimings, current_rollout_observation, measure_rollout
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import TrainingPhase, TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.utils.fd_monitor import start_fd_monitor
from skyrl_train.utils.utils import configure_ray_worker_logging
from skyrl_train.worker_setup import configure_worker_process

_Result = TypeVar("_Result")

# Ray labels every node with its node ID, so a placement-group bundle can name the node it must use.
NODE_ID_LABEL = "ray.io/node-id"


class RolloutWorkers(Protocol):
    """The rollout processes a coordinator hands tasks to."""

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        """Generate and write one group, returning its response token count."""
        ...


def detached_config(config: DictConfig) -> DictConfig:
    """Return a resolved copy without OmegaConf parent references, to send to a rollout worker."""
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


@dataclass(frozen=True)
class WorkerShard:
    """A worker's position in its pool, for runners that divide per-process limits among the workers."""

    index: int
    count: int


class RunnerSpec(Protocol):
    """Picklable inputs that build a trajectory runner inside a rollout worker process."""

    config: DictConfig

    def build(self, tokenizer: PreTrainedTokenizerBase, shard: WorkerShard) -> TrajectoryRunner: ...


@dataclass(frozen=True)
class RolloutWorkerResources:
    """Size of a rollout worker pool, how its workers start, and how long a worker may go without completing a request.

    Workers start one at a time, each ``start_interval_seconds`` after the previous one is up, because each imports
    its runner stack and loads its tokenizer from a shared filesystem, and spacing the starts keeps those page-ins
    from overlapping each other and the engines' weight loads. An interval of zero starts every worker at once.
    The pool fails to start when the workers' CPUs are not reserved within ``placement_timeout_seconds``.
    """

    num_workers: int
    cpus_per_worker: int
    executor_threads: int
    progress_timeout_seconds: float
    start_interval_seconds: float
    placement_timeout_seconds: float

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutWorkerResources:
        workers = config.trajectory_runner.rollout_workers
        return cls(
            num_workers=int(workers.num_workers),
            cpus_per_worker=int(workers.cpus_per_worker),
            executor_threads=int(workers.executor_threads),
            progress_timeout_seconds=float(workers.progress_timeout_seconds),
            start_interval_seconds=float(workers.start_interval_seconds),
            placement_timeout_seconds=float(config.trainer.distributed.placement_group_timeout_seconds),
        )


class RolloutWorkerStalledError(TimeoutError):
    """A rollout worker completed no request before its progress deadline."""


class RolloutWorkerPlacementError(RuntimeError):
    """The Ray cluster cannot hold the requested rollout workers."""


@dataclass(frozen=True)
class RayNode:
    """A live Ray node and the CPUs it offers."""

    node_id: str
    address: str
    cpus: float


def live_ray_nodes() -> list[RayNode]:
    return [
        RayNode(node["NodeID"], node["NodeManagerAddress"], float(node["Resources"].get("CPU", 0)))
        for node in ray.nodes()
        if node["Alive"]
    ]


def rollout_worker_nodes(nodes: Sequence[RayNode], num_workers: int, cpus_per_worker: int) -> list[str]:
    """Assign each rollout worker a node ID, round-robin over the nodes in address order.

    Every worker runs its tasks' sandboxes on its own host, so spreading the workers spreads that load. A node
    takes only as many workers as its CPUs hold; the others keep taking turns until every worker has a node.
    """
    ordered = sorted(nodes, key=lambda node: (node.address, node.node_id))
    free_slots = {node.node_id: int(node.cpus // cpus_per_worker) for node in ordered}
    if sum(free_slots.values()) < num_workers:
        raise RolloutWorkerPlacementError(
            f"{num_workers} rollout workers of {cpus_per_worker} CPUs do not fit on the live Ray nodes: "
            + ", ".join(f"{node.address}={node.cpus:g} CPUs" for node in ordered)
        )
    assignment: list[str] = []
    while len(assignment) < num_workers:
        for node in ordered:
            if len(assignment) < num_workers and free_slots[node.node_id]:
                free_slots[node.node_id] -= 1
                assignment.append(node.node_id)
    return assignment


async def reserve_rollout_workers(resources: RolloutWorkerResources) -> PlacementGroup:
    """Reserve one CPU bundle per worker, bundle ``i`` on worker ``i``'s node, and wait until Ray grants them all.

    Raises ``RolloutWorkerPlacementError`` when the live nodes cannot hold the workers or the reservation is not
    granted within the placement timeout, instead of leaving workers pending.
    """
    nodes = live_ray_nodes()
    assignment = rollout_worker_nodes(nodes, resources.num_workers, resources.cpus_per_worker)
    addresses = {node.node_id: node.address for node in nodes}
    workers_per_node = dict(Counter(addresses[node_id] for node_id in assignment))
    group = placement_group(
        [{"CPU": resources.cpus_per_worker} for _ in assignment],
        strategy="SPREAD",
        bundle_label_selector=[{NODE_ID_LABEL: node_id} for node_id in assignment],
    )
    try:
        await asyncio.wait_for(group.ready(), timeout=resources.placement_timeout_seconds)
    except TimeoutError as error:
        remove_placement_group(group)
        raise RolloutWorkerPlacementError(
            f"Ray did not reserve {resources.cpus_per_worker} CPUs for each rollout worker within "
            f"{resources.placement_timeout_seconds:g}s; workers per node: {workers_per_node}"
        ) from error
    logger.info("Rollout worker CPUs reserved: workers per node {}", workers_per_node)
    return group


@ray.remote
class RolloutWorker:
    """One rollout process; its event loop runs many requests concurrently on one trajectory runner."""

    def __init__(self, spec: RunnerSpec, shard: WorkerShard, sink: RetentionSink | None, executor_threads: int):
        configure_worker_process()
        configure_ray_worker_logging()
        start_fd_monitor()
        self._executor_threads = executor_threads
        self._runner = spec.build(tokenizer_from_config(spec.config), shard)
        if sink is not None:
            self._runner.set_trajectory_sink(sink)

    async def startup(self) -> None:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=self._executor_threads, thread_name_prefix="rollout-worker")
        )
        await self._runner.startup()

    async def shutdown(self) -> None:
        await self._runner.shutdown()

    async def run(
        self, input_batch: TrajectoryRequestBatch, observe: bool
    ) -> tuple[TrajectoryBatch, RolloutTimings | None]:
        with measure_rollout(enabled=observe) as observation:
            output = await self._runner.run(input_batch)
        return output, None if observation is None else observation.timings()

    async def run_task(
        self, task: RolloutTask, writer: RolloutWriter, observe: bool
    ) -> tuple[int, RolloutTimings | None]:
        with measure_rollout(enabled=observe) as observation:
            response_tokens = await self._runner.run_task(task, writer)
        return response_tokens, None if observation is None else observation.timings()

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None,
    ) -> None:
        await self._runner.start_eval_session(run_name=run_name, eval_step=eval_step, val_set_name=val_set_name)

    async def stop_eval_session(self) -> None:
        await self._runner.stop_eval_session()


class RolloutWorkerPool:
    """Rollout worker actors that the trainer uses as its trajectory runner.

    Each training task or ``run`` request goes whole to the least-loaded worker; training tasks write their groups
    straight to the rollout buffer. Evaluation drains training requests and uses the same entire pool;
    new training submissions wait until evaluation finishes. A request fails with
    ``RolloutWorkerStalledError`` when its worker completes nothing for the progress timeout.

    Workers are spread over the live Ray nodes (see ``rollout_worker_nodes``) and reach the rollout buffer, the
    trajectory sink, and the inference engines through Ray.
    """

    def __init__(self, spec: RunnerSpec, resources: RolloutWorkerResources):
        self._spec = spec
        self._resources = resources
        self._sink: RetentionSink | None = None
        self._actors: list[ActorHandle] = []
        self._placement: PlacementGroup | None = None
        # Ray async actors accept every request concurrently, and a runner may queue them behind its own limits, so
        # a worker is stalled only when none of its requests completes, not when one request is slow.
        self._pending = [0] * resources.num_workers
        self._last_progress: list[float | None] = [None] * resources.num_workers
        self._routing = asyncio.Condition()
        self._eval_session_active = False

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        """Retain trajectories inside each worker, where its runner produces them.

        Workers receive the sink when they start, so a different sink cannot be attached afterwards.
        """
        if self._actors and sink is not self._sink:
            raise RuntimeError("attach the trajectory sink before the rollout workers start")
        self._sink = sink

    async def startup(self) -> None:
        self._placement = await reserve_rollout_workers(self._resources)
        count = self._resources.num_workers
        interval = self._resources.start_interval_seconds
        if interval == 0:
            actors = [self._launch_worker(index) for index in range(count)]
            await asyncio.gather(*(actor.startup.remote() for actor in actors))
            self._actors = actors
            logger.info("Rollout workers started: count={}", count)
            return
        for index in range(count):
            if index:
                await asyncio.sleep(interval)
            actor = self._launch_worker(index)
            await actor.startup.remote()
            self._actors.append(actor)
            logger.info("Rollout worker {}/{} started", index + 1, count)

    def _launch_worker(self, index: int) -> ActorHandle:
        bundle = PlacementGroupSchedulingStrategy(self._placement, placement_group_bundle_index=index)
        worker = RolloutWorker.options(num_cpus=self._resources.cpus_per_worker, scheduling_strategy=bundle)
        return worker.remote(
            self._spec, WorkerShard(index, self._resources.num_workers), self._sink, self._resources.executor_threads
        )

    async def shutdown(self) -> None:
        actors, self._actors = self._actors, []
        placement, self._placement = self._placement, None
        # Kill every worker before reporting any that failed to shut down cleanly.
        results = await asyncio.gather(*(actor.shutdown.remote() for actor in actors), return_exceptions=True)
        for actor in actors:
            ray.kill(actor)
        if placement is not None:
            remove_placement_group(placement)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("rollout worker shutdown failed", errors)

    async def run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        return await self._observed(
            _training_phase(input_batch), lambda actor, observe: actor.run.remote(input_batch, observe)
        )

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await self._observed(
            _training_phase(task.request), lambda actor, observe: actor.run_task.remote(task, writer, observe)
        )

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
    ) -> None:
        """Give evaluation priority, drain every worker, and begin one frozen session."""
        async with self._routing:
            if self._eval_session_active:
                raise RuntimeError("an eval session is already active")
            self._eval_session_active = True
            self._eval_dispatch_counts = [0] * len(self._actors)
            logger.info(
                "Evaluation session requested: run={} step={} workers={}", run_name, eval_step, len(self._actors)
            )
        try:
            async with self._routing:
                await self._routing.wait_for(lambda: not any(self._pending))

            async def start(actor):
                await actor.start_eval_session.remote(run_name=run_name, eval_step=eval_step, val_set_name=val_set_name)

            async with asyncio.TaskGroup() as tasks:
                for actor in self._actors:
                    tasks.create_task(start(actor))
        except BaseException:
            await self.stop_eval_session()
            raise

    async def stop_eval_session(self) -> None:
        try:
            await asyncio.gather(*(actor.stop_eval_session.remote() for actor in self._actors))
        finally:
            logger.info("Evaluation worker dispatch counts: {}", self._eval_dispatch_counts)
            await self._release_eval_worker()

    async def _release_eval_worker(self) -> None:
        async with self._routing:
            self._eval_session_active = False
            self._routing.notify_all()

    def _training_worker(self) -> int | None:
        """The least-loaded worker that may take training work now, if any."""
        if self._eval_session_active:
            return None
        return min(range(len(self._actors)), key=self._pending.__getitem__, default=None)

    async def _observed(
        self,
        phase: TrainingPhase,
        submit: Callable[[ActorHandle, bool], Awaitable[tuple[_Result, RolloutTimings | None]]],
    ) -> _Result:
        """Run one request on a worker, adding the waits and phases it measured to the caller's rollout observation.

        Rollout waits happen inside the worker, but the caller's observation publishes them with the call.
        """
        observation = current_rollout_observation()
        result, timings = await self._dispatch(phase, lambda actor: submit(actor, observation is not None))
        if observation is not None and timings is not None:
            observation.absorb(timings)
        return result

    async def _dispatch(self, phase: TrainingPhase, submit: Callable[[ActorHandle], Awaitable[_Result]]) -> _Result:
        async with self._routing:
            if phase == "eval":
                if not self._eval_session_active:
                    raise RuntimeError("evaluation request received without an active eval session")
                index = min(range(len(self._actors)), key=self._pending.__getitem__)
                self._eval_dispatch_counts[index] += 1
            else:
                await self._routing.wait_for(lambda: self._training_worker() is not None)
                index = self._training_worker()
            loop = asyncio.get_running_loop()
            if self._pending[index] == 0:
                self._last_progress[index] = loop.time()
            self._pending[index] += 1
        try:
            return await self._await_progress(index, submit(self._actors[index]))
        finally:
            async with self._routing:
                self._pending[index] -= 1
                if self._pending[index] == 0:
                    self._last_progress[index] = None
                self._routing.notify_all()

    async def _await_progress(self, index: int, request: Awaitable[_Result]) -> _Result:
        """Wait for one request, failing it when its worker completes nothing for the progress timeout."""
        loop = asyncio.get_running_loop()
        result = asyncio.ensure_future(request)
        try:
            while True:
                observed = self._last_progress[index]
                assert observed is not None
                deadline = asyncio.timeout_at(observed + self._resources.progress_timeout_seconds)
                try:
                    async with deadline:
                        output = await asyncio.shield(result)
                except TimeoutError as error:
                    if not deadline.expired():
                        raise
                    if self._last_progress[index] != observed:
                        # Another request on this worker completed while this one waited, so the worker is live.
                        continue
                    # Cancel the remote request too, so the runner unwinds its work instead of leaving it detached.
                    ray.cancel(request, force=False, recursive=True)
                    raise RolloutWorkerStalledError(
                        f"rollout worker {index} completed no request for {self._resources.progress_timeout_seconds:g}s"
                    ) from error
                self._last_progress[index] = loop.time()
                return output
        except asyncio.CancelledError:
            ray.cancel(request, force=False, recursive=True)
            raise
        finally:
            # An abandoned request's result would otherwise surface later as an unretrieved error.
            result.cancel()


def _training_phase(input_batch: TrajectoryRequestBatch) -> TrainingPhase:
    metadata = input_batch.batch_metadata
    return metadata.training_phase if metadata is not None else "train"
