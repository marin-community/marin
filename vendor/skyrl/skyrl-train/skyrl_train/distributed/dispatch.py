"""Defines dispatch and collect logic for distributed training"""

import threading
from dataclasses import dataclass
from ray.actor import ActorHandle
from typing import List, Tuple, Optional, Dict, Type, Any
import asyncio
from abc import ABC, abstractmethod
import ray
from ray import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from skyrl_train.utils.logging_utils import log_exception_as_text
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
import inspect
from loguru import logger
from marinskyrl.runtime_options import R3Transport


class DispatchPutTimeoutError(RuntimeError):
    """Raised when a dispatch-loop ``ray.put()`` exceeds its configured timeout.

    WHY THIS EXISTS (2026-07-10, 80B v4/v5 wedge): `MeshDispatch.dispatch` below is a
    plain synchronous Python `for` loop calling `ray.put()` inline once per dp-group
    (the R3-resident-set fix, ac4b3806/6cfee800). It is called *inline* (not via
    `asyncio.to_thread`) from `Worker.async_run_method` / `async_run_ray_method`
    (worker.py), so it runs directly on the trainer's own asyncio event-loop thread.
    If this call stalls because of object-store capacity, eviction pressure,
    a slow/unresponsive R2 spill target, or an object still pinned by a stuck
    consumer task, the loop never reaches the next dp-group's `ray.put()`, so those
    dp-groups' actors are NEVER dispatched a `forward.remote()` call at all -- and
    the failure is invisible until some unrelated downstream watchdog (NCCL
    heartbeat, hours later) finally fires. See
    agent_logs/2026-07-09_80b_v5_98k_nccl_wedge_kill.md: the finelog shows
    `R3_RESIDENT_SET method=forward dp=0` exactly once and dp=1.. never, and the
    unconditional post-loop `MESH_DISPATCH ... issued forward.remote() to N actors`
    summary (logged only once every actor_info has been dispatched) never fires even
    once across the whole 928MB log -- i.e. this loop provably never completes.

    This does NOT claim to fix the underlying trigger (still uncertain -- see the
    commit message); it converts a silent, multi-hour, whole-job-idle stall into a
    fast, loud, retryable failure, matching this codebase's dominant "nothing blocks
    forever without an explicit timeout" idiom.
    """


@dataclass(frozen=True)
class DispatchSettings:
    r3_transport: R3Transport
    r3_dispatch_put_timeout_seconds: float


def _ray_put_bounded(obj, timeout_seconds: float, what: str) -> ObjectRef:
    """Run ``ray.put(obj)`` with a timeout.

    `ray.put()` has no native timeout param, so the put runs on a daemon helper
    thread and this function waits on it with `Thread.join(timeout=...)`. On timeout
    we raise `DispatchPutTimeoutError` immediately -- we do NOT wait for or cancel
    the helper thread (Ray gives no cancel API for an in-flight `ray.put()`); the
    thread is a daemon so it cannot block process exit, and the caller is expected to
    let the exception propagate and the process restart (the launcher's
    `--max-retries` already handles this).

    ``timeout_seconds <= 0`` disables the bound entirely, matching a bare
    `ray.put(obj)` call (including the exact same exception behavior on failure).
    """
    if timeout_seconds <= 0:
        return ray.put(obj)

    result: Dict[str, Any] = {}

    def _do_put() -> None:
        try:
            result["ref"] = ray.put(obj)
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread below
            result["exc"] = e

    t = threading.Thread(target=_do_put, name=f"skyrl-dispatch-put-{what}", daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        raise DispatchPutTimeoutError(
            f"ray.put() for {what} did not return within {timeout_seconds:.0f}s. "
            f"This is the dispatch-loop stall signature "
            f"documented in agent_logs/2026-07-09_80b_v5_98k_nccl_wedge_kill.md: failing "
            f"loud+fast here instead of hanging silently (previously only surfaced hours "
            f"later via an unrelated NCCL watchdog, with dp-groups after this one never "
            f"dispatched at all). The stuck put() keeps running in the background (no "
            f"cancel API) -- this process should be restarted."
        )
    if "exc" in result:
        raise result["exc"]
    return result["ref"]


# ---------------------------------------------------------------------------
# R3 decentralization
# ---------------------------------------------------------------------------
#
# WHY (2026-07-10, 80B head-plasma overflow). Resident R3 transport
# (ac4b3806) deduped the per-actor R3 fan-out to one `ray.put` per dp-group, but
# that `ray.put` runs ON THE DRIVER, so the multi-GB `rollout_routed_experts`
# (R3) chunk still lands in the DRIVER (head-node) plasma and stays PINNED there
# for the whole ~800s forward (the live `forward.remote()` tasks BORROW the
# driver-owned object). At 80B (8 dp-groups x ~4.6GB + the async-generation
# backlog) this overflows the head object store -> the dp=1 put stalls ->
# DispatchPutTimeoutError at global_step 1 (see
# agent_logs/2026-07-09_80b_v5_98k_nccl_wedge_kill.md). Prior frameworks
# (prime-rl / verl / slime) never centralize R3 in the head.
#
# WHAT this does. With decentralized transport, the per-dp-group chunk is
# materialized into the plasma of a CONSUMER (dp-group) NODE instead of the
# driver: a tiny NodeAffinity-scheduled task runs on that node and RETURNS the
# chunk unchanged. Ray stores a task's return value in the EXECUTING worker's
# node object store (the driver owns only the metadata / ref-count, not the
# bytes). The dp-group's `forward.remote()` calls then borrow a
# consumer-node-resident object, so the HEAD plasma holds ~0 R3 for the forward's
# duration. The driver's transient arg copy (the implicit put Ray makes to ship
# the chunk to the relocate task) is unreferenced the moment that task fetches it
# (seconds), NOT pinned for the ~800s forward -> head footprint becomes O(1) in
# model scale.
#
# CORRECTNESS. This changes only WHERE the chunk's bytes reside, never the value.
# The relocate task is a pure pass-through (`return chunk`), so the object the
# forward actors dereference is byte-identical to the driver-side
# `ray.put(chunk)` it replaces (same `data.chunk` rows, same Ray serialization).
# Row order and DP slicing are unchanged for dense and compact route payloads.
# Select resident transport to force the old
# driver-put behavior (strict A/B isolation); decentral is the default so
# the head-plasma DispatchPutTimeout footgun does not recur at scale.
#
# SCOPE (honest). This removes the PINNED driver residency (the wedge cause) but
# not the driver's TRANSIENT ship of each chunk (the driver still assembles the
# batch and puts each dp-chunk once to ship it). Moving payload fetch and batch
# conversion to trainers would remove that driver hop.

# Per-actor node-id cache: get_ray_node_id is a stable actor property, resolve once.
_ACTOR_NODE_ID_CACHE: Dict[str, str] = {}


@ray.remote(num_cpus=0)
def _relocate_chunk_to_node(chunk: TrainingInputBatch) -> TrainingInputBatch:
    """Return `chunk` unchanged; used purely to place the chunk's object-store
    copy on the executing (dp-group consumer) node rather than the driver.

    Scheduled with NodeAffinity onto a consumer node of the target dp-group. Ray
    stores a task's return value in the executing worker's node plasma, owned by
    the caller (driver) but RESIDENT on the consumer node -- so the driver never
    holds the R3 bytes resident in the head plasma for the forward's duration.
    `num_cpus=0` so it always schedules on the GPU-saturated training nodes.
    """
    return chunk


def _resolve_actor_node_id(handle: ActorHandle, timeout_seconds: float) -> Optional[str]:
    """Best-effort resolve (and cache) the Ray node id an actor lives on.

    Returns None on any failure/timeout so the caller can fall back to the bounded
    driver put -- we NEVER want to lose the loud-fail (DispatchPutTimeoutError)
    property or hang here. ``timeout_seconds <= 0`` waits unbounded (matches the disabled
    dispatch-put bound).
    """
    try:
        key = handle._actor_id.hex()
    except Exception:  # noqa: BLE001 - non-Worker/handle without a stable id -> no decentral
        return None
    if key in _ACTOR_NODE_ID_CACHE:
        return _ACTOR_NODE_ID_CACHE[key]
    try:
        ref = handle.get_ray_node_id.remote()
        node_id = ray.get(ref, timeout=(timeout_seconds if timeout_seconds > 0 else None))
    except Exception as e:  # noqa: BLE001 - actor lacks the method / call failed / timed out
        logger.warning(
            f"Decentralized R3 transport could not resolve node id for actor {key[:8]} "
            f"({e!r}); falling back to the bounded driver ray.put for this dp-group."
        )
        return None
    if not isinstance(node_id, str) or not node_id:
        return None
    _ACTOR_NODE_ID_CACHE[key] = node_id
    return node_id


@dataclass
class MeshRank:
    """Represents a rank in the device mesh.

    This is a tuple of (DP, SP, TP, PP) ranks.
    """

    dp: int
    sp: int
    tp: int
    pp: int

    world_size: int
    dp_size: int
    pp_size: int

    def is_collection_dp_rank(self) -> bool:
        """Check if this rank is a DP rank to collect from

        This is the rank with (SP=0, TP=0, PP=pp_size-1)

        Note: double check this for ETP > 1 (but this is not a typically used case)
        """
        return self.tp == 0 and self.pp == self.pp_size - 1 and self.sp == 0

    def __str__(self) -> str:
        return f"MeshRank(dp={self.dp}, sp={self.sp}, tp={self.tp}, pp={self.pp}, world_size={self.world_size}, dp_size={self.dp_size}, pp_size={self.pp_size})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ActorInfo:
    """Actor information for distributed training.

    This includes the actor handle and the rank in the device mesh.
    """

    handle: ActorHandle
    rank: MeshRank


class WorkerGroupTaskError(RuntimeError):
    """A distributed actor task failed and its peer group was terminated."""

    def __init__(self, operation: str, actor_index: int, mesh_rank: MeshRank) -> None:
        self.operation = operation
        self.actor_index = actor_index
        self.mesh_rank = mesh_rank
        super().__init__(operation, actor_index, mesh_rank)

    def __str__(self) -> str:
        return (
            f"{self.operation} failed on actor index {self.actor_index} ({self.mesh_rank}); "
            "terminating the worker group so an outer retry can rebuild the communicator"
        )


def collect_actor_results(actor_infos: List[ActorInfo], object_refs: List[ObjectRef], *, operation: str) -> List[Any]:
    """Collect a distributed actor gang and terminate every peer on one task error."""
    if len(actor_infos) != len(object_refs):
        raise ValueError("actor_infos and object_refs must have the same length")

    pending = {object_ref: index for index, object_ref in enumerate(object_refs)}
    results: List[Any] = [None] * len(object_refs)
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, fetch_local=False)
        object_ref = ready[0]
        actor_index = pending.pop(object_ref)
        try:
            results[actor_index] = ray.get(object_ref)
        except Exception as error:
            mesh_rank = actor_infos[actor_index].rank
            group_error = WorkerGroupTaskError(operation, actor_index, mesh_rank)
            # This is the last point that still owns Ray's initiating remote exception.
            # Peer teardown can otherwise leave only secondary actor-death and process-signal receipts.
            log_exception_as_text(str(group_error), error)
            for actor_info in actor_infos:
                try:
                    ray.kill(actor_info.handle, no_restart=True)
                except Exception:
                    logger.exception("Failed to terminate a peer after a distributed actor task error")
            raise group_error from error
    return results


class Dispatch(ABC):
    """Base class for dispatch types

    Dispatch types are responsible for:
    - dispatching method calls to actors handling data sharding if necessary
    - collecting results from actors and concatenating results if necessary
    - validating arguments for dispatch
    """

    @classmethod
    @abstractmethod
    def dispatch(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        *args,
        settings: DispatchSettings,
        **kwargs,
    ) -> List[ObjectRef]:
        """Dispatches method calls to the actors with data sharing if necessary."""
        pass

    @classmethod
    @abstractmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        """Collects results from the actors asynchronously in an asyncio-compatible way."""
        pass

    @classmethod
    @abstractmethod
    def sync_collect(cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]) -> Optional[TrainingOutputBatch]:
        """Collects results from the actors synchronously and returns a `TrainingOutputBatch`."""
        pass

    @classmethod
    @abstractmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        """Validate and process arguments for dispatch.

        Returns:
            Tuple of (args, kwargs) to be passed to dispatch
        """
        pass


class MeshDispatch(Dispatch):
    """Mesh dispatch type to dispatch data to a group of actors along the device mesh.

    Supports DP (Data Parallel), SP (Sequence Parallel), TP (Tensor Parallel) and PP (Pipeline Parallel) parallelism.
    The actor method should accept a single argument - the data batch.

    For data dispatch:

    * The input data is chunked into `dp_size` equal chunks, where `dp_size` is the size of data parallelism.
    * Each actor with the same DP rank processes the same data chunk in parallel.

    For data collection:

    * Data is collected only from the primary rank of each model/sequence parallel group.
    * The primary rank is defined as the rank with (SP=0, TP=0, PP=0).
    * The collected chunks are concatenated in order of DP rank to reconstruct the full data.

    Example: For a world size of 8, with DP size=2, SP size=2, TP size=2, PP size=1:

    * Data dispatch: The data is chunked into 2 chunks. All actors with DP rank 0 process the first chunk,
      and all actors with DP rank 1 process the second chunk.
    * Data collection: Only two actors contribute to the final output - the primary rank from each DP group:
      (DP=0, SP=0, TP=0, PP=0) and (DP=1, SP=0, TP=0, PP=0). Their chunks are concatenated in order.

    """

    @classmethod
    def dispatch(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        data: TrainingInputBatch,
        *,
        settings: DispatchSettings,
    ) -> List[ObjectRef]:
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = []
        dp_size = actor_infos[0].rank.dp_size
        assert len(data) % dp_size == 0, "data batch size must be divisible by dp_size, got {} and {}".format(
            len(data), dp_size
        )
        chunk_size = len(data) // dp_size
        data_chunks: List[TrainingInputBatch] = data.chunk(chunk_size)

        # DISPATCH FAN-OUT INSTRUMENT (ungated, only for `forward` to avoid log spam).
        # The 131k MoE-RL wedge (FR-proven 2026-06-30) showed only rank 0 RAN the
        # per-step forward after the weight-sync drain, while a peer rank sat idle
        # in `select`. The open question was whether
        # the DRIVER only KEYED the forward to rank 0 (a dispatch-keying bug) or
        # whether it fanned to all 32 actors but only rank 0's async-actor loop
        # SCHEDULED the dispatched task (a post-drain re-occupation race). This log
        # makes the answer unambiguous on the next run: it prints every (dp,sp,tp,pp)
        # rank the `forward.remote()` call is issued to from the driver. If all 32
        # appear here but only rank 0 logs WORKER_FORWARD_ENTER -> scheduling race
        # (the fix below addresses it). If only rank 0 appears here -> keying bug.
        log_dispatch = method == "forward"
        dispatched_ranks = [] if log_dispatch else None

        # Serialize each DP chunk once and share its ObjectRef across the actors
        # processing those rows. Passing the same Python batch by value still
        # serializes a new object per actor, including batches without R3 routes.
        # Ray dereferences positional ObjectRefs before calling the worker, so
        # slicing and worker signatures stay the same. BY_VALUE is the opt-out.
        resident = settings.r3_transport is not R3Transport.BY_VALUE and len(data_chunks) > 0
        # Bound on each per-dp-group `ray.put()` below (see `_ray_put_bounded` /
        # `DispatchPutTimeoutError` docstrings for the full incident writeup). Default
        # 600s is generous relative to observed local put durations for a single
        # multi-GB dp-chunk (low single-digit seconds even under object-store
        # pressure) while still being well inside the existing NCCL/collective
        # collective timeout budgets this codebase already uses -- i.e. a stuck put fails loud
        # well before the watchdog would otherwise silently wait out the full hour.
        # <=0 disables the bound (byte-identical to today's bare `ray.put()`).
        # Under ``r3_transport=decentral``, materialize each dp-chunk on a consumer node instead of the
        # driver plasma. Only meaningful alongside `resident` (it is the R3
        # transport that overflows the head). See the module-level block above.
        # ``resident`` retains the centralized driver-put behavior for diagnostics.
        decentral = (
            resident and data_chunks[0].routed_experts is not None and settings.r3_transport is R3Transport.DECENTRAL
        )
        chunk_refs: List[Optional[ObjectRef]] = [None] * len(data_chunks)
        for actor_info in actor_infos:
            # index into tensordict to get the correct data to send
            dp = actor_info.rank.dp
            if resident:
                # Put the dp-chunk ONCE; share the single ObjectRef across all
                # actors in this dp-group (no per-actor re-serialization / spill).
                if chunk_refs[dp] is None:
                    _r3 = data_chunks[dp].routed_experts
                    nbytes = int(_r3.nbytes) if _r3 is not None else 0
                    dtype = _r3.dtype if _r3 is not None else None
                    node_id = (
                        _resolve_actor_node_id(actor_info.handle, settings.r3_dispatch_put_timeout_seconds)
                        if decentral
                        else None
                    )
                    if decentral and node_id is not None:
                        # Materialize the chunk's object-store copy on a CONSUMER
                        # node (this actor's node), NOT the driver head plasma.
                        # Byte-identical value (pure pass-through) -> alignment
                        # inherited unchanged; head holds ~0 R3 for the forward.
                        chunk_refs[dp] = _relocate_chunk_to_node.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=True)
                        ).remote(data_chunks[dp])
                        # UNGATED per-dp-group marker (distinct from R3_RESIDENT_SET)
                        # so the smoke test can confirm the DECENTRAL path is taken
                        # AND that the head-resident R3_RESIDENT_SET put is NOT.
                        logger.info(
                            f"R3_DECENTRAL_SET method={method} dp={dp} nbytes={nbytes} dtype={dtype} node={node_id[:8]}"
                        )
                    else:
                        # Default / fallback: bounded driver-side ray.put (today's
                        # behavior; keeps the loud DispatchPutTimeoutError on stall).
                        chunk_refs[dp] = _ray_put_bounded(
                            data_chunks[dp],
                            settings.r3_dispatch_put_timeout_seconds,
                            what=f"method={method} dp={dp}",
                        )
                        # UNGATED per-dp-group marker so we can SEE the resident set
                        # install (target: one line per dp-group, on every step) and
                        # confirm the R3 bytes are put once, not per-actor.
                        if _r3 is not None:
                            logger.info(f"R3_RESIDENT_SET method={method} dp={dp} nbytes={nbytes} dtype={dtype}")
                        else:
                            logger.info(f"BATCH_RESIDENT_SET method={method} dp={dp}")
                data_to_send = chunk_refs[dp]
            else:
                data_to_send = data_chunks[dp]
            object_refs.append(getattr(actor_info.handle, method).remote(data_to_send))
            if log_dispatch:
                r = actor_info.rank
                dispatched_ranks.append(f"(dp={r.dp},sp={r.sp},tp={r.tp},pp={r.pp})")
        if log_dispatch:
            logger.info(
                f"MESH_DISPATCH method=forward issued forward.remote() to "
                f"{len(dispatched_ranks)} actors: {dispatched_ranks}"
            )
        return object_refs

    @classmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        assert len(actor_infos) == len(object_refs), "`actor_infos` and `object_refs` must have the same length"
        all_objects = await asyncio.gather(*object_refs)
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        return

    @classmethod
    def sync_collect(cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]) -> Optional[TrainingOutputBatch]:
        assert len(actor_infos) == len(object_refs), "`actor_infos` and `object_refs` must have the same length"
        all_objects = collect_actor_results(actor_infos, object_refs, operation=f"{cls.__name__} dispatch")
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        # all should be none
        assert all(obj is None for obj in all_objects), "Got a mix of `None` and non-`None` objects"
        return

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        sig = inspect.signature(cls.dispatch)
        # Dispatch settings are supplied by PPORayActorGroup after user arguments are validated.
        validation_settings = DispatchSettings(
            r3_transport=R3Transport.BY_VALUE,
            r3_dispatch_put_timeout_seconds=0,
        )
        bound_args = sig.bind([], "dummy", *args, settings=validation_settings, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments.pop("settings")

        # Check if there are any extra arguments
        if len(bound_args.arguments) > 3:  #  data, actor_infos, method_name
            # remove actor_infos and method_name - not added by user
            bound_args.arguments.pop("actor_infos")
            bound_args.arguments.pop("method")
            raise ValueError(f"MeshDispatch only accepts 'data' as an argument, got extra args: {bound_args.arguments}")

        data = bound_args.arguments.get("data")
        if not isinstance(data, TrainingInputBatch):
            raise ValueError(f"For MeshDispatch, `data` entry should be a `TrainingInput`, got {data}")
        args = (data,)
        kwargs = {}
        return args, kwargs


class PassThroughDispatch(Dispatch):
    """PassThrough dispatch type to dispatch data to a group of actors without any sharding.

    This is useful for cases where we want to run the same method on all the actors.
    Supports methods with any number of arguments.
    """

    @classmethod
    def dispatch(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        *args,
        settings: DispatchSettings,
        **kwargs,
    ) -> List[ObjectRef]:
        return [getattr(actor_info.handle, method).remote(*args, **kwargs) for actor_info in actor_infos]

    @classmethod
    async def async_collect(
        cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
    ) -> Optional[TrainingOutputBatch]:
        all_objects = await asyncio.gather(*object_refs)
        if len(all_objects) and all_objects[0] is not None:
            return concatenate_outputs_after_mesh_dispatch(actor_infos, all_objects)
        return

    @classmethod
    def sync_collect(cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]) -> Optional[TrainingOutputBatch]:
        data_batches = ray.get(object_refs)
        if len(data_batches) > 0 and data_batches[0] is not None:
            assert isinstance(data_batches[0], TrainingOutputBatch), (
                "data_batches must be a list of `TrainingOutputBatch` objects"
            )
            return concatenate_outputs_after_mesh_dispatch(actor_infos, data_batches)
        # all should be none
        assert all(obj is None for obj in data_batches), "Got a mix of `None` and non-`None` objects"
        return

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # no validation needed just pass everything
        return args, kwargs


class DispatchRegistry:
    _registry: Dict[str, Type[Dispatch]] = {"mesh": MeshDispatch, "pass_through": PassThroughDispatch}

    @classmethod
    def register(cls, name: str, dispatch_class: Type[Dispatch]) -> None:
        """Register a new dispatch type."""
        assert issubclass(dispatch_class, Dispatch)
        cls._registry[name] = dispatch_class

    @classmethod
    def get(cls, name: str) -> Type[Dispatch]:
        """Get a registered dispatch type."""
        if name not in cls._registry:
            raise KeyError(f"Dispatch type '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[Dispatch]]:
        """List all registered dispatch types."""
        return cls._registry


def register_dispatch_type(name: str, dispatch_class: Type) -> None:
    DispatchRegistry.register(name, dispatch_class)


def concatenate_outputs_after_mesh_dispatch(
    actor_infos: List[ActorInfo], data_batches: List[TrainingOutputBatch]
) -> TrainingOutputBatch:
    """Concatenate data batches from different ranks after mesh dispatch.

    - Data is collected only from the primary DP rank.
    - The collected chunks are concatenated in order of DP rank to reconstruct the full data.
    """
    assert len(actor_infos) == len(data_batches), "`actor_infos` and `data_batches` must have the same length"
    shards = []
    # collect in-order
    dp_rank_to_shard = {}
    for actor_info, data_batch in zip(actor_infos, data_batches):
        if actor_info.rank.is_collection_dp_rank():
            dp_rank = actor_info.rank.dp
            dp_rank_to_shard[dp_rank] = data_batch
    for i in range(actor_infos[0].rank.dp_size):
        shards.append(dp_rank_to_shard[i])
    return TrainingOutputBatch.cat(shards)
