# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import faulthandler
import gc
import json
import logging
import os
import pathlib
import queue
import resource
import sys
import threading
import time
import tracemalloc
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, Optional, ParamSpec, Sequence, TypeVar, Union

import equinox
import fsspec
import haliax.partitioning
import humanfriendly
import jax
import jax.numpy as jnp
from draccus import field
from fsspec import AbstractFileSystem
from haliax.jax_utils import broadcast_one_to_all, is_in_jit, is_jax_array_like
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jaxtyping import PyTree

from rigging.filesystem import StoragePath

from levanter._debug_logging import flush_debug_output
from levanter.tensorstore_serialization import (
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)
from levanter.utils.types import FilterSpec

logger = logging.getLogger(__name__)

PathLike = Union[str, pathlib.Path]

M = TypeVar("M", bound=PyTree)
Sig = ParamSpec("Sig")


def _format_bytes_human_readable(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None
    return humanfriendly.format_size(num_bytes, binary=True)


def _current_process_rss_bytes() -> int | None:
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    return rss * 1024


CheckpointDebugStateProvider = Callable[[], Mapping[str, Any]]
_DEBUG_CHECKPOINTER_STATE_PROVIDERS: dict[str, CheckpointDebugStateProvider] = {}
_DEBUG_CHECKPOINTER_STATE_PROVIDER_LOCK = threading.Lock()


def register_debug_checkpointer_state_provider(name: str, provider: CheckpointDebugStateProvider) -> None:
    """Register a debug state provider that contributes extra checkpoint diagnostics."""
    with _DEBUG_CHECKPOINTER_STATE_PROVIDER_LOCK:
        _DEBUG_CHECKPOINTER_STATE_PROVIDERS[name] = provider


def unregister_debug_checkpointer_state_provider(name: str) -> None:
    """Unregister a previously registered checkpoint debug state provider."""
    with _DEBUG_CHECKPOINTER_STATE_PROVIDER_LOCK:
        _DEBUG_CHECKPOINTER_STATE_PROVIDERS.pop(name, None)


def _collect_debug_checkpointer_state() -> dict[str, Any]:
    with _DEBUG_CHECKPOINTER_STATE_PROVIDER_LOCK:
        providers = list(_DEBUG_CHECKPOINTER_STATE_PROVIDERS.items())

    snapshots: dict[str, Any] = {}
    for name, provider in providers:
        try:
            snapshots[name] = dict(provider())
        except Exception as exc:
            snapshots[name] = {"provider_error": f"{type(exc).__name__}: {exc}"}
    return snapshots


def _manager_debug_state(manager: GlobalAsyncCheckpointManager | None) -> dict[str, Any]:
    if manager is None:
        return {}

    thread = getattr(manager, "_thread", None)
    futures = getattr(manager, "_commit_futures", None)
    exception = getattr(manager, "_exception", None)

    commit_futures_count = None
    if futures is not None:
        try:
            commit_futures_count = len(futures)
        except Exception:
            commit_futures_count = None

    commit_futures_done = None
    if futures is not None:
        try:
            commit_futures_done = sum(1 for future in futures if hasattr(future, "done") and future.done())
        except Exception:
            commit_futures_done = None

    return {
        "previous_async_commit_alive": bool(thread is not None and thread.is_alive()),
        "previous_async_commit_thread_name": getattr(thread, "name", None),
        "previous_async_commit_futures": commit_futures_count,
        "previous_async_commit_futures_done": commit_futures_done,
        "previous_async_commit_exception": None if exception is None else f"{type(exception).__name__}: {exception}",
    }


def _checkpoint_debug_json(state: Mapping[str, Any]) -> str:
    return json.dumps(dict(state), sort_keys=True, default=str)


def _tracemalloc_memory_state() -> dict[str, str | None]:
    if not tracemalloc.is_tracing():
        return {}

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    return {
        "python_tracemalloc_current": _format_bytes_human_readable(current_bytes),
        "python_tracemalloc_peak": _format_bytes_human_readable(peak_bytes),
    }


def _format_tracemalloc_growth(snapshot: tracemalloc.Snapshot | None, limit: int) -> list[str]:
    if snapshot is None or not tracemalloc.is_tracing() or limit <= 0:
        return []

    current_snapshot = tracemalloc.take_snapshot()
    stats = [stat for stat in current_snapshot.compare_to(snapshot, "lineno") if stat.size_diff > 0][:limit]
    return [
        (
            f"{stat.traceback[0].filename}:{stat.traceback[0].lineno} "
            f"size_diff={_format_bytes_human_readable(stat.size_diff)} count_diff={stat.count_diff}"
        )
        for stat in stats
        if stat.traceback
    ]


def _run_debug_gc() -> dict[str, Any]:
    rss_before = _format_bytes_human_readable(_current_process_rss_bytes())
    gc_counts_before = gc.get_count()
    tracked_objects_before = len(gc.get_objects())
    tracemalloc_before = _tracemalloc_memory_state()

    collected = gc.collect()

    rss_after = _format_bytes_human_readable(_current_process_rss_bytes())
    gc_counts_after = gc.get_count()
    tracked_objects_after = len(gc.get_objects())
    tracemalloc_after = _tracemalloc_memory_state()

    return {
        "collected_objects": collected,
        "gc_counts_before": gc_counts_before,
        "gc_counts_after": gc_counts_after,
        "tracked_objects_before": tracked_objects_before,
        "tracked_objects_after": tracked_objects_after,
        "gc_garbage_len": len(gc.garbage),
        "rss_before": rss_before,
        "rss_after": rss_after,
        **{f"{key}_before": value for key, value in tracemalloc_before.items()},
        **{f"{key}_after": value for key, value in tracemalloc_after.items()},
    }


class _CheckpointProgressLogger:
    def __init__(
        self,
        *,
        step: int,
        checkpoint_path: str,
        manager: GlobalAsyncCheckpointManager | None = None,
        interval: float = 60.0,
        dump_stacks_after: float | None = None,
        top_allocations: int = 8,
        flush_logs: bool = True,
    ):
        self.step = step
        self.checkpoint_path = checkpoint_path
        self.manager = manager
        self.interval = interval
        self.phase = "starting"
        self.started_at = time.time()
        self.phase_started_at = self.started_at
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._stack_dumped = False
        self._dump_after = dump_stacks_after
        self._top_allocations = top_allocations
        self._flush_logs = flush_logs
        self._tracemalloc_baseline = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None

    def _log(self, level: int, message: str, *args: Any) -> None:
        logger.log(level, message, *args)
        if self._flush_logs:
            flush_debug_output(logger)

    def _log_memory_state(self, event: str, *, include_top_allocations: bool = False) -> None:
        state: dict[str, Any] = {
            "event": event,
            "phase": self.phase,
            "rss": _format_bytes_human_readable(_current_process_rss_bytes()),
            "gc_counts": gc.get_count(),
            "gc_garbage_len": len(gc.garbage),
            **_tracemalloc_memory_state(),
        }
        manager_state = _manager_debug_state(self.manager)
        if manager_state:
            state["manager"] = manager_state
        extra_state = _collect_debug_checkpointer_state()
        if extra_state:
            state["providers"] = extra_state

        self._log(
            logging.INFO,
            "Checkpoint debug snapshot: step=%d state=%s",
            self.step,
            _checkpoint_debug_json(state),
        )

        if include_top_allocations:
            top_growth = _format_tracemalloc_growth(self._tracemalloc_baseline, self._top_allocations)
            if top_growth:
                self._log(
                    logging.INFO,
                    "Checkpoint debug tracemalloc growth: step=%d top=%s",
                    self.step,
                    json.dumps(top_growth, default=str),
                )

    def log_gc_snapshot(self, event: str) -> None:
        self._log(
            logging.INFO,
            "Checkpoint debug gc: step=%d state=%s",
            self.step,
            _checkpoint_debug_json({"event": event, **_run_debug_gc()}),
        )

    def reset_tracemalloc_baseline(self, reason: str) -> None:
        if not tracemalloc.is_tracing():
            return
        self._tracemalloc_baseline = tracemalloc.take_snapshot()
        self._log(
            logging.INFO,
            "Checkpoint debug tracemalloc baseline reset: step=%d reason=%s",
            self.step,
            reason,
        )

    def start(self) -> None:
        self._log(
            logging.INFO,
            "PHASE: CHECKPOINT step=%d phase=%s path=%s",
            self.step,
            self.phase,
            self.checkpoint_path,
        )
        self._log_memory_state("checkpoint_start")
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self.phase_started_at = time.time()
        self._log(
            logging.INFO,
            "PHASE: CHECKPOINT step=%d phase=%s path=%s",
            self.step,
            phase,
            self.checkpoint_path,
        )
        self._log_memory_state(f"phase_{phase}", include_top_allocations=True)

    def finish(self, status: str) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        elapsed = time.time() - self.started_at
        self._log_memory_state(f"checkpoint_{status}", include_top_allocations=True)
        self._log(
            logging.INFO,
            "PHASE: CHECKPOINT step=%d phase=%s path=%s elapsed=%.2fs",
            self.step,
            status,
            self.checkpoint_path,
            elapsed,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            with self._lock:
                phase = self.phase
                phase_elapsed = time.time() - self.phase_started_at

            total_elapsed = time.time() - self.started_at
            rss = _format_bytes_human_readable(_current_process_rss_bytes())
            rss_suffix = f", rss={rss}" if rss is not None else ""
            self._log(
                logging.INFO,
                "Checkpoint still running: step=%d phase=%s total_elapsed=%.1fs phase_elapsed=%.1fs%s",
                self.step,
                phase,
                total_elapsed,
                phase_elapsed,
                rss_suffix,
            )
            self._log_memory_state("checkpoint_progress", include_top_allocations=True)

            if self._dump_after is not None and not self._stack_dumped and total_elapsed >= self._dump_after:
                self._stack_dumped = True
                self._log(
                    logging.WARNING,
                    "Checkpoint exceeded %.1fs at step %d; dumping Python thread stacks",
                    self._dump_after,
                    self.step,
                )
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                if self._flush_logs:
                    flush_debug_output(logger)


@dataclass(frozen=True)
class CheckpointInterval:
    every: int  # how often to checkpoint
    until: Optional[int] = None  # until what step to save checkpoints with this policy, None means forever


@dataclass(frozen=True)
class CheckpointCandidate:
    """A complete checkpoint discovered under a checkpoint root."""

    path: str
    step: int
    timestamp: datetime.datetime
    metadata: Mapping[str, Any]


@dataclass
class CheckpointDebugConfig:
    enabled: bool = False
    log_interval: float = 60.0
    dump_stacks_after: float | None = None
    tracemalloc_frames: int = 25
    top_allocations: int = 8
    force_gc_before_serialize: bool = True
    flush_logs: bool = True

    def validate(self) -> None:
        assert self.log_interval > 0, "checkpoint debug log_interval must be positive"
        if self.dump_stacks_after is not None:
            assert self.dump_stacks_after > 0, "checkpoint debug dump_stacks_after must be positive when set"
        assert self.tracemalloc_frames > 0, "checkpoint debug tracemalloc_frames must be positive"
        assert self.top_allocations >= 0, "checkpoint debug top_allocations must be non-negative"

    def __post_init__(self) -> None:
        self.validate()


@dataclass(frozen=True)
class _TemporaryCheckpointRecord:
    path: str
    step: int
    timestamp: datetime.datetime


def _temporary_checkpoint_sort_key(record: _TemporaryCheckpointRecord) -> tuple[int, datetime.datetime, str]:
    return (record.step, record.timestamp, record.path)


def _temporary_checkpoint_record(checkpoint_path: str) -> _TemporaryCheckpointRecord | None:
    try:
        metadata = _load_metadata(checkpoint_path)
    except FileNotFoundError:
        logger.warning("Could not load metadata for checkpoint %s.", checkpoint_path)
        return None

    if not metadata.get("is_temporary", False):
        return None

    return _TemporaryCheckpointRecord(
        path=checkpoint_path,
        step=int(metadata["step"]),
        timestamp=datetime.datetime.fromisoformat(metadata["timestamp"]),
    )


class Checkpointer:
    """
    A checkpointer class that saves checkpoints with two different, but overlapping policies: time and step.

    Note that this class is stateful: it keeps track of the last time a checkpoint was saved, and the last step
    a checkpoint was saved at.
    """

    base_path: str
    save_interval: Optional[datetime.timedelta]  # we save at least this frequently
    step_policies: Sequence[CheckpointInterval] = dataclasses.field(
        default_factory=lambda: [CheckpointInterval(every=1000)]
    )

    _temporary_checkpoints: list["_TemporaryCheckpointRecord"]

    def __init__(
        self,
        base_path: PathLike,
        save_interval: Optional[datetime.timedelta],
        step_policies: Sequence[CheckpointInterval],
        *,
        temporary_base_path: Optional[PathLike] = None,
        keep_params: PyTree[FilterSpec] = True,
        dt_now_injection: Optional[Callable[[], datetime.datetime]] = None,
        delete_old_temp_checkpoints: bool = True,
        keep_last_temporary_checkpoints: int = 1,
        debug: CheckpointDebugConfig | None = None,
    ):
        """
        Class for managing checkpoints. Saves checkpoints according to two policies: time and step.

        Time policy: we save a checkpoint at least every `save_interval` seconds.
        Step policy: we save a checkpoint every `every` steps, until `until` steps have been reached.

        Time checkpoints are deleted after the next checkpoint is saved. Step checkpoints are never deleted.

        Args:
            base_path: the base path to save checkpoints to. may be gcs, local, or anything that tensorstore supports
            save_interval: the minimum amount of time between checkpoints (for time)
            step_policies: the step policies to use
            temporary_base_path: separate base path for time-policy (temporary) checkpoints. When set,
                temporary checkpoints are written here instead of base_path. Permanent (step-policy)
                checkpoints always go to base_path. If None, all checkpoints go to base_path.
            keep_params: a PyTree of FilterSpecs that specifies which parameters to keep in the checkpoint
            dt_now_injection: a function that returns the current time. useful for testing
            delete_old_temp_checkpoints: if True, carry forward a temporary checkpoint discovered at startup so the
                next successful save can clean it up according to keep_last_temporary_checkpoints.
            keep_last_temporary_checkpoints: number of complete temporary checkpoints to retain after a temporary
                checkpoint commits successfully. Set to 0 to delete temporary checkpoints after they commit. Permanent
                checkpoints still clean up temporary checkpoints because they supersede them for recovery.
        """
        if keep_last_temporary_checkpoints < 0:
            raise ValueError("keep_last_temporary_checkpoints must be non-negative")

        self.base_path = str(base_path)
        self.temporary_base_path = str(temporary_base_path) if temporary_base_path is not None else None
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self.keep_params = keep_params
        self._dt_now_injection = dt_now_injection or datetime.datetime.now
        self._last_save_time = self._dt_now_injection()
        self._last_save_step = 0
        self.keep_last_temporary_checkpoints = keep_last_temporary_checkpoints
        self.debug = debug or CheckpointDebugConfig()
        self._temporary_checkpoints = []

        # ensure that the step_policies are sorted. We could sort, but instead we'll just insist that they are sorted
        # since it's probably a typo if they aren't
        for i in range(1, len(step_policies)):
            # factor these out so pyrefly can figure it out
            prev_until = step_policies[i - 1].until
            until = step_policies[i].until
            if prev_until is None:
                raise ValueError("Only the last step policy can have an 'until' value of None")
            if until is None:
                continue
            if prev_until >= until:
                raise ValueError("Step policies must be sorted by 'until' value")

        # The default of 5 minutes is too short even for modestly sized models for some reason
        self._manager = GlobalAsyncCheckpointManager(timeout_secs=60 * 30)

        if jax.process_index() == 0:
            self._async_checkpoint_remover_queue: queue.Queue[str] = queue.Queue(maxsize=-1)
            self._async_checkpoint_remover_thread = threading.Thread(
                target=self._async_checkpoint_remover, daemon=True
            )
            self._async_checkpoint_remover_thread.start()
            self._checkpoint_being_removed = None

        if jax.process_index() == 0 and delete_old_temp_checkpoints:
            self._temporary_checkpoints = self._discover_temporary_checkpoints()
            if self._temporary_checkpoints:
                logger.info(
                    "Found prior temporary checkpoints %s. They will be cleaned up after saving a new checkpoint.",
                    [record.path for record in self._temporary_checkpoints],
                )

    def load_checkpoint(
        self,
        state: M,
        checkpoint_path: PathLike,
        *,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> M:
        return load_checkpoint(state, checkpoint_path, axis_mapping=axis_mapping, mesh=mesh)

    def load_model(
        self,
        model: M,
        checkpoint_path: PathLike,
        *,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> M:
        """
        Convenience method/holdover from  previous API for loading checkpoints.
        Loads just the model assuming the model is in the `model` subdir of the checkpoint.
        """
        ret_dict = self.load_checkpoint({"model": model}, checkpoint_path, axis_mapping=axis_mapping, mesh=mesh)
        return ret_dict["model"]

    def on_step(self, *, tree: PyTree, step: int, force: bool = False):
        step = int(step)

        if step == 0:
            self._last_save_time = self._dt_now_injection()
            if not force:
                return  # don't save checkpoint at step 0 unless forced

        if step == self._last_save_step and not force:
            # we've already saved a checkpoint at this step
            return

        # two reasons we can save: time or step
        # they have different behaviors for retention.
        # if the previous checkpoint was a temporary checkpoint (i.e. saved b/c of time), we can delete it

        # there's a potential clock skew issue here: if we save by time, and the clock is skewed across processes,
        # then we could end up with a situation where one process saves a checkpoint, and then another process
        # saves a checkpoint for the next step, etc. This leads to partial checkpoints, no good.
        # we fix by having process 0 make the decision
        my_should_save = force
        my_save_permanent_ckpt = force

        if not force:
            current_every = self._get_current_step_save_interval(step)
            last_save_time = self._dt_now_injection() - self._last_save_time
            if current_every is not None and step % current_every == 0:
                my_should_save = True
                my_save_permanent_ckpt = True
            elif self.save_interval and last_save_time >= self.save_interval:
                my_should_save = True
                my_save_permanent_ckpt = False

        should_save, save_permanent_ckpt = broadcast_one_to_all(
            jnp.array([my_should_save, my_save_permanent_ckpt], dtype=jnp.bool_)
        )
        # this comes out as np.bool_, so we need to convert it to a regular bool so json serialization works
        save_permanent_ckpt = bool(save_permanent_ckpt)

        # log the decision
        if should_save:
            if save_permanent_ckpt:
                logger.info(f"Saving checkpoint at step {step}.")
            else:
                logger.info(f"Saving temporary checkpoint at step {step}.")

        if should_save:
            destination = f"step-{step}"

            # Route temporary checkpoints to temporary_base_path when configured
            if not save_permanent_ckpt and self.temporary_base_path is not None:
                save_base_path = self.temporary_base_path
            else:
                save_base_path = self.base_path

            checkpoint_path = os.path.join(save_base_path, destination)

            def callback():
                if jax.process_index() == 0:
                    if not save_permanent_ckpt:
                        self._record_temporary_checkpoint(checkpoint_path)
                        self._prune_temporary_checkpoints(self.keep_last_temporary_checkpoints)
                    else:
                        self._prune_temporary_checkpoints(0)

            self.save_checkpoint(
                tree=tree,
                step=step,
                destination=destination,
                commit_callback=callback,
                is_temporary=not save_permanent_ckpt,
                base_path_override=save_base_path,
            )

    def _discover_temporary_checkpoints(self) -> list["_TemporaryCheckpointRecord"]:
        search_paths = [self.base_path]
        if self.temporary_base_path is not None:
            search_paths.append(self.temporary_base_path)

        records_by_path: dict[str, _TemporaryCheckpointRecord] = {}
        for search_path in search_paths:
            for checkpoint_path in _discover_checkpoint_paths_single(search_path):
                record = _temporary_checkpoint_record(checkpoint_path)
                if record is not None:
                    records_by_path[record.path] = record

        return sorted(records_by_path.values(), key=_temporary_checkpoint_sort_key)

    def _record_temporary_checkpoint(self, checkpoint_path: str) -> None:
        record = _temporary_checkpoint_record(checkpoint_path)
        if record is None:
            logger.warning(
                "New checkpoint %s was expected to be temporary but metadata does not say so.", checkpoint_path
            )
            return

        records_by_path = {existing.path: existing for existing in self._temporary_checkpoints}
        records_by_path[record.path] = record
        self._temporary_checkpoints = sorted(records_by_path.values(), key=_temporary_checkpoint_sort_key)

    def _prune_temporary_checkpoints(self, keep: int) -> None:
        if keep == 0:
            retained: list[_TemporaryCheckpointRecord] = []
            to_delete = self._temporary_checkpoints
        else:
            retained = self._temporary_checkpoints[-keep:]
            to_delete = self._temporary_checkpoints[:-keep]

        for record in to_delete:
            try:
                metadata = _load_metadata(record.path)
            except FileNotFoundError:
                logger.warning("Could not load metadata for temporary checkpoint %s.", record.path)
                continue

            if metadata.get("is_temporary", False):
                logger.info("Deleting old temporary checkpoint %s after saving new checkpoint.", record.path)
                self._rm_checkpoint(record.path)
            else:
                logger.info("Not deleting checkpoint %s because it is no longer temporary.", record.path)
                retained.append(record)

        self._temporary_checkpoints = sorted(retained, key=_temporary_checkpoint_sort_key)

    def _get_current_step_save_interval(self, step):
        # binary search for the correct interval
        # we assume that the intervals are sorted by until
        current_policy = next(filter(lambda p: p.until is None or p.until >= step, self.step_policies), None)
        if current_policy is None:
            return None
        return current_policy.every

    def wait_until_finished(self):
        self._manager.wait_until_finished()
        if jax.process_index() == 0:
            while self._checkpoint_being_removed is not None or not self._async_checkpoint_remover_queue.empty():
                time.sleep(0.2)

    def _rm_checkpoint(self, checkpoint):
        if jax.process_index() == 0:
            logger.info(f"Removing checkpoint {checkpoint}")
            self._async_checkpoint_remover_queue.put(checkpoint)

    def _do_rm_checkpoint(self, cp_path):
        # have to strip protocol from path because fsspec filesystems don't like them
        fs, plain_path = _get_fs_and_plain_path(cp_path)

        try:
            logger.info(f"Deleting old checkpoint from {cp_path}")
            time_in = time.time()
            fs.rm(plain_path, recursive=True)
            time_out = time.time()
            logger.info(f"Deleted old checkpoint from {cp_path} in {time_out - time_in:.2f} seconds")
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Failed to delete checkpoint {cp_path}")

    def save_checkpoint(
        self,
        tree: PyTree,
        step: int,
        destination: str,
        commit_callback: Optional[Callable[[], None]] = None,
        *,
        is_temporary: bool = False,
        base_path_override: Optional[str] = None,
    ):
        base = base_path_override if base_path_override is not None else self.base_path
        path = os.path.join(base, destination)
        logger.info(f"Saving checkpoint at step {step} to {path}")

        save_checkpoint(
            tree,
            step=step,
            checkpoint_path=path,
            manager=self._manager,
            commit_callback=commit_callback,
            is_temporary=is_temporary,
            debug=self.debug,
        )
        self._last_save_step = step
        self._last_save_time = self._dt_now_injection()

    def _async_checkpoint_remover(self):
        while True:
            checkpoint = self._async_checkpoint_remover_queue.get(block=True)
            self._checkpoint_being_removed = checkpoint
            self._do_rm_checkpoint(checkpoint)
            self._checkpoint_being_removed = None


def save_checkpoint(
    tree: M,
    step: int,
    checkpoint_path: PathLike,
    manager: Optional[GlobalAsyncCheckpointManager] = None,
    *,
    commit_callback: Optional[Callable[[], None]] = None,
    is_temporary: bool = True,
    debug: CheckpointDebugConfig | None = None,
):
    """
    Save a checkpoint to a given path using TensorStore with OCDBT.
    Old checkpoints (non-OCDBT) can still be loaded for backward compatibility.

    If the path does not exist, it will be created.

    This method is jax.Array-aware and will save shards in a way that can be restored.

    Args:
        tree: the PyTree to save
        step: the step to save the checkpoint at
        checkpoint_path: the path to save the checkpoint to
        manager: the GlobalAsyncCheckpointManager to use for saving the checkpoint
        commit_callback: a callback to call after the checkpoint has been saved
        is_temporary: whether the checkpoint is temporary
    """
    step = int(step)
    checkpoint_path = str(checkpoint_path)
    checkpoint_debug = debug or CheckpointDebugConfig()
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")
    progress_logger: _CheckpointProgressLogger | None = None
    if checkpoint_debug.enabled:
        if not tracemalloc.is_tracing():
            tracemalloc.start(checkpoint_debug.tracemalloc_frames)
        progress_logger = _CheckpointProgressLogger(
            step=step,
            checkpoint_path=checkpoint_path,
            manager=manager,
            interval=checkpoint_debug.log_interval,
            dump_stacks_after=checkpoint_debug.dump_stacks_after,
            top_allocations=checkpoint_debug.top_allocations,
            flush_logs=checkpoint_debug.flush_logs,
        )
        progress_logger.start()

    fs: AbstractFileSystem
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    fs.makedirs(plain_path, exist_ok=True)
    if progress_logger is not None:
        progress_logger.set_phase("filesystem_ready")

    def my_callback():
        if progress_logger is not None:
            progress_logger.set_phase("metadata_write")
        status = "completed"
        try:
            _save_metadata(checkpoint_path, fs, step, is_temporary)
            logger.info(f"Saved checkpoint to {checkpoint_path} for step {step}")

            if commit_callback is not None:
                commit_callback()
        except Exception:
            status = "failed"
            raise
        finally:
            if progress_logger is not None:
                progress_logger.finish(status)

    tree = equinox.filter(tree, lambda x: is_jax_array_like(x) or isinstance(x, (int, float, bool, complex)))

    try:
        if progress_logger is not None:
            if checkpoint_debug.force_gc_before_serialize:
                progress_logger.log_gc_snapshot("pre_tensorstore_serialize")
                progress_logger.reset_tracemalloc_baseline("post_gc_pre_tensorstore_serialize")
            progress_logger.set_phase("tensorstore_serialize")
        tree_serialize_leaves_tensorstore(
            checkpoint_path,
            tree,
            manager,
            commit_callback=my_callback,
            debug_checkpointer=checkpoint_debug.enabled,
        )
        if progress_logger is not None:
            progress_logger.set_phase("async_commit_in_flight")
    except Exception:
        if progress_logger is not None:
            progress_logger.finish("failed")
        raise

    return checkpoint_path


def _save_metadata(checkpoint_path, fs, step, is_temporary):
    metadata = {"step": step, "timestamp": datetime.datetime.now().isoformat(), "is_temporary": is_temporary}
    if jax.process_index() == 0:
        with fs.open(os.path.join(checkpoint_path, "metadata.json"), "w") as json_out:
            json.dump(metadata, json_out)


def load_checkpoint(
    tree: M,
    checkpoint_path: PathLike,
    *,
    subpath: Optional[str] = None,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    allow_partial: bool = False,
) -> M:
    """
    Load a checkpoint from a given path using TensorStore.

    Supports both OCDBT (new format) and non-OCDBT (old format) checkpoints through automatic
    format detection.

    This function expects ``checkpoint_path`` to already point at a concrete checkpoint directory.
    Use ``discover_latest_checkpoint`` or ``latest_checkpoint_path`` before calling when accepting
    a parent directory.

    Args:
        tree: an exemplar of the tree to load. Can be a PyTree[ShapeDTypeStruct] instead of a PyTree[Any]
        checkpoint_path: the concrete checkpoint directory to load from
        subpath: the subpath to load from the checkpoint
        axis_mapping: the axis mapping to use for loading the checkpoint
        mesh: the mesh to use for loading the checkpoint
        allow_partial: if True, allow partial loading of the checkpoint. If False, all parameters must be present in the checkpoint.
    Returns:
        the loaded checkpoint, with the same structure as the exemplar tree

    """
    checkpoint_path = str(checkpoint_path)

    if is_in_jit():
        logger.warning("Loading checkpoint in jit. This is not recommended and probably won't work.")

    if not StoragePath(checkpoint_path).exists():
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if subpath:
        checkpoint_path = os.path.join(checkpoint_path, subpath)

    ser, non_ser = equinox.partition(tree, is_jax_array_like)
    tree = tree_deserialize_leaves_tensorstore(
        checkpoint_path, ser, axis_mapping=axis_mapping, mesh=mesh, allow_missing=allow_partial
    )
    tree = equinox.combine(tree, non_ser)
    return tree


def load_checkpoint_or_initialize(
    init_fn: Callable[Sig, M],
    checkpoint_search_paths: Sequence[PathLike],
    *,
    subpath: Optional[str] = None,
    discover_latest=True,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    is_checkpointed: FilterSpec = True,
    donate_args: FilterSpec = True,
    donate_kwargs: Optional[FilterSpec] = None,
    do_load: Optional[bool] = None,
    allow_partial: bool = False,
) -> Callable[Sig, M]:
    """
    Load from checkpoint search paths, or initialize from scratch when no checkpoint is available.
    If discover_latest is True, the latest checkpoint across the search paths will be loaded. If
    subpath is not None, only that subpath of the checkpoint is loaded. This is useful for loading,
    e.g., just the model and not the entire training state.

    This function supports "partial" checkpoint loading, where only a subset of the parameters of the
    state is loaded from the checkpoint. This is useful for initializing just some parameters.
    (Note that you have to declare which parameters you are expecting to load via is_checkpointed.
     Things can't just be missing from the checkpoint.)

    init_fn will be called inside eval_shape and inside jit, so it should be a pure function. In particular,
    it should not do any I/O.

    This function is commonly used for initializing training state from a possibly non-existent checkpoint, but it can be used
    for initializing any state from a checkpoint.

    By default, this function will donate all arguments to init_fn that are not present in the checkpoint.

    Args:
        init_fn: a function to initialize if needed
        checkpoint_search_paths: paths to search for a checkpoint. If discover_latest is False, this must contain exactly one concrete checkpoint path.
        subpath: the subpath to load from the checkpoint
        discover_latest: whether to discover the latest checkpoint in the search paths
        axis_mapping: the axis mapping to use for loading the checkpoint
        mesh: the mesh to use for loading the checkpoint
        is_checkpointed: a FilterSpec that specifies which parameters are checkpointed
        donate_args: a FilterSpec that specifies which arguments to donate to init_fn if we need to initialize
        donate_kwargs: a FilterSpec that specifies which kwargs to donate to init_fn if we need to initialize
        do_load: if True, always load the checkpoint. If False, always initialize. If None, load if the checkpoint exists, otherwise initialize
        allow_partial: if True, allow partial loading of the checkpoint. If False, all parameters must be present in the checkpoint.

    Returns:
        A function that takes the same arguments as init_fn, but loads the checkpoint if it exists and returns the
        loaded state.

    """
    if len(checkpoint_search_paths) == 0:
        raise ValueError("checkpoint_search_paths must contain at least one path")
    checkpoint_search_paths = [str(path) for path in checkpoint_search_paths]

    # some state might not be initialized, so we need to initialize it
    # JAX will be smart and only do the compute for things we actually need
    @haliax.named_jit(
        axis_resources=axis_mapping,
        out_axis_resources=axis_mapping,
        donate_args=donate_args,
        donate_kwargs=donate_kwargs,
    )
    def init_and_merge(state, *args, **kwargs):
        init_state = init_fn(*args, **kwargs)
        # remove all ShapeDTypeStructs from the state
        state = equinox.filter(state, lambda x: not isinstance(x, jax.ShapeDtypeStruct))
        return equinox.combine(state, init_state)

    def load_or_init(*args, **kwargs):
        # we need to call init_fn to get the shape, dtype, and structure of the state
        # we'll use this to deserialize the checkpoint
        state_shape = equinox.filter_eval_shape(init_fn, *args, **kwargs)

        # we need to filter the state to get the parameters we want to load
        # we'll use this to deserialize the checkpoint
        filtered_state_shape = equinox.filter(state_shape, is_checkpointed)
        # strip out all the shape stuff, leaving only the dtype and structure
        loaded_state = equinox.filter(state_shape, lambda _: False)

        if do_load is not False:
            # now we can load the checkpoint
            try:
                if discover_latest:
                    checkpoint_path = latest_checkpoint_path(checkpoint_search_paths[0], *checkpoint_search_paths[1:])
                else:
                    if len(checkpoint_search_paths) != 1:
                        raise ValueError("discover_latest=False requires exactly one checkpoint search path")
                    checkpoint_path = checkpoint_search_paths[0]

                loaded_state = load_checkpoint(
                    filtered_state_shape,
                    checkpoint_path,
                    subpath=subpath,
                    axis_mapping=axis_mapping,
                    mesh=mesh,
                    allow_partial=allow_partial,
                )
            except FileNotFoundError:
                if do_load is True:
                    raise
                logger.info(f"Checkpoint not found in {checkpoint_search_paths}. Initializing from scratch.")

        state = init_and_merge(loaded_state, *args, **kwargs)

        return state

    return load_or_init


def _load_metadata(checkpoint_path, fs=None):
    if fs is None:
        fs, _, _ = fsspec.get_fs_token_paths(str(checkpoint_path))
    with fs.open(os.path.join(checkpoint_path, "metadata.json")) as metadata_in:
        metadata = json.load(metadata_in)
    return metadata


def discover_checkpoint_candidates(
    checkpoint_path: PathLike,
    *additional_paths: PathLike,
    exclude_paths: Sequence[PathLike] = (),
    max_step: int | None = None,
) -> list[CheckpointCandidate]:
    """Return complete checkpoint candidates across one or more roots.

    A complete candidate is a checkpoint directory with a readable
    ``metadata.json`` containing a parseable integer ``step`` and ISO-format
    ``timestamp``. Results are sorted by numeric step, then timestamp, then path.
    This function is intentionally importable for operational one-liners, e.g.:

    ```bash
    uv run python -c 'from levanter.checkpoint import latest_checkpoint_path; print(latest_checkpoint_path("gs://..."))'
    ```
    """
    all_paths = [str(checkpoint_path)] + [str(p) for p in additional_paths]
    candidates_by_path: dict[str, CheckpointCandidate] = {}

    for cp_path in all_paths:
        for candidate in _discover_checkpoint_candidates_single(cp_path):
            if max_step is not None and candidate.step > max_step:
                continue
            if _is_path_under_any(candidate.path, exclude_paths):
                continue
            candidates_by_path[candidate.path] = candidate

    return sorted(candidates_by_path.values(), key=_checkpoint_candidate_sort_key)


def _checkpoint_candidate_sort_key(candidate: CheckpointCandidate) -> tuple[int, datetime.datetime, str]:
    return (candidate.step, candidate.timestamp, candidate.path)


def _is_path_under_any(path: str, roots: Sequence[PathLike]) -> bool:
    normalized_path = path.rstrip("/")
    for root in roots:
        normalized_root = str(root).rstrip("/")
        if normalized_path == normalized_root or normalized_path.startswith(f"{normalized_root}/"):
            return True
    return False


def discover_latest_checkpoint(
    checkpoint_path: PathLike,
    *additional_paths: PathLike,
    exclude_paths: Sequence[PathLike] = (),
    max_step: int | None = None,
) -> Optional[str]:
    """
    Discover the latest checkpoint across one or more root paths.

    When additional_paths are provided, all roots are searched and the highest
    numeric-step complete checkpoint across all roots is returned.
    """
    all_paths = [str(checkpoint_path)] + [str(p) for p in additional_paths]
    candidates = discover_checkpoint_candidates(
        checkpoint_path, *additional_paths, exclude_paths=exclude_paths, max_step=max_step
    )

    if candidates:
        latest = candidates[-1].path
        logger.info("Discovered latest checkpoint at %s", latest)
        return latest
    else:
        logger.warning(f"No checkpoints found in {all_paths}")
        return None


def latest_checkpoint_path(
    checkpoint_path: PathLike,
    *additional_paths: PathLike,
    exclude_paths: Sequence[PathLike] = (),
    max_step: int | None = None,
) -> str:
    """Return the latest concrete checkpoint path across one or more search roots."""
    latest = discover_latest_checkpoint(
        checkpoint_path, *additional_paths, exclude_paths=exclude_paths, max_step=max_step
    )
    if latest is None:
        search_paths = [str(checkpoint_path)] + [str(path) for path in additional_paths]
        raise FileNotFoundError(f"Could not discover checkpoint under any of: {search_paths}")
    return latest


def _discover_checkpoint_candidates_single(checkpoint_path: str) -> list[CheckpointCandidate]:
    """Discover complete checkpoint candidates in a single root path."""
    candidates: list[CheckpointCandidate] = []

    for ckpt_dir in _discover_checkpoint_paths_single(checkpoint_path):
        try:
            metadata = _load_metadata(ckpt_dir)
            step = int(metadata["step"])
            timestamp = datetime.datetime.fromisoformat(metadata["timestamp"])
        except Exception:
            logger.exception("Error loading metadata for discovered checkpoint %s", ckpt_dir)
            continue

        candidates.append(CheckpointCandidate(path=ckpt_dir, step=step, timestamp=timestamp, metadata=metadata))

    return sorted(candidates, key=_checkpoint_candidate_sort_key)


def _discover_checkpoint_paths_single(checkpoint_path: str) -> list[str]:
    """Discover valid checkpoint directories in a single root path."""
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    def is_checkpoint_dir(path: str):
        return fs.exists(os.path.join(path, "metadata.json"))

    def maybe_unstrip_protocol(path: str):
        base_path_protocol = urllib.parse.urlparse(str(checkpoint_path)).scheme
        if base_path_protocol != "" and urllib.parse.urlparse(path).scheme == "":
            return f"{base_path_protocol}://{path}"
        return path

    ckpt_dirs = [maybe_unstrip_protocol(d) for d in fs.glob(os.path.join(checkpoint_path, "*")) if fs.isdir(d)]
    ckpt_dirs.append(checkpoint_path)
    return sorted(d for d in ckpt_dirs if is_checkpoint_dir(d))


def _get_fs_and_plain_path(path, fs=None):
    if fs is None:
        fs, _, (path_to_open,) = fsspec.get_fs_token_paths(str(path))
    else:
        path_to_open = path
    return fs, path_to_open


@dataclass
class CheckpointerConfig:
    base_path: str = "checkpoints/"
    temporary_base_path: Optional[str] = None
    """Separate base path for temporary (time-policy) checkpoints. When set, temporary checkpoints
    are written here instead of base_path, allowing use of region-local storage with lifecycle TTL."""

    save_interval: Optional[timedelta] = timedelta(minutes=15)
    """Minimum time between temporary checkpoints. None disables time-policy saves
    (only `keep` intervals and forced saves, e.g. the final checkpoint, still run)."""
    # TODO: I'd like to write this, but it's not supported by draccus
    # keep: List[CheckpointInterval] = field(default_factory=lambda: [CheckpointInterval(every=1000)])
    keep: Optional[List[dict]] = field(default_factory=lambda: [dict(every=10000)])
    """Permanent checkpoint intervals. None means only forced checkpoints, such as the final checkpoint."""

    append_run_id_to_base_path: bool = True
    delete_old_temp_checkpoints: bool = True
    """
    If True, delete old checkpoints from prior attempts at this run. If False, keep them.

    This is useful if the run is being preempted and restarted, and you want to keep the old checkpoints.
    """
    keep_last_temporary_checkpoints: int = 1
    """Number of complete temporary checkpoints to retain after a successful temporary checkpoint commit."""
    debug: CheckpointDebugConfig = field(default_factory=CheckpointDebugConfig)
    """Checkpoint-path diagnostics. Disabled by default."""

    def expanded_path(self, run_id) -> str:
        if self.append_run_id_to_base_path:
            return os.path.expanduser(os.path.join(self.base_path, run_id))
        return os.path.expanduser(self.base_path)

    def expanded_temporary_path(self, run_id) -> Optional[str]:
        if self.temporary_base_path is None:
            return None
        if self.append_run_id_to_base_path:
            return os.path.expanduser(os.path.join(self.temporary_base_path, run_id))
        return os.path.expanduser(self.temporary_base_path)

    def create(self, run_id) -> Checkpointer:
        keeps = [CheckpointInterval(**k) for k in self.keep or []]
        return Checkpointer(
            base_path=self.expanded_path(run_id),
            save_interval=self.save_interval,
            step_policies=keeps,
            temporary_base_path=self.expanded_temporary_path(run_id),
            delete_old_temp_checkpoints=self.delete_old_temp_checkpoints,
            keep_last_temporary_checkpoints=self.keep_last_temporary_checkpoints,
            debug=self.debug,
        )

    def __post_init__(self):
        # Workaround for Executor using placeholder types.
        if isinstance(self.base_path, str):
            self.base_path = os.path.expanduser(self.base_path)
        if isinstance(self.temporary_base_path, str):
            self.temporary_base_path = os.path.expanduser(self.temporary_base_path)
        if isinstance(self.debug, dict):
            self.debug = CheckpointDebugConfig(**self.debug)
        if self.keep_last_temporary_checkpoints < 0:
            raise ValueError("keep_last_temporary_checkpoints must be non-negative")

        # validate the checkpoint intervals.
        # we want to make sure that the intervals are monotonic. only the last one can be None
        prev_interval = None
        for interval in self.keep or []:
            if prev_interval is not None:
                assert prev_interval["until"] is not None, "Only the last checkpoint interval can be None"
                assert (
                    interval["until"] is None or interval["until"] > prev_interval["until"]
                ), "Checkpoint intervals must be monotonic"
            prev_interval = interval

        self.debug.validate()


def is_checkpoint_path(path: str) -> bool:
    """
    Check if a given path is a checkpoint path.
    """
    try:
        if not StoragePath(path).exists():
            return False
        # Sometimes we have incomplete checkpoints due to preemption or other issues.
        # try to find a metadata file in the path
        fs, plain_path = _get_fs_and_plain_path(path)
        metadata_path = os.path.join(plain_path, "metadata.json")
        if fs.exists(metadata_path):
            return True
        # glob
        # if we don't find a metadata file, we can check if the path has any subdirectories
        metadata_files = fs.glob(os.path.join(plain_path, "*", "metadata.json"))
        if len(metadata_files) > 0:
            return True
        else:
            logger.warning(
                f"While checkpoint path {path} exists, it does not contain a metadata.json file or subdirectories with"
                " metadata files. Most likely, this path has other data or incomplete checkpoints. Acting as if it is"
                " not a checkpoint path."
            )
            return False

    except Exception:  # noqa
        logger.exception(f"Error checking if {path} is a checkpoint path")
        raise
