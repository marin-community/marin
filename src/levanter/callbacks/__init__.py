# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
import time
from typing import Callable, Optional
from contextlib import contextmanager
import os

import jax
from tqdm_loggable.auto import tqdm

import levanter.tracker
from levanter.callbacks._core import Callback, CBInfo, JitCallback, LambdaCallback, StepInfo
from levanter.callbacks._metrics import (
    _tqdm_logging_one_time_setup,
    log_epoch_progress,
    log_performance_stats,
    log_step_info,
    logger,
    pbar_logger,
)
from levanter.data import DataLoader
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import barrier_sync
from levanter.utils.logging import save_xla_dumps_to_wandb


def eval_loss_loop(loss_fn, model, dataset, max_batches: Optional[int] = None, name: Optional[str] = None):
    total_loss = 0.0
    total_load_time = 0.0
    total_loss_time = 0.0
    n = 0

    if name is not None:
        desc = f"eval {name}"
    else:
        desc = "eval"

    _tqdm_logging_one_time_setup()
    pbar = tqdm(dataset, desc=desc, position=1, leave=False, total=max_batches)

    iter_ = iter(pbar)
    while True:
        time_in = time.time()
        batch = next(iter_, None)
        if batch is None:
            break
        load_time = time.time() - time_in
        total_load_time += load_time
        loss = loss_fn(model, batch)
        total_loss += loss.item()
        n += 1
        loss_time = time.time() - time_in - load_time
        total_loss_time += loss_time

        pbar.set_postfix(loss=total_loss / n)

        if max_batches is not None and n >= max_batches:
            break

    if n > 0:
        total_loss /= n

    return total_loss


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataset: DataLoader,
    max_batches: Optional[int] = None,
    name: Optional[str] = None,
):
    def compute_loss(info: StepInfo):
        loss = eval_loss_loop(loss_fn, info.eval_model, dataset, max_batches=max_batches, name=name)

        prefix = "eval"
        if name:
            prefix += "/" + name
        levanter.tracker.log({f"{prefix}/loss": loss}, step=info.step)

        if name:
            logger.info(f"{name} validation loss: {loss:.3f}")
        else:
            logger.info(f"validation loss: {loss:.3f}")

        return loss

    return compute_loss


def wandb_xla_logger(config: WandbConfig):
    import wandb

    last_mtime = wandb.run and wandb.run.start_time or time.time()

    def log_xla_to_wandb(step: StepInfo):
        nonlocal last_mtime
        save_xla_dumps_to_wandb(last_mtime)
        # update time to now
        last_mtime = time.time()

    if config.save_xla_dumps:
        return log_xla_to_wandb
    else:
        return lambda x: None


def profile(path: str, start_step: int, num_steps: int, create_perfetto_link: bool) -> Callable[[StepInfo], None]:
    def profiler_callback_fn(step: StepInfo):
        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
            logger.info(f"Starting profiler until step {start_step + num_steps}.")
            jax.profiler.start_trace(path, create_perfetto_link=_create_perfetto_link, create_perfetto_trace=True)
        elif step.step == start_step + num_steps - 1:
            if create_perfetto_link:
                logger.info(
                    f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}"
                )
            else:
                logger.info("Stopping profiler.")
            # so, annoyingly, gcloud ssh doesn't reliably flush stdout here, so we need to spin up
            # a thread to flush and print periodically until we make it past stop_trace
            # (note: stop_trace blocks if perfetto is enabled)
            event = threading.Event()
            if create_perfetto_link and jax.process_index() == 0:
                _flush_while_waiting(event)

            jax.profiler.stop_trace()

            if create_perfetto_link and jax.process_index() == 0:
                event.set()

            levanter.tracker.current_tracker().log_artifact(path, type="jax_profile")
            barrier_sync()

    return profiler_callback_fn


def _flush_while_waiting(event):
    def flush_stdout():
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)
        while not event.is_set():
            print("Waiting...", flush=True)
            print("\n", file=sys.stderr, flush=True)
            time.sleep(5)

    thread = threading.Thread(target=flush_stdout)
    thread.start()


@contextmanager
def profile_ctx(
    path: str,
    create_perfetto_link: bool = False,
    *,
    host_profile: bool = False,
    host_profile_basename: str = "host_profile",
    host_profile_topn: int = 0,
):
    """Context manager for JAX profiling traces.

    Starts a JAX profiler trace on enter and stops it on exit, mirroring the
    behavior of the callback returned by ``profile(...)``.

    Args:
        path: Filesystem path where the profile trace will be written.
        create_perfetto_link: If True, process 0 creates a Perfetto link and we
            print periodic messages while waiting for trace finalization.

    Notes:
        - Only process 0 creates the Perfetto link when ``create_perfetto_link`` is True.
        - After stopping the trace, logs the artifact to the current tracker as type
          "jax_profile" and performs a cross-process barrier.
    """
    _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
    logger.info("Starting profiler.")

    # Ensure destination exists
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

    jax.profiler.start_trace(path, create_perfetto_link=_create_perfetto_link, create_perfetto_trace=True)

    event = None
    pr = None
    stats_path = None
    txt_summary_path = None
    if host_profile:
        try:
            import cProfile  # type: ignore

            pr = cProfile.Profile()
            pr.enable()
            # Primary .pstats file and a human-readable txt summary
            stats_path = os.path.join(path, f"{host_profile_basename}.pstats")
            txt_summary_path = os.path.join(path, f"{host_profile_basename}.txt")
        except Exception as e:  # pragma: no cover - optional/diagnostic path
            logger.warning(f"Failed to start cProfile host profiler: {e}")
    try:
        yield
    finally:
        # Stop host profiler and write artifacts
        # Do this first because jax.profiler can be very slow to finish
        if pr is not None and stats_path is not None:
            try:
                pr.disable()
                pr.dump_stats(stats_path)
                if host_profile_topn and txt_summary_path is not None:
                    import pstats  # type: ignore

                    s = pstats.Stats(stats_path)
                    s.strip_dirs().sort_stats("cumtime")
                    with open(txt_summary_path, "w") as f:
                        s.stream = f  # type: ignore
                        s.print_stats(host_profile_topn)
            except Exception:  # pragma: no cover - optional/diagnostic path
                logger.warn("Failed to log host profile stats", exc_info=True)

        # Start periodic flushing before stop_trace since it may block when perfetto is enabled
        if create_perfetto_link and jax.process_index() == 0:
            event = threading.Event()
            _flush_while_waiting(event)

        if create_perfetto_link:
            logger.info(f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}")
        else:
            logger.info("Stopping profiler.")

        jax.profiler.stop_trace()

        if event is not None:
            event.set()

        levanter.tracker.current_tracker().log_artifact(path, type="jax_profile")
        # Log host stats if available
        if stats_path is not None and os.path.exists(stats_path):
            try:
                levanter.tracker.current_tracker().log_artifact(stats_path, type="host_profile")
            except Exception:
                logger.warn("Failed to log host profile stats", exc_info=True)
        if txt_summary_path is not None and os.path.exists(txt_summary_path):
            try:
                levanter.tracker.current_tracker().log_artifact(txt_summary_path, type="host_profile")
            except Exception:
                logger.warn("Failed to log host profile summary", exc_info=True)
        barrier_sync()


__all__ = [
    "eval_loss_loop",
    "compute_validation_loss",
    "wandb_xla_logger",
    "profile",
    "profile_ctx",
    "Callback",
    "CBInfo",
    "JitCallback",
    "LambdaCallback",
    "StepInfo",
    "log_epoch_progress",
    "log_performance_stats",
    "log_step_info",
    "pbar_logger",
]
