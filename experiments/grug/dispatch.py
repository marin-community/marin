# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.v2.client import current_client
from fray.v2.types import Entrypoint, JobRequest, create_environment
from fray.v2.types import GpuConfig, TpuConfig

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray/Iris-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def _default_environment_extras(resources: ResourceConfig) -> list[str]:
    if isinstance(resources.device, TpuConfig):
        return ["tpu"]
    if isinstance(resources.device, GpuConfig):
        return ["gpu"]
    return []


def _start_xla_dump_uploader(local_dir: str, gcs_dst_template: str) -> None:
    """Periodically copy local XLA dumps to GCS so they survive a silent kill.

    Diagnostic for issue #5319. Workers can SIGABRT below Python without flushing
    output, so atexit isn't reliable; a background thread uploads every 30s.
    `gcs_dst_template` may contain `{worker}` which is replaced with the JAX
    process index (defaults to PID before jax.distributed.initialize runs).
    """
    import os
    import sys
    import threading
    import time

    def _emit(msg: str) -> None:
        sys.stderr.write(f"[xla-dump-uploader] {msg}\n")
        sys.stderr.flush()

    def _resolve_worker_id() -> str:
        try:
            import jax

            if jax.distributed.is_initialized():
                return f"worker{jax.process_index()}"
        except Exception:
            pass
        return f"pid{os.getpid()}"

    def _upload_with_gcsfs(src_dir: str, dst_url: str) -> tuple[int, int]:
        """Upload via gcsfs. Returns (file_count, byte_count)."""
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        # Strip gs:// prefix for fs.put behavior
        dst = dst_url.removeprefix("gs://").rstrip("/")
        files = 0
        total_bytes = 0
        for root, _dirs, fnames in os.walk(src_dir):
            for fname in fnames:
                local = os.path.join(root, fname)
                rel = os.path.relpath(local, src_dir)
                remote = f"{dst}/{rel}"
                try:
                    fs.put(local, remote)
                    files += 1
                    total_bytes += os.path.getsize(local)
                except Exception as exc:
                    _emit(f"failed to upload {local}: {exc}")
        return files, total_bytes

    def _loop() -> None:
        _emit(f"thread started, local_dir={local_dir} gcs_dst_template={gcs_dst_template}")
        last_count = -1
        while True:
            time.sleep(30)
            if not os.path.isdir(local_dir):
                _emit(f"local_dir={local_dir} does not exist yet")
                continue
            try:
                file_count = sum(len(fs) for _, _, fs in os.walk(local_dir))
            except Exception:
                file_count = -1
            if file_count == 0:
                _emit(f"local_dir={local_dir} is empty")
                continue
            dst = gcs_dst_template.format(worker=_resolve_worker_id())
            _emit(f"uploading {file_count} files from {local_dir} to {dst} (was {last_count})")
            try:
                uploaded, total_bytes = _upload_with_gcsfs(local_dir, dst)
                _emit(f"uploaded {uploaded} files, {total_bytes} bytes to {dst}")
                last_count = file_count
            except Exception as exc:
                _emit(f"upload failed: {exc}")

    t = threading.Thread(target=_loop, name="xla-dump-uploader", daemon=True)
    t.start()


def _with_jax_distributed_init(fn: Callable[[ConfigT], None], config: ConfigT) -> None:
    """Wrapper that initializes JAX distributed before running the entrypoint.

    On multi-host TPU, Fray's fn_thunk subprocess doesn't auto-initialize
    JAX distributed. Calling jax.distributed.initialize() without args
    lets JAX auto-detect the TPU topology. On single-host this is a no-op.

    Also enables faulthandler so SIGABRT/SIGSEGV from libtpu (which can hard-abort
    the process when the TPU runtime detects launch-group/scheckne mismatches)
    print a Python traceback to stderr instead of dying silently, and starts a
    periodic XLA-dump uploader if XLA_DUMP_GCS_DST is set. Diagnostic for issue
    #5319.
    """
    import faulthandler
    import os
    import signal
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
    for sig in (signal.SIGABRT, signal.SIGSEGV, signal.SIGFPE, signal.SIGBUS, signal.SIGILL):
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=True)
        except Exception:
            pass

    xla_dump_gcs_dst = os.environ.get("XLA_DUMP_GCS_DST")
    if xla_dump_gcs_dst:
        _start_xla_dump_uploader("/tmp/xla_dump", xla_dump_gcs_dst)

    # Issue #5319 test: pre-init wandb BEFORE jax.distributed.initialize so the
    # asymmetric resume-fetch on worker 0 finishes before TPU/JAX state exists
    # to diverge.
    _preinit_wandb_if_configured(config)

    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize()
    fn(config)


def _preinit_wandb_if_configured(config) -> None:
    """Run wandb.init() before jax.distributed comes up. Issue #5319 test."""
    import os

    tracker = getattr(config, "tracker", None)
    run_id = getattr(config, "run_id", None)
    if tracker is None or run_id is None:
        return
    from levanter.tracker.wandb import WandbConfig

    if not isinstance(tracker, WandbConfig):
        return

    # Apply the same id mangling that _resolve_tracker does at trainer config
    # time, so the pre-init id matches the id the trainer will look for.
    from experiments.grug.moe.launch import _resolve_tracker

    resolved = _resolve_tracker(tracker, run_id)
    if not isinstance(resolved, WandbConfig):
        return

    rank = int(os.environ.get("JAX_PROCESS_ID", "0"))
    mode = resolved.mode if rank == 0 else "offline"

    import wandb

    logger.info(
        "issue #5319 preinit-wandb: starting wandb.init id=%s mode=%s resume=%s",
        resolved.id,
        mode,
        resolved.resume,
    )
    wandb.init(
        entity=resolved.entity,
        project=resolved.project,
        name=resolved.name,
        tags=list(resolved.tags or []),
        id=resolved.id,
        group=resolved.group,
        resume=resolved.resume,
        mode=mode,
        allow_val_change=True,
    )
    logger.info("issue #5319 preinit-wandb: wandb.init complete")
    os.environ["LEVANTER_WANDB_PREINITIALIZED"] = "1"


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    extras = _default_environment_extras(resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(_with_jax_distributed_init, args=[local_entrypoint, config]),
        resources=resources,
        environment=create_environment(extras=extras),
        max_retries_failure=max_retries_failure,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
