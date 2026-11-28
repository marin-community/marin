# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX compilation cache and node-local GPU autotune cache setup."""

import atexit
import enum
import logging
import os
import pathlib
import shutil
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version

from finestore.fileset import FineStoreDirectory, fetch_file_set
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import LAUNCH_PROVENANCE_ENV
from rigging.timing import Deadline, Duration, ExponentialBackoff

from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.runtime.env import SCRATCH_CACHE_PATH
from iris.env_resources import TaskResources, _read_iris_resource_proto
from iris.jax.autotune_snapshot import AUTOTUNE_CACHE_SUBDIR, autotune_job_dir, cleanup_shared_autotune_cache
from iris.jax.multigpu import IRIS_MULTIGPU_PROCESS_COUNT_ENV, IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)

_COMPILATION_CACHE_SUBDIR = "compilation-cache"
_XLA_AUTOTUNE_CACHE_DIR_FLAG = "--xla_gpu_per_fusion_autotune_cache_dir"
_XLA_AUTOTUNE_LEVEL_FLAG = "--xla_gpu_autotune_level"
_XLA_AUTOTUNE_MODE_FLAG = "--xla_gpu_experimental_autotune_cache_mode"
_XLA_AUTOTUNE_READ_MODE = "AUTOTUNE_CACHE_MODE_READ"
IRIS_AUTOTUNE_SNAPSHOT_ID = "IRIS_AUTOTUNE_SNAPSHOT_ID"
IRIS_AUTOTUNE_STORE_ROOT = "IRIS_AUTOTUNE_STORE_ROOT"
# Object-store home for the GPU/JAX/CUDA-compatible FineStore file set.
_XLA_AUTOTUNE_REMOTE_PREFIX = "xla-per-fusion-autotune"
_XLA_AUTOTUNE_CACHE_TTL_DAYS = 30


class _AutotuneCacheRole(enum.StrEnum):
    UPLOADER = "uploader"
    FETCHER = "fetcher"
    NONE = "none"


def configure_jax_compilation_cache() -> None:
    """Place JAX's compilation cache on object storage and XLA's autotune cache on the node.

    The JAX cache defaults to a subdirectory of the active Marin prefix, unless
    ``JAX_COMPILATION_CACHE_DIR`` or ``jax.config`` already names one. It has to
    be somewhere every process can read: JAX writes it only from process 0, so a
    node-local copy would leave every other node cold.

    A remote JAX cache additionally disables JAX's XLA sub-cache derivation,
    which would otherwise hand XLA's C++ filesystem layer a URL it cannot open.
    Multi-process GPU jobs always use a pinned, node-local autotune snapshot.
    """
    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or jax.config.jax_compilation_cache_dir
    if not cache_dir:
        cache_dir = prefix_join(marin_prefix(), _COMPILATION_CACHE_SUBDIR)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info("JAX compilation cache: %s", cache_dir)

    if "://" in cache_dir:
        if "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in os.environ:
            jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
        _enable_xla_autotune_cache()
    else:
        job_info = get_job_info()
        process_count = int(os.environ.get(IRIS_MULTIGPU_PROCESS_COUNT_ENV, str(job_info.num_tasks if job_info else 1)))
        if process_count > 1:
            _enable_xla_autotune_cache()


def _enable_xla_autotune_cache() -> None:
    """Point XLA's per-fusion autotune cache at the node-local mount.

    Goes through ``XLA_FLAGS`` because JAX derives this path from the compilation
    cache dir, which is remote. The flag is on ``xla_flags_to_exclude_from_cache_key``,
    so it stays out of the compilation cache key. XLA opens the directory from C++
    through ``tsl::Env``, which cannot read an object store, so the live directory
    is always node-local and FineStore publishes its files as transaction batches.

    GPU only: a TPU or CPU jaxlib aborts on an unknown ``--xla_gpu`` flag.

    The mount arrives with the worker, and VM-cluster workers restart on their own
    daily schedule. A missing mount aborts a multi-process GPU job because a
    rank may otherwise compile from different inputs.
    """
    if TaskResources.from_environment().gpu_count == 0:
        return

    if not os.path.isdir(SCRATCH_CACHE_PATH):
        job_info = get_job_info()
        process_count = int(os.environ.get(IRIS_MULTIGPU_PROCESS_COUNT_ENV, str(job_info.num_tasks if job_info else 1)))
        if process_count > 1:
            raise FileNotFoundError(f"multi-process GPU autotune snapshot requires {SCRATCH_CACHE_PATH}")
        logger.info("XLA autotune cache disabled: %s is not mounted", SCRATCH_CACHE_PATH)
        return

    xla_flags = os.environ.get("XLA_FLAGS", "")
    job_info = get_job_info()
    process_count = int(os.environ.get(IRIS_MULTIGPU_PROCESS_COUNT_ENV, str(job_info.num_tasks if job_info else 1)))
    if process_count > 1:
        if job_info is None:
            raise ValueError("multi-process GPU autotune cache requires Iris job information")
        _enable_shared_autotune_cache(xla_flags, job_info)
        return
    if any(flag.partition("=")[0] == _XLA_AUTOTUNE_CACHE_DIR_FLAG for flag in xla_flags.split()):
        return

    autotune_dir = f"{SCRATCH_CACHE_PATH}/{AUTOTUNE_CACHE_SUBDIR}"
    if job_info is not None:
        autotune_dir = str(autotune_job_dir(job_info, SCRATCH_CACHE_PATH) / "working")
    try:
        os.makedirs(autotune_dir, exist_ok=job_info is None)
    except OSError as exc:
        logger.info("XLA autotune cache disabled: cannot create %s: %s", autotune_dir, exc)
        return
    if job_info is not None:
        atexit.register(cleanup_shared_autotune_cache, job_info, SCRATCH_CACHE_PATH)

    os.environ["XLA_FLAGS"] = f"{xla_flags} {_XLA_AUTOTUNE_CACHE_DIR_FLAG}={autotune_dir}".strip()
    logger.info("XLA per-fusion autotune cache: %s", autotune_dir)

    # One process per task populates its node-local mount before distributed init's
    # barrier; only global process 0 uploads additions shared by all equivalent ranks.
    # Off a real launch the published provenance is absent and the cache stays local.
    if os.environ.get(LAUNCH_PROVENANCE_ENV):
        role = _autotune_cache_role()
        if role is _AutotuneCacheRole.UPLOADER:
            sync_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)
        elif role is _AutotuneCacheRole.FETCHER:
            fetch_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)


def _enable_shared_autotune_cache(xla_flags: str, job_info: JobInfo) -> None:
    """Stage one pinned, read-only autotune input set per node and task attempt."""
    flags: dict[str, str] = {}
    for flag in xla_flags.split():
        name, _, value = flag.partition("=")
        if name in {_XLA_AUTOTUNE_CACHE_DIR_FLAG, _XLA_AUTOTUNE_LEVEL_FLAG, _XLA_AUTOTUNE_MODE_FLAG}:
            if name in flags:
                raise ValueError(f"duplicate XLA autotune flag: {name}")
            flags[name] = value
    if _XLA_AUTOTUNE_CACHE_DIR_FLAG in flags:
        raise ValueError("multi-process GPU jobs cannot set a per-rank XLA autotune cache directory")
    if _XLA_AUTOTUNE_LEVEL_FLAG in flags and flags[_XLA_AUTOTUNE_LEVEL_FLAG] != "0":
        raise ValueError("multi-process GPU jobs require --xla_gpu_autotune_level=0")
    if _XLA_AUTOTUNE_MODE_FLAG in flags and flags[_XLA_AUTOTUNE_MODE_FLAG] != _XLA_AUTOTUNE_READ_MODE:
        raise ValueError("multi-process GPU jobs require AUTOTUNE_CACHE_MODE_READ")
    snapshot_id = os.environ.get(IRIS_AUTOTUNE_SNAPSHOT_ID)
    if snapshot_id is None:
        raise ValueError(f"multi-process GPU job is missing launcher-pinned {IRIS_AUTOTUNE_SNAPSHOT_ID}")
    root = os.environ.get(IRIS_AUTOTUNE_STORE_ROOT)
    if snapshot_id != "empty" and not root:
        raise ValueError(f"{IRIS_AUTOTUNE_STORE_ROOT} is required for snapshot {snapshot_id}")

    job_dir = autotune_job_dir(job_info, SCRATCH_CACHE_PATH)
    snapshot_dir = job_dir / "snapshot"
    ready = job_dir / ".ready"
    failed = job_dir / ".failed"
    process_index = int(os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV, str(job_info.task_index)))
    process_count = int(os.environ.get(IRIS_MULTIGPU_PROCESS_COUNT_ENV, str(job_info.num_tasks)))
    if process_count % job_info.num_tasks:
        raise ValueError(f"GPU process count {process_count} is not divisible by task count {job_info.num_tasks}")
    processes_per_task = process_count // job_info.num_tasks
    local_rank = process_index % processes_per_task
    job_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0 and not ready.exists():
        staged = pathlib.Path(tempfile.mkdtemp(prefix="staging-", dir=job_dir))
        try:
            if snapshot_id == "empty":
                files: set[str] = set()
            else:
                assert root is not None
                files = fetch_file_set(root, str(staged), snapshot_id=snapshot_id)
            for path in staged.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            staged.chmod(0o555)
            if snapshot_dir.exists():
                raise FileExistsError(f"unpublished XLA autotune snapshot directory exists: {snapshot_dir}")
            staged.rename(snapshot_dir)
            with atomic_rename(str(ready)) as staged_ready:
                pathlib.Path(staged_ready).write_text(f"{snapshot_id}\n{len(files)}\n")
        except Exception as exc:
            failed.write_text(f"{type(exc).__name__}: {exc}\n")
            raise
        finally:
            if staged.exists():
                shutil.rmtree(staged)
    deadline = Deadline.from_now(Duration.from_seconds(1800))
    backoff = ExponentialBackoff(initial=0.1, maximum=2)
    while not ready.exists():
        if failed.exists():
            raise RuntimeError(f"XLA autotune snapshot fetch failed: {failed.read_text().strip()}")
        if deadline.expired():
            raise TimeoutError(f"timed out waiting for XLA autotune snapshot in {job_dir}")
        time.sleep(backoff.next_interval())
    pinned_id, count = ready.read_text().splitlines()
    if pinned_id != snapshot_id:
        raise ValueError(f"XLA autotune snapshot mismatch in {job_dir}: {pinned_id} != {snapshot_id}")
    additions = [f"{_XLA_AUTOTUNE_CACHE_DIR_FLAG}={snapshot_dir}"]
    if _XLA_AUTOTUNE_LEVEL_FLAG not in flags:
        additions.append(f"{_XLA_AUTOTUNE_LEVEL_FLAG}=0")
    if _XLA_AUTOTUNE_MODE_FLAG not in flags:
        additions.append(f"{_XLA_AUTOTUNE_MODE_FLAG}={_XLA_AUTOTUNE_READ_MODE}")
    os.environ["XLA_FLAGS"] = " ".join([xla_flags, *additions]).strip()
    logger.info("XLA autotune snapshot_id=%s file_count=%s path=%s", snapshot_id, count, snapshot_dir)
    if processes_per_task == 1:
        atexit.register(cleanup_shared_autotune_cache, job_info, SCRATCH_CACHE_PATH)


def _file_set_cache_root(prefix: str) -> str | None:
    resources = _read_iris_resource_proto()
    if resources is None or not resources.HasField("device") or not resources.device.HasField("gpu"):
        return None
    try:
        jaxlib_version = version("jaxlib")
        try:
            cuda_version = version("nvidia-cuda-runtime")
        except PackageNotFoundError:
            cuda_version = version("nvidia-cuda-runtime-cu12")
    except PackageNotFoundError:
        return None
    key = f"{resources.device.gpu.variant.lower()}/{jaxlib_version}/{cuda_version}"
    return prefix_join(marin_temp_bucket(_XLA_AUTOTUNE_CACHE_TTL_DAYS, prefix), key)


def sync_file_set_cache(prefix: str, local: str) -> FineStoreDirectory | None:
    """Start a file-set synchronizer, or return ``None`` when remote caching is unavailable."""
    root = _file_set_cache_root(prefix)
    if root is None:
        return None
    try:
        return FineStoreDirectory(root, local)
    except OSError as exc:
        logger.warning("XLA autotune cache is unavailable; continuing with the node-local cache: %s", exc)
        return None


def fetch_file_set_cache(prefix: str, local: str) -> None:
    """Fetch one compatibility-keyed file set without starting an uploader."""
    root = _file_set_cache_root(prefix)
    if root is not None:
        try:
            fetch_file_set(root, local)
        except OSError as exc:
            logger.warning("XLA autotune cache fetch failed; starting cold: %s", exc)


def _autotune_cache_role() -> _AutotuneCacheRole:
    """Select one cache fetcher per task and one uploader for the whole job."""
    job_info = get_job_info()
    raw_process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if raw_process_index is None:
        if job_info is None or job_info.task_index == 0:
            return _AutotuneCacheRole.UPLOADER
        return _AutotuneCacheRole.FETCHER

    process_index = int(raw_process_index)
    if process_index == 0:
        return _AutotuneCacheRole.UPLOADER

    process_count = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV])
    num_tasks = job_info.num_tasks if job_info is not None else 1
    processes_per_task = process_count // num_tasks
    if process_index % processes_per_task == 0:
        return _AutotuneCacheRole.FETCHER
    return _AutotuneCacheRole.NONE
