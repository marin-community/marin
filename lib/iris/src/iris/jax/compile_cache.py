# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX compilation cache and node-local GPU autotune cache setup."""

import enum
import logging
import os

from finestore.fileset import FileSetFetchLimits, FineStoreDirectory, fetch_file_set
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import LAUNCH_PROVENANCE_ENV, launch_provenance

from iris.cluster.client.job_info import get_job_info
from iris.cluster.runtime.env import SCRATCH_CACHE_PATH
from iris.env_resources import TaskResources
from iris.jax.multigpu import IRIS_MULTIGPU_PROCESS_COUNT_ENV, IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)

_COMPILATION_CACHE_SUBDIR = "compilation-cache"
_XLA_AUTOTUNE_CACHE_SUBDIR = "xla/per-fusion-autotune"
_XLA_AUTOTUNE_CACHE_DIR_FLAG = "--xla_gpu_per_fusion_autotune_cache_dir"
_XLA_AUTOTUNE_LEVEL_FLAG = "--xla_gpu_autotune_level"
# Object-store home for the per-build FineStore file set.
_XLA_AUTOTUNE_REMOTE_PREFIX = "xla-per-fusion-autotune"
_XLA_AUTOTUNE_CACHE_TTL_DAYS = 30
_XLA_AUTOTUNE_FETCH_LIMITS = FileSetFetchLimits(max_files=500_000, max_bytes=2 * 1024**3, timeout=300)


class _AutotuneCacheRole(enum.StrEnum):
    UPLOADER = "uploader"
    FETCHER = "fetcher"
    NONE = "none"


def _xla_flag_value(flags: str, name: str) -> str | None:
    """Return the last value, or ``None`` if absent; a bare flag has an empty value."""
    tokens = _normalize_xla_flags(flags).split()
    value = None
    for token in tokens:
        if token.startswith(f"{name}="):
            value = token.partition("=")[2]
        elif token == name:
            value = ""
    return value


def _normalize_xla_flags(flags: str) -> str:
    """Convert space-separated values to the equals form required by XLA_FLAGS."""
    tokens = flags.split()
    normalized = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.startswith("--")
            and "=" not in token
            and index + 1 < len(tokens)
            and not tokens[index + 1].startswith("--")
        ):
            normalized.append(f"{token}={tokens[index + 1]}")
            index += 2
        else:
            normalized.append(token)
            index += 1
    return " ".join(normalized)


def configure_jax_compilation_cache() -> None:
    """Place JAX's compilation cache on object storage and XLA's autotune cache on the node.

    The JAX cache defaults to a subdirectory of the active Marin prefix, unless
    ``JAX_COMPILATION_CACHE_DIR`` or ``jax.config`` already names one. It has to
    be somewhere every process can read: JAX writes it only from process 0, so a
    node-local copy would leave every other node cold.

    A remote JAX cache additionally disables JAX's XLA sub-cache derivation,
    which would otherwise hand XLA's C++ filesystem layer a URL it cannot open,
    and redirects XLA's per-fusion autotune cache to node-local disk instead.
    """
    if raw_xla_flags := os.environ.get("XLA_FLAGS"):
        os.environ["XLA_FLAGS"] = _normalize_xla_flags(raw_xla_flags)

    import jax  # noqa: PLC0415  # optional dep: jax (iris does not depend on jax)

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or jax.config.jax_compilation_cache_dir
    if not cache_dir:
        cache_dir = prefix_join(marin_prefix(), _COMPILATION_CACHE_SUBDIR)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info("JAX compilation cache: %s", cache_dir)

    if "://" not in cache_dir:
        return

    if "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES" not in os.environ:
        jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
    _enable_xla_autotune_cache()


def _enable_xla_autotune_cache() -> None:
    """Point XLA's per-fusion autotune cache at the node-local mount and mirror it remotely.

    Goes through ``XLA_FLAGS`` because JAX derives this path from the compilation
    cache dir, which is remote. The flag is on ``xla_flags_to_exclude_from_cache_key``,
    so it stays out of the compilation cache key. XLA opens the directory from C++
    through ``tsl::Env``, which cannot read an object store, so the live directory
    is always node-local and FineStore publishes its files as transaction batches.

    GPU only: a TPU or CPU jaxlib aborts on an unknown ``--xla_gpu`` flag.

    The mount arrives with the worker, and VM-cluster workers restart on their own
    daily schedule. For up to a day after a rollout a task can land on a worker
    whose mount is absent or unwritable; skip the cache there.
    """
    if TaskResources.from_environment().gpu_count == 0:
        return

    if not os.path.isdir(SCRATCH_CACHE_PATH):
        logger.info("XLA autotune cache disabled: %s is not mounted", SCRATCH_CACHE_PATH)
        return

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if _xla_flag_value(xla_flags, _XLA_AUTOTUNE_CACHE_DIR_FLAG) is not None:
        return

    # A multigpu task runs one Python process per GPU on the same node-local mount. XLA moves
    # temporary entries into the cache as it autotunes, so concurrent writers in one directory can
    # race and leave another process with NOT_FOUND for its temporary file. Isolate the writers;
    # repeating autotuning per process is cheaper than losing the distributed job.
    autotune_dir = f"{SCRATCH_CACHE_PATH}/{_XLA_AUTOTUNE_CACHE_SUBDIR}"
    if os.environ.get(LAUNCH_PROVENANCE_ENV) and (tree_hash := launch_provenance().tree_hash):
        autotune_dir = f"{autotune_dir}/{tree_hash}"
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        autotune_dir = f"{autotune_dir}/process-{process_index}"
    try:
        os.makedirs(autotune_dir, exist_ok=True)
    except OSError as exc:
        logger.info("XLA autotune cache disabled: cannot create %s: %s", autotune_dir, exc)
        return

    os.environ["XLA_FLAGS"] = f"{xla_flags} {_XLA_AUTOTUNE_CACHE_DIR_FLAG}={autotune_dir}".strip()
    logger.info("XLA per-fusion autotune cache: %s", autotune_dir)

    # One process per task populates its node-local mount before distributed init's
    # barrier; only global process 0 uploads additions shared by all equivalent ranks.
    # Off a real launch the published provenance is absent and the cache stays local.
    if os.environ.get(LAUNCH_PROVENANCE_ENV):
        role = _autotune_cache_role()
        autotune_level = _xla_flag_value(xla_flags, _XLA_AUTOTUNE_LEVEL_FLAG)
        # XLA looks up cached configs before choosing a fallback when tuning is disabled.
        if role is _AutotuneCacheRole.UPLOADER and autotune_level != "0":
            sync_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)
        elif role is not _AutotuneCacheRole.NONE:
            fetch_file_set_cache(_XLA_AUTOTUNE_REMOTE_PREFIX, autotune_dir)


def _file_set_cache_root(prefix: str) -> str | None:
    tree_hash = launch_provenance().tree_hash
    if not tree_hash:
        return None
    return prefix_join(marin_temp_bucket(_XLA_AUTOTUNE_CACHE_TTL_DAYS, prefix), tree_hash)


def sync_file_set_cache(prefix: str, local: str) -> FineStoreDirectory | None:
    """Start a file-set synchronizer when a build hash is available."""
    root = _file_set_cache_root(prefix)
    if root is None:
        return None
    return FineStoreDirectory(root, local, fetch_limits=_XLA_AUTOTUNE_FETCH_LIMITS)


def fetch_file_set_cache(prefix: str, local: str) -> None:
    """Fetch one build's committed file set without starting an uploader."""
    root = _file_set_cache_root(prefix)
    if root is not None:
        fetch_file_set(root, local, limits=_XLA_AUTOTUNE_FETCH_LIMITS)


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
