"""RL job data-staging + topology helpers for the Iris launcher.

- ``resolve_rl_train_data``: extract HF task datasets to local task directories.
- ``compute_num_inference_engines``: placement math.
- ``check_rl_environment``: locate an optional standalone RL venv.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from cloud.iris.hf_datasets import resolve_hf_dataset_selector
from cloud.iris.tasks_parquet import from_parquet
from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.resource_locator import parse_hf_dataset_selector
from marinskyrl.task_sources import DirectoryDataSource, TaskTroveParquetSource, data_source, data_source_dict


@dataclass(frozen=True)
class ResolvedRLData:
    """Runtime data entries paired with their immutable or local source references."""

    paths: tuple[str | dict[str, Any], ...]
    sources: tuple[str | dict[str, Any], ...]


def _stage_remote_file(uri: str, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        filesystem, source_path = filesystem_and_path(uri)
        with filesystem.open(source_path, "rb") as src, open(staging, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def _stage_remote_task_parquet(
    uri: str,
    tasks_base: Path,
    *,
    on_exist: str,
    verbose: bool,
) -> Path:
    if not uri.split("?", 1)[0].endswith(".parquet"):
        raise ValueError(f"Remote task data must name a parquet file: {uri!r}")
    identity = hashlib.sha256(uri.encode()).hexdigest()[:20]
    local_parquet = tasks_base / ".remote" / identity / "tasks.parquet"
    output_dir = tasks_base / f"remote-{identity}"
    _stage_remote_file(uri, local_parquet, overwrite=on_exist == "overwrite")
    if verbose:
        print(f"[rl_data] Staged remote task parquet: {uri} -> {local_parquet}")
    from_parquet(str(local_parquet), str(output_dir), on_exist=on_exist)
    _fix_task_permissions(output_dir, verbose=verbose)
    return output_dir


def resolve_rl_train_data(
    train_data: List[str | dict[str, Any]],
    scratch_dir: Optional[str] = None,
    on_exist: str = "skip",
    verbose: bool = True,
    kind: str = "tasks",
) -> List[str | dict[str, Any]]:
    """Resolve data into local paths or structured packed-task sources."""
    return list(
        resolve_rl_train_data_with_sources(
            train_data,
            scratch_dir=scratch_dir,
            on_exist=on_exist,
            verbose=verbose,
            kind=kind,
        ).paths
    )


def resolve_rl_train_data_with_sources(
    train_data: List[str | dict[str, Any]],
    scratch_dir: Optional[str] = None,
    on_exist: str = "skip",
    verbose: bool = True,
    kind: str = "tasks",
) -> ResolvedRLData:
    """Resolve train_data paths for the configured data kind.

    ``sources`` preserves local inputs and replaces Hub inputs with canonical
    selectors pinned to their resolved commit.

    ``kind="tasks"`` (default, terminal_bench agentic RL): SkyRL's
    TerminalBenchTaskDataset expects local directory paths where each subdirectory is a
    task containing an ``instruction.md`` file. HuggingFace dataset identifiers are
    extracted to ``$SCRATCH/tasks/<repo-name>/`` via
    ``cloud.iris.extract_tasks_from_parquet``, permissions fixed, and local paths returned.
    Object-store parquet URIs resolve to local directories containing the extracted tasks.

    ``kind="parquet"`` (single-turn RLVR, e.g. main_base + aime): the entries are
    SkyRL-shaped parquet paths that ``PromptDataset`` loads via ``datasets.load_dataset``.
    Never task-extracted. Local paths and HF ids pass through UNCHANGED; an object-store URI
    (``s3://`` / ``gs://`` / http(s)) is STAGED to node-local disk and the local path returned —
    ``datasets.load_dataset`` refuses a remote URI under ``HF_HUB_OFFLINE=1`` (the mode set so
    the prestaged, template-rewritten model reads its warm cache), raising
    ``OfflineModeIsEnabled`` before training. Staging pulls it via fsspec (the pod's object-store
    creds + endpoint) so the offline flag only gates weights, not the dataset read.
    """
    if not train_data:
        return ResolvedRLData((), ())

    if kind == "parquet":
        resolved: List[str] = []
        sources: List[str | dict[str, Any]] = []
        stage_root = Path(scratch_dir) / "rl_parquet" if scratch_dir else Path("/tmp/skyrl_rl_parquet")
        for entry in train_data:
            if isinstance(entry, Mapping):
                source = data_source(entry)
                if not isinstance(source, DirectoryDataSource):
                    raise ValueError(f"Structured parquet data must use a directory source, found {source.kind!r}")
                resolved.append(source.resolved_path())
                sources.append(data_source_dict(source))
                continue
            # A local path or a bare HF dataset id is read directly by PromptDataset.
            if "://" not in entry or entry.startswith("file://"):
                resolved.append(entry)
                sources.append(entry)
                continue
            # Object-store URI: pull to node-local disk (offline mode blocks the remote read).
            local = stage_root / Path(entry.split("?", 1)[0]).name
            _stage_remote_file(entry, local, overwrite=on_exist == "overwrite")
            if verbose:
                print(f"[rl_data] data.kind=parquet: staged {entry} -> {local}")
            resolved.append(str(local))
            sources.append(entry)
        return ResolvedRLData(tuple(resolved), tuple(sources))

    # Determine the scratch directory for extracted tasks. It MUST be a shared
    # filesystem visible to all compute nodes; /tmp is node-local (last resort).
    if scratch_dir is None:
        for env_var in ["SCRATCH", "DCFT", "DCFT_PRIVATE", "HOME"]:
            if os.environ.get(env_var):
                scratch_dir = os.environ[env_var]
                break
        else:
            scratch_dir = "/tmp"
            print(
                "[rl_data] WARNING: Using /tmp for task extraction. "
                "This is local to each node and may fail on multi-node jobs. "
                "Set $SCRATCH, $DCFT, or $DCFT_PRIVATE to a shared filesystem path."
            )
    tasks_base = Path(scratch_dir) / "tasks"

    resolved_paths = []
    sources = []

    for data_path in train_data:
        if isinstance(data_path, Mapping):
            source = data_source(data_path)
            packed_source = data_source_dict(source)
            if isinstance(source, DirectoryDataSource):
                resolved_paths.append(source.resolved_path())
            elif isinstance(source, TaskTroveParquetSource):
                resolved_paths.append(packed_source)
            sources.append(packed_source)
            continue
        if "://" in data_path and not data_path.startswith("file://"):
            output_dir = _stage_remote_task_parquet(data_path, tasks_base, on_exist=on_exist, verbose=verbose)
            resolved_paths.append(str(output_dir))
            sources.append(data_path)
            continue
        if parse_hf_dataset_selector(data_path) is not None:
            selector = resolve_hf_dataset_selector(data_path)
            canonical_source = selector.canonical()
            output_dir = tasks_base / selector.cache_name()

            if verbose:
                print(f"[rl_data] Extracting HF dataset: {canonical_source}")
                print(f"[rl_data] Output directory: {output_dir}")

            if on_exist == "skip" and output_dir.exists() and any(output_dir.iterdir()):
                if verbose:
                    print(f"[rl_data] Tasks already extracted, skipping: {output_dir}")
                resolved_paths.append(str(output_dir))
                sources.append(canonical_source)
                continue

            cmd = [
                sys.executable,
                "-m",
                "cloud.iris.extract_tasks_from_parquet",
                "--parquet",
                canonical_source,
                "--output_dir",
                str(output_dir),
                "--on_exist",
                on_exist,
            ]

            if verbose:
                print(f"[rl_data] Running: {' '.join(cmd)}")

            # A stalled HF download inside the extractor (mid-download socket hang)
            # would block subprocess.run forever, so a per-attempt timeout converts
            # a stall into a retry; HF resumes the partial `.incomplete` shard on the
            # next attempt, so a killed-mid-download attempt loses nothing. 600s
            # covers a clean extract yet fits several retries inside a gang-join budget.
            extract_attempt_timeout = 600
            last_err = ""
            for attempt in range(1, 7):
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=extract_attempt_timeout,
                    )
                    if verbose and result.stdout:
                        print(result.stdout)
                    break
                except subprocess.TimeoutExpired:
                    last_err = (
                        f"extract stalled > {extract_attempt_timeout}s "
                        "(mid-download socket hang); killed, retrying (HF resumes the partial shard)"
                    )
                    print(f"[rl_data] extract attempt {attempt}/6 TIMED OUT for {data_path}: {last_err}")
                    time.sleep(min(30, 2**attempt))
                except subprocess.CalledProcessError as e:
                    last_err = f"stdout: {e.stdout}\n  stderr: {e.stderr}"
                    print(f"[rl_data] extract attempt {attempt}/6 failed for {data_path}:")
                    print(f"  {last_err}")
                    time.sleep(min(30, 2**attempt))
            else:
                raise RuntimeError(f"Failed to extract HF dataset after 6 attempts: {data_path}: {last_err}")

            _fix_task_permissions(output_dir, verbose=verbose)
            resolved_paths.append(str(output_dir))
            sources.append(canonical_source)
        else:
            local_path = Path(data_path)
            if local_path.exists():
                _fix_task_permissions(local_path, verbose=verbose)
            resolved_paths.append(data_path)
            sources.append(data_path)

    return ResolvedRLData(tuple(resolved_paths), tuple(sources))


def _fix_task_permissions(task_dir: Path, verbose: bool = True) -> None:
    """Run ``chmod -R a+rX`` on a task tree so files are readable + dirs traversable.

    Idempotency guard: a single cheap ``stat`` of the top-level dir short-circuits
    when its perms already satisfy ``a+rX``, avoiding a ~100K-write recursive chmod
    over an already-correct tree (which has wedged the launcher under GPFS metadata
    contention). Any dir that genuinely needs the fix still gets the full recursive
    chmod. Never uses find/du.
    """
    if not task_dir.exists():
        return

    rx_bits = stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    try:
        mode = task_dir.stat().st_mode
        if (mode & rx_bits) == rx_bits:
            if verbose:
                print(f"[rl_data] Permissions already a+rX on top-level dir, skipping recursive chmod: {task_dir}")
            return
    except OSError as e:
        if verbose:
            print(f"[rl_data] stat probe failed on {task_dir} ({e}); running recursive chmod to be safe.")

    if verbose:
        print(f"[rl_data] Fixing permissions on: {task_dir}")

    try:
        subprocess.run(
            ["chmod", "-R", "a+rX", str(task_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # Don't fail the whole job for permission issues.
        print(f"[rl_data] Warning: chmod failed on {task_dir}: {e.stderr}")


def compute_num_inference_engines(
    num_nodes: int,
    gpus_per_node: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Compute the number of vLLM inference engines: total GPUs // tensor_parallel_size."""
    total_gpus = num_nodes * gpus_per_node
    return total_gpus // tensor_parallel_size


def check_rl_environment() -> Optional[Path]:
    """Locate an optional standalone RL venv, or None.

    Checks ``$DCFT_RL_ENV``, ``$DCFT/envs/rl``, then ``<repo>/envs/rl``. Returns
    None when no such venv exists (the caller then uses ``sys.executable``, which
    on Iris is already the frozen task venv's Python).
    """
    candidates = []
    if os.environ.get("DCFT_RL_ENV"):
        candidates.append(Path(os.environ["DCFT_RL_ENV"]))
    if os.environ.get("DCFT"):
        candidates.append(Path(os.environ["DCFT"]) / "envs" / "rl")
    candidates.append(Path(__file__).resolve().parents[2] / "envs" / "rl")

    for candidate in candidates:
        if candidate.exists() and (candidate / "bin" / "activate").exists():
            return candidate

    return None
