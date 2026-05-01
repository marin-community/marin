#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
A script to download a HuggingFace dataset and upload it to a specified fsspec path,
using HfFileSystem for direct streaming of data transfer.
"""

import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field

import huggingface_hub
from fray import ResourceConfig
from huggingface_hub.errors import HfHubHTTPError
from packaging.version import Version
from rigging.filesystem import open_url, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

HF_PROTOCOL_PREFIX = "hf://"
HF_BUCKET_PATH_PREFIX = "buckets/"

# HF returns 401 when no credentials are sent and 403 when the caller's token
# lacks access (e.g. gated dataset, accept-license required). Neither is fixed
# by retrying — fail fast so the worker surfaces an actionable error instead of
# stalling for hours behind exponential backoff.
_HF_AUTH_ERROR_STATUSES = frozenset({401, 403})


def _hf_auth_error(exc: BaseException, file_path: str) -> str | None:
    """Return an actionable error message if `exc` is an unrecoverable HF auth failure, else None."""
    if not isinstance(exc, HfHubHTTPError) or exc.response is None:
        return None
    status_code = exc.response.status_code
    if status_code not in _HF_AUTH_ERROR_STATUSES:
        return None
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        hint = (
            "HF_TOKEN is set but lacks access — confirm the token's account has accepted "
            "the dataset license and has read access."
        )
    else:
        hint = (
            "HF_TOKEN is not set in the worker environment. `huggingface-cli login` only "
            "writes ~/.cache/huggingface/token, which iris does not forward to workers; "
            "export HF_TOKEN before submitting the job."
        )
    return f"HuggingFace returned HTTP {status_code} for {file_path} (gated/auth-required). {hint}"


@dataclass(frozen=True)
class DownloadConfig:
    # fmt: off

    # HuggingFace Dataset Parameters
    hf_dataset_id: str                                      # HF Dataset to Download (as `$ORG/$DATASET` on HF Hub)

    revision: str  # (Short) Commit Hash (from HF Dataset Repo; 7 characters)
    hf_urls_glob: list[str] = field(default_factory=list)
    # List of Glob Patterns to Match Files in HF Dataset, If empty we get all the files in a hf repo

    gcs_output_path: str = THIS_OUTPUT_PATH
    """
    Path to store raw data in persistent storage (e.g. gs://$BUCKET/...).
    This works with any fsspec-compatible path, but for backwards compatibility, we call it gcs_output_path.
    """

    append_sha_to_path: bool = False
    """If true, write outputs under ``gcs_output_path/<revision>`` instead of directly under ``gcs_output_path``."""

    # Job Control Parameters, used only for non-gated dataset transfers done via STS
    wait_for_completion: bool = True                        # if True, will block until job completes

    # fmt: on
    hf_repo_type_prefix: str = (
        "datasets"  # The repo_type_prefix is datasets/ for datasets,
        # spaces/ for spaces, and models do not need a prefix in the URL.
    )

    zephyr_max_parallelism: int = 8
    """Maximum parallelism of the Zephyr download job"""

    worker_resources: ResourceConfig | None = None
    """Optional Zephyr worker resource pinning for the download workers."""

    read_timeout_seconds: float = 120.0
    """Socket read timeout while streaming each HF file. Timeout failures trigger retries."""

    progress_log_interval_seconds: float = 60.0
    """Log a heartbeat for each in-flight shard every N seconds while bytes are flowing."""

    read_chunk_size_mib: int = 8
    """Chunk size for each streaming read from HF."""

    source_url_override: str | None = None
    """Optional fsspec URL to read from instead of HuggingFace. Bypasses HF-specific
    listing and revision handling; mainly intended for hermetic tests."""


def _strip_hf_protocol(path: str) -> str:
    return path.removeprefix(HF_PROTOCOL_PREFIX).lstrip("/")


def _resolve_hf_source_path(cfg: DownloadConfig) -> str:
    source_path = (
        os.path.join(cfg.hf_repo_type_prefix, cfg.hf_dataset_id) if cfg.hf_repo_type_prefix else cfg.hf_dataset_id
    )
    return _strip_hf_protocol(source_path)


def _assert_bucket_support_available(source_path: str) -> None:
    if not source_path.startswith(HF_BUCKET_PATH_PREFIX):
        return

    if Version(huggingface_hub.__version__) < Version("1.6.0"):
        raise RuntimeError(
            f"Bucket paths require huggingface_hub>=1.6.0, found {huggingface_hub.__version__}. "
            "Upgrade the runtime environment to a buckets-capable huggingface_hub version."
        )


def _relative_path_in_source(file_path: str, source_path: str) -> str:
    normalized_file = _strip_hf_protocol(file_path)
    normalized_source = _strip_hf_protocol(source_path).rstrip("/")

    source_prefix = f"{normalized_source}/"
    if normalized_file.startswith(source_prefix):
        return normalized_file.removeprefix(source_prefix)

    source_parts = [segment for segment in normalized_source.split("/") if segment]
    file_parts = [segment for segment in normalized_file.split("/") if segment]

    if len(file_parts) >= len(source_parts):
        matches_source = True
        for source_segment, file_segment in zip(source_parts, file_parts, strict=False):
            if source_segment == file_segment:
                continue
            if file_segment.split("@", 1)[0] == source_segment:
                continue
            matches_source = False
            break

        if matches_source:
            return "/".join(file_parts[len(source_parts) :])

    # Backwards-compatible fallback for historical dataset path layout.
    return normalized_file.split("/", 3)[-1]


def ensure_fsspec_path_writable(output_path: str) -> None:
    """Check if the fsspec path is writable by trying to create and delete a temporary file."""
    fs, _ = url_to_fs(output_path)
    try:
        fs.mkdirs(output_path, exist_ok=True)
        test_path = os.path.join(output_path, "test_write_access")
        with fs.open(test_path, "w") as f:
            f.write("test")
        fs.rm(test_path)
    except Exception as e:
        raise ValueError(f"No write access to fsspec path: {output_path} ({e})") from e


def stream_file_to_fsspec(
    gcs_output_path: str,
    file_path: str,
    fsspec_file_path: str,
    expected_size: int | None = None,
    read_timeout_seconds: float = 120.0,
    progress_log_interval_seconds: float = 60.0,
    read_chunk_size_mib: int = 8,
):
    """Stream a file from HfFileSystem to another fsspec path using atomic write.

    Uses atomic_rename to write to a temp file first, then rename on success.
    This enables recovery across individual files if the job is interrupted.

    Args:
        gcs_output_path: Base output path for the download.
        file_path: Source file path on HuggingFace.
        fsspec_file_path: Target file path on the destination filesystem.
        expected_size: Expected file size in bytes for validation. If provided,
            the download will fail if the downloaded size doesn't match.
    """
    target_fs, _ = url_to_fs(gcs_output_path)
    chunk_size = max(1, int(read_chunk_size_mib)) * 1024 * 1024
    max_retries = 20
    # 15 minutes max sleep
    max_sleep = 15 * 60
    # Minimum base wait time to avoid too-fast retries
    min_base_wait = 5

    # Retry when there is an error, such as hf rate limit
    last_exception = None
    for attempt in range(max_retries):
        try:
            target_fs.mkdirs(os.path.dirname(fsspec_file_path), exist_ok=True)
            bytes_written = 0
            with atomic_rename(fsspec_file_path) as temp_path:
                previous_socket_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(read_timeout_seconds)
                try:
                    with (
                        open_url(file_path, "rb", block_size=chunk_size) as src_file,
                        open_url(temp_path, "wb") as dest_file,
                    ):
                        start_time = time.monotonic()
                        next_progress_log = start_time + progress_log_interval_seconds
                        while True:
                            try:
                                chunk = src_file.read(chunk_size)
                            except TimeoutError as timeout_error:
                                raise TimeoutError(
                                    f"Timed out reading from {file_path} after "
                                    f"{read_timeout_seconds:.1f}s with {bytes_written} bytes written"
                                ) from timeout_error
                            if not chunk:
                                break
                            dest_file.write(chunk)
                            bytes_written += len(chunk)
                            now = time.monotonic()
                            if progress_log_interval_seconds > 0 and now >= next_progress_log:
                                elapsed = max(now - start_time, 1e-9)
                                speed_mib_s = (bytes_written / (1024**2)) / elapsed
                                logger.info(
                                    f"Streaming {file_path}: {bytes_written / (1024**2):.1f} MiB written "
                                    f"in {elapsed:.1f}s ({speed_mib_s:.2f} MiB/s)"
                                )
                                next_progress_log = now + progress_log_interval_seconds
                finally:
                    socket.setdefaulttimeout(previous_socket_timeout)

                # Validate file size BEFORE atomic_rename commits the file
                if expected_size is not None and bytes_written != expected_size:
                    raise ValueError(
                        f"Size mismatch for {file_path}: expected {expected_size} bytes, got {bytes_written} bytes"
                    )

            logger.info(f"Streamed {file_path} successfully to {fsspec_file_path} ({bytes_written} bytes)")
            return {"file_path": file_path, "status": "success", "size": bytes_written}
        except Exception as e:
            auth_error = _hf_auth_error(e, file_path)
            if auth_error:
                raise RuntimeError(auth_error) from e

            last_exception = e
            # Base wait: min 5s, then exponential: 5, 10, 20, 40, 80, 160, 320, 600 (capped)
            wait_base = max(min_base_wait, min_base_wait * (2**attempt))

            error_type = type(e).__name__
            error_msg = str(e)
            status_code = -1

            if isinstance(e, HfHubHTTPError):
                status_code = e.response.status_code
                TOO_MANY_REQUESTS = 429
                if status_code == TOO_MANY_REQUESTS:
                    # NOTE: RateLimit "api\|pages\|resolvers";r=[remaining];t=[seconds remaining until reset]
                    try:
                        rate_limit_wait = int(e.response.headers["RateLimit"].split(";")[-1].split("=")[-1])
                        wait_base = max(wait_base, rate_limit_wait + 10)  # Add buffer to rate limit wait
                    except Exception:
                        logger.warning("Failed to parse rate limit header, using default wait period")

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {file_path}: "
                f"{error_type} (status={status_code}): {error_msg}"
            )

            jitter = random.uniform(0, min(wait_base * 0.25, 30))  # Up to 25% jitter, max 30s
            wait_time = min(wait_base + jitter, max_sleep)

            logger.info(f"Retrying {file_path} in {wait_time:.1f}s...")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to download {file_path} after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )


def download_hf(cfg: DownloadConfig) -> None:

    configure_logging(level=logging.INFO)

    # Set cfg.append_sha_to_path=True to mimic the older behavior of writing to gcs_output_path/<revision>.
    # Some historical datasets were written that way, so this flag keeps backwards compatibility when needed.

    # Ensure the output path is writable
    try:
        output_path = os.path.join(cfg.gcs_output_path, cfg.revision) if cfg.append_sha_to_path else cfg.gcs_output_path
        ensure_fsspec_path_writable(output_path)
    except ValueError as e:
        logger.exception(f"Output path validation failed: {e}")
        raise e

    # Resolve source URL and filesystem. For production this is an hf:// URL backed
    # by HfFileSystem; tests can set source_url_override to a local/fsspec path.
    logger.info("Identifying files to download...")
    if cfg.source_url_override is not None:
        source_url = cfg.source_url_override
        source_fs, source_root = url_to_fs(source_url)
        list_kwargs: dict = {}
    else:
        hf_source_path = _resolve_hf_source_path(cfg)
        _assert_bucket_support_available(hf_source_path)
        source_url = f"hf://{hf_source_path}"
        source_fs, source_root = url_to_fs(source_url)
        list_kwargs = {"revision": cfg.revision}

    if not cfg.hf_urls_glob:
        files = source_fs.find(source_root, **list_kwargs)
    else:
        files = []
        for url_glob in cfg.hf_urls_glob:
            pattern = os.path.join(source_root, url_glob)
            files += source_fs.glob(pattern, **list_kwargs)

    if not files:
        raise ValueError(f"No files found for dataset `{cfg.hf_dataset_id}. Used glob patterns: {cfg.hf_urls_glob}")

    # Get file sizes for validation
    logger.info("Getting file sizes for validation...")
    file_sizes: dict[str, int | None] = {}
    for file in files:
        try:
            info = source_fs.info(file, **list_kwargs)
            file_sizes[file] = info.get("size") or None
        except Exception as e:
            logger.warning(f"Could not get size for {file}: {e}")
            file_sizes[file] = None  # Will skip validation for this file

    download_tasks = []

    for file in files:
        try:
            relative_file_path = _relative_path_in_source(file, source_root)
            if relative_file_path.startswith(".."):
                raise ValueError(f"Computed path escapes source root: source={hf_source_path}, file={file}")
            fsspec_file_path = os.path.join(output_path, relative_file_path)
            expected_size = file_sizes.get(file)
            # Fully-qualify the source URL so subprocess workers can open it via fsspec
            # without having to reconstruct HfFileSystem / revision state.
            worker_source_url = file if cfg.source_url_override is not None else f"hf://{file}"
            download_tasks.append(
                (
                    output_path,
                    worker_source_url,
                    fsspec_file_path,
                    expected_size,
                    cfg.read_timeout_seconds,
                    cfg.progress_log_interval_seconds,
                    cfg.read_chunk_size_mib,
                )
            )
        except Exception as e:
            logging.exception(f"Error preparing task for {file}: {e}")

    total_files = len(download_tasks)
    total_size_gb = sum(s for s in file_sizes.values() if s is not None) / (1024**3)
    logger.info(f"Total number of files to process: {total_files} ({total_size_gb:.2f} GB)")

    pipeline = (
        Dataset.from_list(download_tasks)
        .map(lambda task: stream_file_to_fsspec(*task))
        .write_jsonl(
            f"{cfg.gcs_output_path}/.metrics/success-part-{{shard:05d}}-of-{{total:05d}}.jsonl", skip_existing=True
        )
    )
    ctx = ZephyrContext(
        name="download-hf",
        max_workers=cfg.zephyr_max_parallelism,
        resources=cfg.worker_resources,
    )
    ctx.execute(pipeline)

    # Write Provenance JSON
    write_provenance_json(
        output_path,
        metadata={"dataset": cfg.hf_dataset_id, "version": cfg.revision, "links": files},
    )

    logger.info(f"Streamed all files and wrote provenance JSON; check {output_path}.")


def download_hf_step(
    name: str,
    *,
    hf_dataset_id: str,
    revision: str,
    hf_urls_glob: list[str] | None = None,
    append_sha_to_path: bool = False,
    zephyr_max_parallelism: int = 8,
    deps: list[StepSpec] | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that downloads a HuggingFace dataset.

    The raw download is preserved as-is in its original format and directory structure.

    Args:
        name: Step name (e.g. "raw/fineweb").
        hf_dataset_id: HuggingFace dataset identifier (e.g. "HuggingFaceFW/fineweb").
        revision: Commit hash from the HF dataset repo.
        hf_urls_glob: Glob patterns to select specific files. Empty means all files.
        append_sha_to_path: If True, write outputs under ``output_path/<revision>``.
        zephyr_max_parallelism: Maximum download parallelism.
        deps: Optional upstream dependencies.
        override_output_path: Override the computed output path entirely.

    Returns:
        A StepSpec whose output_path contains the raw downloaded files.
    """
    resolved_glob = hf_urls_glob or []

    def _run(output_path: str) -> None:
        download_hf(
            DownloadConfig(
                hf_dataset_id=hf_dataset_id,
                revision=revision,
                hf_urls_glob=resolved_glob,
                gcs_output_path=output_path,
                append_sha_to_path=append_sha_to_path,
                zephyr_max_parallelism=zephyr_max_parallelism,
            )
        )

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={
            "hf_dataset_id": hf_dataset_id,
            "revision": revision,
            "hf_urls_glob": resolved_glob,
            "append_sha_to_path": append_sha_to_path,
        },
        override_output_path=override_output_path,
    )
