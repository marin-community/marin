"""Validation and ordered publication for Hugging Face safetensors models."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger
from marinskyrl.hf_model import normalize_fast_tokenizer_metadata
from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME, MODEL_MANIFEST_FILENAME, write_local_model_manifest
from marinskyrl.resource_locator import join_resource_path

from skyrl_train.io import io

HF_WEIGHT_FILENAME = "model.safetensors"


def verify_hf_model_export(export_path: str) -> None:
    """Reject an HF export unless its safetensors weights are all present."""
    local_export = not io.is_cloud_path(export_path)
    if local_export:
        export_root = Path(export_path)
        export_files = {path.name for path in export_root.iterdir()} if export_root.is_dir() else set()
    else:
        try:
            export_files = {Path(path).name for path in io.list_dir(export_path)}
        except FileNotFoundError:
            export_files = set()

    index_path = join_resource_path(export_path, HF_WEIGHT_INDEX_FILENAME)
    if HF_WEIGHT_INDEX_FILENAME not in export_files:
        if HF_WEIGHT_FILENAME in export_files:
            return
        raise RuntimeError(f"HF export has no safetensors weights at {export_path}")

    try:
        source_context = Path(index_path).open() if local_export else io.open_file(index_path, "r")
        with source_context as source:
            index = json.load(source)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"HF export has an unreadable safetensors index at {index_path}: {error}") from error

    if not isinstance(index, dict):
        raise RuntimeError(f"HF export safetensors index must be a JSON object at {index_path}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"HF export safetensors weight_map must be a JSON object at {index_path}")
    if not weight_map:
        raise RuntimeError(f"HF export has an empty safetensors weight map at {index_path}")

    shard_values = list(weight_map.values())
    invalid_shards = [shard for shard in shard_values if not isinstance(shard, str) or Path(shard).name != shard]
    if invalid_shards:
        raise RuntimeError(f"HF export index contains invalid safetensors shard paths: {invalid_shards}")
    shards = sorted(set(shard_values))
    missing_shards = [shard for shard in shards if shard not in export_files]
    if missing_shards:
        raise RuntimeError(
            f"HF export is missing {len(missing_shards)} referenced safetensors shard(s): {missing_shards[:5]}"
        )


def _upload_hf_model_directory(local_path: str, cloud_path: str) -> None:
    verify_hf_model_export(local_path)

    source_root = Path(local_path)
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    index_files = [path for path in files if path.relative_to(source_root).as_posix() == HF_WEIGHT_INDEX_FILENAME]
    manifest_files = [path for path in files if path.relative_to(source_root).as_posix() == MODEL_MANIFEST_FILENAME]
    weight_files = [path for path in files if path.suffix == ".safetensors"]
    other_files = [
        path for path in files if path not in weight_files and path not in index_files and path not in manifest_files
    ]

    def publish_file(path: Path) -> None:
        relative_path = path.relative_to(source_root).as_posix()
        destination_uri = join_resource_path(cloud_path, relative_path)
        io.upload_file(str(path), destination_uri)

    for shard_index, path in enumerate(weight_files, start=1):
        relative_path = path.relative_to(source_root).as_posix()
        size = path.stat().st_size
        logger.info(f"Publishing HF weight shard {shard_index}/{len(weight_files)}: {relative_path} ({size} bytes)")
        started = time.monotonic()
        publish_file(path)
        logger.info(
            f"Published HF weight shard {shard_index}/{len(weight_files)}: {relative_path} "
            f"({size} bytes in {time.monotonic() - started:.1f}s)"
        )
    for path in [*other_files, *index_files, *manifest_files]:
        publish_file(path)

    logger.info(f"Published HF model directory to {cloud_path}")


def _remove_completion_markers(output_path: str) -> None:
    for filename in (HF_WEIGHT_INDEX_FILENAME, MODEL_MANIFEST_FILENAME):
        path = join_resource_path(output_path, filename)
        if io.exists(path):
            io.remove(path)
            logger.info(f"Removed HF export completion marker: {path}")


@contextmanager
def local_hf_model_dir(output_path: str):
    """Invalidate the prior index and yield a local directory for the completed export."""
    _remove_completion_markers(output_path)

    try:
        with io.local_output_dir(output_path, _upload_hf_model_directory) as work_dir:
            yield work_dir
            normalize_fast_tokenizer_metadata(Path(work_dir))
            write_local_model_manifest(work_dir)
    except BaseException as export_error:
        try:
            _remove_completion_markers(output_path)
        except Exception as cleanup_error:
            raise ExceptionGroup(
                f"HF export failed and its completion markers could not be removed: {output_path}",
                [export_error, cleanup_error],
            ) from export_error
        raise
