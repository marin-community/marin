"""Validation and ordered publication for Hugging Face safetensors models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger
from marinskyrl.hf_model import (
    GENERATION_CONFIG_NAME,
    GenerationConfigTokenizer,
    export_generation_config,
    normalize_fast_tokenizer_metadata,
    write_generation_config,
)
from marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    ModelManifest,
    build_safetensors_weight_index,
    read_safetensors_header,
    write_local_model_manifest,
)
from marinskyrl.resource_locator import join_resource_path
from safetensors.torch import save_file
from torch import Tensor, contiguous_format
from transformers.utils import cached_file

from skyrl_train.io import io

HF_WEIGHT_FILENAME = "model.safetensors"


def save_megatron_hf_model(
    bridge,
    model,
    output_dir: str,
    *,
    rank_zero: bool,
    local_state_source_type: type,
    save_configs: Callable[[str], None],
) -> None:
    """Export a Megatron model and publish its completed HF files on rank zero."""
    source = getattr(getattr(bridge.hf_pretrained, "state", None), "source", None)
    use_bridge_writer = isinstance(source, local_state_source_type) and bool(source.key_to_filename_map)
    weights = None if use_bridge_writer else bridge.export_hf_weights(model, cpu=True)
    if rank_zero:
        model_dir = local_hf_model_dir(output_dir)
    elif weights is None:
        # Bridge's save is collective even when only rank zero publishes.
        model_dir = TemporaryDirectory() if io.is_cloud_path(output_dir) else nullcontext(output_dir)
    else:
        for _ in weights:
            pass
        return

    with model_dir as work_dir:
        if weights is None:
            bridge.save_hf_weights(model, work_dir)
        else:
            write_hf_weight_shards(weights, work_dir)
        if rank_zero:
            save_configs(work_dir)


def write_hf_weight_shards(
    weights: Iterable[tuple[str, Tensor]], output_dir: str, *, shard_size_bytes: int = 5_000_000_000
) -> None:
    """Write a stream of CPU weights as Safetensors shards for the export finalizer."""
    directory = Path(output_dir)
    shard = {}
    size = 0
    shard_number = 1

    def flush():
        nonlocal size, shard_number
        save_file(shard, directory / f"model-{shard_number:05d}.safetensors", metadata={"format": "pt"})
        shard.clear()
        size = 0
        shard_number += 1

    for name, tensor in weights:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if shard and size + tensor_bytes > shard_size_bytes:
            flush()
        shard[name] = tensor.clone(memory_format=contiguous_format)
        size += tensor_bytes
    if shard:
        flush()


def write_export_generation_config(
    work_dir: str, source_model: str | None, tokenizer: GenerationConfigTokenizer, chat_template: str | None
) -> None:
    """Write ``generation_config.json``, preserving explicit values from the source model's generation config."""
    source_config = None
    if source_model:
        source_path = cached_file(source_model, GENERATION_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        if source_path is not None:
            source_config = json.loads(Path(source_path).read_text())
    config = export_generation_config(tokenizer, chat_template=chat_template, explicit=source_config)
    write_generation_config(work_dir, config)
    logger.info(f"Wrote {GENERATION_CONFIG_NAME} to {work_dir}: {config}")


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


def verify_completed_hf_model_export(export_path: str) -> None:
    """Verify completion markers and each indexed shard header before reuse."""
    manifest_path = join_resource_path(export_path, MODEL_MANIFEST_FILENAME)
    index_path = join_resource_path(export_path, HF_WEIGHT_INDEX_FILENAME)
    if not io.exists(manifest_path) or not io.exists(index_path):
        raise RuntimeError(f"HF export lacks completion markers at {export_path}")
    verify_hf_model_export(export_path)
    with io.open_file(manifest_path, "r") as source:
        manifest = ModelManifest.from_mapping(json.load(source), manifest_path)
    with io.open_file(index_path, "rb") as source:
        index_bytes = source.read()
    weight_map = json.loads(index_bytes)["weight_map"]
    manifest_files = {entry.path: entry.size for entry in manifest.files}
    if missing := {
        path for path in manifest_files if not io.exists(join_resource_path(export_path, path))
    }:
        raise RuntimeError(f"HF export is missing manifest files at {export_path}: {sorted(missing)[:5]}")
    shards = set(weight_map.values())
    if not shards.issubset(manifest_files):
        raise RuntimeError(f"HF export manifest omits indexed shards at {export_path}")
    if HF_WEIGHT_INDEX_FILENAME not in manifest_files:
        raise RuntimeError(f"HF export manifest omits its index at {export_path}")
    shard_headers = {}
    for shard in sorted(shards):
        with io.open_file(join_resource_path(export_path, shard), "rb") as source:
            _header, keys = read_safetensors_header(source, shard)
            source.seek(0, 2)
            size = source.tell()
        if size != manifest_files[shard]:
            raise RuntimeError(f"HF export shard size differs from manifest at {export_path}: {shard}")
        shard_headers[shard] = (size, keys)
    build_safetensors_weight_index(shard_headers, export_path, existing=index_bytes)


def has_completed_hf_model_export(export_path: str) -> bool:
    """Return whether a published export can be reused without conversion."""
    if not io.exists(join_resource_path(export_path, MODEL_MANIFEST_FILENAME)):
        return False
    if not io.exists(join_resource_path(export_path, HF_WEIGHT_INDEX_FILENAME)):
        return False
    try:
        verify_completed_hf_model_export(export_path)
    except (RuntimeError, ValueError) as error:
        logger.warning(f"HF export at {export_path} is not reusable: {error}")
        return False
    return True


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
