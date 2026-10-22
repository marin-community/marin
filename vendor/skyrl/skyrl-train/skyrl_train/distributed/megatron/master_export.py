"""FP32 master export from a Megatron torch-dist checkpoint.

Weight analysis reads the FP32 masters a run saved with its optimizer state. The
exporter verifies, from the checkpoint metadata, that every weight it exports is an
FP32 master record before reading any tensor bytes, so a checkpoint that only carries
BF16 model weights fails loudly instead of exporting a silently upcast copy.
"""

import json
from urllib.parse import urlparse

from loguru import logger
import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from marinskyrl.remote_io import create_s3_filesystem
from marinskyrl.resource_locator import join_resource_path
from safetensors.torch import save_file

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master
from skyrl_train.io import io
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader

SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
MAX_SHARD_BYTES = 5 * 2**30


def load_fp32_masters(ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Read every FP32 master record, keyed by model-space weight name.

    Raises ValueError unless the checkpoint metadata shows a full set of FP32 master
    records; there is no model-weight fallback, so a BF16 checkpoint cannot pass as an
    FP32 export. Holds every master in memory; large checkpoints need a streaming
    reader.
    """
    metadata = _storage_reader(ckpt_dir).read_metadata()
    masters = _master_records(metadata, ckpt_dir)
    state = {key: torch.empty(record.size, dtype=torch.float32) for key, record in masters.items()}
    dcp.load(
        state,
        storage_reader=_storage_reader(ckpt_dir),
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )
    return state


def export_fp32_masters(ckpt_dir: str, output_dir: str) -> None:
    """Write every FP32 master record to ``output_dir`` as sharded safetensors.

    The entry point weight analysis uses to read a training checkpoint: masters are
    verified against the checkpoint metadata before any tensor is read, then saved in
    model-space key order with a safetensors index.
    """
    masters = load_fp32_masters(ckpt_dir)
    shards = _shard_groups(masters)
    with io.local_work_dir(output_dir) as work_dir:
        for index, keys in enumerate(shards, start=1):
            shard_name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            save_file(
                {key: masters[key] for key in keys},
                join_resource_path(work_dir, shard_name),
                metadata={"format": "pt"},
            )
        index_payload = {
            "metadata": {"total_size": sum(weight.numel() * weight.element_size() for weight in masters.values())},
            "weight_map": {
                key: f"model-{index:05d}-of-{len(shards):05d}.safetensors"
                for index, keys in enumerate(shards, start=1)
                for key in keys
            },
        }
        io.write_bytes_atomic(
            join_resource_path(work_dir, SAFETENSORS_INDEX_FILE),
            json.dumps(index_payload, indent=2, sort_keys=True).encode(),
        )
    logger.info("Exported {} FP32 masters from {} to {}", len(masters), ckpt_dir, output_dir)


def _storage_reader(ckpt_dir: str):
    if urlparse(ckpt_dir).scheme != "s3":
        return FileSystemReader(ckpt_dir)
    reader = RecordCheckpointReader(ckpt_dir)
    reader.fs = FsspecFileSystem()
    reader.fs.fs = create_s3_filesystem(default_cache_type="none")
    reader.path = ckpt_dir
    return reader


def _master_records(metadata: Metadata, ckpt_dir: str) -> dict[str, TensorStorageMetadata]:
    """Map each model-space weight name to its FP32 master record, raising on any doubt."""
    masters: dict[str, TensorStorageMetadata] = {}
    for key, record in metadata.state_dict_metadata.items():
        model_key = model_key_for_master(key)
        if model_key is None:
            continue
        if not isinstance(record, TensorStorageMetadata):
            raise ValueError(f"master record {key} in {ckpt_dir} is not a tensor record")
        if record.properties.dtype is not torch.float32:
            raise ValueError(
                f"master record {key} in {ckpt_dir} stores {record.properties.dtype}, not float32; "
                "exporting it would silently upcast non-master weights"
            )
        if model_key in masters:
            raise ValueError(f"multiple FP32 master records for {model_key} in {ckpt_dir}")
        masters[model_key] = record
    if not masters:
        raise ValueError(
            f"no FP32 master records in {ckpt_dir}; export_fp32_masters only reads checkpoints saved "
            "with optimizer state under the fully_reshardable format"
        )
    return masters


def _shard_groups(masters: dict[str, torch.Tensor]) -> list[list[str]]:
    """Group sorted keys into safetensors shards of at most ``MAX_SHARD_BYTES``."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key, weight in sorted(masters.items()):
        weight_bytes = weight.numel() * weight.element_size()
        if current and current_bytes + weight_bytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += weight_bytes
    if current:
        shards.append(current)
    return shards
