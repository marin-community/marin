"""FP32 master records of a Megatron torch-dist checkpoint, read without a model.

This reader returns masters under Megatron's own parameter names, which is what a
checkpoint-level check needs. HF-named FP32 exports for weight analysis come from
``MegatronStrategy.save_fp32_master_hf_model``, which converts through Bridge.
"""

from urllib.parse import urlparse

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from marinskyrl.remote_io import create_s3_filesystem

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader



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

