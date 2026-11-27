from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    MCoreSavePlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
    _replace_sharded_keys_with_state_dict_keys,
    _replace_state_dict_keys_with_sharded_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor,
    mcore_to_pyt_state_dict,
)
import torch
from torch.distributed import ProcessGroup, checkpoint
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.metadata import Metadata

from marinskyrl.remote_io import create_s3_filesystem
from skyrl_train.distributed.megatron.checkpoint_writer import StagedMasterWeightsWriter
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader


# MCore 0.18 does not expose a storage-writer hook. Keep the adapter narrow: it
# preserves MCore's state translation and planner and replaces only the writer.
class DirectS3TorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Save MCore torch-dist shards directly to S3 through PyTorch DCP."""

    def __init__(
        self, checkpoint_dir: str, *, staging_bytes: int, checkpoint_group: ProcessGroup
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.staging_bytes = staging_bytes
        self.checkpoint_group = checkpoint_group
        self.pending_write: Future[Metadata] | None = None

    def save(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path) -> None:
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pytorch_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        filesystem = create_s3_filesystem()
        planner = MCoreSavePlanner(
            dedup_replicated_tensors=not self.keep_only_main_replica,
            flatten_state_dict=False,
            flatten_sharded_tensors=False,
        )
        # The copy runs on another thread and must see this step's optimizer writes.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        staged_writer = StagedMasterWeightsWriter(
            self.checkpoint_dir, filesystem=filesystem, max_staged_bytes=self.staging_bytes
        )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self.pending_write = executor.submit(
            checkpoint.save,
            pytorch_state_dict,
            storage_writer=staged_writer,
            planner=planner,
            process_group=self.checkpoint_group,
        )

        def on_finished(_):
            # A failed plan or copy can finish without reaching the staging handoff.
            staged_writer.staged.set()
            executor.shutdown(wait=False)

        self.pending_write.add_done_callback(on_finished)
        staged_writer.staged.wait()


class DirectS3TorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    """Read the DCP records needed by this rank directly from S3."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        source = urlparse(checkpoint_dir)
        if source.scheme != "s3" or not source.netloc or not source.path.strip("/") or source.query or source.fragment:
            raise ValueError("Megatron restore requires a CoreWeave S3 training checkpoint")
        self.filesystem = create_s3_filesystem(default_cache_type="none")
        endpoint = self.filesystem.endpoint_url or self.filesystem.client_kwargs.get("endpoint_url", "")
        resolved = urlparse(endpoint)
        if (
            resolved.scheme not in ("http", "https")
            or resolved.hostname not in ("cwobject.com", "cwlota.com")
            or resolved.path not in ("", "/")
            or resolved.username
            or resolved.query
            or resolved.fragment
        ):
            raise ValueError("Megatron restore requires a resolved cwobject.com or cwlota.com storage endpoint")
        self.checkpoint_dir = checkpoint_dir

    def load(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path, async_strategy: str = "mcore"):
        del async_strategy  # Required by the Megatron sharded-load strategy interface.
        original = sharded_state_dict
        tensors = [value for value in nested_values(original) if isinstance(value, ShardedTensor)]
        converted, flat_mapping, rename_mapping = _replace_state_dict_keys_with_sharded_keys(original)
        pytorch_state_dict = mcore_to_pyt_state_dict(converted, True)

        reader = RecordCheckpointReader(self.checkpoint_dir)
        reader.fs = FsspecFileSystem()
        reader.fs.fs = self.filesystem
        reader.path = self.checkpoint_dir
        checkpoint.load(
            pytorch_state_dict,
            storage_reader=reader,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=[value for value in tensors if not value.allow_shape_mismatch],
                allow_shape_mismatch_sharded_tensors={
                    value.key: value for value in tensors if value.allow_shape_mismatch
                },
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
            no_dist=True,
        )
        restored = {name: _unwrap_pyt_sharded_tensor(value) for name, value in pytorch_state_dict.items()}
        restored = _replace_sharded_keys_with_state_dict_keys(restored, flat_mapping, rename_mapping)
        _restore_dict_types(restored, original)
        return restored
