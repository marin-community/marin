from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedStateDict, ShardedTensor
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
from megatron.core.dist_checkpointing.strategies.state_dict_saver import save_state_dict_async_plan
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
import torch.distributed as dist
from loguru import logger
from torch.distributed import checkpoint
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.utils import _DistWrapper

from marinskyrl.remote_io import create_s3_filesystem
from skyrl_train.distributed.megatron.checkpoint_writer import MasterWeightsWriter
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader
from skyrl_train.telemetry import record_event


def _checkpoint_structure(value: object) -> object:
    if isinstance(value, ShardedBase):
        tensor_properties = None
        if isinstance(value, ShardedTensor) and value.data is not None:
            tensor_properties = (
                tuple(value.data.shape),
                str(value.data.layout),
                tuple(value.data.stride()),
                value.data.requires_grad,
                value.data.is_pinned(),
            )
        return (repr(value.without_data()), tensor_properties)
    if isinstance(value, dict):
        return tuple(
            (repr(key), _checkpoint_structure(value[key]))
            for key in sorted(value, key=lambda key: (type(key).__name__, repr(key)))
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_checkpoint_structure(item) for item in value))
    raise TypeError(f"Unexpected unsharded checkpoint leaf: {type(value).__name__}")


class DirectS3TorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Reuse MCore save plans while writing each checkpoint to fresh regional objects."""

    def __init__(self, checkpoint_dir: str, *, cached_metadata: bool) -> None:
        super().__init__(cached_metadata=cached_metadata)
        self.checkpoint_dir = checkpoint_dir
        self.model_aliases: dict[str, str] = {}
        self.save_count = 0
        self.structure_digest = ""
        self.structure_invalidated = True
        self.fingerprint_seconds = 0.0
        self.distribution_seconds = 0.0
        self.serialization_started: float | None = None
        self.preprocess_seconds: float | None = None

    def invalidate_cache(self) -> None:
        self.cached_central_plan = None
        self.cached_local_plan = None
        self.cached_global_metadata = None
        self.validated_cache_reuse = False
        self.model_aliases = {}

    def save(self, sharded_state_dict: ShardedStateDict, _checkpoint_dir: Path) -> None:
        translation_started = perf_counter()
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pytorch_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        translation_seconds = perf_counter() - translation_started
        writer = MasterWeightsWriter(
            self.checkpoint_dir, filesystem=create_s3_filesystem(), model_aliases=self.model_aliases
        )
        planner = MCoreSavePlanner(
            dedup_replicated_tensors=not self.keep_only_main_replica,
            # Cross-rank master aliases require the writer's global-plan hook.
            can_run_decentralized_global_plan=False,
            flatten_state_dict=False,
            flatten_sharded_tensors=False,
        )
        cache_reused = bool(
            self.use_cached_ckpt_structure and self.validated_cache_reuse and self.cached_central_plan is not None
        )
        planning_started = perf_counter()
        try:
            result, central_plan, local_plan, validated, _ = save_state_dict_async_plan(
                pytorch_state_dict,
                writer,
                planner=planner,
                cached_ckpt_structure=(self.cached_central_plan, self.cached_local_plan, self.validated_cache_reuse)
                if self.use_cached_ckpt_structure
                else None,
            )
            _, metadata, distributed = result
            if metadata is None:
                metadata = self.cached_global_metadata
            planning_seconds = perf_counter() - planning_started

            def finish(results):
                assert metadata is not None, "Coordinator checkpoint metadata is missing"
                # finish installs current storage records and aliases. Cached metadata must
                # retain the logical plan, never a previous destination's storage records.
                writer.finish(deepcopy(metadata), results)

            writing_started = perf_counter()
            distributed.all_reduce("write", writer.write_planned_data, finish)
            writing_seconds = perf_counter() - writing_started
        except BaseException:
            # CheckpointException inherits BaseException. Never reuse partly advanced plans.
            self.invalidate_cache()
            raise
        self.cached_central_plan = central_plan
        self.cached_local_plan = local_plan
        self.cached_global_metadata = metadata
        self.validated_cache_reuse = validated
        self.model_aliases = writer.model_aliases
        self.save_count += 1
        self.serialization_started = None
        fields = {
            "checkpoint_dir": self.checkpoint_dir,
            "rank": dist.get_rank(),
            "save_index": self.save_count,
            "cache_enabled": self.use_cached_ckpt_structure,
            "cache_reused": cache_reused,
            "cache_validated": validated,
            "cache_invalidated": self.structure_invalidated,
            "structure_digest": self.structure_digest,
            "fingerprint_seconds": self.fingerprint_seconds,
            "distribution_seconds": self.distribution_seconds,
            "preprocess_seconds": self.preprocess_seconds,
            "translation_seconds": translation_seconds,
            "planning_seconds": planning_seconds,
            "writing_seconds": writing_seconds,
            "master_aliases": len(self.model_aliases),
        }
        record_event("megatron_checkpoint_save", fields, attributes={"role": "trainer"})
        logger.bind(checkpoint_save=fields).info("DCP checkpoint save: {}", fields)


class CachedFullyParallelSaveStrategy(FullyParallelSaveStrategyWrapper):
    """Invalidate distribution, plans and aliases together when any rank changes structure."""

    def __init__(
        self, strategy: DirectS3TorchDistSaveShardedStrategy, parallelization_group: dist.ProcessGroup
    ) -> None:
        super().__init__(strategy, parallelization_group, do_cache_distribution=True)
        self.direct_strategy = strategy
        self.structure_digest: str | None = None

    def apply_saving_parallelization(self, sharded_state_dict: ShardedStateDict) -> None:
        started = perf_counter()
        self.direct_strategy.preprocess_seconds = (
            None
            if self.direct_strategy.serialization_started is None
            else started - self.direct_strategy.serialization_started
        )
        signature = None

        def changed_structure():
            nonlocal signature
            topology = (dist.get_world_size(), tuple(dist.get_process_group_ranks(self.parallelization_group)))
            signature = hashlib.sha256(repr((topology, _checkpoint_structure(sharded_state_dict))).encode()).hexdigest()
            return signature != self.structure_digest

        # Plans are global even when distribution uses a smaller DP+CP group.
        # DCP propagates fingerprint failures to every rank instead of leaving peers
        # waiting in a cache-invalidation collective.
        invalidated = _DistWrapper(None, True, 0).all_reduce("checkpoint_structure", changed_structure, any)
        assert signature is not None
        if invalidated:
            self.cached_distribution = None
            self.direct_strategy.invalidate_cache()
        self.structure_digest = signature
        self.direct_strategy.structure_digest = signature
        self.direct_strategy.structure_invalidated = invalidated
        self.direct_strategy.fingerprint_seconds = perf_counter() - started
        started = perf_counter()
        super().apply_saving_parallelization(sharded_state_dict)
        self.direct_strategy.distribution_seconds = perf_counter() - started

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> None:
        try:
            super().save(sharded_state_dict, checkpoint_dir)
        except BaseException:
            self.structure_digest = None
            self.cached_distribution = None
            self.direct_strategy.invalidate_cache()
            raise


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
