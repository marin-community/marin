# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export native Hero weights on one CPU host using chunk-aligned TensorStore reads."""

import logging
import math
from dataclasses import dataclass

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from levanter.compat.hf_checkpoints import DEFAULT_MAX_SHARD_SIZE, _shard_hf_checkpoint
from levanter.compat.hf_export import save_host_hf_shards
from levanter.tensorstore_serialization import ARRAY_DRIVER, KVSTORE_DRIVER, build_kvstore_spec
from levanter.utils.jax_utils import leaf_key_paths
from rigging.filesystem.cross_region import record_transfer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import (
    GrugModelConfig,
    InferenceWeightLayout,
    InferenceWeightSpec,
    Transformer,
    grugmoe_inference_weight_specs,
)
from experiments.grug.moe_hero_ep.ops.export_metadata import (
    EXPORT_VERSION,
    artifact_config,
    bf16_hf_config,
    execution_config,
    prepare_export,
    split_expert_names,
    write_export_metadata,
)
from experiments.grug.moe_hero_ep.weights import (
    PENDING_QB_BETAS_KEY,
    checkpoint_weight_path,
    validate_checkpoint_metadata,
)

logger = logging.getLogger(__name__)
BF16_BYTES = 2
EXPERT_PATH = "/expert_mlp/"


@dataclass(frozen=True)
class CpuExportConfig:
    checkpoint: str
    metadata_digest: str
    model: GrugModelConfig
    destination: str
    source_revision: str
    read_host_budget_bytes: int = 96 * 1024**3
    export_host_budget_bytes: int = 32 * 1024**3
    max_concurrent_shards: int = 2
    request_concurrency: int = 16


def _native_shapes(config: GrugModelConfig) -> dict[str, tuple[int, ...]]:
    cpu = jax.devices("cpu")[0]
    mesh = jax.sharding.Mesh(
        np.asarray([cpu]).reshape((1,) * 5),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 5,
    )
    with jax.default_device(cpu), jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    paths = jax.tree.leaves(leaf_key_paths(template))
    return {path.replace(".", "/"): value.shape for path, value in zip(paths, jax.tree.leaves(template), strict=True)}


class CpuCheckpointWeights:
    """Bound ordinary BF16 weights and one chunk-aligned expert block in host memory.

    The writer owns detached shard copies. Its separate host-byte budget bounds
    those copies and serialization buffers while this reader advances to the next
    source block. This reader is called serially by the export's main thread.
    """

    def __init__(self, config: CpuExportConfig):
        validate_checkpoint_metadata(config.checkpoint, config.metadata_digest)
        if not (StoragePath(config.checkpoint) / "manifest.ocdbt").exists():
            raise ValueError("CPU export requires an OCDBT checkpoint")
        self.config = config
        self.weight_path = checkpoint_weight_path(config.checkpoint)
        self.specs = {spec.name: spec for spec in grugmoe_inference_weight_specs(config.model)}
        self.context = ts.Context(
            {
                "cache_pool": {"total_bytes_limit": 1024**3},
                "data_copy_concurrency": {"limit": config.request_concurrency},
                "s3_request_concurrency": {"limit": config.request_concurrency},
                "gcs_request_concurrency": {"limit": config.request_concurrency},
                "file_io_concurrency": {"limit": config.request_concurrency},
            }
        )
        expected = _native_shapes(config.model)
        self.stores = {
            path: self._open(f"{self.weight_path}/{path}")
            for path in dict.fromkeys(spec.path for spec in self.specs.values())
        }
        for path, store in self.stores.items():
            if tuple(store.shape) != expected[path]:
                raise ValueError(
                    f"Checkpoint shape differs from model config: {path}: {store.shape} != {expected[path]}"
                )
            if tuple(store.domain.inclusive_min) != (0,) * store.ndim:
                raise ValueError(f"Checkpoint array must have a zero origin: {path}")
        parent = self.weight_path.rpartition("/")[0]
        pending_path = f"{parent}/{PENDING_QB_BETAS_KEY}" if parent else PENDING_QB_BETAS_KEY
        self.pending = self._open(pending_path)
        if tuple(self.pending.shape) != (config.model.num_layers, config.model.num_experts):
            raise ValueError("Pending router-bias shape differs from model config")
        self.ordinary_bytes = sum(
            math.prod(store.shape) * BF16_BYTES for path, store in self.stores.items() if EXPERT_PATH not in path
        )
        largest_block = max(
            (
                min(self.layer_chunk(path), store.shape[0]) * math.prod(store.shape[1:]) * BF16_BYTES
                for path, store in self.stores.items()
                if EXPERT_PATH in path
            ),
            default=0,
        )
        if self.ordinary_bytes + largest_block > config.read_host_budget_bytes:
            raise ValueError(
                f"Chunk-aligned reads need {self.ordinary_bytes + largest_block} bytes; "
                f"read_host_budget_bytes={config.read_host_budget_bytes}"
            )
        self._ordinary: dict[str, np.ndarray] | None = None
        self._expert_block: np.ndarray | None = None
        self._expert_key: tuple[str, int] | None = None
        self.read_bytes = 0
        self.expert_blocks_read = 0

    def _open(self, path: str) -> ts.TensorStore:
        return ts.open(
            {
                "driver": ARRAY_DRIVER,
                "kvstore": {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(self.config.checkpoint), "path": path},
            },
            open=True,
            read=True,
            write=False,
            context=self.context,
        ).result()

    def layer_chunk(self, path: str) -> int:
        shape = self.stores[path].chunk_layout.read_chunk.shape
        assert shape is not None
        return int(shape[0])

    def _read_bf16(self, store: ts.TensorStore) -> np.ndarray:
        size = math.prod(store.shape) * np.dtype(store.dtype.numpy_dtype).itemsize
        record_transfer(size, self.config.checkpoint)
        self.read_bytes += size
        return ts.cast(store, ts.bfloat16).read().result()

    def _load_ordinary(self) -> dict[str, np.ndarray]:
        if self._ordinary is None:
            self._ordinary = {
                path: self._read_bf16(store) for path, store in self.stores.items() if EXPERT_PATH not in path
            }
            # Match apply_qb_betas: use float32 centering before the BF16 cast.
            pending_bytes = math.prod(self.pending.shape) * np.dtype(self.pending.dtype.numpy_dtype).itemsize
            record_transfer(pending_bytes, self.config.checkpoint)
            self.read_bytes += pending_bytes
            pending = self.pending.read().result()
            with jax.default_device(jax.devices("cpu")[0]):
                bias = -jnp.asarray(pending, dtype=jnp.float32)
                bias = bias - jnp.mean(bias, axis=-1, keepdims=True)
                self._ordinary["stacked_blocks/stacked/mlp/router_bias"] = np.asarray(bias.astype(jnp.bfloat16))
        return self._ordinary

    def _weight(self, spec: InferenceWeightSpec) -> np.ndarray:
        ordinary = self._load_ordinary()
        if EXPERT_PATH in spec.path:
            assert spec.layer_index is not None
            start = spec.layer_index // self.layer_chunk(spec.path) * self.layer_chunk(spec.path)
            key = (spec.path, start)
            if self._expert_key != key:
                # Release the preceding source block before allocating the next one.
                self._expert_block = None
                self._expert_key = None
                store = self.stores[spec.path]
                stop = min(start + self.layer_chunk(spec.path), store.shape[0])
                logger.info("Read expert block %s layers [%d, %d)", spec.path, start, stop)
                self._expert_block = self._read_bf16(store[start:stop])
                self._expert_key = key
                self.expert_blocks_read += 1
            assert self._expert_block is not None
            value = self._expert_block[spec.layer_index - start]
        else:
            value = ordinary[spec.path]
            if spec.layer_index is not None:
                value = value[spec.layer_index]
        if spec.layout == InferenceWeightLayout.LINEAR:
            value = np.swapaxes(value, -1, -2)
        # Writer threads must not retain a view into the reader's reusable block.
        return np.array(value, order="C", copy=True)

    def load_shard(self, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        ordered = sorted(names, key=lambda name: self.shard_order((name,)))
        return {name: self._weight(self.specs[name]) for name in ordered}

    def shapes(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {
            name: jax.ShapeDtypeStruct(spec.output_shape(tuple(self.stores[spec.path].shape)), jnp.bfloat16)
            for name, spec in self.specs.items()
        }

    def shard_order(self, names: tuple[str, ...]) -> tuple[str, int]:
        # Keep all layers sharing one stored expert chunk adjacent. Ordinary
        # tensors remain resident and can be emitted in any shard order.
        experts = [self.specs[name] for name in names if EXPERT_PATH in self.specs[name].path]
        if not experts:
            return ("", -1)
        return min((spec.path, int(spec.layer_index)) for spec in experts)


def export(config: CpuExportConfig) -> None:
    hf_config = bf16_hf_config(config.model)
    encoded = draccus.encode(config)
    request = {
        "export_version": EXPORT_VERSION,
        "implementation": "cpu",
        "config": artifact_config(encoded),
        "hf_config": hf_config,
    }
    resume = prepare_export(config.destination, request)
    reader = CpuCheckpointWeights(config)
    shapes = reader.shapes()
    shards, _ = _shard_hf_checkpoint(shapes, max_shard_size=DEFAULT_MAX_SHARD_SIZE)
    ordered = dict(sorted(shards.items(), key=lambda item: reader.shard_order(tuple(item[1]))))
    records = save_host_hf_shards(
        ordered,
        reader.load_shard,
        config.destination,
        export_host_budget_bytes=config.export_host_budget_bytes,
        max_concurrent_shards=config.max_concurrent_shards,
        resume=resume,
        tensor_names={name: split_expert_names(name, config.model.num_experts) for name in shapes},
    )
    by_filename = {record.filename: record for record in records}
    write_export_metadata(
        resume,
        request,
        [by_filename[name] for name in shards],
        sum(value.size * value.dtype.itemsize for value in shapes.values()),
        {
            "executor": "cpu",
            "execution": execution_config(encoded),
            "read_bytes": reader.read_bytes,
            "expert_blocks_read": reader.expert_blocks_read,
        },
    )


@draccus.wrap()
def main(config: CpuExportConfig) -> None:
    configure_coreweave_s3()
    configure_logging(logging.INFO)
    export(config)


if __name__ == "__main__":
    main()
