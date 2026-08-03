# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral artifacts and quantization for document embeddings."""

import numpy as np
from marin.datakit.source_key import DatakitArtifactPath
from pydantic import BaseModel, Field
from rigging.filesystem import StoragePath

EMBEDDING_ATTR_DATA_VERSION = 2


class EmbeddingAttrData(BaseModel):
    """Co-partitioned per-source embedding Parquet shards.

    One output shard exists for each normalized source shard. The output uses
    the same basename and row order, so consumers can join by shard and row.

    Attributes:
        output_dir: Directory that contains the output shards.
        source_key: Identity of the normalized source that the output mirrors.
        model_name: Embedding model name.
        model_revision: Immutable model revision or model digest.
        embedding_dim: Vector dimension.
        quantization_scale: Scale for conversion from int8 to float32.
        quantization_range: Float range before symmetric quantization.
        batch_size: Inference batch size.
        counters: Aggregated Zephyr counters.
    """

    version: str = f"v{EMBEDDING_ATTR_DATA_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    model_name: str
    model_revision: str = ""
    embedding_dim: int = Field(ge=1)
    quantization_scale: float = Field(gt=0)
    quantization_range: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    counters: dict[str, int | float] = Field(default_factory=dict)

    def shard_paths(self) -> list[str]:
        """Return the sorted Parquet shard paths."""
        return sorted(str(path) for path in StoragePath(f"{self.output_dir.rstrip('/')}/*.parquet").glob())


def quantize_to_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    """Quantize float values to 255 symmetric int8 levels."""
    if scale <= 0:
        raise ValueError("The quantization scale must be positive")
    return np.clip(np.round(arr / scale), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float) -> np.ndarray:
    """Convert quantized int8 values to float32."""
    if scale <= 0:
        raise ValueError("The quantization scale must be positive")
    return arr.astype(np.float32) * scale
