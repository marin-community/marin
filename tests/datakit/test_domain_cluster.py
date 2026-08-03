# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.cluster.domain.v0.train import load_sample_embeddings
from experiments.datakit.embeddings.artifact import EmbeddingAttrData


def _embedding_artifact(output_dir: str, source_key: str, embedding_dim: int) -> EmbeddingAttrData:
    return EmbeddingAttrData(
        output_dir=output_dir,
        source_key=source_key,
        model_name="test/model",
        model_revision="revision",
        embedding_dim=embedding_dim,
        quantization_scale=0.01,
        quantization_range=1.27,
        batch_size=8,
    )


def test_sample_centroid_inputs_rejects_mixed_embedding_spaces(tmp_path) -> None:
    embeddings = {
        "source-a": _embedding_artifact(str(tmp_path / "a"), "a", 192),
        "source-b": _embedding_artifact(str(tmp_path / "b"), "b", 256),
    }

    with pytest.raises(ValueError, match="embedding space for 'source-b' does not match 'source-a'"):
        sample_centroid_inputs(str(tmp_path / "sample"), embeddings, n_per_source=10)


def test_load_sample_embeddings_uses_configured_dimension_and_scale(tmp_path) -> None:
    quantized = np.arange(512, dtype=np.int16).astype(np.int8).reshape(2, 256)
    fixed_size_embeddings = pa.FixedSizeListArray.from_arrays(pa.array(quantized.ravel()), 256)
    table = pa.table({"source": ["a", "b"], "embedding": fixed_size_embeddings})
    pq.write_table(table, tmp_path / "sample.parquet")

    actual = load_sample_embeddings(str(tmp_path), embedding_dim=256, quantization_scale=0.01)

    np.testing.assert_array_equal(actual, quantized.astype(np.float32) * 0.01)


def test_load_sample_embeddings_rejects_wrong_dimension(tmp_path) -> None:
    quantized = np.zeros((2, 256), dtype=np.int8)
    fixed_size_embeddings = pa.FixedSizeListArray.from_arrays(pa.array(quantized.ravel()), 256)
    table = pa.table({"source": ["a", "b"], "embedding": fixed_size_embeddings})
    pq.write_table(table, tmp_path / "sample.parquet")

    with pytest.raises(ValueError, match="does not match configured dimension 192"):
        load_sample_embeddings(str(tmp_path), embedding_dim=192, quantization_scale=0.01)
