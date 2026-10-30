# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
import pytest

from experiments.post_training.weight_analysis.checkpoint_io import TensorSpec, write_sharded_checkpoint
from experiments.post_training.weight_analysis.delta_stats import bf16_export_delta_blocks, delta_blocks
from experiments.post_training.weight_analysis.expert_metrics import seed_ceiling


def test_export_cosine_reflects_bf16_rounding_without_relaxing_master_checks(tmp_path):
    name = "model.layers.0.self_attn.q_proj.weight"
    arrays = {
        "base": np.array([[1.0, 1.0]], dtype=ml_dtypes.bfloat16),
        "seed0": np.array([[1.001, 1.01]], dtype=ml_dtypes.bfloat16),
        "seed1": np.array([[1.01, 1.001]], dtype=ml_dtypes.bfloat16),
    }
    bias_name = "model.layers.0.mlp.router.bias"
    base_bias = 2**-12 + 2**-27
    biases = {label: np.array([base_bias + (2**-26 if label == "seed0" else 0)], dtype=np.float32) for label in arrays}
    for label, array in arrays.items():
        tensors = [(name, array), (bias_name, biases[label])]
        write_sharded_checkpoint(
            tensors,
            str(tmp_path / label),
            plan={key: TensorSpec(value.shape, value.dtype) for key, value in tensors},
            shard_size_bytes=1024,
        )
    streams = [bf16_export_delta_blocks(str(tmp_path / label), str(tmp_path / "base")) for label in ("seed0", "seed1")]
    result = seed_ceiling(streams, near_zero=0.0)
    # The small update rounds away in opposite coordinates: exported deltas
    # are orthogonal although their underlying FP32 updates were not.
    assert result.overall.cosine == 0.0
    deltas = {
        block.name: block.array for block in bf16_export_delta_blocks(str(tmp_path / "seed0"), str(tmp_path / "base"))
    }
    # Router bias stays FP32 even though the surrounding served weights are BF16.
    np.testing.assert_array_equal(deltas[bias_name], np.array([2**-26], dtype=np.float32))
    with pytest.raises(ValueError, match="C8 needs fp32 masters"):
        list(delta_blocks(str(tmp_path / "seed0"), str(tmp_path / "base")))
