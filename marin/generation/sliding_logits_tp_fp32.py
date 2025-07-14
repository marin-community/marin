from __future__ import annotations

from dataclasses import dataclass

import ray

from marin.generation.sliding_logits_tp import (
    Precision,
    SlidingLogitsTPConfig,
    compute_sliding_logits_tp,
)
from marin.utils import remove_tpu_lockfile_on_exit


@dataclass
class SlidingLogitsTPFP32Config(SlidingLogitsTPConfig):
    """Configuration for FP32 tensor-parallel sliding logits on multi-host TPUs."""

    precision: Precision = Precision.FLOAT32
    num_devices: int | None = 16
    mesh_shape: tuple[int, int] | None = (1, 16)


@remove_tpu_lockfile_on_exit
def compute_sliding_logits_tp_fp32(cfg: SlidingLogitsTPFP32Config) -> None:
    """Run tensor-parallel sliding window forward pass with FP32 precision."""

    compute_sliding_logits_tp(cfg)


compute_sliding_logits_tp_fp32_remote = ray.remote(
    # 70B model with FP32 requires more memory
    memory=256 * 1024 * 1024 * 1024,  # 256 GB
    resources={"TPU": 16, "TPU-v6e-16-head": 1},
)(compute_sliding_logits_tp_fp32)
