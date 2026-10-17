# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured local-device GDN-2 tiles; callers choose the untuned fallback."""

from dataclasses import dataclass, replace
from types import MappingProxyType

from .candidate.configs import KernelConfig, ScoreLayout


@dataclass(frozen=True)
class TuningBucket:
    batch_min: int
    batch_max: int
    sequence_length: int
    heads: int
    head_dim: int
    config: KernelConfig

    def matches(self, shape: tuple[int, int, int, int]) -> bool:
        batch, length, heads, dim = shape
        return (
            self.batch_min <= batch <= self.batch_max
            and length == self.sequence_length
            and heads == self.heads
            and dim == self.head_dim
        )


# Both endpoints passed forward/all-input-gradient gates in FP32 and BF16,
# then reproduced the winning tiles with a second seed (v5/v6 also reversed
# implementation order).
# B5-B7 interpolate between measured B4/B8; they are not hardware validations.
# Evidence: docs/reports/data/gdn2-tpu-20260922/README.md and linked JSONLs.
# JAX 0.11.1; scoped VMEM limit 50000 KiB on v5, 98304 KiB on v6, none on v4.
# v4-full1 confirms the forward+backward winner at both seeds. One forward
# timing row has periodic stalls; this choice does not promise stable tails.
_LONG_SEQUENCE = TuningBucket(4, 8, 4096, 6, 128, KernelConfig(bt=128, bc=64, mb=16))
TUNED_BLOCK_SIZES = MappingProxyType(
    {
        "TPU v5 lite": (_LONG_SEQUENCE,),
        "TPU v5": (_LONG_SEQUENCE,),
        "TPU v6 lite": (replace(_LONG_SEQUENCE, config=KernelConfig(bt=128, bc=64, mb=32)),),
        "TPU v4": (
            replace(
                _LONG_SEQUENCE,
                config=KernelConfig(bt=128, bc=64, mb=16, score_layout=ScoreLayout.FEATURE_FIRST),
            ),
        ),
    }
)


def select_kernel_config(
    device_kind: str,
    dtype_name: str,
    shape: tuple[int, int, int, int],
    *,
    fallback: KernelConfig,
) -> KernelConfig:
    """Select tiles from measured endpoints and an interpolated batch bucket.

    Device names are JAX ``device_kind`` values. This lookup does not compile,
    benchmark, validate an unmeasured shape, or change the kernel backend.
    Unmatched inputs return the caller's explicit untuned configuration.
    """
    if dtype_name not in ("float32", "bfloat16"):
        return fallback
    for bucket in TUNED_BLOCK_SIZES.get(device_kind, ()):
        if bucket.matches(shape):
            return bucket.config
    return fallback
