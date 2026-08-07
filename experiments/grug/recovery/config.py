# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run configuration for the grug hang-recovery test.

Kept import-light (the only heavy import is ``GrugModelConfig``, which the
trainer needs anyway) and picklable so the supervisor can ship it to the trainer
subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from levanter.recovery.detection import DetectionConfig

from experiments.grug.base.model import GrugModelConfig

# Llama-3.1-8B-shaped dims scaled to ~10B by depth; synthetic data means only the
# shapes (and thus the sharded collective + snapshot volume) matter, not the vocab.
DEFAULT_VOCAB_SIZE = 128256


# Depth/width presets, GQA with 8 KV heads and head_dim 128 throughout. ``tiny``
# is a ~4-layer model whose collectives still cross the mesh, so it reproduces a
# multi-rack NCCL hang while compiling in seconds. 1B/3B/10B scale depth+width for
# the snapshot-volume and step-time numbers.
_MODEL_PRESETS = {
    "tiny": dict(hidden_dim=1024, intermediate_dim=4096, num_layers=4, num_heads=8, num_kv_heads=8, head_dim=128),
    "1b": dict(hidden_dim=2048, intermediate_dim=5632, num_layers=16, num_heads=16, num_kv_heads=8, head_dim=128),
    "3b": dict(hidden_dim=3072, intermediate_dim=8192, num_layers=28, num_heads=24, num_kv_heads=8, head_dim=128),
    "10b": dict(hidden_dim=4096, intermediate_dim=14336, num_layers=40, num_heads=32, num_kv_heads=8, head_dim=128),
}


def grug_model_preset(name: str, *, vocab_size: int = DEFAULT_VOCAB_SIZE, max_seq_len: int = 2048) -> GrugModelConfig:
    """Return a named dense grug model preset (``tiny`` / ``1b`` / ``3b`` / ``10b``)."""
    if name not in _MODEL_PRESETS:
        raise ValueError(f"unknown model preset {name!r}; choose from {sorted(_MODEL_PRESETS)}")
    return GrugModelConfig(vocab_size=vocab_size, max_seq_len=max_seq_len, **_MODEL_PRESETS[name])


def grug_10b_model(vocab_size: int = DEFAULT_VOCAB_SIZE, max_seq_len: int = 2048) -> GrugModelConfig:
    """A ~10B dense grug transformer (hidden 4096, 40 layers, GQA 32/8)."""
    return grug_model_preset("10b", vocab_size=vocab_size, max_seq_len=max_seq_len)


@dataclass(frozen=True)
class RecoveryRunConfig:
    """Everything the trainer subprocess needs to run one recovery test."""

    model: GrugModelConfig = field(default_factory=grug_10b_model)
    num_steps: int = 40
    batch_size: int = 8
    seq_len: int = 2048
    snapshot_every_steps: int = 4
    learning_rate: float = 1e-4
    seed: int = 0

    # Mesh: default (replica=1, expert=1, model=model_axis) so params shard FSDP
    # over the data axis (all local GPUs) with optional tensor-parallel width.
    replica_axis_size: int | None = None
    model_axis_size: int = 1

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    # Fallback ProgressWatchdog (fixed timeout — no EMA, unsafe under PGLE).
    watchdog_step_timeout_seconds: float = 120.0
    watchdog_startup_grace_seconds: float = 600.0

    log_every: int = 1
