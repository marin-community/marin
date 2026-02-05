# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model size and resource presets for Kelp tree diffusion.

Provides pre-configured model sizes targeting different compute environments.
"""

from dataclasses import dataclass

from experiments.kelp.model.config import TreeDiffusionConfig
from fray.cluster import ResourceConfig


@dataclass(frozen=True)
class ModelPreset:
    """A preset combining model config with training resources."""

    name: str
    """Human-readable name."""

    config: TreeDiffusionConfig
    """Model configuration."""

    resource: ResourceConfig
    """Compute resources for training."""

    batch_size: int
    """Global batch size."""

    learning_rate: float
    """Base learning rate."""

    description: str = ""
    """Description of use case."""


# Default vocab size (LLaMA-3 tokenizer)
DEFAULT_VOCAB_SIZE = 128256


def toy_preset() -> ModelPreset:
    """Tiny preset for unit testing (~1M params)."""
    return ModelPreset(
        name="toy",
        config=TreeDiffusionConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=64,
            intermediate_dim=256,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=128,
            num_diffusion_steps=10,
        ),
        resource=ResourceConfig.with_cpu(cpu=2),
        batch_size=2,
        learning_rate=1e-3,
        description="Toy model for testing infrastructure",
    )


def overnight_cpu_preset() -> ModelPreset:
    """Preset optimized for overnight CPU training (~10M params).

    Designed to complete ~30k+ steps in 8 hours on a laptop CPU.
    """
    return ModelPreset(
        name="overnight_cpu",
        config=TreeDiffusionConfig(
            vocab_size=256,  # Byte-level tokenizer for faster training
            hidden_dim=256,
            intermediate_dim=1024,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=512,
            num_diffusion_steps=50,
        ),
        resource=ResourceConfig.with_cpu(cpu=8),
        batch_size=16,
        learning_rate=1e-3,
        description="For overnight CPU training runs",
    )


def laptop_preset() -> ModelPreset:
    """Small preset for laptop development (~125M params)."""
    return ModelPreset(
        name="laptop",
        config=TreeDiffusionConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=512,
            intermediate_dim=2048,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            max_seq_len=1024,
            num_diffusion_steps=100,
        ),
        resource=ResourceConfig.with_cpu(cpu=8),
        batch_size=4,
        learning_rate=3e-4,
        description="For CPU/laptop iteration",
    )


def single_gpu_preset() -> ModelPreset:
    """Medium preset for single GPU (~300M params)."""
    return ModelPreset(
        name="single_gpu",
        config=TreeDiffusionConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=768,
            intermediate_dim=3072,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            max_seq_len=2048,
            num_diffusion_steps=100,
        ),
        resource=ResourceConfig.with_gpu("a100-40gb", count=1),
        batch_size=16,
        learning_rate=3e-4,
        description="For single A100 training",
    )


def tpu_v4_8_preset() -> ModelPreset:
    """Large preset for v4-8 TPU (~1B params)."""
    return ModelPreset(
        name="tpu_v4_8",
        config=TreeDiffusionConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=2048,
            intermediate_dim=8192,
            num_layers=24,
            num_heads=16,
            num_kv_heads=16,
            max_seq_len=4096,
            num_diffusion_steps=100,
        ),
        resource=ResourceConfig.with_tpu("v4-8"),
        batch_size=64,
        learning_rate=1e-4,
        description="For v4-8 TPU pod training",
    )


def tpu_v5p_8_preset() -> ModelPreset:
    """8B preset for v5p-8 TPU pod (~8B params, matching Marin 8b)."""
    return ModelPreset(
        name="tpu_v5p_8",
        config=TreeDiffusionConfig(
            vocab_size=DEFAULT_VOCAB_SIZE,
            hidden_dim=4096,
            intermediate_dim=14336,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=8192,
            num_diffusion_steps=100,
        ),
        resource=ResourceConfig.with_tpu("v5p-8"),
        batch_size=128,
        learning_rate=5e-5,
        description="For v5p-8 TPU pod, 8B scale",
    )


PRESETS = {
    "toy": toy_preset,
    "overnight_cpu": overnight_cpu_preset,
    "laptop": laptop_preset,
    "single_gpu": single_gpu_preset,
    "tpu_v4_8": tpu_v4_8_preset,
    "tpu_v5p_8": tpu_v5p_8_preset,
}


def get_preset(name: str) -> ModelPreset:
    """Get a preset by name.

    Args:
        name: Preset name.

    Returns:
        ModelPreset instance.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]()
