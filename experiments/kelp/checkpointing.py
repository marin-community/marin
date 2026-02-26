# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Checkpoint save/load for Kelp tree diffusion models.

Checkpoints are stored as directories containing:
- config.json: Model configuration (TreeDiffusionConfig as JSON)
- params.pkl: Model parameters as pickled JAX arrays
"""

import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from levanter.grug.attention import RotaryConfig

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tree.edit_model import EditModelParams

logger = logging.getLogger(__name__)


def save_checkpoint(params: EditModelParams, model_config: TreeDiffusionConfig, ckpt_dir: Path) -> None:
    """Save model parameters and config to a checkpoint directory.

    Args:
        params: Model parameters to save.
        model_config: Model configuration to save.
        ckpt_dir: Directory to save checkpoint into (will be created).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(model_config)
    if hasattr(model_config.rope, "__dict__"):
        config_dict["rope"] = asdict(model_config.rope)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    params_np = jax.tree.map(lambda x: jnp.array(x), params)
    with open(ckpt_dir / "params.pkl", "wb") as f:
        pickle.dump(params_np, f)

    logger.info(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path) -> tuple[EditModelParams, TreeDiffusionConfig]:
    """Load model parameters and config from a checkpoint directory.

    Args:
        ckpt_dir: Directory containing config.json and params.pkl.

    Returns:
        Tuple of (params, config).
    """
    config_path = ckpt_dir / "config.json"
    params_path = ckpt_dir / "params.pkl"

    if not config_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_dir}")

    with open(config_path) as f:
        config_dict = json.load(f)

    rope_val = config_dict.pop("rope", None)
    if isinstance(rope_val, dict):
        config_dict["rope"] = RotaryConfig(**rope_val)
    elif rope_val is not None:
        config_dict["rope"] = rope_val

    config = TreeDiffusionConfig(**config_dict)

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    logger.info(f"Loaded checkpoint from {ckpt_dir} (vocab_size={config.vocab_size})")
    return params, config


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the checkpoint with the highest step number.

    Args:
        checkpoint_dir: Parent directory containing step-XXXXXX subdirectories.

    Returns:
        Path to the best checkpoint, or None if no checkpoints found.
    """
    ckpt_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpt_dirs:
        return None
    return ckpt_dirs[-1]
