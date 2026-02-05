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

"""AR-to-tree-diffusion weight transfer and adaptation.

Transfers weights from a pretrained AR LLM (like Marin 8b) to initialize
a tree diffusion model. Following DiffuGPT and DREAM approaches.

Key modifications:
1. Remove causal attention mask (use bidirectional attention)
2. Add timestep embeddings (initialized fresh)
3. All other weights copied directly
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import PRNGKeyArray

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import (
    TreeDiffusionAttentionParams,
    TreeDiffusionBlockParams,
    TreeDiffusionModelParams,
)
from experiments.kelp.transfer.config import ARToTreeDiffusionTransferConfig

logger = logging.getLogger(__name__)


def load_llama_weights(model_path: str) -> dict[str, Any]:
    """Load weights from a LLaMA-style HuggingFace model.

    Args:
        model_path: Path to HuggingFace model directory.

    Returns:
        Dictionary of weight tensors.
    """
    import fsspec
    from safetensors import safe_open

    fs = fsspec.filesystem(model_path.split(":")[0] if "://" in model_path else "file")

    weight_files = fs.glob(f"{model_path}/*.safetensors")
    if not weight_files:
        weight_files = fs.glob(f"{model_path}/model*.safetensors")

    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    weights = {}
    for weight_file in weight_files:
        logger.info(f"Loading weights from {weight_file}")
        with fs.open(weight_file, "rb") as f:
            with safe_open(f, framework="numpy") as sf:
                for key in sf.keys():
                    weights[key] = jnp.array(sf.get_tensor(key))

    return weights


def init_timestep_embeddings(
    num_steps: int,
    hidden_dim: int,
    method: str = "random",
    std: float = 0.02,
    key: PRNGKeyArray | None = None,
) -> jax.Array:
    """Initialize timestep embeddings.

    Args:
        num_steps: Number of diffusion steps.
        hidden_dim: Hidden dimension.
        method: Initialization method ('random', 'zeros', 'sinusoidal').
        std: Standard deviation for random init.
        key: PRNG key for random init.

    Returns:
        Timestep embedding array of shape (num_steps, hidden_dim).
    """
    if method == "zeros":
        return jnp.zeros((num_steps, hidden_dim))

    elif method == "sinusoidal":
        positions = jnp.arange(num_steps)[:, None]
        dim_indices = jnp.arange(hidden_dim)[None, :]
        div_term = 10000.0 ** (dim_indices / hidden_dim)
        embeddings = jnp.where(
            dim_indices % 2 == 0,
            jnp.sin(positions / div_term),
            jnp.cos(positions / div_term),
        )
        return embeddings

    elif method == "random":
        if key is None:
            key = random.PRNGKey(0)
        return std * random.truncated_normal(key, -3, 3, (num_steps, hidden_dim))

    else:
        raise ValueError(f"Unknown timestep embedding method: {method}")


def map_llama_key(key: str) -> tuple[str, str, int | None]:
    """Map a LLaMA weight key to tree diffusion component.

    Args:
        key: LLaMA weight key.

    Returns:
        Tuple of (component, subcomponent, layer_idx).
        component: 'embed', 'layers', 'norm', 'lm_head'
        subcomponent: specific weight name
        layer_idx: layer index (for 'layers' component)
    """
    if key == "model.embed_tokens.weight":
        return ("embed", "tokens", None)
    elif key == "lm_head.weight":
        return ("lm_head", "weight", None)
    elif key == "model.norm.weight":
        return ("norm", "weight", None)
    elif key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = int(parts[2])
        rest = ".".join(parts[3:])
        return ("layers", rest, layer_idx)
    else:
        return ("unknown", key, None)


def transfer_ar_to_tree_diffusion(
    ar_model_path: str,
    target_config: TreeDiffusionConfig,
    transfer_config: ARToTreeDiffusionTransferConfig | None = None,
    key: PRNGKeyArray | None = None,
) -> TreeDiffusionModelParams:
    """Transfer weights from AR model to tree diffusion model.

    Args:
        ar_model_path: Path to AR model.
        target_config: Target tree diffusion config.
        transfer_config: Transfer configuration.
        key: PRNG key.

    Returns:
        TreeDiffusionModelParams initialized from AR weights.
    """
    if transfer_config is None:
        transfer_config = ARToTreeDiffusionTransferConfig(source_model_path=ar_model_path)

    if key is None:
        key = random.PRNGKey(42)

    logger.info(f"Loading AR model from {ar_model_path}")
    ar_weights = load_llama_weights(ar_model_path)

    logger.info(f"Transferring to tree diffusion with {target_config.num_layers} layers")

    token_embed = ar_weights["model.embed_tokens.weight"]
    output_proj = ar_weights.get("lm_head.weight", token_embed).T
    final_norm = ar_weights["model.norm.weight"]

    key, time_key = random.split(key)
    timestep_embed = init_timestep_embeddings(
        target_config.num_diffusion_steps,
        target_config.hidden_dim,
        method=transfer_config.init_timestep_embed,
        std=transfer_config.timestep_embed_std,
        key=time_key,
    )

    head_dim = target_config.inferred_head_dim

    blocks = []
    for layer_idx in range(target_config.num_layers):
        prefix = f"model.layers.{layer_idx}"

        w_q = ar_weights[f"{prefix}.self_attn.q_proj.weight"].T
        w_k = ar_weights[f"{prefix}.self_attn.k_proj.weight"].T
        w_v = ar_weights[f"{prefix}.self_attn.v_proj.weight"].T
        w_o = ar_weights[f"{prefix}.self_attn.o_proj.weight"].T

        attn = TreeDiffusionAttentionParams(w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o)

        rms_attn = ar_weights[f"{prefix}.input_layernorm.weight"]
        rms_mlp = ar_weights[f"{prefix}.post_attention_layernorm.weight"]

        mlp_gate = ar_weights[f"{prefix}.mlp.gate_proj.weight"].T
        mlp_up = ar_weights[f"{prefix}.mlp.up_proj.weight"].T
        mlp_down = ar_weights[f"{prefix}.mlp.down_proj.weight"].T

        block = TreeDiffusionBlockParams(
            attn=attn,
            rms_attn=rms_attn,
            rms_mlp=rms_mlp,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
        )
        blocks.append(block)

    params = TreeDiffusionModelParams(
        token_embed=token_embed,
        timestep_embed=timestep_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )

    logger.info("Transfer complete!")
    log_param_stats(params)

    return params


def log_param_stats(params: TreeDiffusionModelParams) -> None:
    """Log statistics about transferred parameters."""
    total_params = 0

    token_params = params.token_embed.size
    total_params += token_params
    logger.info(f"  token_embed: {token_params:,} params")

    time_params = params.timestep_embed.size
    total_params += time_params
    logger.info(f"  timestep_embed: {time_params:,} params (fresh)")

    output_params = params.output_proj.size
    total_params += output_params
    logger.info(f"  output_proj: {output_params:,} params")

    norm_params = params.final_norm.size
    total_params += norm_params

    block_params = 0
    for block in params.blocks:
        block_params += block.attn.w_q.size
        block_params += block.attn.w_k.size
        block_params += block.attn.w_v.size
        block_params += block.attn.w_o.size
        block_params += block.mlp_gate.size
        block_params += block.mlp_up.size
        block_params += block.mlp_down.size
        block_params += block.rms_attn.size
        block_params += block.rms_mlp.size

    total_params += block_params
    logger.info(f"  blocks ({len(params.blocks)} layers): {block_params:,} params")

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Total parameters: {total_params / 1e9:.2f}B")


def verify_transfer(
    ar_model_path: str,
    transferred_params: TreeDiffusionModelParams,
    config: TreeDiffusionConfig,
) -> dict:
    """Verify that the transfer was successful.

    Args:
        ar_model_path: Path to original AR model.
        transferred_params: Transferred parameters.
        config: Target config.

    Returns:
        Dictionary with verification results.
    """
    ar_weights = load_llama_weights(ar_model_path)

    results = {
        "embed_match": jnp.allclose(transferred_params.token_embed, ar_weights["model.embed_tokens.weight"]),
        "norm_match": jnp.allclose(transferred_params.final_norm, ar_weights["model.norm.weight"]),
        "num_layers_match": len(transferred_params.blocks) == config.num_layers,
    }

    if transferred_params.blocks:
        first_block = transferred_params.blocks[0]
        results["first_layer_q_match"] = jnp.allclose(
            first_block.attn.w_q, ar_weights["model.layers.0.self_attn.q_proj.weight"].T
        )

    return results
