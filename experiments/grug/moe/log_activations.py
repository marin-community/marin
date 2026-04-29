# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log activations from a checkpoint: residual stream after attn and MLP,
plus router logits for each token. Saves to GCS as numpy arrays.

Usage:
    python -m experiments.grug.moe.log_activations \
        --checkpoint gs://marin-us-central1/grug/moe-v16-compute-opt-d1024-9.00e+18-d3bc52/checkpoints/step-12648 \
        --output gs://marin-us-central1/grug/activations/d1024-step12648 \
        --text "The quick brown fox jumps over the lazy dog."
"""

import argparse
import io
import json

import gcsfs
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig, Transformer, _batch_spec
from levanter.grug.attention import AttentionMask


def _tokenize(text: str, max_tokens: int = 100) -> np.ndarray:
    """Tokenize text with llama3 tokenizer, truncate to max_tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return np.array(ids, dtype=np.int32)


def _forward_with_activations(
    model: Transformer,
    token_ids: jax.Array,
) -> dict[str, np.ndarray]:
    """Run forward pass, capturing activations at every sublayer."""
    cfg = model.config
    batch_spec = _batch_spec()

    hidden = model.token_embed.at[token_ids].get(out_sharding=batch_spec)
    hidden = model.embed_gated_norm(model.embed_norm(hidden))

    activations = {"embed": np.array(hidden)}

    segment_ids = None
    short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
    long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

    router_logits_all = []

    for i, block in enumerate(model.blocks):
        layer_mask = long_mask if i % 4 == 3 else short_mask

        # Attention sublayer
        attn_in = block.attn_gated_norm(block.rms_attn(hidden))
        hidden = hidden + block.attn(attn_in, layer_mask)
        activations[f"layer_{i}_post_attn"] = np.array(hidden)

        # MLP sublayer
        mlp_in = block.mlp_gated_norm(block.rms_mlp(hidden))

        # Capture router logits before MoE dispatch
        from einops import rearrange

        x_flat = rearrange(mlp_in, "b s d -> (b s) d")
        from jax.sharding import reshard

        router_logits = jnp.einsum(
            "td,de->te", x_flat, reshard(block.mlp.router, P(None, None))
        ).astype(jnp.float32)
        router_logits_all.append(np.array(router_logits))

        mlp_out, _ = block.mlp(mlp_in)
        if block.shared is not None:
            from levanter.utils.activation import ActivationFunctionEnum

            mlp_out = mlp_out + block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        hidden = hidden + mlp_out
        activations[f"layer_{i}_post_mlp"] = np.array(hidden)

    # Final norm
    hidden = model.final_gated_norm(model.final_norm(hidden))
    activations["final"] = np.array(hidden)

    # Stack router logits: (num_layers, num_tokens, num_experts)
    activations["router_logits"] = np.stack(router_logits_all, axis=0)

    return activations


def _save_to_gcs(activations: dict[str, np.ndarray], output_path: str, token_ids: np.ndarray, text: str):
    """Save activations as .npy files to GCS."""
    fs = gcsfs.GCSFileSystem()

    # Save metadata
    metadata = {
        "text": text,
        "num_tokens": len(token_ids),
        "token_ids": token_ids.tolist(),
        "keys": list(activations.keys()),
    }
    with fs.open(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save each activation as .npy
    for key, arr in activations.items():
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        with fs.open(f"{output_path}/{key}.npy", "wb") as f:
            f.write(buf.read())
        print(f"Saved {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Save token ids
    buf = io.BytesIO()
    np.save(buf, token_ids)
    buf.seek(0)
    with fs.open(f"{output_path}/token_ids.npy", "wb") as f:
        f.write(buf.read())

    print(f"\nAll activations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Log model activations to GCS")
    parser.add_argument("--checkpoint", required=True, help="GCS path to checkpoint step dir")
    parser.add_argument("--output", required=True, help="GCS path to save activations")
    parser.add_argument("--text", required=True, help="Input text to process")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Model hidden dim")
    parser.add_argument("--budget", type=float, required=True, help="Compute budget used for this model")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to process")
    args = parser.parse_args()

    # Tokenize
    token_ids = _tokenize(args.text, args.max_tokens)
    print(f"Tokenized {len(token_ids)} tokens")

    # Build model config
    model_cfg, _, _, _ = build_from_heuristic(budget=args.budget, hidden_dim=args.hidden_dim)
    print(f"Model: d={model_cfg.hidden_dim}, L={model_cfg.num_layers}, E={model_cfg.num_experts}")

    # Load checkpoint
    from levanter.utils.mesh import MeshConfig

    mesh_config = MeshConfig(axes={"expert": 1})
    with mesh_config.axis_resources():
        mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
        model = Transformer.init(model_cfg, key=jax.random.PRNGKey(0))

        # Load from checkpoint
        from experiments.grug.checkpointing import load_checkpoint

        model = load_checkpoint(model, args.checkpoint)
        model = mp.cast_to_compute(model)

        # Prepare input: (1, seq_len)
        token_ids_jax = jnp.array(token_ids[None, :], dtype=jnp.int32)

        # Forward with activations
        activations = _forward_with_activations(model, token_ids_jax)

    # Squeeze batch dim
    for key in activations:
        if activations[key].ndim >= 2 and activations[key].shape[0] == 1:
            activations[key] = activations[key].squeeze(0)

    # Save
    _save_to_gcs(activations, args.output, token_ids, args.text)


if __name__ == "__main__":
    main()
