# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log activations from a checkpoint: residual stream after attn and MLP,
plus router logits for each token. Saves to GCS as numpy arrays.

Usage:
    python -m experiments.grug.moe.log_activations
"""

import dataclasses
import io
import json
from dataclasses import dataclass

import gcsfs
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import Transformer
from levanter.grug.attention import AttentionMask


def _tokenize(text: str, max_tokens: int = 100) -> np.ndarray:
    """Tokenize text with llama3 tokenizer, truncate to max_tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    ids = tokenizer.encode(text, add_special_tokens=False)[: max_tokens - 1]
    ids = [tokenizer.bos_token_id, *ids]
    return np.array(ids, dtype=np.int32)


def _make_forward_fn(model: Transformer):
    """Build a JIT-compiled forward pass that returns all intermediate activations."""
    cfg = model.config

    @jax.jit
    def forward(token_ids):
        from einops import rearrange

        from experiments.grug.moe.model import _batch_spec

        batch_spec = _batch_spec()

        hidden = model.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = model.embed_gated_norm(model.embed_norm(hidden))

        residual_snapshots = [hidden]

        segment_ids = None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

        router_logits_all = []

        for i, block in enumerate(model.blocks):
            layer_mask = long_mask if i % 4 == 3 else short_mask

            attn_in = block.attn_gated_norm(block.rms_attn(hidden))
            hidden = hidden + block.attn(attn_in, layer_mask)
            residual_snapshots.append(hidden)

            mlp_in = block.mlp_gated_norm(block.rms_mlp(hidden))

            x_flat = rearrange(mlp_in, "b s d -> (b s) d")
            router_logits = jnp.einsum("td,de->te", x_flat, block.mlp.router).astype(jnp.float32)
            router_logits_all.append(router_logits)

            mlp_out, _ = block.mlp(mlp_in)
            if block.shared is not None:
                from levanter.utils.activation import ActivationFunctionEnum

                mlp_out = mlp_out + block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
            hidden = hidden + mlp_out
            residual_snapshots.append(hidden)

        hidden = model.final_gated_norm(model.final_norm(hidden))
        residual_snapshots.append(hidden)

        return residual_snapshots, router_logits_all

    return forward


def _forward_with_activations(
    forward_fn,
    token_ids: jax.Array,
) -> dict[str, np.ndarray]:
    """Run JIT'd forward pass and collect activations as numpy arrays."""
    residual_snapshots, router_logits_all = forward_fn(token_ids)

    return {
        "residual_stream": np.stack([np.array(s) for s in residual_snapshots], axis=0),
        "router_logits": np.stack([np.array(r) for r in router_logits_all], axis=0),
    }


def _save_to_gcs(
    activations: dict[str, np.ndarray], output_path: str, token_ids: np.ndarray, text: str, num_layers: int
):
    """Save activations to GCS: metadata + residual_stream + router_logits + token_ids."""
    fs = gcsfs.GCSFileSystem()

    residual_labels = ["embed"]
    for i in range(num_layers):
        residual_labels.append(f"layer_{i}_post_attn")
        residual_labels.append(f"layer_{i}_post_mlp")
    residual_labels.append("final")

    metadata = {
        "text": text,
        "num_tokens": len(token_ids),
        "token_ids": token_ids.tolist(),
        "num_layers": num_layers,
        "residual_labels": residual_labels,
        "residual_stream_shape": list(activations["residual_stream"].shape),
        "router_logits_shape": list(activations["router_logits"].shape),
    }
    with fs.open(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for key in ["residual_stream", "router_logits"]:
        buf = io.BytesIO()
        np.save(buf, activations[key])
        buf.seek(0)
        with fs.open(f"{output_path}/{key}.npy", "wb") as f:
            f.write(buf.read())
        print(f"Saved {key}: shape={activations[key].shape}, dtype={activations[key].dtype}")

    buf = io.BytesIO()
    np.save(buf, token_ids)
    buf.seek(0)
    with fs.open(f"{output_path}/token_ids.npy", "wb") as f:
        f.write(buf.read())

    print(f"\nSaved to {output_path} (4 files)")


def _get_nemotron_tokens(max_tokens: int = 100) -> tuple[np.ndarray, str]:
    """Get first max_tokens from the nemotron mix training data."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    import levanter.store as store

    nemotron_path = "marin-us-central1/tokenized/nemotron_cc/hq_actual-5af4cc"
    try:
        cache = store.TreeCache.load(f"gs://{nemotron_path}", mode="r")
        first_doc = cache[0]
        token_ids = np.array(first_doc["input_ids"][:max_tokens], dtype=np.int32)
        text = tokenizer.decode(token_ids)
        return token_ids, text
    except Exception as e:
        print(f"Warning: Could not load nemotron tokens ({e}), using fallback text")
        fallback = "The United States of America is a country primarily located in North America."
        return _tokenize(fallback, max_tokens), fallback


def _run_and_save(forward_fn, model_cfg, token_ids: np.ndarray, text: str, output_path: str):
    """Run forward pass and save activations."""
    seq_len = len(token_ids)
    # Splash attention requires seq_len to be a multiple of 128
    padded_len = ((seq_len + 127) // 128) * 128
    padded_ids = np.zeros(padded_len, dtype=np.int32)
    padded_ids[:seq_len] = token_ids

    # Replicate input across batch dim to match data-sharded mesh (batch must be divisible by num_devices)
    num_devices = len(jax.devices())
    batch_ids = np.tile(padded_ids, (num_devices, 1))
    token_ids_jax = jnp.array(batch_ids, dtype=jnp.int32)
    activations = _forward_with_activations(forward_fn, token_ids_jax)

    # Take first batch element and trim padding
    for key in activations:
        if activations[key].ndim == 4:
            activations[key] = activations[key][:, 0, :seq_len, :]
        elif activations[key].ndim == 3:
            activations[key] = activations[key][:, :seq_len, :]

    _save_to_gcs(activations, output_path, token_ids, text, num_layers=model_cfg.num_layers)


@dataclass(frozen=True)
class LogActivationsConfig:
    checkpoint: str
    output_path: str
    hidden_dim: int
    budget: float
    text: str | None = None
    max_tokens: int = 100
    resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig.with_tpu("v5p-8"))


def _run_log_activations_local(config: LogActivationsConfig) -> None:
    """Runs on TPU: load checkpoint, forward pass, save activations."""
    from levanter.utils.mesh import create_mesh_from_axis_specs

    model_cfg, _, _, _ = build_from_heuristic(budget=config.budget, hidden_dim=config.hidden_dim)
    print(f"Model: d={model_cfg.hidden_dim}, L={model_cfg.num_layers}, E={model_cfg.num_experts}")

    from jax.sharding import AxisType

    import haliax.partitioning

    # Match the training mesh exactly: all devices, expert=1, data absorbs rest
    num_devices = len(jax.devices())
    mesh = create_mesh_from_axis_specs(
        ici_axes={"data": num_devices, "replica": 1, "model": 1, "expert": 1},
        dcn_axes={},
        axis_types=tuple(AxisType.Explicit for _ in range(4)),
    )
    with haliax.partitioning.set_mesh(mesh):
        mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
        model = Transformer.init(model_cfg, key=jax.random.PRNGKey(0))

        from levanter.checkpoint import load_checkpoint

        model = load_checkpoint(
            model,
            config.checkpoint,
            subpath="params",
            discover_latest=False,
            mesh=mesh,
        )
        model = mp.cast_to_compute(model)

        # Build JIT'd forward pass — runs under the Explicit mesh like training
        forward_fn = _make_forward_fn(model)

        # 1. Custom text (if provided)
        if config.text:
            print("\n=== Custom text ===")
            custom_ids = _tokenize(config.text, config.max_tokens)
            print(f"Tokenized {len(custom_ids)} tokens")
            _run_and_save(forward_fn, model_cfg, custom_ids, config.text, f"{config.output_path}/custom")

        # 2. Nemotron mix first tokens
        print("\n=== Nemotron mix ===")
        nemotron_ids, nemotron_text = _get_nemotron_tokens(config.max_tokens)
        print(f"Loaded {len(nemotron_ids)} tokens from nemotron")
        _run_and_save(forward_fn, model_cfg, nemotron_ids, nemotron_text, f"{config.output_path}/nemotron")


def run_log_activations(config: LogActivationsConfig) -> None:
    """Dispatch activation logging through Fray to get TPU access."""
    from experiments.grug.dispatch import dispatch_grug_training_run

    dispatch_grug_training_run(
        run_id=f"log-activations-d{config.hidden_dim}",
        config=config,
        local_entrypoint=_run_log_activations_local,
        resources=config.resources,
        max_retries_failure=1,
    )


# --- Define the step for d512 compute-optimal checkpoint ---
_d512_checkpoint = "gs://marin-us-central1/grug/moe-v16-compute-opt-d512-2.19e+17-d3b963/checkpoints/step-6387"

log_activations_step = ExecutorStep(
    name="grug/activations-d512",
    fn=run_log_activations,
    config=LogActivationsConfig(
        checkpoint=versioned(_d512_checkpoint),
        output_path="gs://marin-us-central1/grug/activations/d512-step6387",
        hidden_dim=versioned(512),
        budget=versioned(2.19e17),
        text=versioned("The quick brown fox jumps over the lazy dog."),
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[log_activations_step],
        description="Log activations from trained MoE checkpoints.",
    )
