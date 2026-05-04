# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log fine-grained activations from checkpoints across training.

Captures per-sublayer residual stream, router logits, per-expert MLP outputs,
and shared expert output. Runs at every 1k step checkpoint for d512 and d1024.

One Fray job per model size, loops over all checkpoints sequentially.
Pinned to us-central1 to avoid cross-region GCS writes.

Usage:
    python -m experiments.grug.moe.log_activations
"""

import dataclasses
import io
import json
from dataclasses import dataclass, field

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


def _expert_mlp(x, w_gate_up, w_down):
    """Run a single expert's SwiGLU MLP."""
    i = w_gate_up.shape[-1] // 2
    hidden = jnp.einsum("...d,di->...i", x, w_gate_up)
    gate, up = hidden[..., :i], hidden[..., i:]
    return jnp.einsum("...i,id->...d", jax.nn.silu(gate) * up, w_down)


def _make_forward_fn(model: Transformer):
    """Build a JIT-compiled forward pass that returns all intermediate activations."""
    cfg = model.config

    @jax.jit
    def forward(token_ids):
        from einops import rearrange
        from jax.sharding import PartitionSpec as P
        from jax.sharding import reshard

        from experiments.grug.moe.model import _batch_spec

        batch_spec = _batch_spec()

        hidden = model.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = model.embed_gated_norm(model.embed_norm(hidden))

        residual_snapshots = [hidden]
        router_logits_all = []
        selected_experts_all = []
        combine_weights_all = []
        per_expert_outputs_all = []
        shared_outputs_all = []

        segment_ids = None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

        k = cfg.num_experts_per_token

        for i, block in enumerate(model.blocks):
            layer_mask = long_mask if i % 4 == 3 else short_mask

            attn_in = block.attn_gated_norm(block.rms_attn(hidden))
            hidden = hidden + block.attn(attn_in, layer_mask)
            residual_snapshots.append(hidden)

            mlp_in = block.mlp_gated_norm(block.rms_mlp(hidden))

            x_flat = rearrange(mlp_in, "b s d -> (b s) d")
            router_logits = jnp.einsum("td,de->te", x_flat, reshard(block.mlp.router, P(None, None))).astype(jnp.float32)
            biased_logits = router_logits + jax.lax.stop_gradient(block.mlp.router_bias)
            _topk_logits, selected = jax.lax.top_k(biased_logits, k + 1)
            selected = selected[:, :-1]
            unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
            weights = jax.nn.sigmoid(unbiased_topk)

            router_logits_all.append(router_logits)
            selected_experts_all.append(selected)
            combine_weights_all.append(weights)

            # Per-expert outputs
            w_gate_up_local = reshard(block.mlp.w_gate_up, P(None, None, None))
            w_down_local = reshard(block.mlp.w_down, P(None, None, None))
            expert_outs = []
            for ki in range(k):
                expert_ids = selected[:, ki]
                w_gu = w_gate_up_local[expert_ids]
                w_d = w_down_local[expert_ids]
                expert_out = _expert_mlp(x_flat, w_gu, w_d)
                expert_outs.append(expert_out * weights[:, ki : ki + 1])
            per_expert_outputs_all.append(jnp.stack(expert_outs, axis=1))

            # Shared expert output
            if block.shared is not None:
                from levanter.utils.activation import ActivationFunctionEnum

                shared_out = block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
                shared_outputs_all.append(rearrange(shared_out, "b s d -> (b s) d"))
            else:
                shared_outputs_all.append(jnp.zeros_like(x_flat))

            # Normal forward
            mlp_out, _ = block.mlp(mlp_in)
            if block.shared is not None:
                from levanter.utils.activation import ActivationFunctionEnum

                mlp_out = mlp_out + block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
            hidden = hidden + mlp_out
            residual_snapshots.append(hidden)

        hidden = model.final_gated_norm(model.final_norm(hidden))
        residual_snapshots.append(hidden)

        return (
            residual_snapshots,
            router_logits_all,
            selected_experts_all,
            combine_weights_all,
            per_expert_outputs_all,
            shared_outputs_all,
        )

    return forward


def _forward_with_activations(forward_fn, token_ids):
    """Run JIT'd forward pass and collect activations as numpy arrays."""
    results = forward_fn(token_ids)
    residuals, router_logits, selected, weights, expert_outs, shared_outs = results
    return {
        "residual_stream": np.stack([np.array(s) for s in residuals], axis=0),
        "router_logits": np.stack([np.array(r) for r in router_logits], axis=0),
        "selected_experts": np.stack([np.array(s) for s in selected], axis=0),
        "combine_weights": np.stack([np.array(w) for w in weights], axis=0),
        "per_expert_outputs": np.stack([np.array(e) for e in expert_outs], axis=0),
        "shared_outputs": np.stack([np.array(s) for s in shared_outs], axis=0),
    }


def _save_to_gcs(activations, output_path, token_ids, text, num_layers):
    """Save all activations to GCS."""
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
    }
    for key in activations:
        metadata[f"{key}_shape"] = list(activations[key].shape)

    with fs.open(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for key, arr in activations.items():
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        with fs.open(f"{output_path}/{key}.npy", "wb") as f:
            f.write(buf.read())
        print(f"Saved {key}: shape={arr.shape}, dtype={arr.dtype}")

    buf = io.BytesIO()
    np.save(buf, token_ids)
    buf.seek(0)
    with fs.open(f"{output_path}/token_ids.npy", "wb") as f:
        f.write(buf.read())

    print(f"Saved to {output_path}")


def _get_nemotron_tokens(max_tokens=100):
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


def _run_and_save(forward_fn, model_cfg, token_ids, text, output_path):
    """Run forward pass and save activations."""
    seq_len = len(token_ids)
    padded_len = ((seq_len + 127) // 128) * 128
    padded_ids = np.zeros(padded_len, dtype=np.int32)
    padded_ids[:seq_len] = token_ids

    num_devices = len(jax.devices())
    batch_ids = np.tile(padded_ids, (num_devices, 1))
    token_ids_jax = jnp.array(batch_ids, dtype=jnp.int32)
    activations = _forward_with_activations(forward_fn, token_ids_jax)

    for key in activations:
        arr = activations[key]
        if arr.ndim == 4 and arr.shape[1] == num_devices:
            activations[key] = arr[:, 0, :seq_len, :]
        elif arr.ndim == 5 and arr.shape[1] == num_devices * padded_len:
            activations[key] = arr[:, :seq_len, :, :]
        elif arr.ndim == 3:
            if arr.shape[1] == num_devices * padded_len:
                activations[key] = arr[:, :seq_len, :]
            elif arr.shape[1] == num_devices:
                activations[key] = arr[:, 0, :seq_len]

    _save_to_gcs(activations, output_path, token_ids, text, num_layers=model_cfg.num_layers)


@dataclass(frozen=True)
class LogActivationsSweepConfig:
    """Config for logging activations across multiple checkpoints for one model size."""

    checkpoint_base: str
    output_base: str
    hidden_dim: int
    budget: float
    steps: list[int] = field(default_factory=list)
    text: str | None = None
    max_tokens: int = 100
    resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig.with_tpu("v5p-8"))


def _run_sweep_local(config: LogActivationsSweepConfig) -> None:
    """Runs on TPU: loop over checkpoints, load each, forward pass, save."""
    import os

    from rigging.filesystem import marin_prefix
    from levanter.utils.mesh import create_mesh_from_axis_specs

    output_base = os.path.join(marin_prefix(), config.output_base)
    print(f"Output base: {output_base}")

    model_cfg, _, _, _ = build_from_heuristic(budget=config.budget, hidden_dim=config.hidden_dim)
    print(f"Model: d={model_cfg.hidden_dim}, L={model_cfg.num_layers}, E={model_cfg.num_experts}")

    from jax.sharding import AxisType

    import haliax.partitioning

    num_devices = len(jax.devices())
    mesh = create_mesh_from_axis_specs(
        ici_axes={"data": num_devices, "replica": 1, "model": 1, "expert": 1},
        dcn_axes={},
        axis_types=tuple(AxisType.Explicit for _ in range(4)),
    )

    custom_ids = _tokenize(config.text, config.max_tokens) if config.text else None
    nemotron_ids, nemotron_text = _get_nemotron_tokens(config.max_tokens)

    from levanter.checkpoint import load_checkpoint

    for step in config.steps:
        checkpoint = f"{config.checkpoint_base}/step-{step}"
        output_path = f"{output_base}/step-{step}"
        print(f"\n{'='*60}")
        print(f"Processing step {step}: {checkpoint}")
        print(f"{'='*60}")

        with haliax.partitioning.set_mesh(mesh):
            mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
            model = Transformer.init(model_cfg, key=jax.random.PRNGKey(0))
            model = load_checkpoint(model, checkpoint, subpath="params", discover_latest=False, mesh=mesh)
            model = mp.cast_to_compute(model)

            forward_fn = _make_forward_fn(model)

            if custom_ids is not None:
                print(f"\n--- Custom text ({len(custom_ids)} tokens) ---")
                _run_and_save(forward_fn, model_cfg, custom_ids, config.text, f"{output_path}/custom")

            print(f"\n--- Nemotron ({len(nemotron_ids)} tokens) ---")
            _run_and_save(forward_fn, model_cfg, nemotron_ids, nemotron_text, f"{output_path}/nemotron")

    print(f"\nDone: processed {len(config.steps)} checkpoints.")


def run_sweep(config: LogActivationsSweepConfig) -> None:
    """Dispatch through Fray to get TPU access."""
    from experiments.grug.dispatch import dispatch_grug_training_run

    dispatch_grug_training_run(
        run_id=f"log-activations-d{config.hidden_dim}",
        config=config,
        local_entrypoint=_run_sweep_local,
        resources=config.resources,
        max_retries_failure=1,
    )


# --- Steps ---
_D512_BASE = "gs://marin-us-central1/grug/moe-v16-compute-opt-d512-2.19e+17-d3b963/checkpoints"
_D1024_BASE = "gs://marin-us-central1/grug/moe-v16-compute-opt-d1024-9.00e+18-d3bc52/checkpoints"
_TEXT = "The quick brown fox jumps over the lazy dog."
_RESOURCES = ResourceConfig.with_tpu("v5p-8")

d512_sweep = ExecutorStep(
    name="grug/activations-d512-sweep",
    fn=run_sweep,
    config=LogActivationsSweepConfig(
        checkpoint_base=versioned(_D512_BASE),
        output_base="grug/activations-v2/d512",
        hidden_dim=versioned(512),
        budget=versioned(2.19e17),
        steps=versioned([1000, 2000, 3000, 4000, 5000, 6000, 6387]),
        text=versioned(_TEXT),
        resources=versioned(_RESOURCES),
    ),
)

d1024_sweep = ExecutorStep(
    name="grug/activations-d1024-sweep",
    fn=run_sweep,
    config=LogActivationsSweepConfig(
        checkpoint_base=versioned(_D1024_BASE),
        output_base="grug/activations-v2/d1024",
        hidden_dim=versioned(1024),
        budget=versioned(9.00e18),
        steps=versioned([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 12649]),
        text=versioned(_TEXT),
        resources=versioned(_RESOURCES),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[d512_sweep, d1024_sweep],
        description="Log fine-grained activations across training: d512 + d1024.",
    )
