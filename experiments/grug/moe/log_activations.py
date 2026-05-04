# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log fine-grained activations from checkpoints across training.

Captures per-sublayer residual stream, router logits, per-expert MLP outputs,
and shared expert output. Runs at every 1k step checkpoint for d512 and d1024.

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


def _expert_mlp(x, w_gate_up, w_down):
    """Run a single expert's SwiGLU MLP: gate_up split, silu(gate)*up, then down."""
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

            # Router logits and top-k selection
            x_flat = rearrange(mlp_in, "b s d -> (b s) d")
            from jax.sharding import PartitionSpec as P
            from jax.sharding import reshard

            router_logits = jnp.einsum("td,de->te", x_flat, reshard(block.mlp.router, P(None, None))).astype(jnp.float32)
            biased_logits = router_logits + jax.lax.stop_gradient(block.mlp.router_bias)
            _topk_logits, selected = jax.lax.top_k(biased_logits, k + 1)
            selected = selected[:, :-1]  # (T, K)
            unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
            weights = jax.nn.sigmoid(unbiased_topk)  # (T, K)

            router_logits_all.append(router_logits)
            selected_experts_all.append(selected)
            combine_weights_all.append(weights)

            # Per-expert outputs: manually run each selected expert
            # Unshard expert weights to avoid duplicate axis in gather
            w_gate_up_local = reshard(block.mlp.w_gate_up, P(None, None, None))
            w_down_local = reshard(block.mlp.w_down, P(None, None, None))
            expert_outs = []
            for ki in range(k):
                expert_ids = selected[:, ki]  # (T,)
                w_gu = w_gate_up_local[expert_ids]  # (T, D, 2I)
                w_d = w_down_local[expert_ids]  # (T, I, D)
                expert_out = _expert_mlp(x_flat, w_gu, w_d)  # (T, D)
                expert_outs.append(expert_out * weights[:, ki : ki + 1])  # weighted
            per_expert_outputs_all.append(jnp.stack(expert_outs, axis=1))  # (T, K, D)

            # Shared expert output
            if block.shared is not None:
                from levanter.utils.activation import ActivationFunctionEnum

                shared_out = block.shared(mlp_in, activation=ActivationFunctionEnum.silu)
                shared_outputs_all.append(rearrange(shared_out, "b s d -> (b s) d"))
            else:
                shared_outputs_all.append(jnp.zeros_like(x_flat))

            # Continue with normal forward pass
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

    print(f"\nSaved to {output_path}")


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

    # Take first batch element and trim padding
    for key in activations:
        arr = activations[key]
        if arr.ndim == 4 and arr.shape[1] == num_devices:
            activations[key] = arr[:, 0, :seq_len, :]
        elif arr.ndim == 5 and arr.shape[1] == num_devices * padded_len:
            # per_expert_outputs: (L, B*S, K, D) -> trim to (L, seq_len, K, D)
            activations[key] = arr[:, :seq_len, :, :]
        elif arr.ndim == 3:
            if arr.shape[1] == num_devices * padded_len:
                # router_logits, selected, weights, shared: (L, B*S, ...) -> (L, seq_len, ...)
                activations[key] = arr[:, :seq_len, :]
            elif arr.shape[1] == num_devices:
                # residual with batch dim
                activations[key] = arr[:, 0, :seq_len]

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

        forward_fn = _make_forward_fn(model)

        if config.text:
            print("\n=== Custom text ===")
            custom_ids = _tokenize(config.text, config.max_tokens)
            print(f"Tokenized {len(custom_ids)} tokens")
            _run_and_save(forward_fn, model_cfg, custom_ids, config.text, f"{config.output_path}/custom")

        print("\n=== Nemotron mix ===")
        nemotron_ids, nemotron_text = _get_nemotron_tokens(config.max_tokens)
        print(f"Loaded {len(nemotron_ids)} tokens from nemotron")
        _run_and_save(forward_fn, model_cfg, nemotron_ids, nemotron_text, f"{config.output_path}/nemotron")


def run_log_activations(config: LogActivationsConfig) -> None:
    """Dispatch activation logging through Fray to get TPU access."""
    from experiments.grug.dispatch import dispatch_grug_training_run

    dispatch_grug_training_run(
        run_id=f"log-activations-d{config.hidden_dim}-{config.checkpoint.split('step-')[-1]}",
        config=config,
        local_entrypoint=_run_log_activations_local,
        resources=config.resources,
        max_retries_failure=1,
    )


# --- Generate steps for all checkpoints ---
_D512_BASE = "gs://marin-us-central1/grug/moe-v16-compute-opt-d512-2.19e+17-d3b963/checkpoints"
_D1024_BASE = "gs://marin-us-central1/grug/moe-v16-compute-opt-d1024-9.00e+18-d3bc52/checkpoints"

_D512_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 6387]
_D1024_STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 12649]

_TEXT = "The quick brown fox jumps over the lazy dog."

_all_steps: list[ExecutorStep] = []
for step in _D512_STEPS:
    _all_steps.append(
        ExecutorStep(
            name=f"grug/activations-d512-step{step}",
            fn=run_log_activations,
            config=LogActivationsConfig(
                checkpoint=versioned(f"{_D512_BASE}/step-{step}"),
                output_path=f"gs://marin-us-central1/grug/activations/d512-step{step}",
                hidden_dim=versioned(512),
                budget=versioned(2.19e17),
                text=versioned(_TEXT),
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            ),
        )
    )

for step in _D1024_STEPS:
    _all_steps.append(
        ExecutorStep(
            name=f"grug/activations-d1024-step{step}",
            fn=run_log_activations,
            config=LogActivationsConfig(
                checkpoint=versioned(f"{_D1024_BASE}/step-{step}"),
                output_path=f"gs://marin-us-central1/grug/activations/d1024-step{step}",
                hidden_dim=versioned(1024),
                budget=versioned(9.00e18),
                text=versioned(_TEXT),
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            ),
        )
    )

if __name__ == "__main__":
    executor_main(
        steps=_all_steps,
        description="Log fine-grained activations across training: d512 + d1024.",
    )
