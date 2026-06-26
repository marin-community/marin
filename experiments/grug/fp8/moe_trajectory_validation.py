# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GFP8-035: focused MoE training-trajectory validation — bf16 vs all-E4M3 f8 expert GEMMs.

The dynamic-range question ("does E4M3 have enough range for MoE gradients over a trajectory?")
is about *quantization* — casting expert operands and the output gradient to E4M3 — which is
bit-identical on any backend (the op accumulates in f32; only the E4M3 rounding matters). So we
answer it with a self-contained MoE student/teacher regression loop instead of the full grug model,
removing the attention/mesh/EP confounds (the full 1-GPU model trips a pre-existing XSA sharding
bug unrelated to f8 — bf16 fails identically).

Two arms differ *only* in the expert grouped-GEMM kernel, via the `_moe_mlp_local` dispatch:
    scatter      bf16 ragged_dot
    scatter_f8   all-E4M3, current/per-step per-tensor scaling (operands AND grads in E4M3)
Identical init, identical per-step data, identical optimizer. Any loss-curve divergence or NaN is
attributable to E4M3. Mixed precision mirrors the real recipe: fp32 master params, bf16 compute.

Run: python experiments/grug/fp8/moe_trajectory_validation.py [--steps N] [--layers K] ...
"""

import argparse
import dataclasses

import jax
import jax.numpy as jnp
import optax

from levanter.grug._moe.local import _moe_mlp_local


@dataclasses.dataclass(frozen=True)
class Config:
    dim: int = 256
    expert_dim: int = 512
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 4
    batch: int = 512
    steps: int = 4000
    lr: float = 3e-3
    seed: int = 0
    teacher_hidden: int = 1024
    compute_dtype: jnp.dtype = jnp.bfloat16


def _init_params(cfg: Config, key):
    """Student params (fp32 master). Each layer: router + concatenated gate/up (w13) + down (w2)."""
    keys = jax.random.split(key, cfg.num_layers * 3 + 2)
    params = {"layers": []}
    i = 0
    for _ in range(cfg.num_layers):
        wr = jax.random.normal(keys[i], (cfg.dim, cfg.num_experts)) * 0.02
        # w13 = [E, dim, 2*expert_dim] (gate||up), w2 = [E, expert_dim, dim]
        w13 = jax.random.normal(keys[i + 1], (cfg.num_experts, cfg.dim, 2 * cfg.expert_dim)) * (cfg.dim**-0.5)
        w2 = jax.random.normal(keys[i + 2], (cfg.num_experts, cfg.expert_dim, cfg.dim)) * (cfg.expert_dim**-0.5)
        params["layers"].append({"wr": wr, "w13": w13, "w2": w2})
        i += 3
    params["w_out"] = jax.random.normal(keys[i], (cfg.dim, cfg.dim)) * (cfg.dim**-0.5)
    return params


def _init_teacher(cfg: Config, key):
    """Fixed random 2-layer MLP teacher: gives a learnable target with realistic gradient structure."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (cfg.dim, cfg.teacher_hidden)) * (cfg.dim**-0.5),
        "w2": jax.random.normal(k2, (cfg.teacher_hidden, cfg.dim)) * (cfg.teacher_hidden**-0.5),
    }


def _teacher_forward(teacher, x):
    return jax.nn.gelu(x @ teacher["w1"]) @ teacher["w2"]


def _moe_layer(layer, x, cfg: Config, implementation):
    """One MoE block: top-k router -> expert MLP (via _moe_mlp_local) -> residual."""
    router_logits = (x @ layer["wr"]).astype(jnp.float32)
    _, selected = jax.lax.top_k(router_logits, cfg.top_k)
    combine = jax.nn.sigmoid(jnp.take_along_axis(router_logits, selected, axis=-1)).astype(x.dtype)
    out, _ = _moe_mlp_local(
        x,
        selected.astype(jnp.int32),
        combine,
        layer["w13"].astype(x.dtype),
        layer["w2"].astype(x.dtype),
        activation_fn=jax.nn.silu,
        num_experts=cfg.num_experts,
        implementation=implementation,
    )
    return x + out


def _student_forward(params, x, cfg: Config, implementation):
    h = x.astype(cfg.compute_dtype)
    for layer in params["layers"]:
        h = _moe_layer(layer, h, cfg, implementation)
    return h @ params["w_out"].astype(cfg.compute_dtype)


def _loss(params, teacher, x, cfg: Config, implementation):
    pred = _student_forward(params, x, cfg, implementation).astype(jnp.float32)
    target = _teacher_forward(teacher, x)
    return jnp.mean((pred - target) ** 2)


def run_arm(cfg: Config, implementation: str):
    key = jax.random.PRNGKey(cfg.seed)
    pkey, tkey = jax.random.split(key)
    params = _init_params(cfg, pkey)  # identical across arms (same seed)
    teacher = _init_teacher(cfg, tkey)

    schedule = optax.cosine_decay_schedule(cfg.lr, cfg.steps, alpha=0.1)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x):
        loss, grads = jax.value_and_grad(_loss)(params, teacher, x, cfg, implementation)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    first_nan_step = -1
    for step in range(cfg.steps):
        # Identical per-step data across arms: seed the batch by step index.
        x = jax.random.normal(jax.random.PRNGKey(1_000_000 + step), (cfg.batch, cfg.dim))
        params, opt_state, loss = train_step(params, opt_state, x)
        lv = float(loss)
        losses.append(lv)
        if first_nan_step < 0 and not jnp.isfinite(loss):
            first_nan_step = step
            break
        if step % max(1, cfg.steps // 20) == 0 or step == cfg.steps - 1:
            print(f"[{implementation}] step {step:5d} loss {lv:.6f}", flush=True)
    return losses, first_nan_step


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if f.type in (int, float):
            ap.add_argument(f"--{f.name}", type=f.type, default=getattr(Config, f.name))
    args = ap.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items()})
    print(f"config: {cfg}", flush=True)

    bf16_losses, bf16_nan = run_arm(cfg, "scatter")
    f8_losses, f8_nan = run_arm(cfg, "scatter_f8")

    n = min(len(bf16_losses), len(f8_losses))
    bf16_final = sum(bf16_losses[max(0, n - 50) : n]) / min(50, n)
    f8_final = sum(f8_losses[max(0, n - 50) : n]) / min(50, n)
    print("\n===== GFP8-035 trajectory verdict =====")
    print(f"steps completed: bf16={len(bf16_losses)} f8={len(f8_losses)}")
    print(f"bf16 NaN step: {bf16_nan}   f8 NaN step: {f8_nan}")
    print(f"final loss (mean last 50): bf16={bf16_final:.6f}  f8={f8_final:.6f}")
    print(f"relative gap (f8-bf16)/bf16: {(f8_final - bf16_final) / (bf16_final + 1e-12):.4%}")
    # Max pointwise relative gap along the trajectory.
    max_rel = max(abs(f - b) / (abs(b) + 1e-9) for b, f in zip(bf16_losses[:n], f8_losses[:n]))
    print(f"max pointwise rel gap over trajectory: {max_rel:.4%}")


if __name__ == "__main__":
    main()
