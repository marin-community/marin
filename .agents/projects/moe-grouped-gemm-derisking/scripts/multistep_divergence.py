#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-step gradient and optimizer-state agreement between two expert-MLP kernels.

Trains the same small MoE stack twice from one initialization, once per expert-MLP
implementation, on identical batches, and reports at each logged step the loss gap and the
maximum relative divergence of the parameters and of the Adam moments. Everything except the
expert MLP kernel is shared, and the kernel is exactly the `_ExpertMlp` callable the ragged EP
backend uses (`_cute_expert_mlp` or `_ragged_dot_expert_mlp`), driven through the same sorted
buffer and group-size bookkeeping.

This is a single-device harness. It answers whether the two kernels' rounding differences
compound over hundreds of steps; it does not exercise the transport. For a hero-shape answer
run two arms of `experiments/grug/moe_hero_ep/small_scale_abl_launch.py --flavor ragged` with
`_select_expert_mlp` pinned to each kernel and diff the W&B loss curves.

    uv run python multistep_divergence.py --steps 300 --impl-a quack --impl-b ragged_dot
    uv run python multistep_divergence.py --steps 20 --impl-a ragged_dot --impl-b ragged_dot   # harness self-check
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

os.environ.setdefault("RAGGED_DOT_IMPL", "xla")

from levanter.grug._moe.ep_ragged_all_to_all import (
    _quack_grouped_gemm_available,
    _ragged_dot_expert_mlp,
)


@dataclass
class Config:
    tokens: int = 2048
    hidden: int = 256
    intermediate: int = 384
    experts: int = 8
    topk: int = 2
    layers: int = 4
    vocab: int = 512
    capacity_factor: float = 1.15
    lr: float = 1e-3
    steps: int = 300
    log_every: int = 10
    seed: int = 0


def _impl(name: str):
    if name == "ragged_dot":
        return _ragged_dot_expert_mlp
    if name == "quack":
        if not _quack_grouped_gemm_available():
            raise SystemExit("quack needs an SM100 GPU with the gpu extra installed")
        from levanter.grug._moe.ep_ragged_all_to_all import _cute_expert_mlp  # noqa: PLC0415

        return _cute_expert_mlp
    raise ValueError(name)


def _init(cfg: Config, key):
    keys = jax.random.split(key, 2 + cfg.layers)
    params = {
        "embed": jax.random.normal(keys[0], (cfg.vocab, cfg.hidden)) * 0.02,
        "router": [jax.random.normal(keys[2 + i], (cfg.hidden, cfg.experts)) * 0.02 for i in range(cfg.layers)],
        "w13": [],
        "w2": [],
    }
    for i in range(cfg.layers):
        k13, k2 = jax.random.split(jax.random.fold_in(keys[1], i))
        params["w13"].append(
            jax.random.normal(k13, (cfg.experts, cfg.hidden, 2 * cfg.intermediate)) / np.sqrt(cfg.hidden)
        )
        params["w2"].append(
            jax.random.normal(k2, (cfg.experts, cfg.intermediate, cfg.hidden)) / np.sqrt(cfg.intermediate)
        )
    return params


def _moe_layer(expert_mlp, cfg: Config, x, router, w13, w2):
    """Top-k routed MoE over a sorted, capacity-limited buffer, the way the EP backend lays it out."""
    tokens = x.shape[0]
    logits = x @ router
    weights, selected = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), cfg.topk)
    flat = selected.reshape(-1)
    order = jnp.argsort(flat)
    group_sizes = jnp.bincount(flat, length=cfg.experts).astype(jnp.int32)
    capacity = int(np.ceil(cfg.capacity_factor * tokens * cfg.topk))
    capacity = max(capacity, cfg.experts)
    assignments = tokens * cfg.topk
    # Sorted assignments beyond capacity are dropped, mirroring the receiver buffer. The buffer is
    # at least as large as the assignment count here, so the slice below never truncates rows.
    capacity = max(capacity, assignments)
    sorted_rows = x[(order // cfg.topk)]
    buffer = jnp.pad(sorted_rows, ((0, capacity - assignments), (0, 0)))
    active = jnp.minimum(group_sizes, jnp.maximum(capacity - (jnp.cumsum(group_sizes) - group_sizes), 0))
    physical = active.at[-1].add(capacity - jnp.sum(active))
    out_buffer = expert_mlp(buffer, w13.astype(x.dtype), w2.astype(x.dtype), physical, active, jax.nn.silu)
    valid = jnp.arange(assignments) < jnp.sum(active)
    gathered = (
        jnp.zeros((assignments, cfg.hidden), jnp.float32)
        .at[order]
        .set(jnp.where(valid[:, None], out_buffer[:assignments].astype(jnp.float32), 0.0))
    )
    return jnp.sum(gathered.reshape(tokens, cfg.topk, cfg.hidden) * weights[:, :, None], axis=1)


def _loss_fn(expert_mlp, cfg: Config, params, batch):
    ids, targets = batch
    h = params["embed"][ids].astype(jnp.bfloat16)
    for i in range(cfg.layers):
        h = h + _moe_layer(
            expert_mlp, cfg, h, params["router"][i].astype(jnp.bfloat16), params["w13"][i], params["w2"][i]
        ).astype(jnp.bfloat16)
    logits = h.astype(jnp.float32) @ params["embed"].T
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def _rel_div(a, b) -> float:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    worst = 0.0
    for x, y in zip(la, lb, strict=True):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        scale = max(np.abs(y).max(), 1e-12)
        worst = max(worst, float(np.abs(x - y).max() / scale))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl-a", default="quack")
    parser.add_argument("--impl-b", default="ragged_dot")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    cfg = Config(steps=args.steps, tokens=args.tokens, hidden=args.hidden)

    key = jax.random.key(cfg.seed)
    params0 = _init(cfg, key)
    opt = optax.adam(cfg.lr)
    arms = {}
    for name in (args.impl_a, args.impl_b):
        loss = functools.partial(_loss_fn, _impl(name), cfg)

        @jax.jit
        def step(params, opt_state, batch, loss=loss):
            value, grads = jax.value_and_grad(loss)(params, batch)
            updates, opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, value

        arms[name] = dict(step=step, params=params0, opt_state=opt.init(params0))

    rows = []
    a, b = args.impl_a, args.impl_b
    for i in range(cfg.steps):
        k = jax.random.fold_in(key, 1 + i)
        ids = jax.random.randint(k, (cfg.tokens,), 0, cfg.vocab)
        targets = jnp.roll(ids, -1)
        for arm in arms.values():
            arm["params"], arm["opt_state"], arm["loss"] = arm["step"](arm["params"], arm["opt_state"], (ids, targets))
        if i % cfg.log_every == 0 or i == cfg.steps - 1:
            row = dict(
                step=i,
                loss_a=float(arms[a]["loss"]),
                loss_b=float(arms[b]["loss"]),
                param_rel_div=_rel_div(arms[a]["params"], arms[b]["params"]),
                opt_state_rel_div=_rel_div(arms[a]["opt_state"], arms[b]["opt_state"]),
            )
            rows.append(row)
            gap = abs(row["loss_a"] - row["loss_b"])
            print(
                f"step {i:5d} loss {row['loss_a']:.5f} vs {row['loss_b']:.5f} (|d|={gap:.2e}) "
                f"params rel-div {row['param_rel_div']:.2e} adam-state rel-div {row['opt_state_rel_div']:.2e}"
            )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
