# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of the MXFP8 routed-expert weight gradients on one GB200 node (EP over 4 GPUs).

Each ``--mode`` runs a small hero model's forward and backward in its own process (one NCCL symmetric-memory pool
per process) and saves the loss and gradients; ``--compare`` reports each mode against ``bf16``::

    for m in bf16 mxfp8 mxfp8_wgrad; do python -m experiments.grug.moe_hero_ep.check_mxfp8_expert_wgrad \
        --mode $m --out /tmp/wg; done
    python -m experiments.grug.moe_hero_ep.check_mxfp8_expert_wgrad --compare --out /tmp/wg

``mxfp8`` also prints the optimized-HLO fingerprint, to compare against a checkout without the weight-gradient
path (the flag-off program must be unchanged).
"""

import argparse
import dataclasses
import hashlib
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh

from experiments.grug.moe_hero_ep import model

_MODES = ("bf16", "mxfp8", "mxfp8_wgrad")


def _config(mode: str, ungated: bool) -> model.GrugModelConfig:
    width = 768 if ungated else 512
    cfg = model.GrugModelConfig(
        vocab_size=1024,
        hidden_dim=1024,
        intermediate_dim=width,
        shared_expert_intermediate_dim=width,
        num_shared_experts=1,
        num_experts=16,
        num_experts_per_token=4,
        latent_dim=512,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        max_seq_len=1024,
        sliding_window=512,
        capacity_factor=1.25,
        moe_implementation="ragged_all_to_all",
        ungated_relu2=ungated,
        fp8_recipe=model.Fp8Recipe.NONE if mode == "bf16" else model.Fp8Recipe.MXFP8_EXPERTS,
    )
    if mode == "mxfp8_wgrad":
        cfg = dataclasses.replace(cfg, mxfp8_expert_wgrad=True)
    return cfg


def _hlo_fingerprint(text: str) -> str:
    text = re.sub(r", metadata=\{[^}]*\}", "", text)
    text = re.sub(r"\s*//.*", "", text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run(mode: str, ungated: bool, out: str) -> None:
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 1, 1, 4, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    cfg = _config(mode, ungated)
    tokens = jax.random.randint(jax.random.key(1), (8, 1024), 0, cfg.vocab_size, dtype=jnp.int32)
    with set_mesh(mesh):
        params = model.Transformer.init(cfg, key=jax.random.key(0))

        def loss_fn(p):
            compute = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if jnp.issubdtype(a.dtype, jnp.floating) else a, p)
            return compute.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32), reduction="mean")

        step = jax.jit(jax.value_and_grad(loss_fn))
        if mode == "mxfp8":
            print(f"hlo fingerprint {_hlo_fingerprint(step.lower(params).compile().as_text())}", flush=True)
        loss, grads = step(params)
        leaves = jax.tree_util.tree_leaves_with_path(grads)
        arrays = {jax.tree_util.keystr(path): np.asarray(jax.device_get(g), np.float32) for path, g in leaves}
    print(f"{mode} loss {float(loss):.6f}", flush=True)
    np.savez(os.path.join(out, f"{mode}_{'relu2' if ungated else 'swiglu'}.npz"), loss=np.float32(loss), **arrays)


def compare(ungated: bool, out: str) -> None:
    name = "relu2" if ungated else "swiglu"
    ref = np.load(os.path.join(out, f"bf16_{name}.npz"))
    for mode in _MODES[1:]:
        got = np.load(os.path.join(out, f"{mode}_{name}.npz"))
        print(f"{name} {mode}: loss {float(got['loss']):.6f} vs bf16 {float(ref['loss']):.6f}")
        for key in ref.files:
            if key == "loss":
                continue
            a, b = got[key], ref[key]
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
            print(f"  {key}: rel L2 {rel:.3e}, max abs err {float(np.max(np.abs(a - b))):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=_MODES)
    parser.add_argument("--ungated", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.compare:
        compare(args.ungated, args.out)
    else:
        run(args.mode, args.ungated, args.out)


if __name__ == "__main__":
    main()
