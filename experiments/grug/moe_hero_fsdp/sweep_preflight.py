# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace every arm of a sweep wave on CPU before it reaches a rack.

Iris ships the local worktree, so one wiring mistake kills all four arms at once, roughly 20
minutes of four racks. This traces ``next_token_loss`` under each arm's launcher flags at a small
shape, on CPU, through :func:`apply_hero_overrides` -- the same override path the launcher uses.

``jax.eval_shape`` never allocates and never launches a kernel, so it cannot catch a numerical or
kernel-level fault. What it does cover is the Python: attribute names, config plumbing, remat
policies, and partition specs, which is where every failure so far has been.

Usage
-----
    uv run python -m experiments.grug.moe_hero_fsdp.sweep_preflight <wave>
"""

import dataclasses
import os
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax import Axis
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_fsdp.heuristic import build_hero_configs
from experiments.grug.moe_hero_fsdp.launch import HeroOverrides, apply_hero_overrides
from experiments.grug.moe_hero_fsdp.sweep import WAVES
from experiments.grug.moe_hero_fsdp.train import BATCH_AXES

# Small enough to trace in about a second, large enough to keep every structural knob meaningful:
# expert_chunks up to 8 divides num_experts, and the two GatedNorm sites keep their rank.
PREFLIGHT_OVERRIDES = {
    "vocab_size": 512,
    "hidden_dim": 256,
    "intermediate_dim": 128,
    "shared_expert_intermediate_dim": 128,
    "num_experts": 8,
    "num_layers": 2,
    "num_heads": 8,
    "num_kv_heads": 4,
    "local_kv_heads": 4,
    "global_kv_heads": 2,
    "head_dim": 32,
    "max_seq_len": 256,
    "sliding_window": 64,
    # The hero's `sonic_cute` MoE and `gpu_fa4_cute` attention are SM100 CUTLASS kernels with no
    # CPU lowering. Their portable equivalents consume the same config fields, including
    # `interleave_before_gather`, so the knob plumbing is still exercised.
    "moe_implementation": "scatter",
    "attention_implementation": "reference",
    # `expert_chunks` is rejected outside `sonic_cute`, so chunk-count arms trace at the default
    # chunking. Their flag plumbing is covered by the launcher's own `click.IntRange`.
    "expert_chunks": 1,
}
# `data` is the only axis a preflight needs to be non-trivial: it carries FSDP.
MESH_AXIS_SIZES = {"replica_dcn": 1, "data": 2, "expert": 1, "model": 1}
PREFLIGHT_BATCH = 2


def preflight_model(arm):
    """The hero config at preflight shape with ``arm``'s launcher flags applied."""
    model, _ = build_hero_configs(num_train_steps=20, batch_size=1024)
    flags = {}
    args = list(arm.args)
    while args:
        flag = args.pop(0).removeprefix("--").replace("-", "_")
        if flag == "interleave_before_gather":
            flags[flag] = True
            continue
        flags[flag] = args.pop(0)
    if "expert_chunks" in flags:
        flags["expert_chunks"] = int(flags["expert_chunks"])
    if "ce_b_block_size" in flags:
        flags["ce_b_block_size"] = int(flags["ce_b_block_size"])
    # Arm flags first, then the portable-backend shape, so a knob the CPU path rejects is dropped
    # rather than fought over. `HeroOverrides` rejects a flag name the launcher would not accept.
    return dataclasses.replace(apply_hero_overrides(model, HeroOverrides(**flags)), **PREFLIGHT_OVERRIDES)


def trace(cfg):
    """Abstractly evaluate the loss and its gradient; raises on any wiring error."""
    mesh = jax.make_mesh(
        tuple(MESH_AXIS_SIZES.values()), tuple(MESH_AXIS_SIZES), axis_types=(AxisType.Explicit,) * len(MESH_AXIS_SIZES)
    )
    # The trainer feeds a batch already sharded over BATCH_AXES. Without that the model's explicit
    # shardings disagree between `lax.cond` branches, which is a harness artifact, not a defect.
    batch = NamedSharding(mesh, P(BATCH_AXES, None))
    tokens = jax.ShapeDtypeStruct((PREFLIGHT_BATCH, cfg.max_seq_len), jnp.int32, sharding=batch)
    weights = jax.ShapeDtypeStruct((PREFLIGHT_BATCH, cfg.max_seq_len), jnp.float32, sharding=batch)

    def loss(token_ids, loss_weight):
        model = cfg.build(Axis("vocab", cfg.vocab_size), key=jax.random.PRNGKey(0))
        return eqx.filter_grad(lambda m: m.next_token_loss(token_ids, loss_weight))(model)

    with jax.set_mesh(mesh):
        jax.eval_shape(loss, tokens, weights)


def check_xla_flags(arm):
    """Raise ``ValueError`` if this XLA build will not accept the arm's ``XLA_FLAGS``.

    XLA aborts the process on a bad flag, so a stale name or an ill-typed value takes down a whole
    16-node gang minutes into startup. The arm's exact flag string goes to XLA's own parser in a CPU
    subprocess, and the verdict is the child's exit status: unknown names and unparsable values abort
    through different messages, and both kill a rack.
    """
    flags = arm.env.get("XLA_FLAGS")
    if not flags:
        return
    probe = subprocess.run(
        [sys.executable, "-c", "import jax; jax.devices()"],
        env={**os.environ, "XLA_FLAGS": flags, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    if probe.returncode:
        raise ValueError(f"XLA rejected the flags: {probe.stderr.strip().splitlines()[-1]}")


def main(wave):
    missing = set(BATCH_AXES) - set(MESH_AXIS_SIZES)
    assert not missing, f"mesh axes drifted from the trainer: {missing}"
    failures = 0
    for arm in WAVES[wave]:
        try:
            check_xla_flags(arm)
            trace(preflight_model(arm))
        except Exception as exc:
            failures += 1
            print(f"{arm.tag:12s} FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}")
            continue
        print(f"{arm.tag:12s} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
