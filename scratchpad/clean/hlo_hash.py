# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Lower the fast_track train step on CPU (8 fake devices, single-process EP) and hash its StableHLO.
#
#   XLA_FLAGS=--xla_force_host_platform_device_count=8 python scratchpad/clean/hlo_hash.py OUT_DIR [CASE...]
#
# Cases: belay-d512 belay-d768 belay-d1024 belay-d1280 baseline-d512 dense-d512. The launcher's own argument
# parsing and build_config build the GrugRunConfig, so the hash covers launch.py, belay_ckpt1.py and train.py.
import hashlib
import json
import os
import re
import sys
import time

# QuACK's symmetric GEMM (cutlass) is GPU-only; on CPU stand in X @ X^T. The stub is the same before and after.
import types as _types
from pathlib import Path

import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from marin.execution.build_context import BuildContext, VersionCodex, build_context

_quack = _types.ModuleType("levanter.grug._moe.quack_symmetric_cute")
_quack.quack_symmetric_gemm = lambda x: jnp.einsum("...ik,...jk->...ij", x, x)
sys.modules["levanter.grug._moe.quack_symmetric_cute"] = _quack

from experiments.grug.fast_track import belay_ckpt1, belay_tracks, launch  # noqa: E402
from experiments.grug.fast_track import train as ft_train  # noqa: E402


class _FakeCtx:
    output_path = "/tmp/hlo_hash_out"
    is_fingerprint = True

    def runtime_arg(self, name):
        return None

    def artifact_path(self, item):
        return f"/tmp/hlo_hash_val/{item.name}"


_RESEARCH = json.loads((Path(__file__).parent / "research_cases.json").read_text())
# Active-research flags that no recorded run on this branch combines: the entropy-head skip and the KV-gate mix.
_SYNTH = {
    "synth-entropy": ["entropy_skip_layers=4,5", "extra_mlp_select=low", "extra_mlp_frac=0.3"],
    "synth-kv": [
        "attn_res_kv_layers=3,4,5",
        "attn_res_kv_source_layers=3",
        "lm_head_grad_probe=true",
        "lm_head_grad_probe_site=both",
        "value_embeds=gated_lambda",
        "attn_res_plain_sum=true",
    ],
}


def _train_step_fn(built):
    """``_build_train_step``'s train step: it returned ``(step, diagnostic_watch_step)`` before the cleanup."""
    return built[0] if isinstance(built, tuple) else built


def _args(case: str) -> list[str]:
    if case.startswith("run:"):
        return ["--run-id", "hlo-run", *_strip_run_id(_RESEARCH[case.removeprefix("run:")])]
    if case in _SYNTH:
        args = ["--run-id", f"hlo-{case}", *belay_ckpt1.recipe_args("d512")]
        for setting in _SYNTH[case]:
            args += ["--model-set", setting]
        return args
    kind, size = case.split("-", 1)
    base = ["--run-id", f"hlo-{case}"]
    if kind == "belay":
        return [*base, *belay_ckpt1.recipe_args(size)]
    if kind == "vanilla":
        return [*base, *belay_tracks.track_args(belay_tracks.Track.VANILLA, size)]
    if kind == "basicmoe":
        return [*base, *belay_tracks.track_args(belay_tracks.Track.BASIC_MOE, size)]
    if kind == "baseline":
        return [*base, "--size", size, "--version", "dev"]
    if kind == "dense":
        return [*base, "--size", size, "--dense", "--version", "dev"]
    raise ValueError(case)


# Settings of deleted fields that the recorded runs set to the field's default.
_DELETED_DEFAULT_SETTINGS = ("moe_drop_renorm=false",)


def _strip_run_id(argv: list[str]) -> list[str]:
    i = argv.index("--run-id")
    argv = argv[:i] + argv[i + 2 :]
    out = []
    for a in argv:
        if a in _DELETED_DEFAULT_SETTINGS and out and out[-1] == "--model-set":
            out.pop()
            continue
        out.append(a)
    return out


def run_config(case: str):
    cmd = launch.main
    ctx = cmd.make_context("launch", _args(case))
    params = dict(ctx.params)
    for k in ("default_version", "overrides", "do_run", "max_concurrent"):
        params.pop(k, None)
    version = _args(case)[_args(case).index("--version") + 1]
    with build_context(BuildContext(versions=VersionCodex(default=version, overrides={}))):
        step = cmd.callback.__wrapped__(**params)
    return step.build_config(_FakeCtx())


def lowered_text(case: str) -> str:
    config = run_config(case)
    trainer = config.trainer.trainer
    optimizer = config.optimizer.build(trainer.num_train_steps)
    train_step_fn = _train_step_fn(ft_train._build_train_step(config, optimizer))
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size, replica_axis_size=config.trainer.replica_axis_size
    )
    batch_size = trainer.train_batch_size
    seq = config.model.max_seq_len
    with set_mesh(mesh):
        state = jax.eval_shape(
            lambda key: ft_train.initial_state(config.model, optimizer=optimizer, mp=trainer.mp, key=key),
            jax.random.PRNGKey(0),
        )
        leaves, treedef = jax.tree.flatten(state)
        bs = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens = jax.ShapeDtypeStruct((batch_size, seq), jnp.int32, sharding=bs)
        weight = jax.ShapeDtypeStruct((batch_size, seq), jnp.float32, sharding=bs)
        seg = jax.ShapeDtypeStruct((batch_size, seq), jnp.int32, sharding=bs)
        batch = GrugLmExample(tokens=tokens, loss_weight=weight, attn_mask=AttentionMask.causal().with_segment_ids(seg))
        lowered = ft_train._flat_train_step(train_step_fn, treedef).lower(leaves, batch)
    return lowered.as_text()


def normalize(text: str) -> str:
    # Source locations (file:line) change with any edit; strip them so only the program is hashed.
    text = re.sub(r"#loc\d* = loc\(.*\)\n", "", text)
    text = re.sub(r" loc\(#loc\d*\)", "", text)
    text = re.sub(r"loc\([^)]*\)", "loc()", text)
    return text


def main() -> None:
    out_dir = sys.argv[1]
    cases = sys.argv[2:] or [
        "belay-d512",
        "belay-d768",
        "belay-d1024",
        "belay-d1280",
        "baseline-d512",
        "dense-d512",
        "vanilla-d512",
        "basicmoe-d512",
        *(f"run:{k}" for k in _RESEARCH),
        *_SYNTH,
    ]
    os.makedirs(out_dir, exist_ok=True)
    assert len(jax.devices()) == 8, jax.devices()
    for case in cases:
        start = time.time()
        text = normalize(lowered_text(case))
        with open(os.path.join(out_dir, f"{case}.mlir"), "w") as f:
            f.write(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"{case} {digest} {len(text)} chars {time.time() - start:.0f}s", flush=True)
        with open(os.path.join(out_dir, "hashes.txt"), "a") as f:
            f.write(f"{case} {digest}\n")


if __name__ == "__main__":
    main()
