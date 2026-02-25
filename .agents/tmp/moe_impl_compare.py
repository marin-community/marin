#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import time
import types
import os
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec as P

BASELINE_COMMIT = "aab2354be"
BASELINE_PATH = "experiments/speedrun/new_grug_moe/baseline_scaled_sharded.py"
BASELINE_LOCAL_PATH = pathlib.Path(".agents/tmp/baseline_scaled_sharded_aab2354be.py")
OUR_MOE_PATH = pathlib.Path("lib/levanter/src/levanter/grug/grug_moe.py")


@dataclass(frozen=True)
class BenchCfg:
    batch: int = int(os.environ.get("BENCH_BATCH", "192"))
    seq: int = int(os.environ.get("BENCH_SEQ", "128"))
    hidden: int = int(os.environ.get("BENCH_HIDDEN", "768"))
    experts: int = int(os.environ.get("BENCH_EXPERTS", "64"))
    num_heads: int = int(os.environ.get("BENCH_NUM_HEADS", "8"))
    topks: tuple[int, ...] = (2, 8)
    warmup: int = int(os.environ.get("BENCH_WARMUP", "1"))
    iters: int = int(os.environ.get("BENCH_ITERS", "3"))
    ep_axis_size: int = int(os.environ.get("BENCH_EP_AXIS_SIZE", "2"))


def _ensure_pkg(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = ModuleSpec(name=name, loader=None, is_package=True)
    mod = module_from_spec(spec)
    mod.__path__ = []
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        parent_mod = _ensure_pkg(parent)
        setattr(parent_mod, child, mod)
    return mod


def _install_baseline_import_stubs() -> None:
    _ensure_pkg("experiments")
    _ensure_pkg("experiments.speedrun")
    _ensure_pkg("experiments.speedrun.new_grug_moe")
    _ensure_pkg("levanter")
    _ensure_pkg("levanter.grug")
    _ensure_pkg("marin")
    _ensure_pkg("marin.execution")
    _ensure_pkg("marin.speedrun")

    if "experiments.llama" not in sys.modules:
        llama_mod = types.ModuleType("experiments.llama")
        llama_mod.llama3_tokenizer_vocab_size = 128_256
        sys.modules["experiments.llama"] = llama_mod
        sys.modules["experiments"].llama = llama_mod

    if "experiments.simple_train_config" not in sys.modules:
        simple_cfg_mod = types.ModuleType("experiments.simple_train_config")

        class SimpleTrainConfig:
            def __init__(self, *args, **kwargs):
                pass

        simple_cfg_mod.SimpleTrainConfig = SimpleTrainConfig
        sys.modules["experiments.simple_train_config"] = simple_cfg_mod
        sys.modules["experiments"].simple_train_config = simple_cfg_mod

    if "levanter.grug.attention" not in sys.modules:
        attn_mod = types.ModuleType("levanter.grug.attention")

        class AttentionMask:
            pass

        def attention(*args, **kwargs):
            raise NotImplementedError("benchmark stub")

        attn_mod.AttentionMask = AttentionMask
        attn_mod.attention = attention
        sys.modules["levanter.grug.attention"] = attn_mod
        sys.modules["levanter.grug"].attention = attn_mod

    if "levanter.grug.loss" not in sys.modules:
        loss_mod = types.ModuleType("levanter.grug.loss")

        def fused_linear_softmax_cross_entropy_loss(*args, **kwargs):
            raise NotImplementedError("benchmark stub")

        loss_mod.fused_linear_softmax_cross_entropy_loss = fused_linear_softmax_cross_entropy_loss
        sys.modules["levanter.grug.loss"] = loss_mod
        sys.modules["levanter.grug"].loss = loss_mod

    if "levanter.grug.sharding" not in sys.modules:
        sharding_mod = types.ModuleType("levanter.grug.sharding")
        sharding_mod.Pbatch = P(("data",))
        sharding_mod.Pvocab = P(None, None)
        sharding_mod.unshard = lambda x: x
        sys.modules["levanter.grug.sharding"] = sharding_mod
        sys.modules["levanter.grug"].sharding = sharding_mod

    if "levanter.tracker" not in sys.modules:
        tracker_mod = types.ModuleType("levanter.tracker")
        sys.modules["levanter.tracker"] = tracker_mod
        sys.modules["levanter"].tracker = tracker_mod

    if "levanter.optim" not in sys.modules:
        optim_mod = types.ModuleType("levanter.optim")

        class GrugMuonConfig:
            def __init__(self, *args, **kwargs):
                pass

        optim_mod.GrugMuonConfig = GrugMuonConfig
        sys.modules["levanter.optim"] = optim_mod
        sys.modules["levanter"].optim = optim_mod

    if "marin.execution.executor" not in sys.modules:
        exec_mod = types.ModuleType("marin.execution.executor")
        exec_mod.executor_main = lambda *args, **kwargs: None
        sys.modules["marin.execution.executor"] = exec_mod
        sys.modules["marin.execution"].executor = exec_mod

    if "marin.speedrun.speedrun" not in sys.modules:
        speedrun_mod = types.ModuleType("marin.speedrun.speedrun")

        class Author:
            def __init__(self, *args, **kwargs):
                pass

        speedrun_mod.Author = Author
        sys.modules["marin.speedrun.speedrun"] = speedrun_mod
        sys.modules["marin.speedrun"].speedrun = speedrun_mod

    helpers_mod = types.ModuleType("experiments.speedrun.new_grug_moe.helpers")
    helpers_mod.build_speedrun = lambda *args, **kwargs: None
    sys.modules["experiments.speedrun.new_grug_moe.helpers"] = helpers_mod
    sys.modules["experiments.speedrun.new_grug_moe"].helpers = helpers_mod


def _load_baseline_module() -> types.ModuleType:
    _install_baseline_import_stubs()
    src = BASELINE_LOCAL_PATH.read_text()

    pkg_name = "experiments.speedrun.new_grug_moe"
    mod_name = f"{pkg_name}.baseline_scaled_sharded"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []
        sys.modules[pkg_name] = pkg

    mod = types.ModuleType(mod_name)
    mod.__package__ = pkg_name
    mod.__file__ = f"<local:{BASELINE_LOCAL_PATH}>"
    sys.modules[mod_name] = mod
    exec(src, mod.__dict__)
    return mod


def _load_our_module() -> types.ModuleType:
    pkg_name = "_bench_grug"
    mod_name = f"{pkg_name}.grug_moe"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []
        sys.modules[pkg_name] = pkg

    attn_mod = types.ModuleType(f"{pkg_name}.attention")

    @dataclass(frozen=True)
    class RotaryConfig:
        theta: float = 10000.0
        scaling_factor: float | None = None

    class AttentionMask:
        pass

    def attention(*args, **kwargs):
        raise NotImplementedError("benchmark stub")

    def apply_rotary_embedding(q, k, *, seq_len, head_dim, rope):
        return q, k

    attn_mod.RotaryConfig = RotaryConfig
    attn_mod.AttentionMask = AttentionMask
    attn_mod.attention = attention
    attn_mod.apply_rotary_embedding = apply_rotary_embedding
    sys.modules[f"{pkg_name}.attention"] = attn_mod

    loss_mod = types.ModuleType(f"{pkg_name}.loss")
    loss_mod.fused_linear_softmax_cross_entropy_loss = lambda *args, **kwargs: None
    sys.modules[f"{pkg_name}.loss"] = loss_mod

    sharding_mod = types.ModuleType(f"{pkg_name}.sharding")
    sharding_mod.Pvocab = P(None, None)
    sharding_mod.unshard = lambda x: x
    sys.modules[f"{pkg_name}.sharding"] = sharding_mod

    src = OUR_MOE_PATH.read_text()
    mod = types.ModuleType(mod_name)
    mod.__package__ = pkg_name
    mod.__file__ = str(OUR_MOE_PATH)
    sys.modules[mod_name] = mod
    exec(src, mod.__dict__)
    return mod


def _bench_grad_step(step_fn, model, x, *, warmup: int, iters: int) -> tuple[float, float]:
    loss, _ = step_fn(model, x)
    jax.block_until_ready(loss)

    for _ in range(warmup):
        loss, _ = step_fn(model, x)
        jax.block_until_ready(loss)

    start = time.perf_counter()
    for _ in range(iters):
        loss, _ = step_fn(model, x)
        jax.block_until_ready(loss)
    elapsed = (time.perf_counter() - start) / iters
    return float(elapsed), float(loss)


def _make_dp_mesh(devices: list[jax.Device]) -> Mesh:
    arr = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        arr,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_ep_mesh(devices: list[jax.Device], ep_axis_size: int) -> Mesh:
    if ep_axis_size <= 0:
        raise ValueError(f"ep_axis_size must be positive, got {ep_axis_size}")
    if len(devices) % ep_axis_size != 0:
        raise ValueError(
            f"Need device count divisible by ep_axis_size, got devices={len(devices)}, ep_axis_size={ep_axis_size}"
        )
    data = len(devices) // ep_axis_size
    arr = np.array(devices).reshape(data, ep_axis_size, 1)
    return Mesh(
        arr,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _baseline_dp(bench: BenchCfg, topk: int, baseline_mod: types.ModuleType, mesh: Mesh) -> tuple[float, float]:
    cfg = baseline_mod.ModelConfig(
        vocab_size=4096,
        hidden_dim=bench.hidden,
        num_layers=1,
        num_heads=bench.num_heads,
        num_kv_heads=bench.num_heads,
        max_seq_len=bench.seq,
        n_routed_experts=bench.experts,
        num_experts_per_tok=topk,
        lbl_coef=0.0,
        rzl_coef=0.0,
    )

    with jax.set_mesh(mesh):
        model = baseline_mod.MOE.init(cfg, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (bench.batch, bench.seq, bench.hidden), dtype=jnp.bfloat16)
        x = jax.sharding.reshard(x, P(("data",), None, None))

        def loss_fn(m, inp):
            out, _extras = m(inp)
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        sec, _loss = _bench_grad_step(step, model, x, warmup=bench.warmup, iters=bench.iters)

    tok_s = (bench.batch * bench.seq) / sec
    return sec, tok_s


def _our_dp(bench: BenchCfg, topk: int, mesh: Mesh) -> tuple[float, float]:
    mod = _load_our_module()
    cfg = mod.GrugMoeModelConfig(
        vocab_size=4096,
        hidden_dim=bench.hidden,
        intermediate_dim=bench.hidden * 3,
        shared_expert_intermediate_dim=0,
        num_experts=bench.experts,
        num_experts_per_token=topk,
        num_layers=1,
        num_heads=bench.num_heads,
        num_kv_heads=bench.num_heads,
        max_seq_len=bench.seq,
    )

    with jax.set_mesh(mesh):
        model = mod.MoEMLP.init(cfg, key=jax.random.key(2))
        x = jax.random.normal(jax.random.key(3), (bench.batch, bench.seq, bench.hidden), dtype=jnp.bfloat16)
        x = jax.sharding.reshard(x, P(("data",), None, None))

        def loss_fn(m, inp):
            out = m(inp)
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        sec, _loss = _bench_grad_step(step, model, x, warmup=bench.warmup, iters=bench.iters)

    tok_s = (bench.batch * bench.seq) / sec
    return sec, tok_s


def _our_ep(bench: BenchCfg, topk: int, mesh: Mesh) -> tuple[float, float]:
    mod = _load_our_module()
    cfg = mod.GrugMoeModelConfig(
        vocab_size=4096,
        hidden_dim=bench.hidden,
        intermediate_dim=bench.hidden * 3,
        shared_expert_intermediate_dim=0,
        num_experts=bench.experts,
        num_experts_per_token=topk,
        num_layers=1,
        num_heads=bench.num_heads,
        num_kv_heads=bench.num_heads,
        max_seq_len=bench.seq,
    )

    with jax.set_mesh(mesh):
        model = mod.MoEMLP.init(cfg, key=jax.random.key(4))
        x = jax.random.normal(jax.random.key(5), (bench.batch, bench.seq, bench.hidden), dtype=jnp.bfloat16)
        x = jax.sharding.reshard(x, P(("data", "expert"), None, None))

        def loss_fn(m, inp):
            out = m(inp)
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        sec, _loss = _bench_grad_step(step, model, x, warmup=bench.warmup, iters=bench.iters)

    tok_s = (bench.batch * bench.seq) / sec
    return sec, tok_s


def main() -> None:
    bench = BenchCfg()
    devices = jax.devices()
    print(f"devices={devices}")
    print(f"bench_cfg={bench}")

    baseline_mod = _load_baseline_module()

    dp_mesh = _make_dp_mesh(devices)
    ep_mesh = _make_ep_mesh(devices, bench.ep_axis_size)

    print("\n=== Throughput (forward_backward, MoE block only) ===")
    print("columns: impl, topk, step_s, tokens_per_s")

    for topk in bench.topks:
        b_s, b_tps = _baseline_dp(bench, topk, baseline_mod, dp_mesh)
        d_s, d_tps = _our_dp(bench, topk, dp_mesh)
        e_s, e_tps = _our_ep(bench, topk, ep_mesh)

        print(f"baseline_dp,topk={topk},step_s={b_s:.6f},tokens_per_s={b_tps:.2f}")
        print(f"our_dp,topk={topk},step_s={d_s:.6f},tokens_per_s={d_tps:.2f}")
        print(f"our_dp_ep,topk={topk},step_s={e_s:.6f},tokens_per_s={e_tps:.2f}")

        print(
            f"speedups(topk={topk}): "
            f"our_dp/baseline_dp={d_tps / b_tps:.3f}x, "
            f"our_dp_ep/baseline_dp={e_tps / b_tps:.3f}x, "
            f"our_dp_ep/our_dp={e_tps / d_tps:.3f}x"
        )


if __name__ == "__main__":
    main()
