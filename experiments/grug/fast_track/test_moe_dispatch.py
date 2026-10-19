# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``moe_dispatch=hero_levanter`` (moe_hero_ep's MoE block) computes the same loss and gradients as fast_track's own
MoE block. On CPU (float32 on the expert-parallel ``fixed_pooled_wave_all_to_all`` backend, since XLA:CPU has no
ragged all-to-all) against the fused placement, plain and inside the AttnRes layer scan; on SM100 GPUs
(``ragged_all_to_all``) against the overlap placement the GB200 bridge runs, in bf16 and in its MXFP8 recipe.

Each configuration runs in its own process: on CPU the expert axis needs several host devices, which XLA fixes at
startup, and on GPU every executable with a ragged all-to-all reserves NCCL symmetric memory."""

import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import Fp8Recipe, GrugModelConfig, MoeDispatch, SharedExpertPlacement, Transformer

_CPU_DEVICES = 4
_UNGATED = dict(moe_ungated_relu2=True, moe_ungated_kernel=True, shared_ungated_relu2=True)
# The bridge's layer stack: the plain-sum AttnRes scan with the routed output saved for the backward.
_SCAN = dict(
    attn_res=True,
    attn_res_plain_sum=True,
    attn_res_num_blocks=1,
    scan_layers=True,
    scan_save_routed_output=True,
    attn_res_fused_source_grads=True,
)
_OVERLAP = dict(moe_shared_placement=SharedExpertPlacement.OVERLAP, moe_overlap_param_gather="before_transport")
_HERO = dict(moe_dispatch=MoeDispatch.HERO_LEVANTER)
_MXFP8 = dict(fp8_recipe=Fp8Recipe.MXFP8, mxfp8_fused_grad_quant=True, mxfp8_expert_wgrad=True)
_CASES = {
    # CPU, pooled-wave EP.
    "fused": dict(),
    "hero": dict(**_HERO),
    "ungated_fused": dict(**_UNGATED),
    "ungated_hero": dict(**_UNGATED, **_HERO),
    "scan_fused": dict(**_UNGATED, **_SCAN),
    "scan_hero": dict(**_UNGATED, **_SCAN, **_HERO),
    # GPU, ragged EP: the bridge's layer stack in bf16 and in its MXFP8 recipe with token-quantized expert wgrads.
    "gpu_overlap": dict(**_UNGATED, **_SCAN, **_OVERLAP),
    "gpu_hero": dict(**_UNGATED, **_SCAN, **_HERO),
    "mxfp8_overlap": dict(**_UNGATED, **_SCAN, **_OVERLAP, **_MXFP8),
    "mxfp8_hero": dict(**_UNGATED, **_SCAN, **_HERO, **_MXFP8),
}


def _on_gpu() -> bool:
    return jax.default_backend() == "gpu"


def _config(gpu: bool, **overrides) -> GrugModelConfig:
    # Widths are multiples of the 32-element MX block and the 128-row capacity alignment on GPU.
    width = 512 if gpu else 32
    kwargs = dict(
        vocab_size=2048 if gpu else 64,
        hidden_dim=2 * width,
        intermediate_dim=width,
        shared_expert_intermediate_dim=width,
        num_shared_experts=2,
        num_experts=16 if gpu else 8,
        num_experts_per_token=4 if gpu else 2,
        latent_dim=width,
        num_layers=4,
        num_heads=8 if gpu else 2,
        num_kv_heads=2 if gpu else 1,
        local_kv_heads=2 if gpu else 1,
        global_kv_heads=2 if gpu else 1,
        head_dim=128 if gpu else 16,
        max_seq_len=512 if gpu else 32,
        sliding_window=512 if gpu else 32,
        global_every=2,
        moe_implementation="ragged_all_to_all" if gpu else "fixed_pooled_wave_all_to_all",
        moe_ragged_chunks=2,
        capacity_factor=2.0,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _run_case(case: str, out_path: str) -> None:
    """Loss and gradient leaves of one configuration's train step, saved to ``out_path``."""
    gpu = _on_gpu()
    devices = jax.devices()
    mesh = Mesh(
        np.array(devices, dtype=object).reshape((1, 1, len(devices), 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = _config(gpu, **_CASES[case])
    dtype = jnp.bfloat16 if gpu else jnp.float32
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        model = jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, model)
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2 * len(devices), cfg.max_seq_len), 0, cfg.vocab_size)

        def loss_fn(m, x):
            return m.next_token_loss(x, jnp.ones(x.shape, dtype))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model, tokens)
    leaves = [np.asarray(g, np.float32) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    np.savez(out_path, loss=np.float64(loss), **{f"g{i}": g for i, g in enumerate(leaves)})


def _case(case: str, tmp_path) -> tuple[float, list[np.ndarray]]:
    out = tmp_path / f"{case}.npz"
    env = dict(os.environ)
    if not _on_gpu():
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_CPU_DEVICES}"
        env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.grug.fast_track.test_moe_dispatch", case, str(out)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    data = np.load(out)
    return float(data["loss"]), [data[f"g{i}"] for i in range(len(data.files) - 1)]


def _assert_close(case: str, reference: str, tmp_path, rtol: float, grad_rtol: float) -> None:
    loss, grads = _case(reference, tmp_path)
    hero_loss, hero_grads = _case(case, tmp_path)
    errors = [
        float(np.linalg.norm(h - g) / max(np.linalg.norm(g), 1e-30)) for h, g in zip(hero_grads, grads, strict=True)
    ]
    print(f"{case} vs {reference}: loss {hero_loss:.7f} vs {loss:.7f}; grad relative L2 max {max(errors):.3e}")
    assert abs(hero_loss - loss) <= rtol * abs(loss)
    assert max(errors) <= grad_rtol


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("case", "reference"),
    [
        ("hero", "fused"),
        ("ungated_hero", "ungated_fused"),
        ("scan_hero", "scan_fused"),
    ],
)
def test_hero_dispatch_matches_fast_track_moe(case, reference, tmp_path):
    if _on_gpu():
        pytest.skip("float32 CPU value test; the GPU checks are the two tests below")
    # float32 on CPU: the two blocks differ only in summation order.
    _assert_close(case, reference, tmp_path, rtol=1e-6, grad_rtol=1e-5)


def _require_sm100_gpus():
    if not _on_gpu() or float(jax.devices()[0].compute_capability) < 10.0 or len(jax.devices()) < 2:
        pytest.skip("needs two or more SM100 GPUs")


@pytest.mark.timeout(900)
def test_hero_dispatch_matches_the_bridge_overlap_placement(tmp_path):
    """bf16 on the ragged EP path: the blocks differ in rounding order (the overlap placement's shared experts run
    inside the EP shard on gathered weights)."""
    _require_sm100_gpus()
    _assert_close("gpu_hero", "gpu_overlap", tmp_path, rtol=1e-3, grad_rtol=0.05)


@pytest.mark.timeout(900)
def test_hero_dispatch_mxfp8_tracks_the_bridge(tmp_path):
    """The bridge's MXFP8 recipe: e4m3 GEMM operands, so the two blocks agree to the quantization noise of the
    projections they no longer fuse (the latent down and shared gate/up quantize per FSDP shard)."""
    _require_sm100_gpus()
    _assert_close("mxfp8_hero", "mxfp8_overlap", tmp_path, rtol=2e-3, grad_rtol=0.15)


if __name__ == "__main__":
    _run_case(sys.argv[1], sys.argv[2])
