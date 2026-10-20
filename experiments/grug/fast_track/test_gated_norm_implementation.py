# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``gated_norm_implementation=fused`` (levanter's fused ``GatedNorm(RMSNorm(x))`` op) computes the same loss and
gradients as the two composed modules, with the same parameter tree. On CPU in float32 (the op's XLA backward), plain
and inside the AttnRes layer scan (plain-sum bridge stack and the 8-block-style fused block mix); on SM100 GPUs (its
Triton backward) on the GB200 bridge stack in bf16 and in the MXFP8 recipe.

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

from experiments.grug.fast_track.model import (
    Fp8Recipe,
    GatedNormImplementation,
    GrugModelConfig,
    SharedExpertPlacement,
    Transformer,
)

_CPU_DEVICES = 4
_UNGATED = dict(moe_ungated_relu2=True, moe_ungated_kernel=True, shared_ungated_relu2=True)
# The S4 bridge's layer stack: the plain-sum AttnRes scan with the routed output saved for the backward.
_SCAN = dict(
    attn_res=True,
    attn_res_plain_sum=True,
    attn_res_num_blocks=1,
    scan_layers=True,
    scan_save_routed_output=True,
    attn_res_fused_source_grads=True,
)
# S5's real Block AttnRes (softmax over block sums) with the fused block mix.
_BLOCKS = dict(_SCAN, attn_res_plain_sum=False, attn_res_num_blocks=2, attn_res_fused_block_mix=True)
_OVERLAP = dict(moe_shared_placement=SharedExpertPlacement.OVERLAP, moe_overlap_param_gather="before_transport")
_MXFP8 = dict(fp8_recipe=Fp8Recipe.MXFP8, mxfp8_fused_grad_quant=True, mxfp8_expert_wgrad=True)
_FUSED = dict(gated_norm_implementation=GatedNormImplementation.FUSED)
_STACKS = {
    "plain": dict(),
    "scan": dict(**_UNGATED, **_SCAN),
    "blocks": dict(**_UNGATED, **_BLOCKS),
    "gpu_bridge": dict(**_UNGATED, **_SCAN, **_OVERLAP),
    "gpu_mxfp8": dict(**_UNGATED, **_SCAN, **_OVERLAP, **_MXFP8),
    "gpu_blocks_mxfp8": dict(**_UNGATED, **_BLOCKS, **_MXFP8),
}
_CASES = {
    f"{stack}_{impl}": dict(kw, **(_FUSED if impl == "fused" else {}))
    for stack, kw in _STACKS.items()
    for impl in ("reference", "fused")
}
# GPU accuracy anchor: the stack's function in float32 (reference GatedNorm, no MXFP8, and the reference attention
# and XLA expert MLP, since FA4 and the ungated expert kernel are bf16-only).
_CASES.update(
    {
        f"{stack}_anchor": {
            **{k: v for k, v in kw.items() if k not in _MXFP8},
            "gpu_attention": "reference",
            "moe_ungated_kernel": False,
        }
        for stack, kw in _STACKS.items()
    }
)


def _on_gpu() -> bool:
    return jax.default_backend() == "gpu"


def _config(gpu: bool, **overrides) -> GrugModelConfig:
    # Widths are multiples of the 32-element MX block, the 128-row capacity alignment and the Triton kernels' 128
    # hidden tile on GPU.
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
    """Loss, gradient leaves and parameter paths/shapes of one configuration's train step, saved to ``out_path``."""
    gpu = _on_gpu()
    devices = jax.devices()
    mesh = Mesh(
        np.array(devices, dtype=object).reshape((1, 1, len(devices), 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = _config(gpu, **_CASES[case])
    dtype = jnp.bfloat16 if gpu and not case.endswith("_anchor") else jnp.float32
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        model = jax.tree.map(lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, model)
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2 * len(devices), cfg.max_seq_len), 0, cfg.vocab_size)

        def loss_fn(m, x):
            return m.next_token_loss(x, jnp.ones(x.shape, dtype))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model, tokens)
    params = jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_inexact_array))[0]
    layout = np.array([f"{jax.tree_util.keystr(path)}:{tuple(leaf.shape)}" for path, leaf in params])
    leaves = [np.asarray(g, np.float32) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    np.savez(out_path, loss=np.float64(loss), layout=layout, **{f"g{i}": g for i, g in enumerate(leaves)})


def _case(case: str, tmp_path) -> tuple[float, list[str], list[np.ndarray]]:
    out = tmp_path / f"{case}.npz"
    env = dict(os.environ)
    if not _on_gpu():
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_CPU_DEVICES}"
        env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.grug.fast_track.test_gated_norm_implementation", case, str(out)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    data = np.load(out)
    grads = [data[f"g{i}"] for i in range(len(data.files) - 2)]
    return float(data["loss"]), list(data["layout"]), grads


def _assert_close(stack: str, tmp_path, rtol: float, grad_rtol: float) -> None:
    loss, layout, grads = _case(f"{stack}_reference", tmp_path)
    fused_loss, fused_layout, fused_grads = _case(f"{stack}_fused", tmp_path)
    assert fused_layout == layout
    errors = {
        name: float(np.linalg.norm(f - g) / max(np.linalg.norm(g), 1e-30))
        for name, f, g in zip(layout, fused_grads, grads, strict=True)
    }
    worst = max(errors, key=errors.get)
    norm_worst = max((n for n in errors if "gated_norm" in n or "rms_" in n or "_norm" in n), key=errors.get)
    print(
        f"{stack}: loss fused {fused_loss:.7f} vs reference {loss:.7f}; grad relative L2 max {errors[worst]:.3e} "
        f"({worst}), norm params max {errors[norm_worst]:.3e} ({norm_worst})"
    )
    assert abs(fused_loss - loss) <= rtol * abs(loss)
    assert errors[worst] <= grad_rtol


@pytest.mark.timeout(900)
@pytest.mark.parametrize("stack", ["plain", "scan", "blocks"])
def test_fused_gated_norm_matches_reference_on_cpu(stack, tmp_path):
    if _on_gpu():
        pytest.skip("float32 CPU check")
    _assert_close(stack, tmp_path, rtol=1e-6, grad_rtol=1e-4)


def _global_error(grads: list[np.ndarray], anchor: list[np.ndarray]) -> float:
    diff = np.sqrt(sum(float(np.sum(np.square(g - a))) for g, a in zip(grads, anchor, strict=True)))
    return diff / np.sqrt(sum(float(np.sum(np.square(a))) for a in anchor))


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("stack", ["gpu_bridge", "gpu_mxfp8", "gpu_blocks_mxfp8"])
def test_fused_gated_norm_is_no_less_accurate_on_gpu(stack, tmp_path):
    """In bf16 (and MXFP8) the two paths round differently, so both are scored against a float32 anchor."""
    if not _on_gpu() or len(jax.devices()) < 2:
        pytest.skip("needs >= 2 GPUs for the ragged EP backend")
    anchor_loss, layout, anchor = _case(f"{stack}_anchor", tmp_path)
    loss, ref_layout, ref = _case(f"{stack}_reference", tmp_path)
    fused_loss, fused_layout, fused = _case(f"{stack}_fused", tmp_path)
    assert fused_layout == ref_layout == layout
    ref_err, fused_err = _global_error(ref, anchor), _global_error(fused, anchor)
    leaf = lambda g: max(  # noqa: E731
        float(np.linalg.norm(x - a) / max(np.linalg.norm(a), 1e-30)) for x, a in zip(g, anchor, strict=True)
    )
    print(
        f"{stack}: loss anchor {anchor_loss:.7f} reference {loss:.7f} fused {fused_loss:.7f}; grad relative L2 vs "
        f"anchor (all leaves) reference {ref_err:.3e} fused {fused_err:.3e}; worst leaf reference {leaf(ref):.3e} "
        f"fused {leaf(fused):.3e}"
    )
    assert abs(fused_loss - anchor_loss) <= abs(loss - anchor_loss) + 1e-3 * abs(anchor_loss)
    assert fused_err <= 1.1 * ref_err


if __name__ == "__main__":
    _run_case(sys.argv[1], sys.argv[2])
