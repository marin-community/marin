# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``ce_grad_in_forward_token_block_size`` (the cross-entropy that computes its gradients in the forward) gives the
same training loss and gradients as the default ``xla_fast_bwd`` kernel, with the lm_head z-loss on, plain and inside
the AttnRes layer scan. CPU, float32, expert-parallel over 4 host devices; each configuration runs in its own process
because XLA fixes the host device count at startup."""

import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import GrugModelConfig, Transformer

_CPU_DEVICES = 4
# A z-loss weight large enough that its gradient is a visible part of the lm_head gradient.
_Z_LOSS = 1e-2
# 64 rows per device: 24-row blocks also exercise the zero-weight padding of the last block.
_BLOCK = 24
_SCAN = dict(
    moe_ungated_relu2=True,
    moe_ungated_kernel=True,
    shared_ungated_relu2=True,
    attn_res=True,
    attn_res_plain_sum=True,
    attn_res_num_blocks=1,
    scan_layers=True,
    scan_save_routed_output=True,
    attn_res_fused_source_grads=True,
)
_STACKS = {"plain": dict(), "scan": _SCAN}
_CASES = {
    f"{stack}_{impl}": dict(kw, **({"ce_grad_in_forward_token_block_size": _BLOCK} if impl == "gif" else {}))
    for stack, kw in _STACKS.items()
    for impl in ("default", "gif")
}


def _run_case(case: str, out_path: str) -> None:
    devices = jax.devices()
    mesh = Mesh(
        np.array(devices, dtype=object).reshape((1, 1, len(devices), 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = GrugModelConfig(
        vocab_size=96,
        hidden_dim=64,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_shared_experts=2,
        num_experts=8,
        num_experts_per_token=2,
        latent_dim=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=32,
        sliding_window=32,
        global_every=2,
        moe_implementation="fixed_pooled_wave_all_to_all",
        capacity_factor=2.0,
        **_CASES[case],
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2 * len(devices), cfg.max_seq_len), 0, cfg.vocab_size)
        loss_weight = jax.random.bernoulli(jax.random.PRNGKey(2), 0.8, tokens.shape).astype(jnp.float32)

        def loss_fn(m):
            return m.next_token_loss(tokens, loss_weight, reduction="mean", logsumexp_weight=_Z_LOSS)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
    params = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_inexact_array))[0]
    names = np.array([jax.tree_util.keystr(path) for path, _ in params])
    np.savez(out_path, loss=np.float64(loss), names=names, **{f"g{i}": np.asarray(g) for i, (_, g) in enumerate(params)})


def _case(case: str, tmp_path):
    out = tmp_path / f"{case}.npz"
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_CPU_DEVICES}"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.grug.fast_track.test_ce_grad_in_forward", case, str(out)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    data = np.load(out)
    return float(data["loss"]), list(data["names"]), [data[f"g{i}"] for i in range(len(data.files) - 2)]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("stack", list(_STACKS))
def test_ce_grad_in_forward_matches_xla_fast_bwd_with_z_loss(stack, tmp_path):
    loss, names, grads = _case(f"{stack}_default", tmp_path)
    gif_loss, gif_names, gif_grads = _case(f"{stack}_gif", tmp_path)
    assert gif_names == names
    errors = {
        n: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        for n, a, b in zip(names, gif_grads, grads, strict=True)
    }
    worst = max(errors, key=errors.get)
    head = [n for n in names if "output_proj" in n]
    print(
        f"{stack}: loss {gif_loss:.7f} vs {loss:.7f}; grad relative L2 max {errors[worst]:.3e} ({worst}); "
        f"lm_head {[f'{errors[n]:.2e}' for n in head]}"
    )
    assert head
    assert abs(gif_loss - loss) <= 1e-6 * abs(loss)
    assert errors[worst] <= 1e-4


if __name__ == "__main__":
    _run_case(sys.argv[1], sys.argv[2])
