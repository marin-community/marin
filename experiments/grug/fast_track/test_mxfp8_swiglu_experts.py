# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``fp8_recipe=mxfp8`` on gated SwiGLU experts (``moe_ungated_relu2=false``): a small hero-like fast_track model's
loss and gradients stay within MXFP8 error of bf16 on the ``ragged_all_to_all`` backend. Needs two or more SM100
GPUs (the expert-parallel path; one device runs the local experts, which have no MXFP8 kernels)."""

import dataclasses
import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import Fp8Recipe, GrugModelConfig, Transformer

_SEQ = 512
_BATCH_PER_DEVICE = 2
_VOCAB = 2048


def _gpus() -> int:
    if jax.default_backend() != "gpu" or float(jax.devices()[0].compute_capability) < 10.0:
        return 0
    return len(jax.devices())


def _config(**overrides) -> GrugModelConfig:
    # The hero's layout scaled down: SwiGLU experts in a latent space, two shared experts, GQA attention.
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_dim=1024,
        intermediate_dim=512,
        shared_expert_intermediate_dim=512,
        num_shared_experts=2,
        num_experts=16,
        num_experts_per_token=4,
        latent_dim=512,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        max_seq_len=_SEQ,
        moe_implementation="ragged_all_to_all",
        capacity_factor=4.0,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _loss_and_grads(mesh, cfg):
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jax.random.randint(jax.random.PRNGKey(1), (_BATCH_PER_DEVICE * mesh.size, _SEQ), 0, _VOCAB)

        def loss_fn(m, x):
            return m.next_token_loss(x, jnp.ones(x.shape, jnp.float32))

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        loss, grads = step(model, tokens)
        hlo = step.lower(model, tokens).as_text()
    leaves = [np.asarray(g, np.float32) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    return float(loss), leaves, hlo


def _mesh(gpus: int) -> Mesh:
    return Mesh(
        np.array(jax.devices(), dtype=object).reshape((1, 1, gpus, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


@pytest.mark.parametrize(
    "mxfp8_overrides",
    [dict(), dict(moe_mxfp8_dispatch=True, mxfp8_fused_grad_quant=True)],
    ids=["mxfp8", "mxfp8_dispatch_fused_grad_quant"],
)
def test_swiglu_mxfp8_loss_and_grads_track_bf16(mxfp8_overrides):
    gpus = _gpus()
    if gpus < 2:
        pytest.skip("needs two or more SM100 GPUs")
    mesh = _mesh(gpus)
    base = _config()
    loss, grads, _ = _loss_and_grads(mesh, base)
    dense_loss, _, _ = _loss_and_grads(mesh, dataclasses.replace(base, fp8_recipe=Fp8Recipe.MXFP8_DENSE))
    mx_loss, mx_grads, _ = _loss_and_grads(
        mesh, dataclasses.replace(base, fp8_recipe=Fp8Recipe.MXFP8, **mxfp8_overrides)
    )
    # The routed experts really ran in MXFP8: the full recipe differs from its dense-only half.
    assert mx_loss != dense_loss
    errors = [float(np.linalg.norm(m - g) / max(np.linalg.norm(g), 1e-30)) for m, g in zip(mx_grads, grads, strict=True)]
    print(f"loss bf16 {loss:.6f} mxfp8 {mx_loss:.6f}; grad relative L2 max {max(errors):.3e} mean {np.mean(errors):.3e}")
    assert abs(mx_loss - loss) / loss < 1e-2
    assert max(errors) < 0.15


def test_ungated_relu2_mxfp8_step_fingerprint():
    """Prints a fingerprint of the ungated ReLU^2 MXFP8 step (lowered HLO and gradients) to compare across commits."""
    gpus = _gpus()
    if gpus < 2:
        pytest.skip("needs two or more SM100 GPUs")
    cfg = _config(
        moe_ungated_relu2=True,
        moe_ungated_kernel=True,
        shared_ungated_relu2=True,
        fp8_recipe=Fp8Recipe.MXFP8,
    )
    loss, grads, hlo = _loss_and_grads(_mesh(gpus), cfg)
    grad_hash = hashlib.sha256(b"".join(g.tobytes() for g in grads)).hexdigest()[:16]
    hlo_hash = hashlib.sha256(hlo.encode()).hexdigest()[:16]
    print(f"ungated relu2 mxfp8: loss {loss!r} grads sha {grad_hash} hlo sha {hlo_hash}")
