# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``fp8_recipe=mxfp8`` on gated SwiGLU experts (``moe_ungated_relu2=false``): a small hero-like fast_track model's
loss and gradients stay within MXFP8 error of bf16 on the ``ragged_all_to_all`` backend. Needs two or more SM100
GPUs (the expert-parallel path; one device runs the local experts, which have no MXFP8 kernels).

Each configuration runs in its own process: every executable with a ragged all-to-all reserves NCCL symmetric
memory, and a few of them in one process exhaust it."""

import hashlib
import re
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from levanter.grug.mxfp8 import Mxfp8ExpertKernels

from experiments.grug.fast_track.model import Fp8Recipe, GrugModelConfig, Transformer

_SEQ = 512
_BATCH_PER_DEVICE = 2
_VOCAB = 2048
_UNGATED = dict(moe_ungated_relu2=True, moe_ungated_kernel=True, shared_ungated_relu2=True)
_CASES = {
    "bf16": dict(),
    "mxfp8_dense": dict(fp8_recipe=Fp8Recipe.MXFP8_DENSE),
    "mxfp8": dict(fp8_recipe=Fp8Recipe.MXFP8),
    "mxfp8_dispatch_fused_grad_quant": dict(
        fp8_recipe=Fp8Recipe.MXFP8, moe_mxfp8_dispatch=True, mxfp8_fused_grad_quant=True
    ),
    "mxfp8_wgrad": dict(
        fp8_recipe=Fp8Recipe.MXFP8, moe_mxfp8_dispatch=True, mxfp8_fused_grad_quant=True, mxfp8_expert_wgrad=True
    ),
    "ungated_relu2_bf16": dict(**_UNGATED),
    "ungated_relu2_mxfp8_dense": dict(fp8_recipe=Fp8Recipe.MXFP8_DENSE, **_UNGATED),
    "ungated_relu2_mxfp8": dict(fp8_recipe=Fp8Recipe.MXFP8, **_UNGATED),
    "ungated_relu2_mxfp8_wgrad": dict(
        fp8_recipe=Fp8Recipe.MXFP8, moe_mxfp8_dispatch=True, mxfp8_expert_wgrad=True, **_UNGATED
    ),
    "ungated_relu2_mxfp8_wgrad_cudnn": dict(
        fp8_recipe=Fp8Recipe.MXFP8,
        mxfp8_expert_wgrad=True,
        moe_mxfp8_expert_kernels=Mxfp8ExpertKernels.CUDNN_FRONTEND,
        **_UNGATED,
    ),
}


def _sm100_gpus() -> int:
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
        capacity_factor=2.0,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _run_case(case: str, out_path: str) -> None:
    """Loss, gradient leaves and optimized-HLO hash of one configuration's train step, saved to ``out_path``."""
    # No source locations in the lowered modules (Pallas kernels embed them), so the HLO hash ignores moved lines.
    jax.config.update("jax_traceback_in_locations_limit", 0)
    gpus = len(jax.devices())
    mesh = Mesh(
        np.array(jax.devices(), dtype=object).reshape((1, 1, gpus, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(_config(**_CASES[case]), key=jax.random.PRNGKey(0))
        # bf16 compute, as training casts the f32 master weights.
        model = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, model)
        tokens = jax.random.randint(jax.random.PRNGKey(1), (_BATCH_PER_DEVICE * gpus, _SEQ), 0, _VOCAB)

        def loss_fn(m, x):
            return m.next_token_loss(x, jnp.ones(x.shape, jnp.bfloat16))

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        loss, grads = step(model, tokens)
        # The optimized HLO without op metadata (source lines).
        hlo = re.sub(r", metadata=\{[^}]*\}", "", step.lower(model, tokens).lowered.compile().as_text())
    leaves = [np.asarray(g, np.float32) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    np.savez(
        out_path,
        loss=np.float64(loss),
        hlo_sha=hashlib.sha256(hlo.encode()).hexdigest()[:16],
        **{f"g{i}": g for i, g in enumerate(leaves)},
    )


def _case(case: str, tmp_path) -> tuple[float, list[np.ndarray], str]:
    out = tmp_path / f"{case}.npz"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.grug.fast_track.test_mxfp8_swiglu_experts", case, str(out)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    data = np.load(out)
    grads = [data[f"g{i}"] for i in range(len(data.files) - 2)]
    return float(data["loss"]), grads, str(data["hlo_sha"])


def _require_gpus():
    if _sm100_gpus() < 2:
        pytest.skip("needs two or more SM100 GPUs")


@pytest.mark.timeout(900)  # three GPU processes, each compiling a train step
@pytest.mark.parametrize(
    "case",
    [
        "mxfp8",
        "mxfp8_dispatch_fused_grad_quant",
        "mxfp8_wgrad",
        "ungated_relu2_mxfp8_wgrad",
        "ungated_relu2_mxfp8_wgrad_cudnn",
    ],
)
def test_swiglu_mxfp8_loss_and_grads_track_bf16(case, tmp_path):
    _require_gpus()
    prefix = "ungated_relu2_" if case.startswith("ungated_relu2_") else ""
    loss, grads, _ = _case(f"{prefix}bf16", tmp_path)
    dense_loss, _, _ = _case(f"{prefix}mxfp8_dense", tmp_path)
    mx_loss, mx_grads, _ = _case(case, tmp_path)
    # The routed experts really ran in MXFP8: the full recipe differs from its dense-only half.
    assert mx_loss != dense_loss
    errors = [float(np.linalg.norm(m - g) / max(np.linalg.norm(g), 1e-30)) for m, g in zip(mx_grads, grads, strict=True)]
    print(
        f"{case}: loss bf16 {loss:.6f} mxfp8_dense {dense_loss:.6f} {case} {mx_loss:.6f}; "
        f"grad relative L2 max {max(errors):.3e} mean {np.mean(errors):.3e}"
    )
    # Every projection and expert GEMM is MXFP8 (e4m3: 3 mantissa bits), against bf16 gradients at init.
    assert abs(mx_loss - loss) / loss < 1e-3
    assert max(errors) < 0.25 and np.mean(errors) < 0.1


@pytest.mark.timeout(300)
def test_ungated_relu2_mxfp8_step_fingerprint(tmp_path):
    """Prints a fingerprint of the ungated ReLU^2 MXFP8 step to compare across commits: the loss and the optimized HLO
    (the gradients' hash also varies run to run on one commit: the ragged transport and wgrads are not bitwise
    deterministic)."""
    _require_gpus()
    loss, grads, hlo_sha = _case("ungated_relu2_mxfp8", tmp_path)
    grad_sha = hashlib.sha256(b"".join(g.tobytes() for g in grads)).hexdigest()[:16]
    print(f"ungated relu2 mxfp8: loss {loss!r} grads sha {grad_sha} hlo sha {hlo_sha}")


if __name__ == "__main__":
    _run_case(sys.argv[1], sys.argv[2])
