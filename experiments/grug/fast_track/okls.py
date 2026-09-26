# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Online KL-Shampoo (OKLS) direction for selected matrix leaves of the fast_track MoE model.

Port of the Tilde Research OKLS release (https://blog.tilderesearch.com/blog/online-kl-shampoo,
https://github.com/tilde-research/online-kl-shampoo-release), first ported on branch ``okls-optimizer``.
Per matrix (m x n) it keeps Nesterov momentum and two Kronecker preconditioner EMAs S_a (m x m),
S_b (n x n); each step it roots them with Scaled CANS coupled Newton-Schulz and whitens the momentum,
``U = S_a^{-1/2} N S_b^{-1/2}``. With ``hyperball`` the whitened direction takes MuonH's Frobenius
norm-preserving step (the ladder's matrices live on that sphere); otherwise the released muP-scaled,
AdamC-decayed update. ``GrugMoeMuonHConfig.okls_targets`` selects which matrix families use it.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec, reshard
from optax import tree_utils as otu

from experiments.grug.fast_track.grugmuon_stacked import _target_named_sharding

# ── CANS coupled Newton-Schulz coefficients (arXiv:2506.10935), 10 coupled steps ──
# Each (a, b): Y_{k+1} = a Y_k + b Y_k P_k ; Z_{k+1} = a Z_k + b P_k Z_k ; P_k = Z_k Y_k.
CANS_COEFFS: tuple[tuple[float, float], ...] = (
    (5.182503604966906, -5.178098480082684),
    (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178),
    (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716),
    (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968),
    (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799),
    (1.5, -0.5),
)


def scaled_cans_inv_sqrt(
    matrix: jax.Array,
    steps: int = 10,
    matmul_dtype=jnp.bfloat16,
) -> jax.Array:
    """Return ``matrix^{-1/2}`` for a single symmetric-PD ``(d, d)`` matrix via Scaled CANS.

    FP32 accumulators; GEMM inputs cast to ``matmul_dtype`` (bf16 on device, fp32 for exact tests).
    """
    d = matrix.shape[-1]
    s = matrix.astype(jnp.float32)
    # Symmetric spectral upper bound: rho(S) <= ||S||_inf = max abs row sum. Scale spectrum into (0, 1].
    max_eig = jnp.max(jnp.sum(jnp.abs(s), axis=-1)) * 1.01
    y = s / max_eig
    z = jnp.eye(d, dtype=jnp.float32)

    # Force true fp32 for the fp32 path so a bf16/tf32 global matmul-precision default cannot silently
    # degrade it; reduced-precision inputs (fp16/bf16) accumulate in fp32 regardless.
    is_fp32 = jnp.dtype(matmul_dtype) == jnp.dtype(jnp.float32)
    precision = jax.lax.Precision.HIGHEST if is_fp32 else None

    def mm(lhs, rhs):
        return jnp.matmul(
            lhs.astype(matmul_dtype),
            rhs.astype(matmul_dtype),
            preferred_element_type=jnp.float32,
            precision=precision,
        )

    for a, b in CANS_COEFFS[:steps]:
        p = mm(z, y)
        y = a * y + b * mm(y, p)
        z = a * z + b * mm(p, z)

    w = z * jax.lax.rsqrt(max_eig)
    return 0.5 * (w + w.T)


def _okls_warm_start(grad: jax.Array, eps: float) -> tuple[jax.Array, jax.Array]:
    """Warm-start covariance factors S_a (m x m), S_b (n x n) from the first gradient ``(m, n)``."""
    m, n = grad.shape
    frob_sq = jnp.sum(jnp.square(grad))

    ggt = grad @ grad.T
    s_a = jnp.sqrt(m / (n * frob_sq + eps)) * ggt
    s_a = 0.5 * (s_a + s_a.T)
    k_a = jnp.sqrt(jnp.sum(jnp.square(s_a))) / jnp.sqrt(m)
    s_a = s_a + (k_a + eps) * jnp.eye(m, dtype=grad.dtype)

    gtg = grad.T @ grad
    s_b = jnp.sqrt(n / (m * frob_sq + eps)) * gtg
    s_b = 0.5 * (s_b + s_b.T)
    k_b = jnp.sqrt(jnp.sum(jnp.square(s_b))) / jnp.sqrt(n)
    s_b = s_b + (k_b + eps) * jnp.eye(n, dtype=grad.dtype)
    return s_a, s_b


def _okls_core_2d(
    grad: jax.Array,
    param: jax.Array,
    momentum: jax.Array,
    s_a: jax.Array,
    s_b: jax.Array,
    p_a_prev: jax.Array,
    p_b_prev: jax.Array,
    is_first: jax.Array,
    *,
    refresh: bool,
    beta1: float,
    beta2: float,
    eps: float,
    cans_steps: int,
    matmul_dtype,
    lr: jax.Array,
    lr_peak: float,
    weight_decay: float,
    hyperball: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One OKLS step on a single 2-D matrix. Returns (delta, momentum, S_a, S_b, P_a, P_b), all fp32.

    ``p_a_prev`` / ``p_b_prev`` are the stored inverse roots from the last refresh. With ``refresh`` (a
    static flag) the roots of the updated S are recomputed and returned; otherwise the stored roots are
    reused (Shampoo's stale-preconditioner schedule). On the first step the stored roots are zeros, so a
    scaled identity from the warm-start factors stands in for them.

    ``delta`` is the parameter increment (optax adds it). Default: muP-scaled, AdamC-weight-decayed
    ``param(1 - wd*lr*lr/lr_peak) - lr*c*s*U``. With ``hyperball``: the whitened direction ``U`` is
    taken through MuonH's Frobenius norm-preserving hyperball step instead (no muP scale, no decoupled
    weight decay -- ``lr`` sets the effective step size, mirroring the MuonH matrix step).
    """
    m, n = grad.shape

    # Warm start on the first step (cheap; no extra ScaledCANS -- selected S is rooted below anyway).
    s_a_warm, s_b_warm = _okls_warm_start(grad, eps)
    s_a = jnp.where(is_first, s_a_warm, s_a)
    s_b = jnp.where(is_first, s_b_warm, s_b)

    # Preconditioners entering this step's EMA: the stored roots (a scaled identity on the first step).
    ident_a = jnp.eye(m, dtype=s_a.dtype) * jax.lax.rsqrt(jnp.mean(jnp.diag(s_a)) + eps)
    ident_b = jnp.eye(n, dtype=s_b.dtype) * jax.lax.rsqrt(jnp.mean(jnp.diag(s_b)) + eps)
    p_a = jnp.where(is_first, ident_a, p_a_prev)
    p_b = jnp.where(is_first, ident_b, p_b_prev)

    # Nesterov momentum.
    momentum = beta1 * momentum + (1.0 - beta1) * grad
    nesterov = beta1 * momentum + (1.0 - beta1) * grad

    # Symmetrized preconditioner EMAs.
    gpb = grad @ p_b
    pag = p_a @ grad
    s_a = beta2 * s_a + ((1.0 - beta2) / n) * (gpb @ gpb.T)
    s_a = 0.5 * (s_a + s_a.T) + eps * jnp.eye(m, dtype=s_a.dtype)
    s_b = beta2 * s_b + ((1.0 - beta2) / m) * (pag.T @ pag)
    s_b = 0.5 * (s_b + s_b.T) + eps * jnp.eye(n, dtype=s_b.dtype)

    # Fresh roots on refresh steps; stored ones otherwise. Then whiten.
    if refresh:
        p_a = scaled_cans_inv_sqrt(s_a, cans_steps, matmul_dtype)
        p_b = scaled_cans_inv_sqrt(s_b, cans_steps, matmul_dtype)
    whitened = p_a @ nesterov @ p_b

    if hyperball:
        # MuonH's Frobenius norm-preserving step on the whitened direction: move along U scaled to the
        # parameter's own norm, then reproject onto the sphere of radius ||param||.
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param)))
        u_norm = jnp.sqrt(jnp.sum(jnp.square(whitened)))
        moved = param - lr * whitened * param_norm / jnp.maximum(u_norm, 1e-10)
        moved_norm = jnp.sqrt(jnp.sum(jnp.square(moved)))
        delta = moved / jnp.maximum(moved_norm, 1e-10) * param_norm - param
        return delta, momentum, s_a, s_b, p_a, p_b

    # Nesterov variance correction and muP shape scale. Here (m, n) = (fan_in, fan_out) in the
    # levanter convention (matmul contracts the leading axis), so d_out = n, d_in = m.
    v_nesterov = ((1.0 - beta1) / (1.0 + beta1)) * (1.0 + 2.0 * beta1 - 2.0 * beta1**3)
    c_momentum = v_nesterov**-0.5
    fan_in, fan_out = m, n
    s_shape = (fan_out / fan_in) ** 0.5 / (fan_in**0.5 + fan_out**0.5)

    wd_coeff = weight_decay * lr * (lr / lr_peak)  # AdamC decoupled weight decay
    delta = -wd_coeff * param - lr * c_momentum * s_shape * whitened
    return delta, momentum, s_a, s_b, p_a, p_b


class ScaleByOklsState(NamedTuple):
    """OKLS state. ``momentum`` mirrors the params; ``S_a`` / ``S_b`` are the (m x m) / (n x n) EMAs."""

    count: chex.Array
    momentum: optax.Updates
    S_a: optax.Updates
    S_b: optax.Updates
    P_a: optax.Updates
    P_b: optax.Updates


def _replicate_matrix_spec(sharding) -> PartitionSpec | None:
    """PartitionSpec that keeps the leading (stack) axes' sharding and replicates the two matrix dims."""
    if sharding is None:
        return None
    spec = sharding.spec
    return PartitionSpec(*tuple(spec[:-2]), None, None)


def _make_factor_zeros(param: jax.Array, side: str) -> jax.Array:
    """Zeroed EMA factor for ``param``: leading stack dims + (m, m) for ``side='a'`` else (n, n)."""
    leading = param.shape[:-2]
    m, n = param.shape[-2], param.shape[-1]
    d = m if side == "a" else n
    zeros = jnp.zeros((*leading, d, d), dtype=jnp.float32)
    spec = _replicate_matrix_spec(_target_named_sharding(param))
    return reshard(zeros, spec) if spec is not None else zeros


def scale_with_grug_okls(
    *,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    cans_steps: int,
    matmul_dtype,
    learning_rate: jax.Array,
    lr_peak: float,
    hyperball: bool = False,
    root_every: int = 1,
) -> optax.GradientTransformation:
    """Online KL-Shampoo transform for the stacked model (2D/3D/4D matrix leaves).

    Returns parameter *increments* (LR, muP scale and decoupled weight decay baked in), matching the
    MuonH group's convention so it plugs into the same ``optax.multi_transform`` slot. The inverse roots
    are stored and recomputed every ``root_every`` steps. The refresh decision is one step-level
    ``lax.cond`` over the whole tree, so non-refresh steps skip Scaled CANS entirely.
    """
    if root_every < 1:
        raise ValueError(f"root_every must be >= 1, got {root_every}")

    def init_fn(params):
        return ScaleByOklsState(
            count=jnp.zeros([], jnp.int32),
            momentum=otu.tree_zeros_like(params),
            S_a=jax.tree.map(lambda p: _make_factor_zeros(p, "a"), params),
            S_b=jax.tree.map(lambda p: _make_factor_zeros(p, "b"), params),
            P_a=jax.tree.map(lambda p: _make_factor_zeros(p, "a"), params),
            P_b=jax.tree.map(lambda p: _make_factor_zeros(p, "b"), params),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_okls requires params (decoupled weight decay + muP scale)")
        is_first = state.count == 0
        is_leaf = lambda x: x is None  # noqa: E731
        flat_updates, treedef = jax.tree.flatten(updates, is_leaf=is_leaf)
        fields = (params, state.momentum, state.S_a, state.S_b, state.P_a, state.P_b)
        rest = [treedef.flatten_up_to(t) for t in fields]

        def run(refresh: bool):
            def leaf(update, param, momentum, s_a, s_b, p_a, p_b):
                if not hasattr(param, "ndim") or param.ndim not in (2, 3, 4):
                    return update, momentum, s_a, s_b, p_a, p_b

                def core(g, p, mo, sa, sb, pa, pb):
                    return _okls_core_2d(
                        g,
                        p,
                        mo,
                        sa,
                        sb,
                        pa,
                        pb,
                        is_first,
                        refresh=refresh,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        cans_steps=cans_steps,
                        matmul_dtype=matmul_dtype,
                        lr=learning_rate,
                        lr_peak=lr_peak,
                        weight_decay=weight_decay,
                        hyperball=hyperball,
                    )

                for _ in range(param.ndim - 2):
                    core = jax.vmap(core)
                spec = _replicate_matrix_spec(_target_named_sharding(param))
                args = [update.astype(jnp.float32), param.astype(jnp.float32), momentum, s_a, s_b, p_a, p_b]
                if spec is not None:
                    args = [reshard(a, spec) for a in args]
                return core(*args)

            results = [leaf(u, *r) for u, *r in zip(flat_updates, *rest, strict=True)]
            return tuple(treedef.unflatten([r[i] for r in results]) for i in range(6))

        if root_every == 1:
            outs = run(True)
        else:
            outs = jax.lax.cond(state.count % root_every == 0, lambda: run(True), lambda: run(False))
        deltas, momenta, s_as, s_bs, p_as, p_bs = outs
        new_state = ScaleByOklsState(
            count=optax.safe_increment(state.count), momentum=momenta, S_a=s_as, S_b=s_bs, P_a=p_as, P_b=p_bs
        )
        return deltas, new_state

    return optax.GradientTransformation(init_fn, update_fn)


OKLS_MATMUL_DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}
