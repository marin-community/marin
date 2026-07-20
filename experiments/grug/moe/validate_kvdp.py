# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU validation for the MLA KV down-projection ablations (#7425).

Checks: (1) slice-first drops w_dkv and runs; (2) alternating slice actually selects different
residual halves per layer parity; (3) freezing w_dkv zeros its update (incl. weight decay).
Run: uv run python -m experiments.grug.moe.validate_kvdp
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.model import MultiheadLatentAttention
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig

HIDDEN, CKV = 1024, 512


def main() -> None:
    devices = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "expert", "model"), axis_types=(AxisType.Explicit,) * 4)
    base = MoeHeuristic().build_model_config(HIDDEN, seq_len=4096)
    base = dataclasses.replace(
        base,
        head_dim=128,
        num_heads=8,
        num_layers=12,
        max_seq_len=4096,
        attention_implementation="reference",
        use_array_stacked_blocks=False,
        use_mla=True,
        kv_lora_rank=CKV,
        q_lora_rank=CKV,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_scale_kv_lora=True,
        mla_scale_q_lora=True,
    )
    from levanter.grug.attention import AttentionMask  # noqa: PLC0415

    mask = AttentionMask.causal()
    # craft x whose two halves are clearly distinct so an alternating slice changes the output
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, HIDDEN))
    x = x.at[..., CKV:].multiply(5.0)
    ok = True
    with jax.set_mesh(mesh):
        # 1. baseline (matrix) + slice-first: structure + finite forward
        for label, over in [("baseline", {}), ("slice-first", {"mla_kv_slice": True})]:
            cfg = dataclasses.replace(base, **over)
            m = MultiheadLatentAttention.init(cfg, key=jax.random.PRNGKey(0))
            has_wdkv = m.w_dkv is not None
            out = jax.jit(lambda mm, xx: mm(xx, mask))(m, x)
            finite = bool(jnp.all(jnp.isfinite(out)))
            want_wdkv = not over.get("mla_kv_slice", False)
            good = finite and out.shape == x.shape and has_wdkv == want_wdkv
            ok = ok and good
            print(f"{label:14s} {'OK' if good else 'FAIL'} w_dkv={'present' if has_wdkv else 'None'} finite={finite}")

        # 2. alternating slice: lower vs upper parity must give different outputs on this x
        cfg = dataclasses.replace(base, mla_kv_slice=True, mla_kv_slice_alternate=True)
        m = MultiheadLatentAttention.init(cfg, key=jax.random.PRNGKey(0))
        out_lo = jax.jit(lambda mm, xx: mm(xx, mask, kv_slice_upper=False))(m, x)
        out_hi = jax.jit(lambda mm, xx: mm(xx, mask, kv_slice_upper=True))(m, x)
        dmax = float(jnp.abs(out_lo - out_hi).max())
        differ = dmax > 1e-3
        print(f"{'alternate':14s} {'OK' if differ else 'FAIL'} lower!=upper max|Δ|={dmax:.3f}")
        ok = ok and differ

        # 3. freeze: w_dkv update must be exactly zero (grad AND weight decay), others nonzero
        m = MultiheadLatentAttention.init(base, key=jax.random.PRNGKey(0))
        params = eqx.filter(m, eqx.is_inexact_array)
        opt = GrugMoeMuonHConfig(learning_rate=0.01, adam_lr=0.001, freeze_substrings=("w_dkv",)).build(100)
        state = opt.init(params)
        grads = jax.tree.map(lambda a: jnp.ones_like(a), params)
        updates, _ = opt.update(grads, state, params)
        du_wdkv = float(jnp.abs(updates.w_dkv).max())
        du_wuk = float(jnp.abs(updates.w_uk).max())
        frozen_ok = du_wdkv == 0.0 and du_wuk > 0.0
        print(
            f"{'freeze w_dkv':14s} {'OK' if frozen_ok else 'FAIL'} |Δw_dkv|={du_wdkv:.2e} (0) |Δw_uk|={du_wuk:.2e} (>0)"
        )
        ok = ok and frozen_ok
    print("\nVALIDATE_OK" if ok else "\nVALIDATE_FAIL")


if __name__ == "__main__":
    main()
