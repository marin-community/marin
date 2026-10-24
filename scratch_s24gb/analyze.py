# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-GEMM table and step-time / memory projection for 2:4 and 2:4T on the lc1 GB200 MXFP8 recipe.

Inputs: bench2.log (torch: cuBLAS dense, cuSPARSELt 2:4, Triton v1 kernels, cuSPARSELt compress),
bench3.log (JAX: our QuACK grouped GEMMs bf16 + MXFP8, Pallas quantizers; torch: v2 fused quant+compress kernels).
Writes page_data.json."""

import json
import os

from load import load

D = os.path.dirname(os.path.abspath(__file__))
E, L, F, H = 6, 3072, 4608, 6144
ME = 109312
R = E * ME  # rows per GPU per layer in the bench (655872; recipe mean 655360, capacity 688128)
LAYERS, FULL_LAYERS = 48, 2  # latent_out_full_layers=4,5 write 6144
STEP = 14.59

b2, b3 = load(os.path.join(D, "bench2.log")), load(os.path.join(D, "bench3.log"))
b4 = load(os.path.join(D, "bench4.log"))


def torch_gemm(shape, M):
    """Per-expert kernel us of every variant at one shape, from the individual RESULT lines."""
    key = "K" if "wgrad" in shape else "M"
    out = {d["name"]: d["us"] for d in b2 if d.get("shape") == shape and d.get(key) == M and "us" in d}
    if not out:
        raise KeyError((shape, M))
    return out


def jax_us(name, m_e, out_dim=L):
    for d in b3:
        if d["name"] == name and d.get("m_e") == m_e and d.get("out_dim") == out_dim and "us" in d:
            return min(d["us"], d["wall_us"])  # profiler sum vs back-to-back wall time: take the cleaner one
    raise KeyError((name, m_e, out_dim))


def v2(name, rows):
    for d in b3:
        if d["name"] == name and d.get("R") == rows and "us" in d:
            return d["us"]
    for d in b2:
        if d["name"] == name and d.get("R") == rows and "us" in d:
            return d["us"]
    raise KeyError((name, rows))


def _get(rows_list, name, rows):
    for d in rows_list:
        if d["name"] == name and d.get("R") == rows and "us" in d:
            return min(d["us"], d.get("wall_us", d["us"]))
    return None


def best(d, *keys):
    vals = [d[k] for k in keys if d.get(k)]
    return min(vals)


def gemm_table(m_e):
    """Per-layer (6 experts) us for each role and variant."""
    rows = []
    up, down = torch_gemm("up", m_e), torch_gemm("down", m_e)
    wg = torch_gemm("wgrad", m_e)
    roles = [
        ("fwd up  X·W_up", "up", "mxfp8_up", "bf16_up", up, "dense", "dense"),
        ("fwd down  post·W_down", "down", "mxfp8_down", "bf16_down", down, "K", "K"),
        ("dpost  dY·W_downᵀ", "dpost", "mxfp8_dpost", "bf16_dpost", up, "dense", "dense"),
        ("dW_down  postᵀ·dY", "dWd", None, "bf16_dWd", wg, "dense", "T"),
        ("dX  dpre·W_upᵀ", "dX", "mxfp8_dX", "bf16_dX", down, "K", "K"),
        ("dW_up  Xᵀ·dpre", "dWu", None, "bf16_dWu", wg, "dense", "T"),
    ]
    for label, role, mx, bf, t, s24, s24t in roles:
        r = dict(role=role, label=label, sparse_24=s24, sparse_24t=s24t, m_e=m_e)
        r["quack_bf16"] = jax_us(bf, m_e)
        r["quack_mxfp8"] = jax_us(mx, m_e) if mx else None
        r["cublas_bf16"] = E * t["bf16_dense"] if "bf16_dense" in t else None  # one log line interleaved away
        r["cublas_mxfp8"] = E * t["mxfp8_dense_cublas"]
        r["cslt_bf16_24"] = E * best(t, "bf16_sparse_cslt", "nvm_bf16_sparse")
        r["cslt_fp8_24"] = E * best(t, "fp8_sparse_cslt", "nvm_fp8_sparse")
        r["cslt_mxfp8_24"] = E * t["nvm_mxfp8_sparse"]
        r["cslt_mxfp8_compress"] = E * t["nvm_mxfp8_compress"]
        r["speedup_mx_vs_cublas"] = t["mxfp8_dense_cublas"] / t["nvm_mxfp8_sparse"]
        r["speedup_bf16_vs_cublas"] = (
            t["bf16_dense"] / best(t, "bf16_sparse_cslt", "nvm_bf16_sparse") if "bf16_dense" in t else None
        )
        rows.append(r)
    return rows


def compress_costs(rows):
    c = dict(R=rows)
    c["quant_rows"] = v2("triton_quant_rows", rows)
    c["quant_rows_cols"] = v2("triton_quant_rows_cols", rows)
    c["sp24_rows"] = min(
        v2(n, rows)
        for n in ("v2_sp24_rows_bm32w4", "v2_sp24_rows_bm64w4", "v2_sp24_rows_bm128w4", "v2_sp24_rows_bm128w8")
    )
    c["sp24t"] = min(
        [v2(n, rows) for n in ("v2_sp24t_w4", "v2_sp24t_w8")]
        + [x for x in (_get(b4, n, rows) for n in ("v3_sp24t_w4", "v3_sp24t_w8")) if x]
    )
    c["sp24_rows"] = min(
        [c["sp24_rows"]] + [x for x in (_get(b4, n, rows) for n in ("v3_sp24_rows_bm32w4", "v3_sp24_rows_bm64w8")) if x]
    )
    c["torch_tile_prune_both"] = _get(b2, "torch_tile_prune_both", rows)
    c["cslt_compress_bf16"] = _get(b2, "cslt_compress_bf16", rows)
    c["cslt_compress_fp8"] = _get(b2, "cslt_compress_fp8", rows)
    c["pallas_quant_rows"] = _get(b3, "pallas_quant_rows", rows)
    c["pallas_quant_rows_cols"] = _get(b3, "pallas_quant_rows_cols", rows)
    c["v1_sp24_rows"] = min(
        filter(None, (_get(b2, n, rows) for n in ("triton_sp24_rows_16x512w4", "triton_sp24_rows_16x256w2"))),
        default=None,
    )
    c["v1_sp24t"] = _get(b2, "triton_sp24t_64x64w4", rows)
    c["copy_bf16"] = v2("copy_bf16", rows)
    c["gbytes_in"] = rows * F * 2 / 1e9
    return c


def projection(tab, tab_full, comp, pallas_cols_3072):
    g = {r["role"]: r for r in tab}
    gf = {r["role"]: r for r in tab_full}
    base = (
        2 * g["up"]["quack_mxfp8"]
        + g["down"]["quack_mxfp8"]
        + g["dpost"]["quack_mxfp8"]
        + g["dX"]["quack_mxfp8"]
        + g["dWu"]["quack_bf16"]
        + g["dWd"]["quack_bf16"]
    )
    # extra epilogue bytes when the GEMM must write bf16 instead of e4m3 for a standalone compress pass: 1 B / element
    extra_write = comp["copy_bf16"] / 4
    out = {"base_layer_us": base}

    def k_sparse(role, how):
        r = g[role]
        return r["cslt_mxfp8_24"] if how == "abs" else r["quack_mxfp8"] / r["speedup_mx_vs_cublas"]

    def t_sparse_bf16(role, how):
        r = g[role]
        return r["cslt_bf16_24"] if how == "abs" else r["quack_bf16"] / r["speedup_bf16_vs_cublas"]

    variants = {}
    unf24 = 2 * (comp["sp24_rows"] + extra_write)
    unf24t = 3 * comp["sp24t"] + 2 * extra_write
    out["unfused_cost_24_us"], out["unfused_cost_24t_us"], out["extra_write_us"] = unf24, unf24t, extra_write
    for how in ("ratio", "abs"):
        dK = (g["down"]["quack_mxfp8"] - k_sparse("down", how)) + (g["dX"]["quack_mxfp8"] - k_sparse("dX", how))
        dT = sum(g[x]["quack_bf16"] - t_sparse_bf16(x, how) for x in ("dWd", "dWu"))
        variants[f"24_{how}_fused"] = dK
        variants[f"24_{how}_unfused"] = dK - unf24
        variants[f"24T_{how}_fused"] = dK + dT
        variants[f"24T_{how}_unfused"] = dK + dT - unf24t
        out[f"dK_{how}"], out[f"dT_bf16_{how}"] = dK, dT
    # 2:4T on top of MXFP8 weight gradients (not the recipe today): dense MXFP8 wgrad (our QuACK block-scaled GEMM, one
    # expert x 6) vs the cuSPARSELt MXFP8 2:4 / cuBLAS MXFP8 ratio
    mx_wg = {x: E * jax_us("mxfp8_dWd_one_expert", ME) for x in ("dWd", "dWu")}
    out["mx_wgrad_dense_layer_us"] = sum(mx_wg.values())
    out["mx_wgrad_sparse_layer_us"] = sum(v * g[x]["cslt_mxfp8_24"] / g[x]["cublas_mxfp8"] for x, v in mx_wg.items())
    # full-width layers: down (K-sparse) with 6144 out; scale the K-sparse saving by the measured full-width ratio
    full_extra = {}
    if gf:
        d = gf["down"]
        full_extra["down_saving"] = d["quack_mxfp8"] - d["quack_mxfp8"] / d["speedup_mx_vs_cublas"]
    out["variants_layer_us"] = variants
    base_full = base
    if gf:
        base_full = (
            base
            + (gf["down"]["quack_mxfp8"] - g["down"]["quack_mxfp8"])
            + (gf["dpost"]["quack_mxfp8"] - g["dpost"]["quack_mxfp8"])
            + (gf["dWd"]["quack_bf16"] - g["dWd"]["quack_bf16"])
        )
    out["base_full_layer_us"] = base_full
    expert_step = ((LAYERS - FULL_LAYERS) * base + FULL_LAYERS * base_full) / 1e6
    out["expert_gemm_step_s"] = expert_step
    out["expert_gemm_frac"] = expert_step / STEP
    out["step_delta_s"] = {}
    for k, v in variants.items():
        dfull = v + (
            (full_extra.get("down_saving", 0) - (g["down"]["quack_mxfp8"] - k_sparse("down", "ratio")))
            if "ratio" in k
            else 0
        )
        sec = ((LAYERS - FULL_LAYERS) * v + FULL_LAYERS * dfull) / 1e6
        out["step_delta_s"][k] = -sec
    # inference / forward only (prefill-sized M): up + down, compress fused into the up epilogue
    fwd = g["up"]["quack_mxfp8"] + g["down"]["quack_mxfp8"]
    out["infer_fwd_saving_ratio"] = (g["down"]["quack_mxfp8"] - k_sparse("down", "ratio")) / fwd
    out["infer_fwd_saving_abs"] = (g["down"]["quack_mxfp8"] - k_sparse("down", "abs")) / fwd
    out["infer_fwd_saving_unfused"] = (
        g["down"]["quack_mxfp8"] - k_sparse("down", "ratio") - comp["sp24_rows"] - extra_write
    ) / fwd
    return out


def memory():
    cap = 688128  # capacity rows per GPU (655360 x 1.05)
    gb = 1e9
    mx_dense = cap * F * (1 + 1 / 32) / gb
    mx_24 = cap * F * (0.5 + 0.125 + 1 / 64) / gb
    bf = cap * F * 2 / gb
    bf_24 = cap * F * (1 + 0.125) / gb
    saved = 65536 * L * 2 / gb
    return dict(
        saved_routed_per_layer_gb=saved,
        saved_routed_total_gb=saved * LAYERS,
        compressed_post_per_layer_gb=mx_24,
        compressed_post_total_gb=mx_24 * LAYERS,
        post_mx_dense_gb=mx_dense,
        post_mx_24_gb=mx_24,
        post_bf16_gb=bf,
        post_bf16_24_gb=bf_24,
        x_cols_mx_gb=cap * L * (1 + 1 / 64) / gb,
        delta_24_gb=-(mx_dense - mx_24),  # dpre e4m3 -> compressed (bwd), post e4m3 -> compressed (fwd)
        delta_24t_bf16wg_gb=-(mx_dense - mx_24) - 2 * (bf - bf_24),
        delta_24t_mxwg_gb=-(mx_dense - mx_24) - 2 * (bf - mx_24) + 2 * cap * L * (1 + 1 / 64) / gb,
    )


def main():
    tabs = {m: gemm_table(m) for m in (4096, 16384, 65536, ME)}
    try:
        tf = gemm_table_full()
    except KeyError:
        tf = []
    comps = {}
    for rows in (24576, 98304, 393216, 655872):
        try:
            comps[rows] = compress_costs(rows)
        except (KeyError, ValueError):
            pass
    pallas_cols = None
    for d in b3:
        if d["name"] == "pallas_quant_cols" and d.get("R") == 393216 and "us" in d:
            pallas_cols = d["us"] * (655872 / 393216) * (L / F)
    proj = projection(tabs[ME], tf, comps[655872], pallas_cols or 0.0)
    pallas = {d["name"] + f"@{d['R']}": d["us"] for d in b3 if d["name"].startswith("pallas_") and "us" in d}
    out = dict(
        tables={str(k): v for k, v in tabs.items()},
        full=tf,
        compress=comps,
        projection=proj,
        memory=memory(),
        pallas=pallas,
        pallas_cols_3072_layer=pallas_cols,
        v2_check=[d for d in b3 if d["name"] == "v2_check"],
        cslt_compress=[d for d in b2 if d["name"].startswith("nvm_") and "compress" in d["name"]],
    )
    with open(os.path.join(D, "page_data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(proj, indent=1))
    print(json.dumps(memory(), indent=1))
    for r in tabs[ME]:
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    for c in comps.values():
        print({a: round(b, 1) if isinstance(b, float) else b for a, b in c.items()})


def gemm_table_full():
    d_down, d_up, d_wg = torch_gemm("down_full", ME), torch_gemm("up_full", ME), torch_gemm("wgrad_full", ME)
    rows = []
    for role, mx, bf, t in (
        ("down", "mxfp8_down", "bf16_down", d_down),
        ("dpost", "mxfp8_dpost", "bf16_dpost", d_up),
        ("dWd", None, "bf16_dWd", d_wg),
    ):
        r = dict(
            role=role,
            quack_bf16=jax_us(bf, ME, H),
            quack_mxfp8=jax_us(mx, ME, H) if mx else None,
            cublas_mxfp8=E * t["mxfp8_dense_cublas"],
            cslt_mxfp8_24=E * t["nvm_mxfp8_sparse"],
            speedup_mx_vs_cublas=t["mxfp8_dense_cublas"] / t["nvm_mxfp8_sparse"],
        )
        rows.append(r)
    return rows


if __name__ == "__main__":
    main()
