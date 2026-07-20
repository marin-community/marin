# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight analysis of the MLA KV down-projection ``w_dkv`` (#7424).

Reads the learned ``w_dkv`` (X -> c_kv, ``[L, D, Ckv]`` = ``[12, 1024, 512]``) and the KV-latent
RMSNorm gain from a restored grug state, and characterizes the 512 latent read-directions in the
D-dim residual stream: (1) per-layer effective rank, (2) within-layer column coherence, (3)
cross-layer subspace alignment. Numeric results go to stdout; raw arrays + plots are saved and
uploaded to the wandb run so they can be fetched afterwards.

Invoked from ``train.py`` after checkpoint restore when ``SCALE_ANALYZE_WDKV`` is set.
"""

import logging
import os

import jax
import numpy as np

logger = logging.getLogger(__name__)


def _extract(params) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``w_dkv`` ``[L, D, Ckv]`` and kv_norm gain ``[L, Ckv]`` as host numpy (scanned or unrolled)."""
    stacked = getattr(params, "stacked_blocks", None)
    if stacked is not None:
        attn = stacked.stacked.attn
        return np.asarray(jax.device_get(attn.w_dkv), np.float64), np.asarray(
            jax.device_get(attn.kv_norm.weight), np.float64
        )
    ws, gs = [], []
    for b in params.blocks:
        ws.append(np.asarray(jax.device_get(b.attn.w_dkv), np.float64))
        gs.append(np.asarray(jax.device_get(b.attn.kv_norm.weight), np.float64))
    return np.stack(ws), np.stack(gs)


def _effective_ranks(s: np.ndarray) -> dict:
    """Effective-rank estimators from singular values ``s`` (sorted descending)."""
    s2 = s**2
    energy = np.cumsum(s2) / np.sum(s2)
    p = s2 / np.sum(s2)
    return {
        "energy90": int(np.searchsorted(energy, 0.90) + 1),
        "energy99": int(np.searchsorted(energy, 0.99) + 1),
        "stable": float(np.sum(s2) / s2[0]),  # ||W||_F^2 / sigma_max^2
        "entropy": float(np.exp(-np.sum(p * np.log(p + 1e-300)))),  # exp(spectral entropy)
        "smax": float(s[0]),
        "smin": float(s[-1]),
        "cond": float(s[0] / s[-1]),
    }


def _coherence(W: np.ndarray) -> dict:
    """Pairwise coherence of the 512 columns (unit-normalized), gain-invariant."""
    C = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    G = C.T @ C
    off = G[~np.eye(G.shape[0], dtype=bool)]
    return {
        "mean_abs": float(np.mean(np.abs(off))),
        "max_abs": float(np.max(np.abs(off))),
        "rms": float(np.sqrt(np.mean(off**2))),
    }


def _left_basis(W: np.ndarray, r: int) -> np.ndarray:
    return np.linalg.svd(W, full_matrices=False)[0][:, :r]


def _subspace_overlap(Ui: np.ndarray, Uj: np.ndarray) -> float:
    """Mean cos^2 of principal angles between two orthonormal bases (1 = identical subspace)."""
    sv = np.linalg.svd(Ui.T @ Uj, compute_uv=False)
    return float(np.mean(np.clip(sv, 0.0, 1.0) ** 2))


def _analyze_set(name: str, W: np.ndarray) -> dict:
    """W: [L, D, Ckv]. Returns per-layer ranks/coherence + cross-layer overlap matrix."""
    L = W.shape[0]
    svals = [np.linalg.svd(W[i], compute_uv=False) for i in range(L)]
    ranks = [_effective_ranks(s) for s in svals]
    coh = [_coherence(W[i]) for i in range(L)]
    r = int(np.clip(round(np.mean([rk["energy99"] for rk in ranks])), 1, W.shape[2]))
    bases = [_left_basis(W[i], r) for i in range(L)]
    overlap = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            overlap[i, j] = overlap[j, i] = _subspace_overlap(bases[i], bases[j])
    return {"name": name, "svals": svals, "ranks": ranks, "coh": coh, "overlap_r": r, "overlap": overlap}


def _random_baseline(D: int, Ckv: int) -> dict:
    rng = np.random.default_rng(0)
    G = rng.standard_normal((D, Ckv))
    s = np.linalg.svd(G, compute_uv=False)
    return {"ranks": _effective_ranks(s), "coh": _coherence(G)}


def _plots(raw: dict, scaled: dict, base: dict, out_dir: str) -> list[str]:
    import matplotlib  # noqa: PLC0415, ICN001

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    L = len(raw["svals"])
    paths = []
    # 1. singular-value spectra (raw), log-y
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(L):
        ax.plot(raw["svals"][i], lw=1, alpha=0.8, label=f"L{i}")
    ax.set_yscale("log")
    ax.set_xlabel("singular index (0..511)")
    ax.set_ylabel("singular value")
    ax.set_title("w_dkv singular-value spectrum per layer (raw)")
    ax.legend(fontsize=6, ncol=2)
    p = f"{out_dir}/wdkv_sv_spectra.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 2. effective rank + coherence vs layer
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    layers = np.arange(L)
    axs[0].plot(layers, [r["energy99"] for r in raw["ranks"]], "o-", label="energy99 (raw)")
    axs[0].plot(layers, [r["energy90"] for r in raw["ranks"]], "s-", label="energy90 (raw)")
    axs[0].plot(layers, [r["entropy"] for r in raw["ranks"]], "^-", label="entropy (raw)")
    axs[0].plot(layers, [r["stable"] for r in raw["ranks"]], "d-", label="stable (raw)")
    axs[0].axhline(base["ranks"]["energy99"], ls="--", c="gray", label="rand energy99")
    axs[0].axhline(512, ls=":", c="k", label="full (512)")
    axs[0].set_xlabel("layer")
    axs[0].set_ylabel("effective rank")
    axs[0].set_title("w_dkv effective rank")
    axs[0].legend(fontsize=7)
    axs[1].plot(layers, [c["mean_abs"] for c in raw["coh"]], "o-", label="mean |cos| (raw)")
    axs[1].plot(layers, [c["max_abs"] for c in raw["coh"]], "s-", label="max |cos| (raw)")
    axs[1].axhline(base["coh"]["mean_abs"], ls="--", c="gray", label="rand mean |cos|")
    axs[1].set_xlabel("layer")
    axs[1].set_ylabel("column coherence")
    axs[1].set_title("w_dkv within-layer column overlap")
    axs[1].legend(fontsize=7)
    p = f"{out_dir}/wdkv_rank_coherence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 3. cross-layer subspace overlap heatmaps (raw + scaled)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, d, ttl in ((axs[0], raw, "raw"), (axs[1], scaled, "gain-scaled")):
        im = ax.imshow(d["overlap"], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"cross-layer subspace overlap ({ttl}), r={d['overlap_r']}")
        ax.set_xlabel("layer j")
        ax.set_ylabel("layer i")
        fig.colorbar(im, ax=ax, fraction=0.046)
    p = f"{out_dir}/wdkv_cross_layer.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def _probe_params(params) -> None:
    """Audit every leaf (path, shape, std, max|abs|, and |w-1| for .weight) to confirm restore populated
    the trained values rather than leaving leaves at init."""
    from jax.tree_util import keystr, tree_leaves_with_path  # noqa: PLC0415

    print("=== PARAM_PROBE_BEGIN ===", flush=True)
    for path, leaf in tree_leaves_with_path(params):
        a = np.asarray(jax.device_get(leaf), np.float64)
        k = keystr(path)
        extra = f" max|w-1|={np.abs(a - 1).max():.3e}" if k.endswith(".weight") else ""
        stats = f"std={a.std():.5f} mean={a.mean():.2e} maxabs={np.abs(a).max():.4f}"
        print(f"{k} shape={a.shape} {stats}{extra}", flush=True)
    print("=== PARAM_PROBE_END ===", flush=True)


def run_wdkv_analysis(params) -> None:
    """Extract w_dkv + kv_norm gain, analyze, print results, save/upload arrays + plots."""
    _probe_params(params)
    wdkv, gain = _extract(params)  # [L,D,Ckv], [L,Ckv]
    L, D, Ckv = wdkv.shape
    wdkv_scaled = wdkv * gain[:, None, :]  # scale each column by its learned kv_norm gain
    logger.info(f"[wdkv] extracted w_dkv {wdkv.shape}, gain {gain.shape}")

    raw = _analyze_set("raw", wdkv)
    scaled = _analyze_set("gain-scaled", wdkv_scaled)
    base = _random_baseline(D, Ckv)

    print("=== WDKV_ANALYSIS_BEGIN ===", flush=True)
    print(f"shape L={L} D={D} Ckv={Ckv}", flush=True)
    print(
        f"random-baseline [{D}x{Ckv}] iid-normal: energy99={base['ranks']['energy99']} "
        f"entropy={base['ranks']['entropy']:.1f} stable={base['ranks']['stable']:.1f} "
        f"mean|cos|={base['coh']['mean_abs']:.4f} max|cos|={base['coh']['max_abs']:.3f}",
        flush=True,
    )
    print("\n-- per-layer effective rank (raw w_dkv), Ckv=512 ceiling --", flush=True)
    print(
        f"{'L':>2} {'e90':>4} {'e99':>4} {'entropy':>8} {'stable':>7} {'smax':>8} {'cond':>8} "
        f"{'mean|cos|':>9} {'max|cos|':>8}",
        flush=True,
    )
    for i in range(L):
        r, c = raw["ranks"][i], raw["coh"][i]
        print(
            f"{i:>2} {r['energy90']:>4} {r['energy99']:>4} {r['entropy']:>8.1f} {r['stable']:>7.2f} "
            f"{r['smax']:>8.3f} {r['cond']:>8.1f} {c['mean_abs']:>9.4f} {c['max_abs']:>8.3f}",
            flush=True,
        )
    print(
        f"\n-- cross-layer subspace overlap (raw, r={raw['overlap_r']}, mean cos^2 of principal angles) --", flush=True
    )
    print("     " + " ".join(f"L{j:>2}" for j in range(L)), flush=True)
    for i in range(L):
        print(f"L{i:>2} " + " ".join(f"{raw['overlap'][i, j]:>4.2f}" for j in range(L)), flush=True)
    om = raw["overlap"][~np.eye(L, dtype=bool)]
    print(f"\ncross-layer off-diagonal overlap: mean={om.mean():.3f} min={om.min():.3f} max={om.max():.3f}", flush=True)
    adj = [raw["overlap"][i, i + 1] for i in range(L - 1)]
    print(f"adjacent-layer overlap (i,i+1): {[round(a, 2) for a in adj]}", flush=True)
    print("=== WDKV_ANALYSIS_END ===", flush=True)

    out_dir = os.environ.get("WDKV_OUT_DIR", "/tmp/wdkv")
    os.makedirs(out_dir, exist_ok=True)
    npz = f"{out_dir}/wdkv_arrays.npz"
    np.savez_compressed(
        npz,
        wdkv=wdkv.astype(np.float32),
        gain=gain.astype(np.float32),
        overlap_raw=raw["overlap"],
        overlap_scaled=scaled["overlap"],
    )
    try:
        paths = _plots(raw, scaled, base, out_dir)
    except Exception as e:  # plotting is best-effort; numeric results already printed
        logger.warning(f"[wdkv] plotting failed: {e}")
        paths = []
    # Upload arrays + plots to the active wandb run so they can be fetched afterwards.
    try:
        import wandb  # noqa: PLC0415

        if wandb.run is not None:
            for f in [npz, *paths]:
                wandb.save(f, base_path=out_dir, policy="now")
            logger.info(f"[wdkv] uploaded {1 + len(paths)} files to wandb run {wandb.run.name}")
    except Exception as e:
        logger.warning(f"[wdkv] wandb upload failed: {e}")
