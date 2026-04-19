# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001

"""Plot distograms produced by `eval_protein_distogram.py`.

Reads the `summary.json` + `distogram_n{N}.npz` files from the eval output dir
and either writes a single multi-page PDF report (default) and/or a directory of
PNGs. Works on CPU; no accelerator needed.

Usage::

    uv run python -m experiments.protein.plot_distogram \\
        --input-dir gs://marin-us-east5/eval/protein-distogram/1qys/run-01 \\
        --output-pdf /tmp/protein-plots/1qys-run-01.pdf
"""

import argparse
import io
import json
import logging
import os
import sys

import fsspec
import numpy as np
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

CB_CONTACT_ANGSTROMS = 8.0


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with fsspec.open(path, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
        return {k: data[k] for k in data.files}


def _load_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _save_fig(fig, path: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    with fsspec.open(path, "wb") as f:
        f.write(buf.getvalue())


def _expected_and_argmax(probs: np.ndarray, midpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exp = (probs * midpoints[None, None, :]).sum(axis=-1)
    argmax = midpoints[probs.argmax(axis=-1)]
    return exp, argmax


def _mask_valid(gt: np.ndarray, max_A: float) -> np.ndarray:
    n = gt.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (ii != jj) & np.isfinite(gt) & (gt <= max_A)


def plot_distogram_heatmaps(expected: np.ndarray, gt: np.ndarray, max_A: float, title: str):
    """Side-by-side expected-distance and GT-distance heatmaps, plus residual."""
    import matplotlib.pyplot as plt

    n = expected.shape[0]
    residual = expected - gt
    residual_masked = np.where(np.isfinite(gt) & (gt <= max_A), residual, np.nan)
    # Clip GT > max_A for display
    gt_display = np.where(gt <= max_A, gt, max_A)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    kwargs = dict(origin="lower", vmin=0, vmax=max_A, cmap="viridis_r")
    im0 = axes[0].imshow(expected, **kwargs, extent=(0.5, n + 0.5, 0.5, n + 0.5))
    axes[0].set_title("Predicted (expected distance, Å)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(gt_display, **kwargs, extent=(0.5, n + 0.5, 0.5, n + 0.5))
    axes[1].set_title("Ground truth CB–CB (Å)")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    vmax = float(np.nanmax(np.abs(residual_masked))) if np.any(np.isfinite(residual_masked)) else 5.0
    vmax = max(vmax, 1.0)
    im2 = axes[2].imshow(
        residual_masked, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r", extent=(0.5, n + 0.5, 0.5, n + 0.5)
    )
    axes[2].set_title("Residual: predicted − GT (Å)")
    axes[2].set_xlabel("j")
    axes[2].set_ylabel("i")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    return fig


def plot_scatter_pred_vs_gt(expected: np.ndarray, gt: np.ndarray, max_A: float, title: str):
    """Scatter of predicted expected-distance vs GT, colored by |i-j|."""
    import matplotlib.pyplot as plt

    n = expected.shape[0]
    mask = _mask_valid(gt, max_A)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sep = np.abs(ii - jj)

    x = gt[mask]
    y = expected[mask]
    s = sep[mask]

    fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    sc = ax.scatter(x, y, c=s, cmap="plasma", s=4, alpha=0.35)
    ax.plot([0, max_A], [0, max_A], "k--", lw=1)
    ax.set_xlim(0, max_A)
    ax.set_ylim(0, max_A)
    ax.set_xlabel("Ground truth CB–CB distance (Å)")
    ax.set_ylabel("Predicted expected distance (Å)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="|i − j|", fraction=0.046)

    mae = float(np.abs(y - x).mean())
    bias = float((y - x).mean())
    ax.set_title(f"{title}\nMAE={mae:.2f}Å  bias={bias:+.2f}Å  n={int(mask.sum())}")
    return fig


def plot_contact_map(probs: np.ndarray, gt: np.ndarray, bin_edges: np.ndarray, contact_A: float, title: str):
    """Heatmap of P(d ≤ contact_A) with GT contact pairs marked."""
    import matplotlib.pyplot as plt

    # Bin k covers ((k+1-1)*0.5, (k+1)*0.5]. Bins covering (0, 8] are k=0..15.
    last_contact_bin = int(np.searchsorted(bin_edges, contact_A, side="left"))
    p_contact = probs[:, :, : last_contact_bin + 1].sum(axis=-1)

    n = p_contact.shape[0]
    gt_contact = (gt < contact_A) & np.isfinite(gt)
    ii, jj = np.where(gt_contact & ~np.eye(n, dtype=bool))

    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    im = ax.imshow(p_contact, origin="lower", vmin=0, vmax=1, cmap="magma", extent=(0.5, n + 0.5, 0.5, n + 0.5))
    # Overlay GT contacts as small white dots
    ax.scatter(jj + 1, ii + 1, s=4, facecolors="none", edgecolors="white", linewidths=0.35)
    ax.set_title(f"{title}\nP(d ≤ {contact_A:.0f} Å); white dots = GT contacts")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046, label=f"P(d ≤ {contact_A:.0f} Å)")
    return fig


def plot_summary_curves(summary: dict):
    """One figure with MAE-vs-N and asym-vs-N overlays."""
    import matplotlib.pyplot as plt

    ns = [r["n_prompt_contacts"] for r in summary["per_n"]]
    exp_mae = [r["metrics"]["expected_mean_abs_err_A"] for r in summary["per_n"]]
    arg_mae = [r["metrics"]["argmax_mean_abs_err_A"] for r in summary["per_n"]]
    asym = [r["metrics"]["expected_order_asymmetry_mean_A"] for r in summary["per_n"]]
    contact_corr = [r["metrics"]["contact_prob_auc_proxy_corr"] for r in summary["per_n"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(ns, exp_mae, "o-", label="Expected MAE")
    axes[0].plot(ns, arg_mae, "s-", label="Argmax MAE")
    axes[0].plot(ns, asym, "^-", label="Order asymmetry (mean)")
    axes[0].set_xlabel("# seeded GT long-range contacts (N)")
    axes[0].set_ylabel("Å")
    axes[0].set_title("Distance errors vs. seeding")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ns, contact_corr, "o-", color="tab:green")
    axes[1].set_xlabel("# seeded GT long-range contacts (N)")
    axes[1].set_ylabel("corr(P(d≤8Å), GT contact)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Contact-prob correlation vs. seeding")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{summary['pdb_id']} — distogram sweep")
    return fig


def plot_per_pair_pmf_grid(
    probs: np.ndarray,
    gt: np.ndarray,
    bin_midpoints: np.ndarray,
    bin_edges: np.ndarray,
    pairs: list[tuple[int, int]],
    title: str,
):
    """Small multiples showing the 64-bin PMF for a handful of (i,j) pairs."""
    import matplotlib.pyplot as plt

    cols = 4
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.3 * rows), constrained_layout=True, squeeze=False)
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]
        pmf = probs[i - 1, j - 1]
        ax.bar(bin_midpoints, pmf, width=0.45, color="tab:blue", alpha=0.85)
        d_gt = gt[i - 1, j - 1] if (0 < i <= probs.shape[0] and 0 < j <= probs.shape[0]) else np.nan
        if np.isfinite(d_gt) and d_gt <= bin_edges[-1]:
            ax.axvline(d_gt, color="red", lw=1.5, label=f"GT={d_gt:.1f}Å")
        exp = float((pmf * bin_midpoints).sum())
        ax.axvline(exp, color="black", lw=1.0, ls="--", label=f"E={exp:.1f}Å")
        ax.set_title(f"(i={i}, j={j})  |i-j|={abs(i-j)}")
        ax.set_xlim(0, bin_edges[-1])
        ax.set_xlabel("Å")
        ax.legend(fontsize=7, loc="upper right")
    # hide unused
    for idx in range(len(pairs), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title)
    return fig


def _pick_example_pairs(gt: np.ndarray, max_A: float, n_each: int = 4) -> list[tuple[int, int]]:
    """Pick a few short-range contacts, medium-range contacts, long-range contacts,
    and long-range non-contacts, for PMF inspection."""
    n = gt.shape[0]
    valid = _mask_valid(gt, max_A)
    pairs: list[tuple[int, int]] = []
    rng = np.random.default_rng(0)

    def _sample(mask, k):
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return []
        sel = rng.choice(len(idx), size=min(k, len(idx)), replace=False)
        return [(int(idx[s, 0]) + 1, int(idx[s, 1]) + 1) for s in sel]

    short = (
        valid
        & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 6)
        & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < 12)
        & (gt < CB_CONTACT_ANGSTROMS)
    )
    medium = (
        valid
        & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 12)
        & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) < 24)
        & (gt < CB_CONTACT_ANGSTROMS)
    )
    long_ = valid & (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 24) & (gt < CB_CONTACT_ANGSTROMS)
    far = valid & (gt > 20.0)

    pairs.extend(_sample(short, n_each))
    pairs.extend(_sample(medium, n_each))
    pairs.extend(_sample(long_, n_each))
    pairs.extend(_sample(far, n_each))
    return pairs


def _title_page(summary: dict):
    """Cover page: run metadata + per-N metrics table."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.5,
        0.95,
        f"{summary['pdb_id']} — Distogram Report",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "CB–CB distance prediction via single-token queries",
        ha="center",
        va="top",
        fontsize=11,
        style="italic",
    )

    meta_lines = [
        f"PDB ID: {summary['pdb_id']}"
        + (f"  chain: {summary['chain_id']}" if summary.get("chain_id") else "  chain: (first)"),
        f"Sequence length: {summary['sequence_length']} residues",
        f"Atom scheme: CB for non-GLY, CA for GLY ({summary['atom_per_residue'].count('CA')} CA, "
        f"{summary['atom_per_residue'].count('CB')} CB)",
        f"Model: {summary['inference']['model']}",
        f"Top-K logprobs: {summary['inference']['top_k_logprobs']}   "
        f"Batch size: {summary['inference']['batch_size']}   "
        f"Total elapsed: {summary['inference']['total_elapsed_seconds']:.1f} s",
        "",
        "Ground truth contact counts:",
        f"    {summary['ground_truth_counts']}",
    ]
    fig.text(0.07, 0.88, "\n".join(meta_lines), ha="left", va="top", fontsize=10, family="monospace")

    # Metrics table
    fig.text(0.07, 0.67, "Per-N metrics", ha="left", va="top", fontsize=13, fontweight="bold")
    header = (
        f"{'N':>3}  {'E|err|':>8}  {'bias':>7}  {'argmax|err|':>12}  "
        f"{'asym(mean)':>11}  {'asym(max)':>10}  {'contact corr':>12}"
    )
    rows = [header, "-" * len(header)]
    for rec in summary["per_n"]:
        m = rec["metrics"]
        rows.append(
            f"{rec['n_prompt_contacts']:>3}  "
            f"{m['expected_mean_abs_err_A']:>7.2f}A  "
            f"{m['expected_mean_signed_err_A']:>+6.2f}A  "
            f"{m['argmax_mean_abs_err_A']:>11.2f}A  "
            f"{m['expected_order_asymmetry_mean_A']:>10.2f}A  "
            f"{m['expected_order_asymmetry_max_A']:>9.2f}A  "
            f"{m['contact_prob_auc_proxy_corr']:>12.3f}"
        )
    fig.text(0.07, 0.63, "\n".join(rows), ha="left", va="top", fontsize=9, family="monospace")

    # Legend / explanation
    legend_lines = [
        "Column definitions:",
        "  N            number of ground-truth long-range contacts prepended to the prompt",
        "  E|err|       mean |expected_predicted − GT| distance (Å), over pairs with GT ≤ 32 Å",
        "  bias         mean (expected_predicted − GT) — positive means the model over-predicts distance",
        "  argmax|err|  same error but using argmax bin instead of posterior expectation",
        "  asym(mean)   |E[d|i,j] − E[d|j,i]| — averaged over valid pairs; should be 0 for a true distance",
        "  asym(max)    max of that across pairs",
        "  contact corr corr(P(d ≤ 8 Å),  1[GT_dist < 8 Å])  — 0 = random, 1 = perfect",
    ]
    fig.text(0.07, 0.38, "\n".join(legend_lines), ha="left", va="top", fontsize=9)

    fig.text(
        0.5,
        0.05,
        "See following pages for cross-N summary, expected-distance montage,\n"
        "and per-N distogram / scatter / contact-map / PMF plots.",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic",
    )
    return fig


def _add_caption(
    fig, text: str, bottom: float = 0.22, y: float = 0.03, fontsize: int = 8, max_chars_per_line: int = 140
):
    """Reserve space at the bottom of `fig` and render a wrapped caption there.

    `fig` may have been built with constrained_layout; we disable the layout
    engine before `subplots_adjust` so it actually takes effect.
    """
    import textwrap

    fig.set_layout_engine(None)
    fig.subplots_adjust(bottom=bottom)
    wrapped = "\n".join(textwrap.wrap(text, width=max_chars_per_line))
    fig.text(0.5, y, wrapped, ha="center", va="bottom", fontsize=fontsize)


def _build_captions(summary: dict) -> dict[str, str]:
    """Compute per-N captions from summary metrics."""
    captions: dict[str, str] = {}
    for rec in summary["per_n"]:
        n = rec["n_prompt_contacts"]
        m = rec["metrics"]
        c = rec["coverage_stats"]
        seeded_desc = (
            "no seeded contacts"
            if n == 0
            else f"{n} seeded GT long-range contacts: " + ", ".join(f"({i},{j})" for _t, i, j in rec["seeded_contacts"])
        )
        captions[f"n{n}_distogram"] = (
            f"N={n} — {seeded_desc}. "
            f"Left: the model's expected CB–CB distance E[d|(i,j)] = Σ_k P(bin_k) · midpoint_k. "
            f"Middle: ground-truth CB–CB distance (CA for GLY), clipped at 32 Å. "
            f"Right: signed residual (pred − GT). "
            f"Overall MAE = {m['expected_mean_abs_err_A']:.2f} Å, bias {m['expected_mean_signed_err_A']:+.2f} Å, "
            f"{m['num_valid_pairs']} valid pairs."
        )
        captions[f"n{n}_scatter_contact"] = (
            f"N={n} — Left: every ordered pair (i,j) with GT ≤ 32 Å, colored by |i−j|. "
            f"The model predicts an expected distance which is plotted against the true CB–CB distance. "
            f"Argmax MAE = {m['argmax_mean_abs_err_A']:.2f} Å (much worse than posterior mean "
            f"because the posterior is diffuse — see PMFs). "
            f"Right: P(d ≤ 8 Å) heatmap; white dots mark the GT contact pairs. "
            f"Contact-prob ↔ GT correlation = {m['contact_prob_auc_proxy_corr']:.3f}."
        )
        captions[f"n{n}_pmfs"] = (
            f"N={n} — Four random pairs per row: short-range contacts (top), medium-range contacts, "
            f"long-range contacts, far pairs (bottom). Blue bars are the 64 distance-bin probabilities "
            f"(0.5 Å bins, range (0, 32] Å). Red line = GT distance, dashed black = posterior expectation. "
            f"Coverage stats for this N: median of {c['missing_bins_median']:.0f}/64 distance bins were "
            f"outside the top-{summary['inference']['top_k_logprobs']} logprobs (approximated as 0); "
            f"median non-distance mass in top-K = {c['non_distance_top_mass_median']:.2e}."
        )
    captions["summary_vs_n"] = (
        "Left: distance-error metrics vs. the number of seeded GT long-range contacts in the prompt. "
        "Right: correlation between the model's P(d ≤ 8 Å) and the binary indicator [GT_dist < 8 Å] vs. N. "
        "Seeding with a handful of real contacts does not meaningfully improve predictions, and N=1 is "
        "actively worse than N=0; contact-probability correlation is essentially flat across N."
    )
    captions["montage"] = (
        "Expected-distance distograms across all N values. Side-by-side lets you confirm that seeding "
        "with a few GT long-range contacts barely perturbs the predicted distance map — the model's "
        "internal prior over distances is dominant."
    )
    return captions


def _write_pdf(pdf_path: str, summary: dict, input_dir: str, ns: list[int]) -> None:
    """Build the multi-page PDF report at pdf_path (local or gs://)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    captions = _build_captions(summary)
    pdb = summary["pdb_id"]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- Title page ---
        fig = _title_page(summary)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Summary vs N ---
        fig = plot_summary_curves(summary)
        _add_caption(fig, captions["summary_vs_n"], bottom=0.28, y=0.04)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Per-N plots + montage data collection ---
        expected_for_montage: dict[int, np.ndarray] = {}
        max_A_for_montage: float = 32.0

        for n_prompt in ns:
            logger.info("Rendering N=%d", n_prompt)
            data = _load_npz(f"{input_dir}/distogram_n{n_prompt}.npz")
            probs = data["probs"].astype(np.float32)
            gt = data["gt_distance"].astype(np.float32)
            bin_midpoints = data["bin_midpoints"].astype(np.float32)
            bin_edges = data["bin_edges"].astype(np.float32)
            max_A = float(bin_edges[-1])
            max_A_for_montage = max_A

            expected, _ = _expected_and_argmax(probs, bin_midpoints)
            expected_for_montage[n_prompt] = expected
            tag = f"{pdb} — N={n_prompt}"

            # Heatmaps page
            fig = plot_distogram_heatmaps(expected, gt, max_A, tag)
            _add_caption(fig, captions[f"n{n_prompt}_distogram"], bottom=0.25, y=0.03)
            pdf.savefig(fig)
            plt.close(fig)

            # Scatter + contact map side by side
            fig_s = plot_scatter_pred_vs_gt(expected, gt, max_A, tag)
            fig_c = plot_contact_map(probs, gt, bin_edges, CB_CONTACT_ANGSTROMS, tag)
            combo = plt.figure(figsize=(12, 5.5))
            ax_s = combo.add_subplot(1, 2, 1)
            ax_c = combo.add_subplot(1, 2, 2)
            # Re-render each plot into the combo figure axes.
            _scatter_on(ax_s, expected, gt, max_A, f"Scatter — {tag}")
            _contact_map_on(combo, ax_c, probs, gt, bin_edges, CB_CONTACT_ANGSTROMS, f"Contact map — {tag}")
            plt.close(fig_s)
            plt.close(fig_c)
            _add_caption(combo, captions[f"n{n_prompt}_scatter_contact"], bottom=0.28, y=0.03)
            pdf.savefig(combo)
            plt.close(combo)

            # Per-pair PMFs
            pairs = _pick_example_pairs(gt, max_A, n_each=4)
            if pairs:
                fig = plot_per_pair_pmf_grid(
                    probs,
                    gt,
                    bin_midpoints,
                    bin_edges,
                    pairs,
                    f"{tag} — example per-pair PMFs (short / medium / long / far)",
                )
                _add_caption(fig, captions[f"n{n_prompt}_pmfs"], bottom=0.18, y=0.03)
                pdf.savefig(fig)
                plt.close(fig)

        # --- Cross-N montage ---
        if expected_for_montage:
            cols = 3
            rows = (len(ns) + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(4.2 * cols, 3.8 * rows), constrained_layout=True, squeeze=False
            )
            for idx, n_prompt in enumerate(ns):
                ax = axes[idx // cols][idx % cols]
                im = ax.imshow(
                    expected_for_montage[n_prompt], origin="lower", vmin=0, vmax=max_A_for_montage, cmap="viridis_r"
                )
                ax.set_title(f"N={n_prompt}")
                ax.set_xlabel("j")
                ax.set_ylabel("i")
                fig.colorbar(im, ax=ax, fraction=0.046)
            for idx in range(len(ns), rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            fig.suptitle(f"{pdb} — expected-distance distograms across N")
            _add_caption(fig, captions["montage"], bottom=0.15, y=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        # --- PDF metadata ---
        meta = pdf.infodict()
        meta["Title"] = f"{pdb} — Distogram report"
        meta["Subject"] = "Marin protein-contacts eval"

    buf.seek(0)
    with fsspec.open(pdf_path, "wb") as f:
        f.write(buf.getvalue())


# Helpers to re-render scatter / contact-map onto an existing axis (for side-by-side
# pages). These mirror plot_scatter_pred_vs_gt / plot_contact_map but operate on a
# caller-supplied axis.


def _scatter_on(ax, expected: np.ndarray, gt: np.ndarray, max_A: float, title: str) -> None:
    n = expected.shape[0]
    mask = _mask_valid(gt, max_A)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sep = np.abs(ii - jj)
    x = gt[mask]
    y = expected[mask]
    s = sep[mask]
    sc = ax.scatter(x, y, c=s, cmap="plasma", s=4, alpha=0.35)
    ax.plot([0, max_A], [0, max_A], "k--", lw=1)
    ax.set_xlim(0, max_A)
    ax.set_ylim(0, max_A)
    ax.set_xlabel("Ground truth CB–CB distance (Å)")
    ax.set_ylabel("Predicted expected distance (Å)")
    ax.set_aspect("equal")
    mae = float(np.abs(y - x).mean())
    bias = float((y - x).mean())
    ax.set_title(f"{title}\nMAE={mae:.2f}Å  bias={bias:+.2f}Å  n={int(mask.sum())}")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046)
    cb.set_label("|i − j|")


def _contact_map_on(
    fig, ax, probs: np.ndarray, gt: np.ndarray, bin_edges: np.ndarray, contact_A: float, title: str
) -> None:
    last_bin = int(np.searchsorted(bin_edges, contact_A, side="left"))
    p_contact = probs[:, :, : last_bin + 1].sum(axis=-1)
    n = p_contact.shape[0]
    gt_contact = (gt < contact_A) & np.isfinite(gt)
    ii, jj = np.where(gt_contact & ~np.eye(n, dtype=bool))
    im = ax.imshow(p_contact, origin="lower", vmin=0, vmax=1, cmap="magma", extent=(0.5, n + 0.5, 0.5, n + 0.5))
    ax.scatter(jj + 1, ii + 1, s=4, facecolors="none", edgecolors="white", linewidths=0.35)
    ax.set_title(f"{title}\nP(d ≤ {contact_A:.0f} Å); white dots = GT contacts")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(f"P(d ≤ {contact_A:.0f} Å)")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dir containing summary.json + distogram_n{N}.npz from eval_protein_distogram.",
    )
    parser.add_argument("--output-pdf", default=None, help="Path (local or gs://) for single PDF report.")
    parser.add_argument("--output-dir", default=None, help="Optional: also write individual PNGs to this directory.")
    args = parser.parse_args(argv)

    if not args.output_pdf and not args.output_dir:
        parser.error("Provide at least one of --output-pdf or --output-dir.")

    in_dir = args.input_dir.rstrip("/")

    summary = _load_json(f"{in_dir}/summary.json")
    pdb = summary["pdb_id"]

    fs, _ = url_to_fs(in_dir)
    found = fs.find(in_dir, detail=True, maxdepth=1)
    npz_basenames = sorted(
        os.path.basename(p) for p in found if os.path.basename(p).startswith("distogram_n") and p.endswith(".npz")
    )
    ns = sorted(int(b[len("distogram_n") : -len(".npz")]) for b in npz_basenames)
    logger.info("Found distograms for N=%s", ns)

    if args.output_pdf:
        logger.info("Building PDF report → %s", args.output_pdf)
        _write_pdf(args.output_pdf, summary, in_dir, ns)

    if args.output_dir:
        import matplotlib.pyplot as plt

        out_dir = args.output_dir.rstrip("/")
        logger.info("Writing individual PNGs → %s", out_dir)

        fig = plot_summary_curves(summary)
        _save_fig(fig, f"{out_dir}/summary_vs_n.png")
        plt.close(fig)

        expected_for_montage: dict[int, np.ndarray] = {}
        max_A_for_montage = 32.0

        for n_prompt in ns:
            data = _load_npz(f"{in_dir}/distogram_n{n_prompt}.npz")
            probs = data["probs"].astype(np.float32)
            gt = data["gt_distance"].astype(np.float32)
            bin_midpoints = data["bin_midpoints"].astype(np.float32)
            bin_edges = data["bin_edges"].astype(np.float32)
            expected, _ = _expected_and_argmax(probs, bin_midpoints)
            expected_for_montage[n_prompt] = expected
            max_A = float(bin_edges[-1])
            max_A_for_montage = max_A
            tag = f"{pdb} — N={n_prompt}"

            fig = plot_distogram_heatmaps(expected, gt, max_A, tag)
            _save_fig(fig, f"{out_dir}/n{n_prompt}_distogram_vs_gt.png")
            plt.close(fig)

            fig = plot_scatter_pred_vs_gt(expected, gt, max_A, tag)
            _save_fig(fig, f"{out_dir}/n{n_prompt}_scatter.png")
            plt.close(fig)

            fig = plot_contact_map(probs, gt, bin_edges, CB_CONTACT_ANGSTROMS, tag)
            _save_fig(fig, f"{out_dir}/n{n_prompt}_contact_map.png")
            plt.close(fig)

            pairs = _pick_example_pairs(gt, max_A, n_each=3)
            if pairs:
                fig = plot_per_pair_pmf_grid(
                    probs, gt, bin_midpoints, bin_edges, pairs, f"{tag} — example per-pair PMFs"
                )
                _save_fig(fig, f"{out_dir}/n{n_prompt}_example_pmfs.png")
                plt.close(fig)

        if expected_for_montage:
            cols = 3
            rows = (len(ns) + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=True, squeeze=False
            )
            for idx, n_prompt in enumerate(ns):
                ax = axes[idx // cols][idx % cols]
                im = ax.imshow(
                    expected_for_montage[n_prompt], origin="lower", vmin=0, vmax=max_A_for_montage, cmap="viridis_r"
                )
                ax.set_title(f"N={n_prompt}")
                ax.set_xlabel("j")
                ax.set_ylabel("i")
                fig.colorbar(im, ax=ax, fraction=0.046)
            for idx in range(len(ns), rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            fig.suptitle(f"{pdb} — expected-distance distograms across N")
            _save_fig(fig, f"{out_dir}/montage_expected_vs_n.png")
            plt.close(fig)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
