# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a PDF report from `eval_protein_contacts.py` output.

Expects the eval output directory to contain:
  - summary.json   (top-level metrics + per-type per-rollout reports)
  - rollouts.json  (raw per-rollout entries)
  - matrices.npz   (per-type frequency + GT matrices)

Produces one PDF with:
  1. Summary page (headline numbers per range).
  2. Per-range 2x2 heatmap grid (GT / consensus-50% / rollout-frequency / GT-vs-consensus overlay).
  3. Combined contact-map page (all ranges overlaid, GT vs rollout-frequency).
  4. Per-rollout precision/recall bar chart.
  5. Invalid-emission diagnostics (out-of-range / duplicate / non-position-token slot counts per type).

Usage::

    python -m experiments.protein.plot_protein_contacts \\
        --input-dir gs://.../eval/protein-contacts/.../<variant>/run-01 \\
        --output /Users/tim/Dropbox/.../reports/contacts-<variant>.pdf
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

RANGES = ["long", "medium", "short"]  # display order
RANGE_COLORS = {"long": "tab:red", "medium": "gold", "short": "tab:blue"}
RANGE_TOKEN = {
    "long": "<long-range-contact>",
    "medium": "<medium-range-contact>",
    "short": "<short-range-contact>",
}


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _read_npz(path: str) -> dict[str, np.ndarray]:
    with fsspec.open(path, "rb") as f:
        buf = io.BytesIO(f.read())
    return dict(np.load(buf, allow_pickle=True))


def _render_summary(pdf: PdfPages, summary: dict, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    pdb = summary["pdb_id"]
    seq_len = summary["sequence_length"]
    inf = summary["inference"]
    counts = summary["ground_truth_counts"]
    override = summary.get("sequence_override")

    lines: list[str] = []
    lines.append(f"Forced-scaffold contact-rollout eval - {pdb} ({seq_len} residues)  [{label}]")
    lines.append("=" * 80)
    lines.append(f"Model:  {inf['model']}")
    lines.append(
        f"Rollouts: {inf['num_rollouts']} @ T={inf['temperature']}, top_k={inf['top_k']}  "
        f"| {inf['total_forced_slots']} forced slots  | {inf['elapsed_seconds']:.0f}s"
    )
    if override is not None:
        lines.append("")
        lines.append(f"Sequence override: {override['target_label']} / {override['method']} #{override['redesign_idx']}")
        lines.append(
            f"  Hamming to native: {override['hamming_distance']}/{seq_len} "
            f"({100 * override['hamming_distance'] / seq_len:.0f}% of residues redesigned)"
        )
    lines.append("")
    lines.append(
        f"Ground-truth contact counts (CB-CB <= 8 A):  "
        f"long={counts['<long-range-contact>']}  "
        f"medium={counts['<medium-range-contact>']}  "
        f"short={counts['<short-range-contact>']}  "
        f"total={sum(counts.values())}"
    )
    lines.append("")
    lines.append("Per-range metrics (precision/recall on residue pairs):")
    lines.append("-" * 80)

    per_type = summary["per_type"]
    for name in RANGES:
        tok = RANGE_TOKEN[name]
        rep = per_type[tok]
        pr_rolls = rep["per_rollout"]
        precs = [r["precision"] for r in pr_rolls]
        recs = [r["recall"] for r in pr_rolls]
        invalids = [r["num_invalid_slots"] for r in pr_rolls]
        oors = [r["num_out_of_range"] for r in pr_rolls]
        selfs = [r["num_self_contacts"] for r in pr_rolls]
        cons = rep["consensus"]
        gt_n = rep["num_ground_truth"]
        lines.append(f"  {name:<7}  (GT={gt_n} pairs)")
        if gt_n == 0:
            lines.append("           - no ground-truth pairs in this range -")
            continue
        med_p = float(np.median(precs)) if precs else 0.0
        med_r = float(np.median(recs)) if recs else 0.0
        med_inv = float(np.median(invalids)) if invalids else 0.0
        med_oor = float(np.median(oors)) if oors else 0.0
        med_self = float(np.median(selfs)) if selfs else 0.0
        lines.append(
            f"    per-rollout median P={med_p:.3f}  R={med_r:.3f}  "
            f"invalid_slots={med_inv:.0f}  out_of_range={med_oor:.0f}  self_contacts={med_self:.0f}"
        )
        lines.append(
            f"    consensus >= {cons['threshold']:.0%}:  "
            f"pred={cons['num_predicted']}  correct={cons['true_positives']}  "
            f"P={cons['precision']:.3f}  R={cons['recall']:.3f}"
        )

    # Overall (union across ranges)
    gt_all = set()
    for name in RANGES:
        for ij in summary["ground_truth_pairs_by_type"][RANGE_TOKEN[name]]:
            gt_all.add(tuple(ij))
    lines.append("")
    lines.append(f"Total GT pairs (all ranges): {len(gt_all)}")

    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=8.5, ha="left", va="top", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _render_per_range_heatmaps(pdf: PdfPages, summary: dict, matrices: dict[str, np.ndarray]) -> None:
    seq_len = summary["sequence_length"]
    cmap_bw = ListedColormap(["white", "black"])

    for name in RANGES:
        tok = RANGE_TOKEN[name]
        rep = summary["per_type"][tok]
        gt_n = rep["num_ground_truth"]
        if gt_n == 0:
            continue

        gt_mat = matrices[f"gt_{name}"]
        freq_mat = matrices[f"freq_{name}"]
        cons_mat = (freq_mat >= summary["per_type"][tok]["consensus"]["threshold"]).astype(np.float32)
        # Overlay: lower triangle = GT, upper triangle = freq (rainbow)
        overlay = np.zeros((seq_len, seq_len), dtype=np.float32)
        overlay[np.tri(seq_len, seq_len, 0, dtype=bool)] = gt_mat[np.tri(seq_len, seq_len, 0, dtype=bool)]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(gt_mat, cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
        axes[0, 0].set_title(f"Ground truth ({name}-range)\n{gt_n} pairs")

        axes[0, 1].imshow(cons_mat, cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
        cons = rep["consensus"]
        axes[0, 1].set_title(
            f"Consensus >= {cons['threshold']:.0%}\nP={cons['precision']:.0%} R={cons['recall']:.0%} "
            f"({cons['true_positives']}/{cons['num_predicted']} pred)"
        )

        im = axes[1, 0].imshow(freq_mat, cmap="viridis", vmin=0, vmax=1, origin="lower")
        axes[1, 0].set_title("Rollout frequency")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Overlay: upper triangle = freq (viridis), lower = GT (BW)
        # We'll render them as two overlapping images
        mask_upper = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=0)
        mask_lower = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=-1)
        axes[1, 1].imshow(np.where(mask_lower, gt_mat, np.nan), cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
        im2 = axes[1, 1].imshow(np.where(mask_upper, freq_mat, np.nan), cmap="viridis", vmin=0, vmax=1, origin="lower")
        axes[1, 1].set_title("GT (lower tri) vs rollout freq (upper tri)")
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        for ax in axes.flat:
            ax.set_xlabel("Residue")
            ax.set_ylabel("Residue")
            ax.set_xlim(0, seq_len)
            ax.set_ylim(0, seq_len)

        override_suffix = ""
        ov = summary.get("sequence_override")
        if ov is not None:
            override_suffix = f"  |  {ov['method']} #{ov['redesign_idx']} (Hamming {ov['hamming_distance']}/{seq_len})"
        fig.suptitle(f"{summary['pdb_id']} - {name}-range contacts{override_suffix}", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _render_combined_heatmap(pdf: PdfPages, summary: dict, matrices: dict[str, np.ndarray]) -> None:
    """Single heatmap: upper tri colored by rollout frequency-by-range, lower tri = GT by range."""
    seq_len = summary["sequence_length"]
    # Build RGB for lower triangle: long=red, medium=yellow, short=blue
    color_map = {"long": (0.84, 0.19, 0.15), "medium": (1.0, 0.83, 0.0), "short": (0.12, 0.47, 0.71)}
    rgb_lower = np.ones((seq_len, seq_len, 3), dtype=np.float32)
    rgb_upper = np.ones((seq_len, seq_len, 3), dtype=np.float32)

    for name in RANGES:
        gt = matrices[f"gt_{name}"]
        freq = matrices[f"freq_{name}"]
        rgb = np.array(color_map[name], dtype=np.float32)[None, None, :]  # (1,1,3)
        # Lower triangle: GT present → solid color, else white
        gt_alpha = gt[:, :, None]  # (n,n,1)
        rgb_lower = np.where(np.tril(np.ones_like(gt, dtype=bool), k=-1)[:, :, None] & (gt_alpha > 0), rgb, rgb_lower)
        # Upper triangle: frequency-weighted color
        f_alpha = freq[:, :, None]
        rgb_upper = np.where(
            np.triu(np.ones_like(gt, dtype=bool), k=0)[:, :, None],
            rgb * f_alpha + (1 - f_alpha),
            rgb_upper * (f_alpha == 0) + rgb_upper * (f_alpha > 0) * (1 - f_alpha) + rgb * f_alpha,
        )

    # Simpler: just blend by taking the max-alpha color per-cell
    rgb_combined = np.ones((seq_len, seq_len, 3), dtype=np.float32)
    for name in RANGES:
        gt = matrices[f"gt_{name}"]
        freq = matrices[f"freq_{name}"]
        rgb = np.array(color_map[name], dtype=np.float32)
        lower = np.tril(np.ones_like(gt, dtype=bool), k=-1)
        upper = np.triu(np.ones_like(gt, dtype=bool), k=0)
        mask = np.zeros_like(gt)
        mask[lower] = gt[lower]  # lower triangle: binary GT
        mask[upper] = freq[upper]  # upper triangle: freq
        rgb_combined = np.where(
            mask[:, :, None] > 0,
            rgb[None, None, :] * mask[:, :, None] + (1 - mask[:, :, None]) * rgb_combined,
            rgb_combined,
        )

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(rgb_combined, origin="lower", aspect="equal")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, seq_len)
    ax.set_title(
        f"{summary['pdb_id']} - combined contact map\n"
        "lower tri = GT, upper tri = rollout frequency  |  "
        "red=long, yellow=medium, blue=short"
    )
    pdf.savefig(fig)
    plt.close(fig)


def _render_per_rollout_pr(pdf: PdfPages, summary: dict) -> None:
    per_type = summary["per_type"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, name in zip(axes, RANGES, strict=True):
        rep = per_type[RANGE_TOKEN[name]]
        per_roll = rep["per_rollout"]
        if not per_roll:
            ax.set_title(f"{name}-range (no data)")
            continue
        idx = np.arange(len(per_roll))
        precs = np.array([r["precision"] for r in per_roll])
        recs = np.array([r["recall"] for r in per_roll])
        w = 0.35
        ax.bar(idx - w / 2, precs, w, color=RANGE_COLORS[name], label="Precision")
        ax.bar(idx + w / 2, recs, w, color=RANGE_COLORS[name], alpha=0.45, label="Recall")
        ax.set_title(f"{name}-range  (GT={rep['num_ground_truth']})")
        ax.set_xlabel("Rollout index")
        ax.set_ylim(0, 1)
        ax.set_xticks(idx)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Precision / Recall")
    fig.suptitle(f"{summary['pdb_id']} - per-rollout precision/recall")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_diagnostics(pdf: PdfPages, summary: dict) -> None:
    """Per-range invalid-emission counts (median across rollouts)."""
    per_type = summary["per_type"]
    names = RANGES
    data = {
        "invalid": [],
        "out_of_range": [],
        "self": [],
        "valid": [],
    }
    gts = []
    for name in names:
        rep = per_type[RANGE_TOKEN[name]]
        per_roll = rep["per_rollout"]
        if not per_roll:
            data["invalid"].append(0)
            data["out_of_range"].append(0)
            data["self"].append(0)
            data["valid"].append(0)
            gts.append(0)
            continue
        data["invalid"].append(float(np.median([r["num_invalid_slots"] for r in per_roll])))
        data["out_of_range"].append(float(np.median([r["num_out_of_range"] for r in per_roll])))
        data["self"].append(float(np.median([r["num_self_contacts"] for r in per_roll])))
        n_slots = [r["num_slots"] for r in per_roll]
        data["valid"].append(
            float(
                np.median(
                    [
                        n_slots[i] - data_i
                        for i, data_i in enumerate(
                            [
                                per_roll[i]["num_invalid_slots"]
                                + per_roll[i]["num_out_of_range"]
                                + per_roll[i]["num_self_contacts"]
                                for i in range(len(per_roll))
                            ]
                        )
                    ]
                )
            )
        )
        gts.append(rep["num_ground_truth"])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros_like(x, dtype=np.float32)
    for key, color in (
        ("valid", "tab:green"),
        ("invalid", "tab:gray"),
        ("out_of_range", "tab:orange"),
        ("self", "tab:purple"),
    ):
        vals = np.array(data[key], dtype=np.float32)
        ax.bar(x, vals, bottom=bottom, label=key, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(GT={gts[i]})" for i, n in enumerate(names)])
    ax.set_ylabel("Median slot count per rollout")
    ax.set_title(f"{summary['pdb_id']} - emission diagnostics per range")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_report(input_dir: str, output_pdf: str, label: str) -> None:
    input_dir = input_dir.rstrip("/")
    summary = _read_json(f"{input_dir}/summary.json")
    matrices = _read_npz(f"{input_dir}/matrices.npz")

    local_path = output_pdf
    if output_pdf.startswith(("gs://", "s3://")):
        import tempfile

        local_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name

    with PdfPages(local_path) as pdf:
        _render_summary(pdf, summary, label=label)
        _render_per_range_heatmaps(pdf, summary, matrices)
        _render_combined_heatmap(pdf, summary, matrices)
        _render_per_rollout_pr(pdf, summary)
        _render_diagnostics(pdf, summary)

    if local_path != output_pdf:
        with open(local_path, "rb") as src, fsspec.open(output_pdf, "wb") as dst:
            dst.write(src.read())
    logger.info("Wrote %s", output_pdf)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    build_report(args.input_dir, args.output, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
