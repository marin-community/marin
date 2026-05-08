# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a PDF report from `eval_protein_proteingym.py` output.

Reads the eval output's `summary.json` plus per-dataset `per_variant.json`
and the ProteinGym leaderboard CSV (mirrored at
`<proteingym-dir>/reference/DMS_substitutions_Spearman_DMS_level.csv`).

Pages:
  1. Headline summary: per-dataset Spearman bar chart with leaderboard
     baselines overlaid (ESM-2 650M, ESM-1v, ProGen2 medium, ...). Sorted by
     this model's Spearman descending.
  2. Position-stratified Spearman bar chart per dataset (4 buckets).
  3. Aggregate perplexity-by-position curve (averaged over datasets, plus
     per-dataset spaghetti).
  4. Per-dataset scatter plots (sample) of model score vs DMS score.
  5. Spearman vs leaderboard scatterplot (this model vs each baseline,
     color = baseline).
  6. Bucket comparison: this model's per-bucket Spearman vs the
     leaderboard's overall Spearman for the same datasets — flags which
     position buckets carry the model's overall signal.

Usage::

    python -m experiments.protein.plot_protein_proteingym \\
        --input-dir gs://marin-us-east5/eval/protein-proteingym/<run>/step-15049 \\
        --proteingym-dir gs://marin-us-east5/protein-structure/proteingym/v1.3 \\
        --output /Users/tim/Dropbox/.../reports/proteingym-step-15049.pdf
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# Baseline methods to overlay. These are column names in the leaderboard CSV.
HEADLINE_BASELINES = [
    "ESM-1v (single)",
    "ESM2 (650M)",
    "ESM2 (3B)",
    "Tranception L",
    "ProGen2 medium",
    "Site-Independent",
]


def _read_text(path: str) -> str:
    with fsspec.open(path, "r") as f:
        return f.read()


def _read_json(path: str):
    return json.loads(_read_text(path))


def _read_leaderboard(path: str) -> dict[str, dict[str, float]]:
    """Return {dms_id: {method_name: spearman}}."""
    text = _read_text(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        dms = r.get("DMS ID") or r.get("DMS_ID") or r.get("DMS_id")
        if not dms:
            continue
        d: dict[str, float] = {}
        for k, v in r.items():
            if k in {
                "DMS ID",
                "DMS_ID",
                "DMS_id",
                "Number of Mutants",
                "Selection Type",
                "UniProt ID",
                "MSA_Neff_L_category",
                "Taxon",
            }:
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                continue
        out[dms] = d
    return out


def _render_headline(pdf, summary: dict, leaderboard: dict[str, dict[str, float]]) -> None:
    datasets = [d for d in summary["datasets"] if "spearman_overall" in d]
    datasets.sort(key=lambda d: -d.get("spearman_overall", float("-inf")))

    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8.5, n * 0.35 + 3), 6))
    x = np.arange(n)
    ours = np.array([d["spearman_overall"] for d in datasets])

    width = 0.7 / (1 + len(HEADLINE_BASELINES))
    ax.bar(x - 0.35, ours, width, label="protein-docs (this run)", color="black")
    palette = plt.cm.tab10.colors
    for k, baseline in enumerate(HEADLINE_BASELINES):
        vals = []
        for d in datasets:
            lb = leaderboard.get(d["dms_id"], {})
            vals.append(lb.get(baseline, np.nan))
        ax.bar(x - 0.35 + (k + 1) * width, vals, width, label=baseline, color=palette[k % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels([d["dms_id"] for d in datasets], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho (DMS score vs model VEP)")
    ax.set_title(f"ProteinGym DMS-substitutions zero-shot ({n} datasets)\n" f"Model: {summary.get('model', 'unknown')}")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_position_buckets(pdf, summary: dict) -> None:
    datasets = [d for d in summary["datasets"] if d.get("spearman_by_position_bucket")]
    if not datasets:
        return
    bucket_labels = ["0-25", "25-50", "50-75", "75-100"]
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8.5, n * 0.35 + 3), 5))
    x = np.arange(n)
    width = 0.18
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    for k, b in enumerate(bucket_labels):
        vals = np.array([d["spearman_by_position_bucket"].get(b, np.nan) for d in datasets])
        ax.bar(x + (k - 1.5) * width, vals, width, label=f"pos {b}%", color=palette[k])
    # Overlay: overall Spearman as black dot
    overall = np.array([d["spearman_overall"] for d in datasets])
    ax.plot(x, overall, "ko", markersize=4, label="overall")
    ax.set_xticks(x)
    ax.set_xticklabels([d["dms_id"] for d in datasets], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho within position bucket")
    ax.set_title("Position-stratified Spearman per dataset")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_perplexity_profile(pdf, summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: per-dataset spaghetti (alpha-blended)
    profiles = []
    for d in summary["datasets"]:
        p = d.get("perplexity_profile_mean")
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32)
        # Plot relative position so different lengths overlay
        rel = np.linspace(0, 1, len(arr))
        axes[0].plot(rel, arr, alpha=0.15, linewidth=0.7)
        profiles.append(arr)

    # Bin into 50 fractional-position buckets and average across datasets
    if profiles:
        n_bins = 50
        accum = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int64)
        for arr in profiles:
            rel = np.linspace(0, 1, len(arr), endpoint=False)
            bins = np.minimum((rel * n_bins).astype(int), n_bins - 1)
            for b, v in zip(bins, arr, strict=True):
                if np.isfinite(v):
                    accum[b] += v
                    counts[b] += 1
        means = accum / np.maximum(counts, 1)
        axes[0].plot(np.linspace(0, 1, n_bins), means, "k", linewidth=2.5, label="mean")
    axes[0].set_xlabel("Relative position in protein (N->C)")
    axes[0].set_ylabel("-log P(WT residue | causal prefix) [nats]")
    axes[0].set_title("Per-position perplexity profile\n(spaghetti = per-dataset, black = mean)")
    axes[0].legend()

    # Right: same data, but cap to first 200 absolute positions
    abs_max = 200
    accum_abs = np.zeros(abs_max, dtype=np.float64)
    counts_abs = np.zeros(abs_max, dtype=np.int64)
    for arr in profiles:
        for i in range(min(len(arr), abs_max)):
            if np.isfinite(arr[i]):
                accum_abs[i] += arr[i]
                counts_abs[i] += 1
    means_abs = accum_abs / np.maximum(counts_abs, 1)
    axes[1].plot(np.arange(1, abs_max + 1), means_abs, color="black", linewidth=2)
    axes[1].set_xlabel("Absolute position (1-indexed)")
    axes[1].set_ylabel("Mean -log P(WT residue) across datasets [nats]")
    axes[1].set_title("Absolute-position perplexity (first 200 res)\nshould drop sharply with prefix length")
    axes[1].axhline(np.log(20), color="red", linestyle="--", linewidth=0.8, label="uniform (log 20)")
    axes[1].legend()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_leaderboard_scatter(pdf, summary: dict, leaderboard: dict) -> None:
    """For each baseline, scatter (this model's Spearman) vs (baseline's Spearman) per dataset."""
    datasets = [d for d in summary["datasets"] if "spearman_overall" in d]
    if not datasets:
        return
    rows = (len(HEADLINE_BASELINES) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for k, baseline in enumerate(HEADLINE_BASELINES):
        ax = axes.flat[k]
        ours = []
        theirs = []
        for d in datasets:
            lb = leaderboard.get(d["dms_id"], {})
            v = lb.get(baseline, np.nan)
            if np.isnan(v):
                continue
            ours.append(d["spearman_overall"])
            theirs.append(v)
        ax.scatter(theirs, ours, color="tab:blue", s=15)
        lim = (-0.1, 1.0)
        ax.plot(lim, lim, "k--", linewidth=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{baseline}")
        ax.set_ylabel("protein-docs")
        ax.set_title(f"vs {baseline}\n({len(ours)} datasets)")
        ax.axhline(0, color="grey", linewidth=0.3)
        ax.axvline(0, color="grey", linewidth=0.3)

    # Hide unused subplots
    for j in range(len(HEADLINE_BASELINES), rows * 3):
        axes.flat[j].axis("off")

    fig.suptitle("Per-dataset Spearman: protein-docs vs leaderboard baselines (above diagonal = we win)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_summary_text(pdf, summary: dict, leaderboard: dict) -> None:
    """First page — text summary."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    datasets = [d for d in summary["datasets"] if "spearman_overall" in d]
    skipped = [d for d in summary["datasets"] if "skipped" in d]

    # Aggregate stats
    spearmans = [d["spearman_overall"] for d in datasets if np.isfinite(d.get("spearman_overall", np.nan))]
    mean_sp = float(np.mean(spearmans)) if spearmans else float("nan")
    median_sp = float(np.median(spearmans)) if spearmans else float("nan")

    # Compare against each baseline (averaged across datasets we both have).
    baseline_means: dict[str, tuple[float, int]] = {}
    for baseline in HEADLINE_BASELINES:
        ours_aligned = []
        theirs_aligned = []
        for d in datasets:
            lb = leaderboard.get(d["dms_id"], {})
            v = lb.get(baseline, np.nan)
            if np.isnan(v) or not np.isfinite(d.get("spearman_overall", np.nan)):
                continue
            ours_aligned.append(d["spearman_overall"])
            theirs_aligned.append(v)
        if ours_aligned:
            baseline_means[baseline] = (
                float(np.mean(ours_aligned) - np.mean(theirs_aligned)),
                len(ours_aligned),
            )

    lines: list[str] = []
    lines.append(f"ProteinGym DMS-substitutions zero-shot — {len(datasets)} datasets evaluated")
    lines.append("=" * 78)
    lines.append(f"Model: {summary.get('model', 'unknown')}")
    lines.append("")
    lines.append(f"Overall Spearman (mean across datasets):   {mean_sp:+.3f}")
    lines.append(f"Overall Spearman (median across datasets): {median_sp:+.3f}")
    if skipped:
        lines.append(f"Skipped: {len(skipped)} datasets ({Counter(d.get('skipped', '?') for d in skipped)})")
    lines.append("")
    lines.append("Mean Spearman gap vs leaderboard baselines (positive = we win, on aligned subset):")
    lines.append(f"  {'baseline':<28} {'gap':>8}  {'n_aligned':>10}")
    for b, (gap, n) in baseline_means.items():
        lines.append(f"  {b:<28} {gap:+8.3f}  {n:>10}")
    lines.append("")
    lines.append("Top-5 datasets by our Spearman:")
    sorted_ds = sorted(datasets, key=lambda d: -d.get("spearman_overall", -np.inf))
    for d in sorted_ds[:5]:
        lines.append(f"  {d['dms_id']:<48}  rho={d['spearman_overall']:+.3f}  n={d['num_variants']}")
    lines.append("")
    lines.append("Bottom-5 datasets:")
    for d in sorted_ds[-5:]:
        lines.append(f"  {d['dms_id']:<48}  rho={d['spearman_overall']:+.3f}  n={d['num_variants']}")
    lines.append("")
    lines.append("Position-bucket effect (mean Spearman per bucket across datasets):")
    bucket_labels = ["0-25", "25-50", "50-75", "75-100"]
    for b in bucket_labels:
        vals = [
            d["spearman_by_position_bucket"].get(b, np.nan) for d in datasets if d.get("spearman_by_position_bucket")
        ]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            lines.append(f"  pos {b}%:  {float(np.mean(vals)):+.3f}  (n={len(vals)})")

    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=9, ha="left", va="top", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


# Lazy import to avoid circular issues
from collections import Counter  # noqa: E402


def build_report(input_dir: str, proteingym_dir: str, output_pdf: str) -> None:
    summary = _read_json(f"{input_dir.rstrip('/')}/summary.json")
    leaderboard = _read_leaderboard(f"{proteingym_dir.rstrip('/')}/reference/DMS_substitutions_Spearman_DMS_level.csv")

    local_path = output_pdf
    if output_pdf.startswith(("gs://", "s3://")):
        import tempfile

        local_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name

    with PdfPages(local_path) as pdf:
        _render_summary_text(pdf, summary, leaderboard)
        _render_headline(pdf, summary, leaderboard)
        _render_position_buckets(pdf, summary)
        _render_perplexity_profile(pdf, summary)
        _render_leaderboard_scatter(pdf, summary, leaderboard)

    if local_path != output_pdf:
        with open(local_path, "rb") as src, fsspec.open(output_pdf, "wb") as dst:
            dst.write(src.read())
    logger.info("Wrote %s", output_pdf)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--proteingym-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    build_report(args.input_dir, args.proteingym_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
