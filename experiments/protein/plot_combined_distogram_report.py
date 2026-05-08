# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One big PDF: distogram metrics across all models x targets x variants.

Reads the layout produced by ``run_distogram_evals.py``::

    gs://marin-us-east5/eval/protein-distogram/v1/
        <model_label>/
            <target_label>/
                <variant_label>/
                    summary.json
                    distogram_n{0..5}.npz

For every (target, variant) pair found, writes:

* An overview page with MAE @ N=0 bars per (target, variant) cell.
* A MAE-vs-N + contact-corr-vs-N overlay page per (target, variant), with all
  models on common axes.
* A heatmap page per (target, variant): 3x3 grid of expected-distance maps,
  with one panel for ground truth and one per model (using the N=0 distogram).
* A perplexity-vs-mean-MAE scatter (one point per model snapshot), with eval
  loss pulled from W&B at each snapshot's training step.

Also writes a wide-table summary CSV at the same time.

Usage::

    uv run python -m experiments.protein.plot_combined_distogram_report \\
        --input-prefix gs://marin-us-east5/eval/protein-distogram/v1 \\
        --output-pdf /tmp/protein-distogram-cross-run.pdf \\
        --output-csv /tmp/protein-distogram-cross-run.csv

The perplexity scatter requires ``WANDB_API_KEY`` in the environment; it's
silently omitted otherwise.
"""

import argparse
import io
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import fsspec
import numpy as np

logger = logging.getLogger(__name__)


# Map ``model_label`` (the leading "<size>" token) to the W&B run name that
# logged training metrics for it. Entries cover both the "size-step-N" labels
# we use in EVAL_RUNS and any historical labels.
WANDB_ENTITY = "timodonnell"
WANDB_PROJECT = "marin"
PERPLEXITY_METRIC = "eval/protein-docs-cd-val/loss"
MODEL_PREFIX_TO_WANDB_RUN: dict[str, str] = {
    "1b": "protein-contacts-1b-3.5e-4-distance-masked-7d355e",
    "30m": "protein-contacts-30m-distance-masked-a7457a",
    "100m": "protein-contacts-100m-distance-masked-917586",
    "400m": "protein-contacts-400m-distance-masked-0de2c1",
    "420m_deep": "protein-contacts-420m-deep-distance-masked-81e865",
    "1_5b": "protein-contacts-1_5b-distance-masked-70f8f5",
    "3b": "protein-contacts-3b-distance-masked-ef3aa5",
    "1b_unmasked": "protein-contacts-1b-3.5e-4-unmasked-8efbcb",
    "100m_unmasked": "protein-contacts-100m-3.5e-4-unmasked-7c3ef7",
}


# Order to use when picking colors / line styles. Earlier entries plot on top.
# Models not in this list still plot but in registry order, after these.
PREFERRED_MODEL_ORDER = (
    "30m",
    "100m",
    "400m",
    "420m_deep",
    "1b",
    "1b_continue_train",
    "1_5b",
    "3b",
)


@dataclass(frozen=True)
class CellResult:
    model_label: str
    target_label: str
    variant_label: str
    summary: dict


# ---- Discovery ----


def discover_cells(input_prefix: str) -> list[CellResult]:
    """Walk ``<prefix>/<model>/<target>/<variant>/summary.json``."""
    fs, root = fsspec.core.url_to_fs(input_prefix.rstrip("/"))
    cells: list[CellResult] = []

    if not fs.exists(root):
        logger.warning("Input prefix %s does not exist; nothing to plot.", input_prefix)
        return cells

    for model_dir in sorted(fs.ls(root, detail=False)):
        model_label = model_dir.rstrip("/").rsplit("/", 1)[-1]
        if not fs.isdir(model_dir):
            continue
        for target_dir in sorted(fs.ls(model_dir, detail=False)):
            target_label = target_dir.rstrip("/").rsplit("/", 1)[-1]
            if not fs.isdir(target_dir):
                continue
            for variant_dir in sorted(fs.ls(target_dir, detail=False)):
                variant_label = variant_dir.rstrip("/").rsplit("/", 1)[-1]
                summary_path = f"{variant_dir.rstrip('/')}/summary.json"
                if not fs.exists(summary_path):
                    logger.debug("No summary.json at %s; skipping.", summary_path)
                    continue
                with fs.open(summary_path, "r") as f:
                    summary = json.load(f)
                cells.append(
                    CellResult(
                        model_label=model_label,
                        target_label=target_label,
                        variant_label=variant_label,
                        summary=summary,
                    )
                )
    logger.info("Discovered %d cells under %s", len(cells), input_prefix)
    return cells


# ---- Indexing ----


def model_sort_key(model_label: str) -> tuple[int, str]:
    if model_label in PREFERRED_MODEL_ORDER:
        return (PREFERRED_MODEL_ORDER.index(model_label), model_label)
    return (len(PREFERRED_MODEL_ORDER), model_label)


def index_by_target_variant(cells: list[CellResult]) -> dict[tuple[str, str], list[CellResult]]:
    out: dict[tuple[str, str], list[CellResult]] = defaultdict(list)
    for c in cells:
        out[(c.target_label, c.variant_label)].append(c)
    for k in out:
        out[k].sort(key=lambda c: model_sort_key(c.model_label))
    return out


# ---- Per-page plot ----


def _per_n(c: CellResult) -> dict[int, dict]:
    """``{n_prompt_contacts: metrics_dict}`` for one cell."""
    return {int(r["n_prompt_contacts"]): r["metrics"] for r in c.summary.get("per_n", [])}


def plot_target_variant_page(target: str, variant: str, cells: list[CellResult]):
    """Page with two panels — MAE-vs-N and contact-corr-vs-N — overlaying all models."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_models = len(cells)
    color_map = cm.get_cmap("viridis")
    colors = {c.model_label: color_map(i / max(1, n_models - 1)) for i, c in enumerate(cells)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for c in cells:
        per_n = _per_n(c)
        ns = sorted(per_n)
        if not ns:
            continue
        exp_mae = [per_n[n].get("expected_mean_abs_err_A") for n in ns]
        argmax_mae = [per_n[n].get("argmax_mean_abs_err_A") for n in ns]
        contact_corr = [per_n[n].get("contact_prob_auc_proxy_corr") for n in ns]
        color = colors[c.model_label]
        axes[0].plot(ns, exp_mae, "o-", color=color, label=f"{c.model_label} (exp)")
        axes[0].plot(ns, argmax_mae, "s--", color=color, alpha=0.5)
        axes[1].plot(ns, contact_corr, "o-", color=color, label=c.model_label)

    axes[0].set_xlabel("# seeded GT long-range contacts (N)")
    axes[0].set_ylabel("MAE (Å)")
    axes[0].set_title("Distance MAE vs. seeding (solid = expected, dashed = argmax)")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("# seeded GT long-range contacts (N)")
    axes[1].set_ylabel("corr(P(d≤8Å), GT contact)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Contact-prob correlation vs. seeding")
    axes[1].legend(fontsize=7, loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{target} — {variant}")
    return fig


def plot_summary_page(cells: list[CellResult]):
    """Title page: per-(target, variant) at N=0 averaged across models, table-style."""
    import matplotlib.pyplot as plt

    # Aggregate at N=0 (no seeding; "raw" model performance) across cells.
    by_tv: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for c in cells:
        per_n = _per_n(c)
        if 0 not in per_n:
            continue
        v = per_n[0].get("expected_mean_abs_err_A")
        if v is None:
            continue
        by_tv[(c.target_label, c.variant_label)].append((c.model_label, float(v)))

    targets = sorted({t for t, _ in by_tv})
    variants = sorted({v for _, v in by_tv})
    if not targets or not variants:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "(no cells found)", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(
        len(targets),
        len(variants),
        figsize=(3.5 * len(variants), 3 * len(targets)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for i, t in enumerate(targets):
        for j, v in enumerate(variants):
            ax = axes[i][j]
            entries = sorted(by_tv.get((t, v), []), key=lambda kv: model_sort_key(kv[0]))
            if not entries:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            xs = list(range(len(entries)))
            ys = [mae for _, mae in entries]
            labels = [model for model, _ in entries]
            ax.bar(xs, ys, color="tab:blue")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{t}\nMAE @ N=0 (Å)", fontsize=9)
            if i == 0:
                ax.set_title(v, fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distogram MAE @ N=0 — overview (one bar per model, faceted by (target, variant))")
    return fig


# ---- Heatmap pages (load distogram_n{N}.npz lazily) ----


def _load_distogram_npz(input_prefix: str, model_label: str, target: str, variant: str, n_seeded: int) -> dict:
    path = f"{input_prefix.rstrip('/')}/{model_label}/{target}/{variant}/distogram_n{n_seeded}.npz"
    with fsspec.open(path, "rb") as f:
        buf = io.BytesIO(f.read())
    with np.load(buf, allow_pickle=True) as data:
        return {k: data[k] for k in data.keys()}


def _expected_distance(probs: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
    return (probs * midpoints[None, None, :]).sum(axis=-1)


def plot_distogram_heatmap_page(
    input_prefix: str,
    target: str,
    variant: str,
    cells: list[CellResult],
    *,
    n_seeded: int = 0,
):
    """3x3 grid: one ground-truth panel + one per model, sharing a 0..32 Å scale.

    Loads ``distogram_n{n_seeded}.npz`` from each cell's eval dir on demand.
    Uses ``viridis_r`` so contacts (small distances) appear bright.
    """
    import matplotlib.pyplot as plt

    sorted_cells = sorted(cells, key=lambda c: model_sort_key(c.model_label))
    if not sorted_cells:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "(no cells)", ha="center", va="center")
        ax.axis("off")
        return fig

    # Get GT from the first cell (all cells for the same target share GT).
    first = _load_distogram_npz(input_prefix, sorted_cells[0].model_label, target, variant, n_seeded)
    gt = first["gt_distance"]
    midpoints = first["bin_midpoints"]
    max_a = float(midpoints[-1] + (midpoints[1] - midpoints[0]) / 2)
    n = gt.shape[0]
    extent = (0.5, n + 0.5, 0.5, n + 0.5)
    im_kwargs = dict(origin="lower", vmin=0, vmax=max_a, cmap="viridis_r", extent=extent)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    flat = axes.flatten()

    gt_display = np.where(gt <= max_a, gt, max_a)
    flat[0].imshow(gt_display, **im_kwargs)
    flat[0].set_title("Ground truth", fontsize=9)
    flat[0].tick_params(labelsize=7)

    for idx, cell in enumerate(sorted_cells):
        ax = flat[idx + 1]
        try:
            data = _load_distogram_npz(input_prefix, cell.model_label, target, variant, n_seeded)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cell.model_label, fontsize=9)
            ax.axis("off")
            continue
        expected = _expected_distance(data["probs"], data["bin_midpoints"])
        ax.imshow(expected, **im_kwargs)
        ax.set_title(cell.model_label, fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide any remaining unused subplots.
    for k in range(len(sorted_cells) + 1, len(flat)):
        flat[k].axis("off")

    fig.suptitle(
        f"{target} — {variant}  •  Expected CB-CB distance (Å), N={n_seeded} seeded contacts",
        fontsize=11,
    )
    return fig


# ---- Perplexity-vs-MAE page (W&B-backed) ----


def _parse_step_from_label(label: str) -> int | None:
    """``"30m-step-50000"`` → 50000."""
    m = re.search(r"step-(\d+)$", label)
    return int(m.group(1)) if m else None


def _model_prefix_from_label(label: str) -> str:
    """``"30m-step-50000"`` → ``"30m"``. Falls back to the full label."""
    return re.sub(r"-step-\d+$", "", label)


def fetch_perplexity_per_model(cells: list[CellResult]) -> dict[str, tuple[int, float]]:
    """For each unique model label, return ``(checkpoint_step, val_perplexity)``.

    Pulls ``eval/protein-docs-cd-val/loss`` from the W&B run that matches the
    model label's prefix; uses the closest log point at-or-before the
    snapshot's checkpoint step. Skips models whose W&B run isn't registered.
    Returns ``{}`` if ``wandb`` import fails (no API key, etc.).
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not importable; skipping perplexity fetch.")
        return {}

    api = wandb.Api()
    out: dict[str, tuple[int, float]] = {}
    seen_labels: set[str] = set()
    for c in cells:
        if c.model_label in seen_labels:
            continue
        seen_labels.add(c.model_label)
        prefix = _model_prefix_from_label(c.model_label)
        run_name = MODEL_PREFIX_TO_WANDB_RUN.get(prefix)
        target_step = _parse_step_from_label(c.model_label)
        if run_name is None or target_step is None:
            logger.info("No W&B mapping for %s (prefix=%s); skipping.", c.model_label, prefix)
            continue
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_name}")
        except Exception as exc:
            logger.warning("Could not load W&B run %s: %s", run_name, exc)
            continue

        # Find the most recent log point with the eval loss at or before our step.
        records = []
        for record in run.scan_history(keys=["_step", PERPLEXITY_METRIC], page_size=10000):
            step = record.get("_step")
            loss = record.get(PERPLEXITY_METRIC)
            if step is None or loss is None:
                continue
            if int(step) > target_step:
                continue
            records.append((int(step), float(loss)))
        if not records:
            logger.warning("No %s logged for %s at or before step %d", PERPLEXITY_METRIC, run_name, target_step)
            continue
        actual_step, loss = max(records, key=lambda kv: kv[0])
        out[c.model_label] = (actual_step, math.exp(loss))
        logger.info(
            "  %s → wandb step %d, val_loss=%.4f, perplexity=%.2f",
            c.model_label,
            actual_step,
            loss,
            out[c.model_label][1],
        )
    return out


def plot_perplexity_vs_mae_page(cells: list[CellResult], perplexity_per_model: dict[str, tuple[int, float]]):
    """Scatter of mean MAE vs validation perplexity (one point per model snapshot)."""
    import matplotlib.pyplot as plt

    if not perplexity_per_model:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "(no perplexity data; set WANDB_API_KEY)", ha="center", va="center")
        ax.axis("off")
        return fig

    # Mean MAE per model_label, averaged across all (target, variant, N).
    mae_per_model: dict[str, float] = {}
    for c in cells:
        for r in c.summary.get("per_n", []):
            v = r.get("metrics", {}).get("expected_mean_abs_err_A")
            if v is None:
                continue
            mae_per_model.setdefault(c.model_label, [])
            mae_per_model[c.model_label].append(float(v))  # pyrefly: ignore
    mae_per_model = {k: float(np.mean(vs)) for k, vs in mae_per_model.items()}  # pyrefly: ignore

    common = sorted(set(mae_per_model) & set(perplexity_per_model), key=model_sort_key)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if not common:
        ax.text(0.5, 0.5, "(no overlap)", ha="center", va="center")
        ax.axis("off")
        return fig

    xs = [perplexity_per_model[m][1] for m in common]
    ys = [mae_per_model[m] for m in common]
    ax.scatter(xs, ys, s=80, c=range(len(common)), cmap="viridis")
    for x, y, label in zip(xs, ys, common, strict=True):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Validation perplexity (eval/protein-docs-cd-val/loss → exp), log scale")
    ax.set_ylabel("Mean expected MAE across all (target, variant, N) (Å)")
    ax.set_title("Perplexity ↔ downstream MAE — one point per (model, snapshot)")
    ax.grid(True, alpha=0.3, which="both")
    return fig


# ---- CSV ----


def write_csv(path: str, cells: list[CellResult]) -> None:
    import csv
    import io

    fieldnames = [
        "model_label",
        "target_label",
        "variant_label",
        "n_prompt_contacts",
        "expected_mean_abs_err_A",
        "argmax_mean_abs_err_A",
        "expected_mean_signed_err_A",
        "expected_order_asymmetry_mean_A",
        "contact_prob_auc_proxy_corr",
        "num_valid_pairs",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    n_rows = 0
    for c in sorted(cells, key=lambda c: (model_sort_key(c.model_label), c.target_label, c.variant_label)):
        for per_n_record in c.summary.get("per_n", []):
            metrics = per_n_record.get("metrics", {})
            writer.writerow(
                {
                    "model_label": c.model_label,
                    "target_label": c.target_label,
                    "variant_label": c.variant_label,
                    "n_prompt_contacts": int(per_n_record["n_prompt_contacts"]),
                    "expected_mean_abs_err_A": metrics.get("expected_mean_abs_err_A"),
                    "argmax_mean_abs_err_A": metrics.get("argmax_mean_abs_err_A"),
                    "expected_mean_signed_err_A": metrics.get("expected_mean_signed_err_A"),
                    "expected_order_asymmetry_mean_A": metrics.get("expected_order_asymmetry_mean_A"),
                    "contact_prob_auc_proxy_corr": metrics.get("contact_prob_auc_proxy_corr"),
                    "num_valid_pairs": metrics.get("num_valid_pairs"),
                }
            )
            n_rows += 1
    with fsspec.open(path, "w") as f:
        f.write(buf.getvalue())
    logger.info("Wrote %d rows to %s", n_rows, path)


# ---- Main ----


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-prefix",
        default="gs://marin-us-east5/eval/protein-distogram/v1",
        help="Root of the model/target/variant tree.",
    )
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-csv", default=None, help="Optional path; skipped if not given.")
    parser.add_argument(
        "--heatmap-n-seeded",
        type=int,
        default=0,
        help="N value (0..5) used for the per-(target,variant) heatmap pages.",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip the per-(target,variant) heatmap pages (saves ~10s of npz reads).",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip the perplexity-vs-MAE scatter (otherwise queries W&B for val loss).",
    )
    args = parser.parse_args(argv)

    cells = discover_cells(args.input_prefix)
    if not cells:
        logger.error("No cells found at %s; nothing to plot.", args.input_prefix)
        return 1

    if args.output_csv:
        write_csv(args.output_csv, cells)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    perplexity_per_model: dict[str, tuple[int, float]] = {}
    if not args.skip_perplexity:
        logger.info("Fetching W&B val perplexity for %d cells...", len(cells))
        perplexity_per_model = fetch_perplexity_per_model(cells)

    indexed = index_by_target_variant(cells)
    n_heatmap_pages = 0
    with PdfPages(args.output_pdf) as pdf:
        # Headline pages first.
        pdf.savefig(plot_summary_page(cells))
        plt.close()
        if perplexity_per_model:
            fig = plot_perplexity_vs_mae_page(cells, perplexity_per_model)
            pdf.savefig(fig)
            plt.close(fig)

        # Per-(target, variant) pages: MAE-vs-N curves, then heatmap grid.
        for target, variant in sorted(indexed):
            fig = plot_target_variant_page(target, variant, indexed[(target, variant)])
            pdf.savefig(fig)
            plt.close(fig)
            if not args.skip_heatmaps:
                fig = plot_distogram_heatmap_page(
                    args.input_prefix,
                    target,
                    variant,
                    indexed[(target, variant)],
                    n_seeded=args.heatmap_n_seeded,
                )
                pdf.savefig(fig)
                plt.close(fig)
                n_heatmap_pages += 1

    perplexity_pages = 1 if perplexity_per_model else 0
    logger.info(
        "Wrote PDF to %s (1 summary + %d perplexity + %d MAE-vs-N + %d heatmap pages)",
        args.output_pdf,
        perplexity_pages,
        len(indexed),
        n_heatmap_pages,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
