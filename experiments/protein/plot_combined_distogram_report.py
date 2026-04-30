# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One big PDF: distogram metrics across all models x targets x variants.

Reads the layout produced by ``run_distogram_evals.py``::

    gs://marin-us-east5/eval/protein-distogram/v1/
        <model_label>/
            <target_label>/
                <variant_label>/
                    summary.json

For every (target, variant) pair found, writes a page with all available
models overlaid:

* MAE-vs-N (expected and argmax) — one curve per model.
* Contact-prob correlation vs. N — one curve per model.

Also writes a wide-table summary CSV at the same time.

Usage::

    uv run python -m experiments.protein.plot_combined_distogram_report \\
        --input-prefix gs://marin-us-east5/eval/protein-distogram/v1 \\
        --output-pdf /tmp/protein-distogram-cross-run.pdf \\
        --output-csv /tmp/protein-distogram-cross-run.csv
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)


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
    args = parser.parse_args(argv)

    cells = discover_cells(args.input_prefix)
    if not cells:
        logger.error("No cells found at %s; nothing to plot.", args.input_prefix)
        return 1

    if args.output_csv:
        write_csv(args.output_csv, cells)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    indexed = index_by_target_variant(cells)
    with PdfPages(args.output_pdf) as pdf:
        pdf.savefig(plot_summary_page(cells))
        plt.close()

        # One page per (target, variant). Sort: target first (alphabetical),
        # variant alphabetical within each target.
        for target, variant in sorted(indexed):
            fig = plot_target_variant_page(target, variant, indexed[(target, variant)])
            pdf.savefig(fig)
            plt.close(fig)

    logger.info("Wrote PDF to %s (1 summary page + %d per-(target,variant) pages)", args.output_pdf, len(indexed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
