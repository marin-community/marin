# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot perplexity ↔ downstream-accuracy from the aggregated CSVs.

Reads ``perplexity.csv`` and ``accuracy.csv`` (produced by
``aggregate_perplexity_vs_accuracy.py``), joins them by
``(model_name, checkpoint_step)``, and writes a multi-panel PDF report:

* Distogram MAE vs. validation perplexity, faceted by ``n_prompt_contacts``.
* Contact F1 (long / med / short) vs. validation perplexity, faceted by
  contact range.
* Validation perplexity vs. parameter count (for context).

Usage::

    uv run python -m experiments.protein.plot_perplexity_vs_accuracy \\
        --input-dir gs://marin-us-east5/eval/protein-perplexity-vs-accuracy/v1 \\
        --output-pdf /tmp/protein-perplexity-vs-accuracy.pdf
"""

import argparse
import logging
import sys

import fsspec
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    with fsspec.open(path, "r") as f:
        return pd.read_csv(f)


def join(perplexity: pd.DataFrame, accuracy: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on (model_name, checkpoint_step). Drops rows where either side
    has step=-1 (eval summary didn't record a checkpoint step)."""
    a = accuracy[accuracy["checkpoint_step"] >= 0].copy()
    p = perplexity[["model_name", "checkpoint_step", "val_loss", "val_perplexity"]]
    return a.merge(p, on=["model_name", "checkpoint_step"], how="inner")


def plot_report(joined: pd.DataFrame, output_pdf: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if joined.empty:
        logger.warning("No joined rows to plot. Did the eval summaries record checkpoint_step?")

    with PdfPages(output_pdf) as pdf:
        # --- Distogram MAE vs. perplexity, faceted by N ---
        dist = joined[joined["eval_kind"] == "distogram"]
        if not dist.empty:
            n_values = sorted(dist["n_prompt_contacts"].dropna().unique())
            ncols = max(1, len(n_values))
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False, sharey=True)
            for ax, n in zip(axes[0], n_values, strict=False):
                sub = dist[dist["n_prompt_contacts"] == n]
                for model, g in sub.groupby("model_name"):
                    ax.scatter(
                        g["val_perplexity"],
                        g["distogram_expected_mae_a"],
                        s=20 + (g["model_size_params"] / 5e7),
                        label=model,
                        alpha=0.8,
                    )
                ax.set_xscale("log")
                ax.set_xlabel("val perplexity (log)")
                ax.set_title(f"N={int(n)}")
                ax.grid(True, alpha=0.3)
            axes[0, 0].set_ylabel("Distogram E[|d_pred - d_gt|] (Å)")
            axes[0, 0].legend(fontsize=8, loc="best")
            fig.suptitle("Distogram MAE vs. validation perplexity (faceted by N seeded contacts)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Contact F1 vs. perplexity, faceted by range ---
        contacts = joined[joined["eval_kind"] == "contacts"]
        if not contacts.empty:
            ranges = ["contact_long_f1", "contact_med_f1", "contact_short_f1"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            for ax, col in zip(axes, ranges, strict=True):
                for model, g in contacts.groupby("model_name"):
                    ax.scatter(
                        g["val_perplexity"],
                        g[col],
                        s=20 + (g["model_size_params"] / 5e7),
                        label=model,
                        alpha=0.8,
                    )
                ax.set_xscale("log")
                ax.set_xlabel("val perplexity (log)")
                ax.set_title(col.replace("contact_", "").replace("_f1", "").upper())
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("Consensus F1")
            axes[0].legend(fontsize=8, loc="best")
            fig.suptitle("Contact F1 vs. validation perplexity (faceted by range)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Perplexity vs. params (context plot) ---
        # Use the latest-step row per model.
        latest = joined.sort_values("checkpoint_step").drop_duplicates("model_name", keep="last")
        if not latest.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            for model, row in latest.set_index("model_name").iterrows():
                ax.scatter(row["model_size_params"], row["val_perplexity"], s=80)
                ax.annotate(model, (row["model_size_params"], row["val_perplexity"]))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Parameters (log)")
            ax.set_ylabel("Val perplexity (log)")
            ax.set_title("Validation perplexity vs. parameter count")
            ax.grid(True, alpha=0.3, which="both")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args(argv)

    in_dir = args.input_dir.rstrip("/")
    perplexity = load_csv(f"{in_dir}/perplexity.csv")
    accuracy = load_csv(f"{in_dir}/accuracy.csv")
    logger.info("Loaded %d perplexity + %d accuracy rows", len(perplexity), len(accuracy))

    joined = join(perplexity, accuracy)
    logger.info("Joined: %d rows", len(joined))

    plot_report(joined, args.output_pdf)
    logger.info("Wrote PDF report to %s", args.output_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
