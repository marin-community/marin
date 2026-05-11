# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Slim, talk-ready PDF showing example distogram rollouts from a single model.

For each chosen ``(target, variant)`` cell, lays out a row of expected-distance
heatmaps:

    Ground truth │ Model @ N=0 (cold start) │ Model @ N=5 (5 seeded contacts)

The leftmost panel is the experimental structure's CB-CB distance map; the
middle panel is the model's purely-language-modeling rollout with no contact
hints; the rightmost panel is the same model conditioned on 5 ground-truth
contacts inserted into the prompt. The progression is the key talk visual:
"this is what the model knows from sequence alone vs. with a handful of
hints."

Defaults to ``1b-step-49999`` (best model) on the three benchmark targets at
their native sequences, but ``--model``, ``--variant``, ``--targets``, and
``--n-values`` let you pick something else.

Usage::

    uv run python -m experiments.protein.plot_rollout_examples \\
        --output-pdf /tmp/rollout-examples.pdf

    # Custom — different model, soluble redesigns, only top7 + ubiquitin
    uv run python -m experiments.protein.plot_rollout_examples \\
        --model 1b-step-49999 \\
        --variant soluble-0 \\
        --targets top7 ubiquitin \\
        --n-values 0 2 5 \\
        --output-pdf /tmp/rollout-examples.pdf
"""

import argparse
import io
import json
import logging
import sys

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

INPUT_PREFIX = "gs://marin-us-east5/eval/protein-distogram/v1"


def _expected_distance(probs: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
    """Per-pair expected distance ``Σ p_b · midpoint_b``. ``probs`` is (N, N, B)."""
    return np.einsum("ijb,b->ij", probs, midpoints)


def _load_distogram_npz(model_label: str, target: str, variant: str, n: int) -> dict:
    path = f"{INPUT_PREFIX}/{model_label}/{target}/{variant}/distogram_n{n}.npz"
    with fsspec.open(path, "rb") as fh:
        buf = io.BytesIO(fh.read())  # pyrefly: ignore
    with np.load(buf, allow_pickle=True) as data:
        return {
            "probs": data["probs"],
            "bin_midpoints": data["bin_midpoints"],
            "gt_distance": data["gt_distance"],
        }


def _load_summary(model_label: str, target: str, variant: str) -> dict:
    path = f"{INPUT_PREFIX}/{model_label}/{target}/{variant}/summary.json"
    with fsspec.open(path, "r") as fh:
        return json.load(fh)  # pyrefly: ignore


def _mae_at(summary: dict, n: int) -> float | None:
    """Pick the per-N metric block matching ``n_prompt_contacts == n``."""
    for r in summary.get("per_n", []):
        if int(r.get("n_prompt_contacts", -1)) == n:
            return r.get("metrics", {}).get("expected_mean_abs_err_A")
    return None


def plot_target_row(model: str, target: str, variant: str, n_values: list[int]):
    """Title page-style row: GT plus one panel per ``n`` in ``n_values``."""
    import matplotlib.pyplot as plt

    summary = _load_summary(model, target, variant)
    pdb_id = summary.get("pdb_id", target)
    seq_len = summary.get("sequence_length", "?")
    sequence_override = summary.get("sequence_override")
    seq_label = "native" if not sequence_override else "redesigned"

    # Use first N to grab GT (all N values share the same GT).
    first = _load_distogram_npz(model, target, variant, n_values[0])
    gt = first["gt_distance"]
    midpoints = first["bin_midpoints"]
    max_a = float(midpoints[-1] + (midpoints[1] - midpoints[0]) / 2)
    n_res = gt.shape[0]
    extent = (0.5, n_res + 0.5, 0.5, n_res + 0.5)
    im_kwargs = dict(origin="lower", vmin=0, vmax=max_a, cmap="viridis_r", extent=extent)

    n_panels = 1 + len(n_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.7 * n_panels, 4.2), constrained_layout=True)
    flat = np.atleast_1d(axes).flatten()

    gt_display = np.where(gt <= max_a, gt, max_a)
    last_im = flat[0].imshow(gt_display, **im_kwargs)
    flat[0].set_title("Ground truth", fontsize=10)
    flat[0].set_xlabel("Residue j")
    flat[0].set_ylabel("Residue i")

    for idx, n in enumerate(n_values):
        ax = flat[idx + 1]
        try:
            data = _load_distogram_npz(model, target, variant, n)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"N={n}", fontsize=10)
            ax.axis("off")
            continue
        expected = _expected_distance(data["probs"], data["bin_midpoints"])
        last_im = ax.imshow(expected, **im_kwargs)
        mae = _mae_at(summary, n)
        title = f"N={n} seeded" if n > 0 else "N=0 (cold start)"
        if mae is not None:
            title += f"\nMAE = {mae:.2f} Å"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Residue j")

    fig.colorbar(
        last_im,
        ax=axes,
        shrink=0.85,
        pad=0.02,
        label="Expected CB-CB distance (Å)",
    )

    fig.suptitle(
        f"{pdb_id} ({seq_label}, length {seq_len}) — model {model}",
        fontsize=12,
    )
    return fig


def plot_title_page(model: str, variant: str, targets: list[str], n_values: list[int]):
    """First page: short text-only summary so the talk reader has context."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    body = (
        f"Distogram rollouts from {model}\n"
        f"Sequence variant: {variant}\n"
        f"Targets shown: {', '.join(targets)}\n"
        f"Seeded-contact counts: {', '.join(str(n) for n in n_values)}\n\n"
        "Each subsequent page shows one target. From left to right:\n"
        "  • Ground-truth CB-CB distance map (from the deposited structure)\n"
        "  • Model's expected-distance prediction with 0..N ground-truth\n"
        "    contacts inserted into the prompt as hints\n\n"
        "Cells are colored by expected distance in Ångströms — darker = closer.\n"
        "Diagonals are 0 by definition; sharper off-diagonal structure means\n"
        "the model has correctly identified secondary-structure contacts."
    )
    ax.text(
        0.05,
        0.95,
        body,
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        transform=ax.transAxes,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--model", default="1b-step-49999", help="Model label (eval cell prefix).")
    parser.add_argument("--variant", default="native", help="Sequence variant (native / soluble-0 / soluble-1).")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["top7", "7bny", "ubiquitin"],
        help="Target labels to include (one row per target).",
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[0, 5],
        help="Seeded-contact counts to show alongside ground truth (in order).",
    )
    args = parser.parse_args(argv)

    from matplotlib.backends.backend_pdf import PdfPages

    logger.info(
        "Building rollout PDF: model=%s variant=%s targets=%s n_values=%s",
        args.model,
        args.variant,
        args.targets,
        args.n_values,
    )
    with PdfPages(args.output_pdf) as pdf:
        pdf.savefig(plot_title_page(args.model, args.variant, args.targets, args.n_values))
        for target in args.targets:
            fig = plot_target_row(args.model, target, args.variant, args.n_values)
            pdf.savefig(fig)
    logger.info("Wrote %s", args.output_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
