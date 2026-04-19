# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot Top7 contact-prediction results for the Marin protein-docs-1b-v3 model.

Reads the outputs produced by ``eval_protein_top7_v3.py`` (JSON files in an
output directory, possibly on gs://) and renders a multi-page PDF with:

  1. Summary page with headline metrics and a comparison to the
     LlamaFold-experiments exp5 checkpoint-125500 numbers.
  2. Contact-map grid: ground truth, greedy (bin_lt4), rollout-consensus map.
  3. Per-rollout precision/recall bar chart.
  4. Distance-bin distribution per rollout.
  5. Top predicted pairs (by cross-rollout frequency).

Usage::

    python -m experiments.protein.plot_protein_top7_v3 \\
        --input-dir gs://marin-us-central1/eval/protein-top7-v3/1qys/run-01 \\
        --output plots/protein-top7-v3-1qys.pdf
"""

import argparse
import json
import logging
import os
import sys

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

# LlamaFold-experiments exp5 Top7 baseline (from visualize_contacts_executed.ipynb).
# Greedy prefix=0; 10-rollout consensus (>50%).
EXP5_BASELINE = {
    "source": "LlamaFold-experiments exp5.ethereal-galaxy-3 / checkpoint-125500",
    "pdb_id": "1QYS",
    "num_rollouts": 10,
    "greedy": {
        "num_contacts": 53,
        "num_lt4_pairs": 39,
        "num_correct": 39,
        "precision": 1.000,
        "recall": 0.152,
    },
    "consensus": {
        "num_pred_pairs": 153,
        "num_correct": 139,
        "precision": 0.908,
        "recall": 0.541,
        "short_range": {"num_pred": 133, "num_correct": 124, "precision": 0.932, "recall": 0.867, "num_gt": 143},
        "long_range": {"num_pred": 20, "num_correct": 15, "precision": 0.750, "recall": 0.132, "num_gt": 114},
    },
    "ground_truth": {"num_pairs": 257, "num_short_range": 143, "num_long_range": 114},
}


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def contacts_to_matrix(pairs, seq_len):
    m = np.zeros((seq_len, seq_len), dtype=np.float32)
    for p1, p2 in pairs:
        if 1 <= p1 <= seq_len and 1 <= p2 <= seq_len:
            m[p1 - 1, p2 - 1] = 1
            m[p2 - 1, p1 - 1] = 1
    return m


def _render_summary(pdf: PdfPages, summary: dict, baseline: dict | None, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    pdb = summary["pdb_id"]
    seq_len = summary["sequence_length"]
    gt = summary["ground_truth"]
    model = summary["model"]
    g = summary["greedy"]
    c = summary["consensus"]
    n_rollouts = summary["inference"]["num_rollouts"]
    temp = summary["inference"]["temperature"]

    lines: list[str] = []
    lines.append(f"Top7 Contact Prediction — {pdb} ({seq_len} residues)  [{label}]")
    lines.append("=" * 70)
    lines.append(f"Marin model:  {model}")
    lines.append(f"Rollouts: {n_rollouts} @ T={temp}  |  consensus threshold: >= {c['threshold']:.0%}")
    lines.append("")
    lines.append(
        f"Ground truth: {gt['num_pairs']} residue pairs in contact (<4 Å) "
        f"(short |i-j|<6: {gt['num_short_range']}; long >=6: {gt['num_long_range']})"
    )
    lines.append("")
    lines.append("Greedy (Marin v3)")
    lines.append("-" * 40)
    lines.append(
        f"  Total contacts: {g['num_contacts']}   unique pairs: {g['num_lt4_pairs']}   correct: {g['num_correct']}"
    )
    lines.append(f"  Precision: {g['precision']:.1%}   Recall: {g['recall']:.1%}")
    sr = g["short_range"]
    lr = g["long_range"]
    lines.append(
        f"  Short-range: P={sr['precision']:.1%} R={sr['recall']:.1%}"
        f" ({sr['num_correct']}/{sr['num_pred']} pred, {sr['num_gt']} GT)"
    )
    lines.append(
        f"  Long-range:  P={lr['precision']:.1%} R={lr['recall']:.1%}"
        f" ({lr['num_correct']}/{lr['num_pred']} pred, {lr['num_gt']} GT)"
    )
    av = g["atom_validity"]
    lines.append(f"  Atom validity: {av['fraction']:.1%} ({av['valid']}/{av['total']})")
    lines.append(f"  Grammar valid: {g['valid_grammar']}")
    lines.append("")
    lines.append(f"Rollout consensus (>= {c['threshold']:.0%} of {n_rollouts})")
    lines.append("-" * 40)
    lines.append(f"  Predicted pairs: {c['num_predicted']}   correct: {c['num_correct']}")
    lines.append(f"  Precision: {c['precision']:.1%}   Recall: {c['recall']:.1%}")
    sr = c["short_range"]
    lr = c["long_range"]
    lines.append(
        f"  Short-range: P={sr['precision']:.1%} R={sr['recall']:.1%}"
        f" ({sr['num_correct']}/{sr['num_pred']} pred, {sr['num_gt']} GT)"
    )
    lines.append(
        f"  Long-range:  P={lr['precision']:.1%} R={lr['recall']:.1%}"
        f" ({lr['num_correct']}/{lr['num_pred']} pred, {lr['num_gt']} GT)"
    )
    lines.append("")
    lines.append("Three-way comparison: current | baseline | LlamaFold exp5")
    lines.append("=" * 70)
    lines.append("Note: Marin v3 trained on random-3-bins docs with a v1")
    lines.append("(deterministic-positives-only) tokenizer, so format-specific tokens")
    lines.append("(correction, bin, plddt, random-3-bins header) collapsed to <UNK>.")
    lines.append("GT and scoring (4 A heavy-atom pair-level) are identical everywhere.")
    lines.append("")
    eg = EXP5_BASELINE["greedy"]
    ec = EXP5_BASELINE["consensus"]

    if baseline is not None:
        bg = baseline["greedy"]
        bc = baseline["consensus"]
        lines.append(f"  {'':<20}{'current':>11}{'baseline':>11}{'exp5':>11}{'d(cur-exp5)':>13}")

        def _row3(lbl, cur, base, exp):
            delta = (cur - exp) * 100
            sign = "+" if delta >= 0 else ""
            return f"  {lbl:<20}{cur:>10.1%} {base:>10.1%} {exp:>10.1%} {sign}{delta:>8.1f}pp"

        lines.append(_row3("Greedy P", g["precision"], bg["precision"], eg["precision"]))
        lines.append(_row3("Greedy R", g["recall"], bg["recall"], eg["recall"]))
        lines.append(_row3("Consensus P", c["precision"], bc["precision"], ec["precision"]))
        lines.append(_row3("Consensus R", c["recall"], bc["recall"], ec["recall"]))
        lines.append(
            _row3(
                "  short-range P",
                c["short_range"]["precision"],
                bc["short_range"]["precision"],
                ec["short_range"]["precision"],
            )
        )
        lines.append(
            _row3(
                "  short-range R",
                c["short_range"]["recall"],
                bc["short_range"]["recall"],
                ec["short_range"]["recall"],
            )
        )
        lines.append(
            _row3(
                "  long-range P",
                c["long_range"]["precision"],
                bc["long_range"]["precision"],
                ec["long_range"]["precision"],
            )
        )
        lines.append(
            _row3(
                "  long-range R",
                c["long_range"]["recall"],
                bc["long_range"]["recall"],
                ec["long_range"]["recall"],
            )
        )
    else:
        lines.append(f"  {'':<28}{'current':>14}{'exp5':>14}{'delta':>10}")

        def _row2(lbl, mv, ev):
            delta = (mv - ev) * 100
            sign = "+" if delta >= 0 else ""
            return f"  {lbl:<28}{mv:>13.1%} {ev:>13.1%} {sign}{delta:>6.1f}pp"

        lines.append(_row2("Greedy precision", g["precision"], eg["precision"]))
        lines.append(_row2("Greedy recall", g["recall"], eg["recall"]))
        lines.append(_row2("Consensus precision", c["precision"], ec["precision"]))
        lines.append(_row2("Consensus recall", c["recall"], ec["recall"]))
        lines.append(_row2("  short-range P", c["short_range"]["precision"], ec["short_range"]["precision"]))
        lines.append(_row2("  short-range R", c["short_range"]["recall"], ec["short_range"]["recall"]))
        lines.append(_row2("  long-range P", c["long_range"]["precision"], ec["long_range"]["precision"]))
        lines.append(_row2("  long-range R", c["long_range"]["recall"], ec["long_range"]["recall"]))
    lines.append("")
    lines.append(f"  (exp5 baseline: {EXP5_BASELINE['source']})")

    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=9, ha="left", va="top", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _render_contact_maps(pdf: PdfPages, summary: dict, greedy_json: dict, rollouts_json: dict) -> None:
    seq_len = summary["sequence_length"]
    gt_pairs = {tuple(p) for p in summary["ground_truth"]["pairs"]}
    gt_mat = contacts_to_matrix(gt_pairs, seq_len)

    # Greedy predicted pairs
    greedy_pairs = [(c["pos1"], c["pos2"]) for c in greedy_json["contacts"]]
    greedy_mat = contacts_to_matrix(greedy_pairs, seq_len)

    # Rollout frequency
    n_rollouts = len(rollouts_json["rollouts"])
    freq = np.zeros((seq_len, seq_len), dtype=np.float32)
    for r in rollouts_json["rollouts"]:
        uniq = {(min(c["pos1"], c["pos2"]), max(c["pos1"], c["pos2"])) for c in r["contacts"]}
        for p1, p2 in uniq:
            if 1 <= p1 <= seq_len and 1 <= p2 <= seq_len:
                freq[p1 - 1, p2 - 1] += 1
                freq[p2 - 1, p1 - 1] += 1
    freq /= max(1, n_rollouts)

    consensus_threshold = summary["consensus"]["threshold"]
    cons_mat = (freq >= consensus_threshold).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    cmap_bw = ListedColormap(["white", "black"])
    axes[0, 0].imshow(gt_mat, cmap=cmap_bw, vmin=0, vmax=1)
    axes[0, 0].set_title(f"Ground truth (<4 Å)\n{len(gt_pairs)} pairs")

    axes[0, 1].imshow(greedy_mat, cmap=cmap_bw, vmin=0, vmax=1)
    g = summary["greedy"]
    axes[0, 1].set_title(
        f"Greedy — P={g['precision']:.0%} R={g['recall']:.0%}\n"
        f"{g['num_lt4_pairs']} pairs, {g['num_correct']} correct"
    )

    axes[1, 0].imshow(cons_mat, cmap=cmap_bw, vmin=0, vmax=1)
    c = summary["consensus"]
    axes[1, 0].set_title(
        f"Consensus (>= {consensus_threshold:.0%} of {n_rollouts})\n"
        f"P={c['precision']:.0%} R={c['recall']:.0%} — {c['num_predicted']} pairs"
    )

    im = axes[1, 1].imshow(freq, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Per-pair frequency across rollouts")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")

    fig.suptitle(f"{summary['pdb_id']} contact maps (Marin protein-docs-1b-v3)", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_per_rollout_metrics(pdf: PdfPages, summary: dict) -> None:
    rollouts = summary["rollouts"]
    n = len(rollouts)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))

    idx = np.arange(n)
    width = 0.35
    prec = np.array([r["precision"] for r in rollouts])
    rec = np.array([r["recall"] for r in rollouts])
    axes[0].bar(idx - width / 2, prec, width, label="Precision", color="tab:blue")
    axes[0].bar(idx + width / 2, rec, width, label="Recall", color="tab:orange")
    axes[0].set_xticks(idx)
    axes[0].set_xlabel("Rollout index")
    axes[0].set_ylabel("Residue-pair accuracy")
    axes[0].set_ylim(0, 1)
    g = summary["greedy"]
    axes[0].axhline(g["precision"], linestyle="--", color="tab:blue", alpha=0.6, label=f"Greedy P={g['precision']:.2f}")
    axes[0].axhline(g["recall"], linestyle="--", color="tab:orange", alpha=0.6, label=f"Greedy R={g['recall']:.2f}")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[0].set_title(f"Per-rollout precision/recall  —  {summary['pdb_id']}")

    # Short/long range per rollout
    sr_p = [r["short_range"]["precision"] for r in rollouts]
    sr_r = [r["short_range"]["recall"] for r in rollouts]
    lr_p = [r["long_range"]["precision"] for r in rollouts]
    lr_r = [r["long_range"]["recall"] for r in rollouts]
    axes[1].plot(idx, sr_p, "o-", label="Short-range P", color="tab:blue")
    axes[1].plot(idx, sr_r, "s-", label="Short-range R", color="tab:blue", alpha=0.5)
    axes[1].plot(idx, lr_p, "o-", label="Long-range P", color="tab:red")
    axes[1].plot(idx, lr_r, "s-", label="Long-range R", color="tab:red", alpha=0.5)
    axes[1].set_xticks(idx)
    axes[1].set_xlabel("Rollout index")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    axes[1].set_title("Short-range vs long-range per rollout")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_rollout_size_and_atom_validity(pdf: PdfPages, summary: dict) -> None:
    rollouts = summary["rollouts"]
    n = len(rollouts)
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    num_contacts = np.array([r["num_contacts"] for r in rollouts])
    num_correct = np.array([r["num_correct"] for r in rollouts])
    axes[0].bar(idx, num_contacts, color="tab:gray", label="Generated contacts")
    axes[0].bar(idx, num_correct, color="tab:green", label="Correct (TP)")
    axes[0].set_xticks(idx)
    axes[0].set_xlabel("Rollout index")
    axes[0].set_ylabel("# contacts")
    axes[0].set_title(f"{summary['pdb_id']}: contacts generated per rollout")
    axes[0].legend(loc="upper right")
    for i, r in enumerate(rollouts):
        valid = "grammar ok" if r.get("valid_grammar") else "grammar bad"
        axes[0].text(i, num_contacts[i] + 3, valid, ha="center", va="bottom", fontsize=6, rotation=90)

    atom_frac = np.array([r["atom_validity"]["fraction"] for r in rollouts])
    axes[1].bar(idx, atom_frac, color="tab:purple")
    axes[1].set_xticks(idx)
    axes[1].set_xlabel("Rollout index")
    axes[1].set_ylabel("Atom validity fraction")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Atom name validity per rollout (atom in residue's heavy-atom set)")
    for i, v in enumerate(atom_frac):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_three_way_bars(
    pdf: PdfPages,
    summary: dict,
    baseline: dict | None,
    current_label: str,
    baseline_label: str,
) -> None:
    """Side-by-side precision/recall bars: current run vs baseline run vs exp5."""
    if baseline is None:
        return

    metrics = [
        ("Greedy P", "greedy", "precision"),
        ("Greedy R", "greedy", "recall"),
        ("Consensus P", "consensus", "precision"),
        ("Consensus R", "consensus", "recall"),
        ("Short-range R", "consensus.short_range", "recall"),
        ("Long-range R", "consensus.long_range", "recall"),
    ]

    def _get(src: dict, path: str, field: str) -> float:
        node = src
        for part in path.split("."):
            node = node[part]
        return float(node[field])

    eg = EXP5_BASELINE["greedy"]
    ec = EXP5_BASELINE["consensus"]

    def _exp(path: str, field: str) -> float:
        if path == "greedy":
            return float(eg[field])
        if path == "consensus":
            return float(ec[field])
        if path == "consensus.short_range":
            return float(ec["short_range"][field])
        if path == "consensus.long_range":
            return float(ec["long_range"][field])
        raise KeyError(path)

    current_vals = [_get(summary, path, field) for _, path, field in metrics]
    baseline_vals = [_get(baseline, path, field) for _, path, field in metrics]
    exp5_vals = [_exp(path, field) for _, path, field in metrics]

    labels = [m[0] for m in metrics]
    idx = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(idx - width, current_vals, width, label=f"current: {current_label}", color="tab:blue")
    ax.bar(idx, baseline_vals, width, label=f"baseline: {baseline_label}", color="tab:orange")
    ax.bar(idx + width, exp5_vals, width, label="exp5 checkpoint-125500", color="tab:green")

    for i, v in enumerate(current_vals):
        ax.text(i - width, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(baseline_vals):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(exp5_vals):
        ax.text(i + width, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Metric value")
    ax.set_title(f"{summary['pdb_id']}: Marin v3 (two prompting strategies) vs LlamaFold exp5")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_top_pairs(pdf: PdfPages, summary: dict) -> None:
    top = summary["consensus"]["top_pairs_by_freq"][:40]
    if not top:
        return

    fig, ax = plt.subplots(figsize=(9, 7.5))
    labels = [f"p{t['i']}-p{t['j']}" for t in top]
    freqs = [t["freq"] for t in top]
    colors = ["tab:green" if t["is_ground_truth"] else "tab:red" for t in top]

    idx = np.arange(len(top))
    ax.barh(idx, freqs, color=colors)
    ax.set_yticks(idx)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of rollouts predicting this pair")
    ax.set_xlim(0, 1)
    ax.set_title(f"{summary['pdb_id']}: top predicted pairs (green = true positive)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with summary/rollouts/greedy JSON.")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional second run to compare side-by-side with the current one.",
    )
    parser.add_argument("--current-label", default="current", help="Label for --input-dir in comparison pages.")
    parser.add_argument("--baseline-label", default="baseline", help="Label for --baseline-dir in comparison pages.")
    parser.add_argument("--output", required=True, help="Output PDF path.")
    args = parser.parse_args(argv)

    input_dir = args.input_dir.rstrip("/")
    logger.info("Reading summary from %s", input_dir)
    summary = _read_json(f"{input_dir}/summary.json")
    rollouts_json = _read_json(f"{input_dir}/rollouts.json")
    greedy_json = _read_json(f"{input_dir}/greedy.json")

    baseline_summary: dict | None = None
    if args.baseline_dir:
        baseline_dir = args.baseline_dir.rstrip("/")
        logger.info("Reading baseline summary from %s", baseline_dir)
        baseline_summary = _read_json(f"{baseline_dir}/summary.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    logger.info("Writing PDF to %s", args.output)
    with PdfPages(args.output) as pdf:
        _render_summary(pdf, summary, baseline_summary, args.current_label)
        _render_three_way_bars(pdf, summary, baseline_summary, args.current_label, args.baseline_label)
        _render_contact_maps(pdf, summary, greedy_json, rollouts_json)
        _render_per_rollout_metrics(pdf, summary)
        _render_rollout_size_and_atom_validity(pdf, summary)
        _render_top_pairs(pdf, summary)

    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
