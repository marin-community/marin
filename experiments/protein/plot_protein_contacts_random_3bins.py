# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a PDF report from `eval_protein_contacts_random_3bins.py` output.

Expects the eval output directory to contain:
  - summary.json   (top-level metrics + per-N summaries)
  - rollouts.json  (raw per-N rollout dumps)
  - matrices.npz   (gt / greedy / frequency matrices per N, plus atom-name freq)

Produces one PDF with:
  1. Summary page (headline numbers per N).
  2. Per-N contact-map grid: GT, greedy (B/W), rollout-frequency heatmap,
     consensus-50% B/W — with per-plot precision/recall.
  3. Per-N bin-colored greedy map (bin_lt4 red, bin_4_12 yellow, bin_gt12 blue).
  4. Per-N by-sequence-range precision/recall bar chart (short <6 vs long >=6).
  5. Per-N correction / pLDDT / atom-validity stats.
  6. Atom-name frequency: GT vs rollouts.

Usage::

    python -m experiments.protein.plot_protein_contacts_random_3bins \\
        --input-dir gs://.../eval/protein-contacts-random-3bins/.../1qys/run-01 \\
        --output /Users/tim/Dropbox/.../reports/exp5-1qys.pdf
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
    gt = summary["ground_truth"]
    override = summary.get("sequence_override")

    lines: list[str] = []
    lines.append(f"Legacy contact-prediction rollouts - {pdb} ({seq_len} residues)  [{label}]")
    lines.append("=" * 78)
    lines.append(f"Model:  {inf['hf_repo']}:{inf['hf_subdir']}")
    lines.append(
        f"Rollouts: {inf['num_rollouts']} @ T={inf['temperature']}, top_k={inf['top_k']}  "
        f"| TP={inf['tensor_parallel_size']}  | total {inf['total_elapsed_seconds']:.0f}s"
    )
    if override is not None:
        lines.append("")
        lines.append(f"Sequence override: {override['target_label']} / {override['method']} #{override['redesign_idx']}")
        lines.append(
            f"  Hamming to native: {override['hamming_distance']}/{seq_len} "
            f"({100*override['hamming_distance']/seq_len:.0f}% of residues redesigned)"
        )
        if override.get("mpnn_score") is not None:
            lines.append(f"  MPNN score: {override['mpnn_score']}")
    lines.append("")
    lines.append(
        f"Ground truth: {gt['num_contacts']} residue pairs in contact "
        f"(<{gt['distance_cutoff_A']:.0f} A heavy-atom, |i-j|>={gt['min_seq_sep']})"
    )
    lines.append("")

    for per_n in summary["per_n"]:
        n = per_n["n_prompt_contacts"]
        g = per_n["greedy"]
        r = per_n["rollouts"]
        lines.append(f"--- N = {n} seeded GT contacts ---")
        # Greedy
        pr = g["lt4_precision_recall"]
        lines.append(
            f"  Greedy:    {g['num_contacts']} contacts  "
            f"(bins {g['bin_counts']})  pLDDT={g['plddt']}  grammar={g['valid_grammar']}"
        )
        lines.append(
            f"             pred lt4 pairs: {pr['n_predicted']}  correct: {pr['n_correct']}  "
            f"P={pr['precision']:.1%}  R={pr['recall']:.1%}"
        )
        gsr, glr = g["by_seq_sep"]["short"], g["by_seq_sep"]["long"]
        lines.append(
            f"             short<6:  P={gsr['precision']:.1%} R={gsr['recall']:.1%} "
            f"({gsr['n_correct']}/{gsr['n_predicted']} pred, {gsr['n_gt']} GT)"
        )
        lines.append(
            f"             long>=6:  P={glr['precision']:.1%} R={glr['recall']:.1%} "
            f"({glr['n_correct']}/{glr['n_predicted']} pred, {glr['n_gt']} GT)"
        )
        # Rollouts
        c50 = r["consensus_50_overall"]
        c10 = r["consensus_10_overall"]
        lines.append(
            f"  Rollouts:  {r['num']} @ median {r['median_num_contacts']:.0f} contacts  "
            f"grammar_ok={r['valid_grammar_frac']:.0%}  atom_valid={r['atom_validity_mean_pct']:.0f}%"
        )
        lines.append(
            f"             consensus>=50%:  pred {c50['n_predicted']}  correct {c50['n_correct']}  "
            f"P={c50['precision']:.1%}  R={c50['recall']:.1%}"
        )
        c50sr = r["consensus_50_by_seq_sep"]["short"]
        c50lr = r["consensus_50_by_seq_sep"]["long"]
        lines.append(f"             consensus>=50% short<6:  P={c50sr['precision']:.1%} R={c50sr['recall']:.1%}")
        lines.append(f"             consensus>=50% long>=6:  P={c50lr['precision']:.1%} R={c50lr['recall']:.1%}")
        lines.append(
            f"             consensus>=10%:  pred {c10['n_predicted']}  correct {c10['n_correct']}  "
            f"P={c10['precision']:.1%}  R={c10['recall']:.1%}"
        )
        pld = r["plddt_counts"]
        lines.append(f"             pLDDT: {pld}")
        lines.append("")

    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=8.5, ha="left", va="top", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _render_contact_maps(
    pdf: PdfPages,
    summary: dict,
    matrices: dict[str, np.ndarray],
    n_prompt: int,
) -> None:
    pdb = summary["pdb_id"]
    seq_len = summary["sequence_length"]
    gt_mat = matrices["gt_pair_matrix"]
    greedy_mat = matrices[f"greedy_pair_matrix_N{n_prompt}"]
    freq = matrices[f"freq_N{n_prompt}"]
    cons_50 = (freq >= 0.5).astype(np.float32)

    per_n = next(p for p in summary["per_n"] if p["n_prompt_contacts"] == n_prompt)
    g = per_n["greedy"]["lt4_precision_recall"]
    c = per_n["rollouts"]["consensus_50_overall"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cmap_bw = ListedColormap(["white", "black"])

    axes[0, 0].imshow(gt_mat, cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
    axes[0, 0].set_title(f"Ground truth (<4 A)\n{summary['ground_truth']['num_contacts']} pairs")

    axes[0, 1].imshow(greedy_mat, cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
    axes[0, 1].set_title(
        f"Greedy bin_lt4 (N={n_prompt})\nP={g['precision']:.0%} R={g['recall']:.0%}  "
        f"({g['n_correct']}/{g['n_predicted']} pred, {g['n_gt']} GT)"
    )

    im = axes[1, 0].imshow(freq, cmap="viridis", vmin=0, vmax=1, origin="lower")
    axes[1, 0].set_title(f"Rollout frequency (N={n_prompt})")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(cons_50, cmap=cmap_bw, vmin=0, vmax=1, origin="lower")
    axes[1, 1].set_title(
        f"Consensus >=50% (N={n_prompt})\nP={c['precision']:.0%} R={c['recall']:.0%}  "
        f"({c['n_correct']}/{c['n_predicted']} pred)"
    )

    for ax in axes.flat:
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, seq_len)

    override_suffix = ""
    ov = summary.get("sequence_override")
    if ov is not None:
        override_suffix = f" | {ov['method']} #{ov['redesign_idx']} " f"(Hamming {ov['hamming_distance']}/{seq_len})"
    fig.suptitle(f"{pdb} - N={n_prompt} seeded{override_suffix}", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_bin_map(
    pdf: PdfPages,
    summary: dict,
    matrices: dict[str, np.ndarray],
    n_prompt: int,
) -> None:
    pdb = summary["pdb_id"]
    seq_len = summary["sequence_length"]
    bin_mat = matrices[f"greedy_bin_matrix_N{n_prompt}"]
    # 0 = empty (white), 1 = bin_lt4 (red), 2 = bin_4_12 (yellow), 3 = bin_gt12 (blue)
    cmap = ListedColormap(["white", "tab:red", "gold", "tab:blue"])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(bin_mat, cmap=cmap, vmin=0, vmax=3, origin="lower")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, seq_len)
    per_n = next(p for p in summary["per_n"] if p["n_prompt_contacts"] == n_prompt)
    bins = per_n["greedy"]["bin_counts"]
    ax.set_title(
        f"{pdb} - Greedy bin assignments (N={n_prompt})\n"
        f"red=bin_lt4 ({bins.get('bin_lt4', 0)}) "
        f"yellow=bin_4_12 ({bins.get('bin_4_12', 0)}) "
        f"blue=bin_gt12 ({bins.get('bin_gt12', 0)})"
    )
    pdf.savefig(fig)
    plt.close(fig)


def _render_range_bars(pdf: PdfPages, summary: dict) -> None:
    per_n_list = summary["per_n"]
    n_vals = [p["n_prompt_contacts"] for p in per_n_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    width = 0.35
    x = np.arange(len(n_vals))

    # Greedy
    gsp = [p["greedy"]["by_seq_sep"]["short"]["precision"] for p in per_n_list]
    glp = [p["greedy"]["by_seq_sep"]["long"]["precision"] for p in per_n_list]
    gsr = [p["greedy"]["by_seq_sep"]["short"]["recall"] for p in per_n_list]
    glr = [p["greedy"]["by_seq_sep"]["long"]["recall"] for p in per_n_list]

    axes[0].bar(x - 1.5 * width, gsp, width, label="short<6 P", color="#1f77b4")
    axes[0].bar(x - 0.5 * width, gsr, width, label="short<6 R", color="#aec7e8")
    axes[0].bar(x + 0.5 * width, glp, width, label="long>=6 P", color="#d62728")
    axes[0].bar(x + 1.5 * width, glr, width, label="long>=6 R", color="#ff9896")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N={n}" for n in n_vals])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Greedy P / R on bin_lt4 pairs")
    axes[0].set_title("Greedy precision/recall by sequence range")
    axes[0].legend(fontsize=8)

    # Consensus 50
    csp = [p["rollouts"]["consensus_50_by_seq_sep"]["short"]["precision"] for p in per_n_list]
    clp = [p["rollouts"]["consensus_50_by_seq_sep"]["long"]["precision"] for p in per_n_list]
    csr = [p["rollouts"]["consensus_50_by_seq_sep"]["short"]["recall"] for p in per_n_list]
    clr = [p["rollouts"]["consensus_50_by_seq_sep"]["long"]["recall"] for p in per_n_list]
    axes[1].bar(x - 1.5 * width, csp, width, label="short<6 P", color="#1f77b4")
    axes[1].bar(x - 0.5 * width, csr, width, label="short<6 R", color="#aec7e8")
    axes[1].bar(x + 0.5 * width, clp, width, label="long>=6 P", color="#d62728")
    axes[1].bar(x + 1.5 * width, clr, width, label="long>=6 R", color="#ff9896")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"N={n}" for n in n_vals])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Consensus >=50% P / R on bin_lt4 pairs")
    axes[1].set_title("Consensus precision/recall by sequence range")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{summary['pdb_id']} - P/R by sequence separation")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_atom_freq(pdf: PdfPages, summary: dict, matrices: dict[str, np.ndarray]) -> None:
    gt_keys = list(matrices["gt_atom_freq_keys"])
    gt_vals = matrices["gt_atom_freq_values"].astype(np.float64)
    ro_keys = list(matrices["rollout_atom_freq_keys"])
    ro_vals = matrices["rollout_atom_freq_values"].astype(np.float64)

    # Normalize to relative frequencies.
    gt_total = gt_vals.sum() or 1.0
    ro_total = ro_vals.sum() or 1.0
    gt_frac = {str(k): v / gt_total for k, v in zip(gt_keys, gt_vals, strict=True)}
    ro_frac = {str(k): v / ro_total for k, v in zip(ro_keys, ro_vals, strict=True)}

    all_atoms = sorted(set(gt_frac) | set(ro_frac))
    # Order by GT frequency (descending), missing at end.
    all_atoms.sort(key=lambda a: (-gt_frac.get(a, 0.0), a))

    x = np.arange(len(all_atoms))
    gt_plot = [gt_frac.get(a, 0.0) for a in all_atoms]
    ro_plot = [ro_frac.get(a, 0.0) for a in all_atoms]

    fig, ax = plt.subplots(figsize=(max(10, len(all_atoms) * 0.28), 4.5))
    width = 0.4
    ax.bar(x - width / 2, gt_plot, width, label="GT (heavy-atom <4A pairs)", color="tab:gray")
    ax.bar(x + width / 2, ro_plot, width, label="Rollouts (all N)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(all_atoms, rotation=45, ha="right")
    ax.set_ylabel("Relative frequency")
    ax.set_title(f"{summary['pdb_id']} - Atom-name usage in contact statements")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_report(input_dir: str, output_pdf: str, label: str) -> None:
    input_dir = input_dir.rstrip("/")
    summary = _read_json(f"{input_dir}/summary.json")
    matrices = _read_npz(f"{input_dir}/matrices.npz")

    # Write local, then copy to final destination if it's remote.
    local_path = output_pdf
    if output_pdf.startswith("gs://") or output_pdf.startswith("s3://"):
        import tempfile

        local_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name

    with PdfPages(local_path) as pdf:
        _render_summary(pdf, summary, label=label)
        for per_n in summary["per_n"]:
            n = per_n["n_prompt_contacts"]
            _render_contact_maps(pdf, summary, matrices, n)
            _render_bin_map(pdf, summary, matrices, n)
        _render_range_bars(pdf, summary)
        _render_atom_freq(pdf, summary, matrices)

    if local_path != output_pdf:
        with open(local_path, "rb") as src, fsspec.open(output_pdf, "wb") as dst:
            dst.write(src.read())
    logger.info("Wrote %s", output_pdf)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="", help="Optional label in the summary page title")
    args = parser.parse_args(argv)
    build_report(args.input_dir, args.output, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
