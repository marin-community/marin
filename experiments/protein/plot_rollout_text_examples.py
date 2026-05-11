# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Talk-ready PDF showing the *literal text* of distogram eval queries.

Each plot heatmap is built from many INDEPENDENT teacher-forced queries that
all share a common prefix (the document header + sequence + any seeded
contacts). For each residue pair (i, j) with i<j we form one query:

    <prefix> + <distance> <p_i> <p_j> <atom_i> <atom_j>

and read the model's next-token probability distribution over the 64 distance
bins. That distribution becomes ``probs[i-1, j-1, :]``. The model never sees
any other pair's prediction; every pair is an independent forward pass.

This PDF makes that procedure concrete: per (target, variant, N) page it
shows the full prefix once, then a handful of representative pair queries
with the model's predicted bin (argmax) and the ground-truth bin for
comparison. Helpful for talks because it disambiguates "we're doing one
generation per heatmap" from what we're actually doing.

Usage::

    uv run python -m experiments.protein.plot_rollout_text_examples \\
        --output-pdf /tmp/rollout-text-examples.pdf
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
DISTANCE_BIN_WIDTH_A = 0.5


def _bin_token(k: int) -> str:
    """Return the distance-bin token name for bin index ``k`` (0-indexed)."""
    return f"<d{(k + 1) * DISTANCE_BIN_WIDTH_A:.1f}>"


def _gt_bin_index(distance_a: float, num_bins: int = 64) -> int | None:
    """Map a real-valued GT distance (Å) to its bin index, or None if out of range."""
    if not np.isfinite(distance_a):
        return None
    k = int(distance_a / DISTANCE_BIN_WIDTH_A) - 1
    if k < 0 or k >= num_bins:
        return None
    return k


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


def build_prefix_tokens(summary: dict, n: int) -> list[str]:
    """Reconstruct the shared prefix used for every pair query at this N."""
    sequence_3letter: list[str] = summary["sequence_3letter_used_in_prompt"]
    per_n_entry = next((r for r in summary["per_n"] if int(r["n_prompt_contacts"]) == n), None)
    if per_n_entry is None:
        raise ValueError(f"No per_n entry with N={n}")
    seeded = per_n_entry["seeded_contacts"]

    toks: list[str] = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    toks.extend(f"<{aa}>" for aa in sequence_3letter)
    toks.append("<begin_statements>")
    for type_tok, i, j in seeded:
        toks.extend([type_tok, f"<p{i}>", f"<p{j}>"])
    return toks


def pair_tail_tokens(i: int, j: int, atom_i: str, atom_j: str) -> list[str]:
    """The 5-token suffix appended to the prefix for the (i, j) query."""
    return ["<distance>", f"<p{i}>", f"<p{j}>", f"<{atom_i}>", f"<{atom_j}>"]


def pick_example_pairs(seq_len: int, n_examples: int = 8) -> list[tuple[int, int]]:
    """Pick a spread of (i, j) pairs (i<j) covering short/medium/long range."""
    candidates = [
        (1, 2),                             # adjacent
        (1, 5),                             # short range
        (1, 10),                            # short range
        (max(1, seq_len // 4), seq_len // 4 + 5),       # short range mid
        (max(1, seq_len // 4), max(2, seq_len // 2)),   # medium range
        (max(1, seq_len // 8), max(2, 7 * seq_len // 8)),  # long range
        (max(1, seq_len // 3), seq_len),                # long range
        (max(1, seq_len // 2), seq_len),                # long range
    ]
    # Drop duplicates, drop invalid (j > seq_len or i==j), keep order.
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i, j in candidates:
        if i < 1 or j > seq_len or i >= j:
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        out.append((i, j))
        if len(out) >= n_examples:
            break
    return out


def render_query_page(
    *,
    title: str,
    prefix: list[str],
    pair_queries: list[dict],
    max_prefix_tokens: int,
):
    """Page layout: header → full prefix (blue) → per-pair query rows.

    Each row shows:  ``<distance> <p_i> <p_j> <atom_i> <atom_j>``  (gray)
                    →  ``<d_x.x>`` (predicted bin, green)
                    GT  ``<d_y.y>`` (ground-truth bin, gray italic)
                    p_top = 0.NN  (top-1 probability mass)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(14.0, 8.5))
    ax = fig.add_axes((0.04, 0.03, 0.92, 0.94))
    ax.axis("off")

    fontsize = 8
    line_height = 0.020
    max_line_chars = 175

    # Header.
    ax.text(0.0, 1.0, title, fontsize=11, family="monospace", weight="bold", va="top", transform=ax.transAxes)
    legend = [
        Patch(facecolor="tab:blue", label="PROMPT (shared prefix; same for every pair query)"),
        Patch(facecolor="dimgray", label="QUERY tail (appended to prefix per pair)"),
        Patch(facecolor="tab:green", label="MODEL prediction (argmax over 64 bins)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, frameon=False)

    # 1) Render the prefix at top, monospace blue, with line wrapping.
    y = 0.95
    ax.text(
        0.0, y,
        f"PROMPT ({len(prefix)} tokens, shown first {min(len(prefix), max_prefix_tokens)}):",
        fontsize=9, family="monospace", weight="bold", color="black", va="top", transform=ax.transAxes,
    )
    y -= line_height + 0.005

    shown_prefix = prefix[:max_prefix_tokens]
    char_pos = 0
    line_tokens: list[tuple[str, int]] = []  # (token, char_offset_in_line)
    for tok in shown_prefix:
        sep = 1 if line_tokens else 0
        if char_pos + sep + len(tok) > max_line_chars and line_tokens:
            for t, off in line_tokens:
                ax.text(off / max_line_chars, y, t, fontsize=fontsize, family="monospace",
                        color="tab:blue", va="top", transform=ax.transAxes)
            y -= line_height
            line_tokens = []
            char_pos = 0
            sep = 0
        line_tokens.append((tok, char_pos + sep))
        char_pos += sep + len(tok)
    if line_tokens:
        for t, off in line_tokens:
            ax.text(off / max_line_chars, y, t, fontsize=fontsize, family="monospace",
                    color="tab:blue", va="top", transform=ax.transAxes)
        y -= line_height
    if len(prefix) > max_prefix_tokens:
        ax.text(0.0, y, f"... [{len(prefix) - max_prefix_tokens} more prefix tokens omitted]",
                fontsize=fontsize, family="monospace", color="gray", va="top", transform=ax.transAxes)
        y -= line_height

    # 2) Header for the query section (split across two lines so it fits).
    y -= 0.012
    ax.text(0.0, y,
            "PER-PAIR QUERIES — each row is an independent forward pass.",
            fontsize=9, family="monospace", weight="bold", color="black", va="top", transform=ax.transAxes)
    y -= line_height
    ax.text(0.0, y,
            "Input = the prompt above + the tail in the row; output = one distance-bin token.",
            fontsize=9, family="monospace", color="black", va="top", transform=ax.transAxes)
    y -= line_height + 0.004

    # 3) Render each pair as a row with stable column alignment.
    # Column boundaries are expressed in monospace character offsets, then
    # mapped to axes-fraction via /max_line_chars so columns never overlap.
    def col(char_offset: int) -> float:
        return char_offset / max_line_chars

    col_pair = col(0)        # "(i=  X, j=  Y)"             — 16 chars
    col_tail = col(20)       # tail tokens                   — up to ~42 chars
    col_arrow = col(64)      # "→"
    col_pred = col(68)       # "<d_x.x> (p=0.NN)"            — ~18 chars
    col_gt = col(94)         # "GT: <d_y.y> (z.zz Å)"        — ~22 chars

    for q in pair_queries:
        i, j = q["i"], q["j"]
        tail = q["tail"]
        pred_tok = q["pred_tok"]
        gt_tok = q["gt_tok"]
        pred_p = q["pred_p"]
        gt_dist = q["gt_distance_a"]

        ax.text(col_pair, y, f"(i={i:3d}, j={j:3d})", fontsize=fontsize,
                family="monospace", color="dimgray", va="top", transform=ax.transAxes)
        ax.text(col_tail, y, " ".join(tail), fontsize=fontsize, family="monospace",
                color="dimgray", va="top", transform=ax.transAxes)
        ax.text(col_arrow, y, "→", fontsize=fontsize, family="monospace", color="black",
                va="top", transform=ax.transAxes)
        ax.text(col_pred, y, f"{pred_tok}  p={pred_p:.2f}", fontsize=fontsize,
                family="monospace", color="tab:green", weight="bold", va="top", transform=ax.transAxes)
        gt_str = f"GT: {gt_tok if gt_tok else '(out-of-range)'}  {gt_dist:.2f} Å"
        ax.text(col_gt, y, gt_str, fontsize=fontsize, family="monospace", color="gray",
                style="italic", va="top", transform=ax.transAxes)

        y -= line_height
        if y < 0.04:
            break

    return fig


def plot_title_page(model: str, variant: str, targets: list[str], n_values: list[int], max_prefix_tokens: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    body = (
        f"Distogram eval queries from {model}\n"
        f"Sequence variant: {variant}\n"
        f"Targets shown: {', '.join(targets)}\n"
        f"Seeded-contact counts: {', '.join(str(n) for n in n_values)}\n"
        f"Prefix display cap: {max_prefix_tokens} tokens per page\n\n"
        "WHAT THE EVAL DOES (this PDF visualizes a representative slice):\n"
        "  • Build a single shared PREFIX once per (target, variant, N):\n"
        "      <contacts-and-distances-v1> <begin_sequence> <AA1>...<AAn>\n"
        "      <begin_statements> [<long-range-contact> <p_i> <p_j>] × N\n"
        "  • For every residue pair (i, j) with i<j, form an independent\n"
        "    teacher-forced query by appending a 5-token tail:\n"
        "      <distance> <p_i> <p_j> <atom_i> <atom_j>\n"
        "  • Read the model's next-token distribution over the 64 distance\n"
        "    bins (<d0.5>...<d32.0>); that distribution becomes\n"
        "    probs[i-1, j-1, :] in distogram_n{N}.npz.\n"
        "  • The model NEVER sees any other pair's prediction. Each pair is\n"
        "    an independent forward pass; the prefix is reused unchanged.\n"
        "    The expected-distance heatmaps are built from these per-pair\n"
        "    distributions — there is no autoregressive rollout.\n\n"
        "Per-page layout (next pages):\n"
        "  • Top: the full PROMPT (shared prefix), monospace blue.\n"
        "  • Bottom: a representative spread of pair queries (short / medium\n"
        "    / long range), each with its tail (gray), the model's argmax\n"
        "    bin and top-1 probability (green), and the ground-truth bin for\n"
        "    comparison (gray italic)."
    )
    ax.text(
        0.05, 0.95, body, ha="left", va="top",
        fontsize=11, family="monospace", transform=ax.transAxes,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--model", default="1b-step-49999")
    parser.add_argument("--variant", default="native")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["top7", "7bny", "ubiquitin"],
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[0, 5],
        help="Seeded-contact counts to render (one page per (target, N)).",
    )
    parser.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=600,
        help="Cap rendered prefix tokens per page (the actual prefix is used in full for inference).",
    )
    parser.add_argument(
        "--n-example-pairs",
        type=int,
        default=8,
        help="How many pair queries to display per page.",
    )
    args = parser.parse_args(argv)

    from matplotlib.backends.backend_pdf import PdfPages

    logger.info(
        "Building rollout-text PDF: model=%s variant=%s targets=%s n_values=%s",
        args.model, args.variant, args.targets, args.n_values,
    )
    with PdfPages(args.output_pdf) as pdf:
        pdf.savefig(plot_title_page(args.model, args.variant, args.targets, args.n_values, args.max_prefix_tokens))
        for target in args.targets:
            for n in args.n_values:
                summary = _load_summary(args.model, target, args.variant)
                npz = _load_distogram_npz(args.model, target, args.variant, n)
                probs = npz["probs"]
                gt_distance = npz["gt_distance"]
                atom_per_residue = summary["atom_per_residue"]
                seq_len = int(summary["sequence_length"])

                prefix = build_prefix_tokens(summary, n)
                pairs = pick_example_pairs(seq_len, n_examples=args.n_example_pairs)

                pair_queries = []
                for i, j in pairs:
                    atom_i = atom_per_residue[i - 1]
                    atom_j = atom_per_residue[j - 1]
                    tail = pair_tail_tokens(i, j, atom_i, atom_j)
                    p_i_j = probs[i - 1, j - 1]
                    pred_k = int(np.argmax(p_i_j))
                    pred_p = float(p_i_j[pred_k])
                    gt_k = _gt_bin_index(float(gt_distance[i - 1, j - 1]))
                    gt_tok = _bin_token(gt_k) if gt_k is not None else None
                    pair_queries.append({
                        "i": i, "j": j, "tail": tail,
                        "pred_tok": _bin_token(pred_k), "pred_p": pred_p,
                        "gt_tok": gt_tok,
                        "gt_distance_a": float(gt_distance[i - 1, j - 1]),
                    })

                title = f"{target} ({args.variant}) — N={n} seeded contacts — model {args.model}"
                fig = render_query_page(
                    title=title,
                    prefix=prefix,
                    pair_queries=pair_queries,
                    max_prefix_tokens=args.max_prefix_tokens,
                )
                pdf.savefig(fig)
    logger.info("Wrote %s", args.output_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
