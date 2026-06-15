# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize the per-document / per-token perplexity gap across the Delphi ladder.

Reads the curated metadata and each scale's ``scored_documents.parquet`` (written
by ``score_perplexity_gap_docs.py``) and produces:

- ``ppl_gap_per_doc_vs_scale.png`` — every curated val doc's mean loss vs compute,
  colored by train-Jaccard band; train near-duplicates dashed.
- ``ppl_gap_per_band_vs_scale.png`` — band-mean loss vs compute (the document-set
  view of the band decomposition).
- ``ppl_gap_per_token_<group>.png`` — per-token loss along a flagship high-Jaccard
  val doc, its train twin, and a clean doc, at small / mid / large compute.
- ``ppl_gap_report_data.json`` — the per-doc loss table + flagship token detail.

Pulls only the small scored parquets (a few MB) from GCS.

    uv run --with matplotlib --with gcsfs --with pyarrow \\
        python scripts/analysis/analyze_perplexity_gap.py
"""

import argparse
import json
import logging
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PPL_GAP_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/perplexity_gap"
CURATED_DOCS = f"{PPL_GAP_ROOT}/curated_docs.jsonl"
SCORES_ROOT = f"{PPL_GAP_ROOT}/scores"

# scale label -> (run, training FLOPs) in ladder order.
LADDER = [
    ("3e18", "delphi-3e18-p33m67-k0p20-lr33-a003", 3e18),
    ("9e18", "delphi-9e18-p33m67-k0p20-lr33-a002", 9e18),
    ("2e19", "delphi-2e19-p33m67-k0p20-lr33-a002", 2e19),
    ("3e19", "delphi-3e19-p33m67-k0p20-lr33-a002", 3e19),
    ("9e19", "delphi-9e19-p33m67-k0p20-lr33-a002", 9e19),
    ("2e20", "delphi-2e20-p33m67-k0p20-lr33-a001", 2e20),
    ("3e20", "delphi-3e20-p33m67-k0p20-lr33-a001", 3e20),
    ("1e21", "delphi-1e21-p33m67-9p25b-lr0.33-58ebcb", 1e21),
    ("1e22", "delphi-1e22-p33m67-32p07b-lr0.33-e9132105", 1e22),
]
BAND_COLOR = {
    "clean": "#2166ac",
    "j050": "#4393c3",
    "j060": "#92c5de",
    "j075": "#f4a582",
    "j088": "#b2182b",
    "twin": "#000000",
}
BAND_ORDER = ["clean", "j050", "j060", "j075", "j088"]


def load_meta() -> list[dict]:
    """Curated docs WITH text (needed to match the scored parquet by text)."""
    with fsspec.open(CURATED_DOCS) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scores(run: str) -> dict[str, dict] | None:
    """text -> {per_byte_loss, token_byte_starts/ends, token_ids, num_bytes, token_loss}."""
    hits = fsspec_glob(f"{SCORES_ROOT}/{run}/step-*/scored_documents.parquet")
    if not hits:
        return None
    with fsspec.open(sorted(hits)[-1], "rb") as f:
        t = pq.read_table(f)
    out: dict[str, dict] = {}
    for row in t.to_pylist():
        pbl = np.asarray(row["per_byte_loss"], dtype=np.float64)
        starts = np.asarray(row["token_byte_starts"], dtype=np.int64)
        ends = np.asarray(row["token_byte_ends"], dtype=np.int64)
        # token i loss = sum of its bytes' loss; token 0 is the unscored prefix.
        token_loss = np.array([pbl[s:e].sum() for s, e in zip(starts, ends, strict=True)], dtype=np.float64)
        out[row["text"]] = {
            "per_byte_loss": pbl,
            "token_byte_starts": starts,
            "token_byte_ends": ends,
            "token_ids": np.asarray(row["token_ids"], dtype=np.int64),
            "num_bytes": int(row["num_bytes"]),
            "token_loss": token_loss,
        }
    return out


def doc_mean_loss(score: dict) -> float:
    """Mean nats/token over scored tokens (token 0 carries no loss)."""
    n_tok = max(len(score["token_ids"]) - 1, 1)
    return float(score["per_byte_loss"].sum() / n_tok)


def collect(meta: list[dict]) -> tuple[list[dict], list[tuple[str, float]]]:
    """Attach per-scale mean loss to each curated doc; return (docs, available scales)."""
    scales: list[tuple[str, float]] = []
    scores_by_scale: dict[str, dict[str, dict]] = {}
    for label, run, flops in LADDER:
        s = load_scores(run)
        if s is None:
            continue
        scales.append((label, flops))
        scores_by_scale[label] = s
    for rec in meta:
        rec["loss"] = {}
        rec["score"] = {}
        for label, _ in scales:
            s = scores_by_scale[label].get(rec["text"])
            if s is not None:
                rec["loss"][label] = doc_mean_loss(s)
                rec["score"][label] = s
    return meta, scales


def plot_per_doc(meta: list[dict], scales: list[tuple[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [f for _, f in scales]
    labels = [lab for lab, _ in scales]
    for rec in meta:
        ys = [rec["loss"].get(lab) for lab in labels]
        if any(y is None for y in ys):
            continue
        band = rec["band"]
        style = "--" if rec["role"] == "train_twin" else "-"
        color = BAND_COLOR.get(band, "0.5")
        ax.plot(xs, ys, style, color=color, alpha=0.7, linewidth=1.3, marker="o", markersize=3)
    handles = [plt.Line2D([], [], color=BAND_COLOR[b], marker="o", label=b) for b in BAND_ORDER]
    handles.append(plt.Line2D([], [], color="0.3", linestyle="--", label="train twin"))
    ax.legend(handles=handles, title="max train Jaccard band", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training compute (FLOPs)")
    ax.set_ylabel("doc mean loss (nats/token)")
    ax.set_title(
        "Per-document loss vs compute, by train-Jaccard band (n=5/band)\n"
        "memorized high-J docs collapse at 1e22; wide within-band spread"
    )
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


def plot_per_band(meta: list[dict], scales: list[tuple[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [f for _, f in scales]
    labels = [lab for lab, _ in scales]
    for band in BAND_ORDER:
        vals = [r for r in meta if r["band"] == band and r["role"] == "val"]
        ys = []
        for lab in labels:
            losses = [r["loss"][lab] for r in vals if lab in r["loss"]]
            ys.append(float(np.mean(losses)) if losses else None)
        if any(y is None for y in ys):
            continue
        ax.plot(xs, ys, "-o", color=BAND_COLOR[band], label=band, linewidth=2, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("training compute (FLOPs)")
    ax.set_ylabel("band mean loss (nats/token)")
    ax.set_title(
        "Per-band mean loss vs compute (curated 5-doc/band sample)\n"
        "high-J bands improve far more at 1e22 (memorization gradient); "
        "absolute crossover is a full-population effect, not resolved by n=5"
    )
    ax.legend(title="band", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


def flagship_group(meta: list[dict]) -> str | None:
    """A j088 group that has both the val doc and its train twin scored."""
    groups = {}
    for r in meta:
        groups.setdefault(r["group"], {})[r["role"]] = r
    for g, roles in groups.items():
        if roles.get("val", {}).get("band") == "j088" and "train_twin" in roles and roles["val"]["loss"]:
            return g
    return None


def plot_per_token(meta: list[dict], scales: list[tuple[str, float]], group: str, clean_rec: dict, out: Path) -> None:
    labels = [lab for lab, _ in scales]
    pick = [labels[0], labels[len(labels) // 2], labels[-1]]
    recs = {r["role"]: r for r in meta if r["group"] == group}
    val, twin = recs["val"], recs.get("train_twin")
    panels = [("high-J val doc", val), ("its train twin", twin), ("clean val doc", clean_rec)]
    panels = [(t, r) for t, r in panels if r is not None]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, rec) in zip(axes, panels, strict=True):
        for lab in pick:
            s = rec["score"].get(lab)
            if s is None:
                continue
            tl = s["token_loss"][1:]  # drop unscored first token
            ax.plot(np.arange(len(tl)), tl, linewidth=0.8, label=lab, alpha=0.85)
        mj = rec.get("max_jaccard")
        ax.set_title(
            f"{title}  (max train J={mj if mj is None else round(mj,3)})  mean loss "
            f"{rec['loss'].get(pick[0],float('nan')):.2f}→{rec['loss'].get(pick[-1],float('nan')):.2f}"
        )
        ax.set_ylabel("token loss")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("token position")
    fig.suptitle("Per-token loss: memorized near-dup collapses at 1e22, clean doc does not")
    fig.tight_layout()
    fig.savefig(out, dpi=110)  # keep this dense multi-panel figure under the 500 KB repo limit
    logger.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    meta, scales = collect(load_meta())
    logger.info("scored scales available: %s", [lab for lab, _ in scales])
    if len(scales) < 2:
        raise RuntimeError(f"need >=2 scored scales, have {len(scales)}")

    plot_per_doc(meta, scales, out_dir / "ppl_gap_per_doc_vs_scale.png")
    plot_per_band(meta, scales, out_dir / "ppl_gap_per_band_vs_scale.png")

    group = flagship_group(meta)
    clean_recs = [r for r in meta if r["band"] == "clean" and r["role"] == "val" and r["loss"]]
    if group and clean_recs:
        plot_per_token(meta, scales, group, clean_recs[0], out_dir / f"ppl_gap_per_token_{group}.png")
    else:
        logger.warning("no flagship j088 group with twin (%s) or no clean doc; skipping per-token plot", group)

    # Dump the per-doc loss table for the report (no score arrays).
    table = [{k: r[k] for k in ("doc_id", "role", "band", "max_jaccard", "group")} | {"loss": r["loss"]} for r in meta]
    with open(out_dir / "ppl_gap_report_data.json", "w") as f:
        json.dump({"scales": [lab for lab, _ in scales], "docs": table}, f, indent=2)
    logger.info("wrote %s", out_dir / "ppl_gap_report_data.json")


if __name__ == "__main__":
    main()
