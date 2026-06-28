# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Architecture sweep for the fast-transformer quality regressor.

Trains a grid of :class:`FastTransformer` variants on the oracle-scored train
split and reports held-out oracle metrics for each, so we can read off the
effect of each design choice (pooling kind, pool window, depth, width) against
the fasttext baseline (AUC 0.846 / Spearman 0.641).

The grid is "one axis at a time" off a fixed anchor config: every entry changes
exactly one knob, which isolates that knob's effect. Results (one JSON record
per run plus a markdown table) are written to ``--out``.

Submit on a single v6e slice:

    uv run iris --cluster=marin job run --no-wait \\
        --tpu v6e-4 --enable-extra-resources --extra marin-core:tpu \\
        --cpu 2 --memory 8GB --region europe-west4 \\
        --job-name ft-quality-sweep -- \\
        python -m experiments.datakit.cluster.quality.fast_transformer.sweep \\
          --train gs://marin-eu-west4/datakit/llm-quality-classifier/scored/train-n7000-seed42-sonnet46.parquet \\
          --eval  gs://marin-eu-west4/datakit/llm-quality-classifier/scored/eval-n1000-seed43-sonnet46.parquet \\
          --out   gs://marin-eu-west4/datakit/llm-quality-classifier/fast_transformer/sweep-v1
"""

import argparse
import json
import logging
from dataclasses import asdict, replace

import jax
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import load_packed
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import RunResult, TrainHParams, train_one

logger = logging.getLogger(__name__)

# Fasttext baseline on the same holdout (model/sonnet46-thr05), for reference.
BASELINE = {"auc": 0.846, "spearman_rho": 0.641, "accuracy": 0.784, "f1": 0.718}

FLOPS_BUDGET = 1_000_000


def build_grid(vocab_size: int) -> list[tuple[str, FastTransformerConfig, TrainHParams]]:
    """One-axis-at-a-time grid off a fixed anchor (meanmaxmin, w=64, d=512, L=4)."""
    hp = TrainHParams()
    anchor = FastTransformerConfig(
        vocab_size=vocab_size,
        max_tokens=1024,
        pool_window=64,
        pool_kind="meanmaxmin",
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        final_pool="mean",
        dropout=0.1,
    )
    variants: list[tuple[str, FastTransformerConfig]] = [
        ("anchor:meanmaxmin-w64-d512-L4", anchor),
        # pooling kind
        ("pool=mean", replace(anchor, pool_kind="mean")),
        ("pool=max", replace(anchor, pool_kind="max")),
        ("pool=attn", replace(anchor, pool_kind="attn")),
        # depth (L=0 == learned-embedding pooling + head, no attention)
        ("L=0", replace(anchor, num_layers=0)),
        ("L=2", replace(anchor, num_layers=2)),
        ("L=6", replace(anchor, num_layers=6)),
        # width
        ("d=256", replace(anchor, hidden_dim=256)),
        ("d=768", replace(anchor, hidden_dim=768, num_heads=12)),
        # pool window
        ("w=16", replace(anchor, pool_window=16)),
        ("w=32", replace(anchor, pool_window=32)),
        ("w=128", replace(anchor, pool_window=128)),
        # final pooling
        ("final=attn", replace(anchor, final_pool="attn")),
        # "neural fasttext": single mean window over all tokens, no transformer
        ("neural-bow:mean-w1024-L0", replace(anchor, pool_kind="mean", pool_window=1024, num_layers=0)),
    ]
    return [(label, cfg, hp) for label, cfg in variants]


def _md_table(results: list[RunResult], labels: list[str]) -> str:
    head = (
        "| variant | params(M) | FLOPs/tok(K) | AUC | Spearman | acc | F1 | val_rho | epoch |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = [
        f"| fasttext (baseline) | — | — | {BASELINE['auc']:.3f} | {BASELINE['spearman_rho']:.3f} | "
        f"{BASELINE['accuracy']:.3f} | {BASELINE['f1']:.3f} | — | — |"
    ]
    for label, r in zip(labels, results, strict=True):
        rows.append(
            f"| {label} | {r.params / 1e6:.2f} | {r.flops_per_token / 1e3:.0f} | "
            f"{r.holdout.auc:.3f} | {r.holdout.spearman_rho:.3f} | {r.holdout.accuracy:.3f} | "
            f"{r.holdout.f1:.3f} | {r.val.spearman_rho:.3f} | {r.best_epoch} |"
        )
    return head + "\n".join(rows) + "\n"


def run_sweep(*, train_path: str, eval_path: str, out_dir: str, tokenizer: str, min_count: int, cache_dir: str) -> None:
    # All anchor variants share max_tokens=1024, so one packing serves the grid.
    data = load_packed(
        train_path=train_path,
        eval_path=eval_path,
        tokenizer_name=tokenizer,
        max_tokens=1024,
        min_count=min_count,
        cache_dir=cache_dir,
    )
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())
    grid = build_grid(data.vocab_size)
    logger.info(
        "running %d configs; baseline AUC=%.3f spearman=%.3f", len(grid), BASELINE["auc"], BASELINE["spearman_rho"]
    )

    results: list[RunResult] = []
    labels: list[str] = []
    records: list[dict] = []
    for label, cfg, hp in grid:
        flops = cfg.flops_per_token()
        if flops > FLOPS_BUDGET:
            logger.warning("SKIP %s: %.0f FLOPs/token over budget", label, flops)
            continue
        logger.info("=== %s (FLOPs/tok=%.0f) ===", label, flops)
        result = train_one(cfg, data, hp)
        results.append(result)
        labels.append(label)
        rec = {"label": label, **asdict(result)}
        records.append(rec)
        # Stream partial results so a long sweep is inspectable mid-run.
        with open_url(out_dir.rstrip("/") + "/results.jsonl", "wb") as fh:
            fh.write(("\n".join(json.dumps(r) for r in records) + "\n").encode())

    table = _md_table(results, labels)
    logger.info("\n%s", table)
    with open_url(out_dir.rstrip("/") + "/results.json", "wb") as fh:
        fh.write(json.dumps(records, indent=2).encode())
    with open_url(out_dir.rstrip("/") + "/results.md", "wb") as fh:
        fh.write(("# Fast-transformer quality sweep\n\n" + table).encode())
    logger.info("wrote results to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True, help="Output dir (GCS) for results.json/md")
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--cache-dir", default="/tmp/ft-quality-cache")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    run_sweep(
        train_path=args.train,
        eval_path=args.eval,
        out_dir=args.out,
        tokenizer=args.tokenizer,
        min_count=args.min_count,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
