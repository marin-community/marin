# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Architecture sweep for the fast-transformer quality regressor.

Trains a grid of :class:`FastTransformer` variants on the oracle-scored train
split and reports held-out oracle metrics for each, so we can read off the
effect of each design choice (pooling kind, pool window, depth, width,
regularization) against the fasttext baseline (AUC 0.846 / Spearman 0.641).

Three grids:

* ``phase1`` -- one axis at a time off a fixed anchor (meanmaxmin / w=64 /
  d=512 / L=4); isolates each architectural knob.
* ``refine`` -- regularization and vocabulary-pruning variants on the anchor,
  to push generalization on the small (5.6k-doc) oracle set.
* ``context`` -- a context-length ladder (1k -> 16k tokens) at the anchor, since
  the model and the baseline only read ~the first 1k of each doc while the oracle
  scored up to ~8k tokens; pooling makes the extra context cheap.

Each grid entry declares the tokenization it needs (``max_tokens``,
``min_count``); the runner packs once per distinct requirement and reuses it.
Results (one JSON record per run plus a markdown table) are written to ``--out``.

Submit on a single v6e slice:

    uv run iris --cluster=marin job run --no-wait \\
        --tpu v6e-4 --enable-extra-resources --extra marin-core:tpu \\
        --cpu 2 --memory 16GB --region europe-west4 \\
        --job-name ft-quality-sweep -- \\
        python -m experiments.datakit.cluster.quality.fast_transformer.sweep \\
          --train gs://marin-eu-west4/datakit/llm-quality-classifier/scored/train-n7000-seed42-sonnet46.parquet \\
          --eval  gs://marin-eu-west4/datakit/llm-quality-classifier/scored/eval-n1000-seed43-sonnet46.parquet \\
          --out   gs://marin-eu-west4/datakit/llm-quality-classifier/fast_transformer/sweep-v1 --grid phase1
"""

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import jax
from rigging.filesystem import open_url, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import load_packed
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import RunResult, TrainHParams, train_one

logger = logging.getLogger(__name__)

# Fasttext baseline on the same holdout (model/sonnet46-thr05), for reference.
BASELINE = {"auc": 0.846, "spearman_rho": 0.641, "accuracy": 0.784, "f1": 0.718}

DEFAULT_FLOPS_BUDGET = 1_000_000


@dataclass(frozen=True)
class GridEntry:
    """One sweep run: a label, the tokenization it needs, and how to build it."""

    label: str
    max_tokens: int
    min_count: int
    build: Callable[[int], FastTransformerConfig]  # vocab_size -> config
    hp: TrainHParams


def _anchor(vocab_size: int) -> FastTransformerConfig:
    return FastTransformerConfig(
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


def build_phase1_grid() -> list[GridEntry]:
    """One-axis-at-a-time grid off the anchor (meanmaxmin, w=64, d=512, L=4)."""
    hp = TrainHParams()
    variants: list[tuple[str, Callable[[FastTransformerConfig], FastTransformerConfig]]] = [
        ("anchor:meanmaxmin-w64-d512-L4", lambda a: a),
        ("pool=mean", lambda a: replace(a, pool_kind="mean")),
        ("pool=max", lambda a: replace(a, pool_kind="max")),
        ("pool=attn", lambda a: replace(a, pool_kind="attn")),
        ("L=0", lambda a: replace(a, num_layers=0)),
        ("L=2", lambda a: replace(a, num_layers=2)),
        ("L=6", lambda a: replace(a, num_layers=6)),
        ("d=256", lambda a: replace(a, hidden_dim=256)),
        ("d=768", lambda a: replace(a, hidden_dim=768, num_heads=12)),
        ("w=16", lambda a: replace(a, pool_window=16)),
        ("w=32", lambda a: replace(a, pool_window=32)),
        ("w=128", lambda a: replace(a, pool_window=128)),
        ("final=attn", lambda a: replace(a, final_pool="attn")),
        ("neural-bow:mean-w1024-L0", lambda a: replace(a, pool_kind="mean", pool_window=1024, num_layers=0)),
    ]
    return [GridEntry(label, 1024, 2, (lambda v, f=f: f(_anchor(v))), hp) for label, f in variants]


def build_refine_grid() -> list[GridEntry]:
    """Regularization + vocab-pruning variants to fight overfitting on 5.6k docs.

    The anchor overfits (train MSE -> 0 by ~epoch 10) and the embedding table is
    the bulk of the params, so we probe stronger dropout / weight decay, a
    smaller embedding, shallower/narrower models, and more aggressive ``minCount``
    vocab pruning (which both shrinks the table and matches fasttext's pruning).
    """
    hp = TrainHParams()
    entries: list[GridEntry] = []
    for mc in (2, 8):
        tag = f"mc{mc}"
        entries += [
            GridEntry(f"{tag}:base", 1024, mc, (lambda v: _anchor(v)), hp),
            GridEntry(f"{tag}:dropout0.3", 1024, mc, (lambda v: replace(_anchor(v), dropout=0.3)), hp),
            GridEntry(f"{tag}:wd0.2", 1024, mc, (lambda v: _anchor(v)), replace(hp, weight_decay=0.2)),
            GridEntry(f"{tag}:embed128", 1024, mc, (lambda v: replace(_anchor(v), embed_dim=128)), hp),
            GridEntry(f"{tag}:d256-L2", 1024, mc, (lambda v: replace(_anchor(v), hidden_dim=256, num_layers=2)), hp),
            GridEntry(
                f"{tag}:reg++",
                1024,
                mc,
                (lambda v: replace(_anchor(v), dropout=0.3, embed_dim=128)),
                replace(hp, weight_decay=0.2),
            ),
        ]
    return entries


def build_context_grid() -> list[GridEntry]:
    """Long-context sweep: see the whole document, not just its head.

    The oracle scored each doc on up to 32k chars (~8k tokens), but the model AND
    the fasttext baseline only read ~the first 1k tokens -- 28% of all tokens
    (median doc 646 tokens, mean 2319, 90th pct ~7.9k). Pooling makes long context
    cheap (per-token cost is roughly flat at fixed window), so feed 4k-16k tokens
    and measure whether the rest of the document carries quality signal. This also
    finally gives attention a real job: 64-256 super-tokens spanning the full doc.

    Batch size scales down with context so the [B, T, E] embedding activation stays
    bounded on a single v6e chip (the model is not sharded); epochs scale down too
    since the smaller batch means more steps per epoch.
    """

    def hp(batch_size: int, epochs: int) -> TrainHParams:
        return replace(TrainHParams(), batch_size=batch_size, epochs=epochs)

    return [
        # Context ladder at the anchor architecture (meanmaxmin / w=64 / d=512 / L=4).
        GridEntry("ctx1024-w64", 1024, 2, lambda v: _anchor(v), hp(512, 60)),
        GridEntry("ctx4096-w64", 4096, 2, lambda v: replace(_anchor(v), max_tokens=4096), hp(128, 40)),
        GridEntry("ctx8192-w64", 8192, 2, lambda v: replace(_anchor(v), max_tokens=8192), hp(64, 30)),
        GridEntry("ctx16384-w64", 16384, 2, lambda v: replace(_anchor(v), max_tokens=16384), hp(32, 25)),
        # Does depth/finer attention pay off now that attention spans the full doc?
        GridEntry("ctx8192-w64-L8", 8192, 2, lambda v: replace(_anchor(v), max_tokens=8192, num_layers=8), hp(64, 30)),
        GridEntry("ctx8192-w32", 8192, 2, lambda v: replace(_anchor(v), max_tokens=8192, pool_window=32), hp(64, 30)),
        GridEntry(
            "ctx16384-w128-L6",
            16384,
            2,
            lambda v: replace(_anchor(v), max_tokens=16384, pool_window=128, num_layers=6),
            hp(32, 25),
        ),
    ]


GRIDS: dict[str, Callable[[], list[GridEntry]]] = {
    "phase1": build_phase1_grid,
    "refine": build_refine_grid,
    "context": build_context_grid,
}


def _md_table(records: list[dict]) -> str:
    head = (
        "| variant | params(M) | FLOPs/tok(K) | AUC | Spearman | acc | F1 | val_rho | epoch |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = [
        f"| fasttext (baseline) | — | — | {BASELINE['auc']:.3f} | {BASELINE['spearman_rho']:.3f} | "
        f"{BASELINE['accuracy']:.3f} | {BASELINE['f1']:.3f} | — | — |"
    ]
    for r in records:
        h = r["holdout"]
        rows.append(
            f"| {r['label']} | {r['params'] / 1e6:.2f} | {r['flops_per_token'] / 1e3:.0f} | "
            f"{h['auc']:.3f} | {h['spearman_rho']:.3f} | {h['accuracy']:.3f} | "
            f"{h['f1']:.3f} | {r['val']['spearman_rho']:.3f} | {r['best_epoch']} |"
        )
    return head + "\n".join(rows) + "\n"


def _write(out_dir: str, name: str, blob: str) -> None:
    with open_url(out_dir.rstrip("/") + "/" + name, "wb") as fh:
        fh.write(blob.encode())


def _load_done(out_dir: str) -> list[dict]:
    """Read any streamed results so a restarted (e.g. OOM-retried) job resumes."""
    path = out_dir.rstrip("/") + "/results.jsonl"
    fs, resolved = url_to_fs(path)
    if not fs.exists(resolved):
        return []
    with fs.open(resolved, "rb") as fh:
        text = fh.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_sweep(
    *,
    grid: list[GridEntry],
    train_path: str,
    eval_path: str,
    out_dir: str,
    tokenizer: str,
    cache_dir: str,
    flops_budget: int,
) -> None:
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())
    logger.info(
        "running %d configs; baseline AUC=%.3f spearman=%.3f", len(grid), BASELINE["auc"], BASELINE["spearman_rho"]
    )

    # Group by tokenization requirement so each distinct packing is built once.
    groups: dict[tuple[int, int], list[GridEntry]] = {}
    for entry in grid:
        groups.setdefault((entry.max_tokens, entry.min_count), []).append(entry)

    records: list[dict] = _load_done(out_dir)
    done = {r["label"] for r in records}
    if done:
        logger.info("resuming: %d configs already done (%s)", len(done), sorted(done))

    for (max_tokens, min_count), entries in groups.items():
        pending = [e for e in entries if e.label not in done]
        if not pending:
            continue
        data = load_packed(
            train_path=train_path,
            eval_path=eval_path,
            tokenizer_name=tokenizer,
            max_tokens=max_tokens,
            min_count=min_count,
            cache_dir=cache_dir,
        )
        for entry in pending:
            cfg = entry.build(data.vocab_size)
            flops = cfg.flops_per_token()
            if flops > flops_budget:
                logger.warning("SKIP %s: %.0f FLOPs/token over budget %d", entry.label, flops, flops_budget)
                continue
            logger.info("=== %s (FLOPs/tok=%.0f) ===", entry.label, flops)
            result: RunResult = train_one(cfg, data, entry.hp)
            records.append({"label": entry.label, **asdict(result)})
            # Stream partial results so a restart resumes and the run is inspectable mid-flight.
            _write(out_dir, "results.jsonl", "\n".join(json.dumps(r) for r in records) + "\n")
            # Release compiled executables for this (distinct) structure to bound host memory.
            jax.clear_caches()

    table = _md_table(records)
    logger.info("\n%s", table)
    _write(out_dir, "results.json", json.dumps(records, indent=2))
    _write(out_dir, "results.md", "# Fast-transformer quality sweep\n\n" + table)
    logger.info("wrote results to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True, help="Output dir (GCS) for results.json/md")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="phase1")
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--cache-dir", default="/tmp/ft-quality-cache")
    parser.add_argument(
        "--flops-budget", type=int, default=DEFAULT_FLOPS_BUDGET, help="Skip configs above this FLOPs/token"
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    run_sweep(
        grid=GRIDS[args.grid](),
        train_path=args.train,
        eval_path=args.eval,
        out_dir=args.out,
        tokenizer=args.tokenizer,
        cache_dir=args.cache_dir,
        flops_budget=args.flops_budget,
    )


if __name__ == "__main__":
    main()
