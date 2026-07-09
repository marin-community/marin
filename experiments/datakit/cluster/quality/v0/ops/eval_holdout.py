# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the trained LLM-quality classifier against held-out oracle labels.

Given:

  * ``--model-bin``: a ``model.bin`` produced by :mod:`llm_quality.train`
  * ``--scored-holdout``: a parquet emitted by :mod:`llm_quality.score`
    on a **fresh** sample (different seed than the training sample)

Computes the model's ``P(label=='1')`` per document and reports:

  * **AUC** of the predicted probability against the binarized LLM label
    (LLM ``score_normalized >= threshold`` → 1).
  * **Spearman rho** between predicted ``P(high)`` and the continuous
    LLM normalized score.
  * **Accuracy / precision / recall / F1** at the same threshold.
  * Per-source breakdown of the same metrics.
  * TSV with one row per doc (``source, id, llm_score, llm_label, p_high``).

These numbers are the real generalization signal -- the val-split metrics
inside :mod:`train` are sampled from the same distribution and tend to be
optimistic. A held-out oracle sample (drawn with a different seed) is
the closest practical proxy for true generalization without spending
another order of magnitude on labels.

Submit:

    uv run iris --cluster=marin job run --no-wait --memory=2G --extra=cpu \\
        --region europe-west4 \\
        --job-name "llm-quality-eval-$(date +%Y%m%d-%H%M%S)" -- \\
        python -m experiments.datakit.cluster.quality.v0.ops.eval_holdout \\
          --model-bin       $BASE/model/sonnet46-thr05/model.bin \\
          --scored-holdout  $BASE/scored/eval-n1000-seed43-sonnet46.parquet \\
          --report          $BASE/eval/holdout-sonnet46-thr05.tsv

where BASE=gs://marin-eu-west4/datakit/llm-quality-classifier
"""

import argparse
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass

import fasttext
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 0.5
MAX_TEXT_CHARS = 100_000


def _patch_numpy_copy_compat() -> None:
    """Same shim as in experiments.datakit.fasttext: NumPy 2 vs fasttext-wheel 0.9.2."""
    if getattr(np, "_fasttext_copy_compat", False):
        return
    _orig = np.array

    def _shim(*args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return _orig(*args, **kwargs)

    np.array = _shim
    np._fasttext_copy_compat = True


def _load_model_local(model_bin_path: str):
    """Stream the model.bin to a temp file and return ``(model, local_path)``."""
    fs, resolved = url_to_fs(model_bin_path)
    fd, local_path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as tmp, fs.open(resolved, "rb") as src:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    return fasttext.load_model(local_path), local_path


def predict_p_high(model, text: str, max_chars: int = MAX_TEXT_CHARS) -> float:
    """Return ``P(label == '__label__1')`` for *text*. 0.0 if no label is "1"."""
    if len(text) > max_chars:
        text = text[:max_chars]
    cleaned = text.replace("\n", " ").replace("\r", " ")
    labels, probs = model.predict(cleaned, k=-1)
    for label, prob in zip(labels, probs, strict=False):
        if label == "__label__1":
            return float(prob)
    return 0.0


def _avg_ranks(xs: list[float]) -> list[float]:
    """Return tie-aware 1-indexed average ranks."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, average-rank handling for ties."""
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    rx = _avg_ranks(x)
    ry = _avg_ranks(y)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def auc(y_true: list[int], y_score: list[float]) -> float:
    """ROC AUC via the rank-based formula (O(n log n))."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(y_score)
    sum_pos_ranks = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class HoldoutMetrics:
    n: int
    n_pos: int
    n_neg: int
    auc: float
    spearman_rho: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float, float]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def _compute_metrics(rows: list[dict], threshold: float) -> HoldoutMetrics:
    llm_norm = [r["llm_score"] for r in rows]
    p_high = [r["p_high"] for r in rows]
    y_true = [1 if s >= threshold else 0 for s in llm_norm]
    y_pred = [1 if p >= 0.5 else 0 for p in p_high]
    acc, prec, rec, f1 = _binary_metrics(y_true, y_pred)
    return HoldoutMetrics(
        n=len(rows),
        n_pos=sum(y_true),
        n_neg=len(y_true) - sum(y_true),
        auc=auc(y_true, p_high),
        spearman_rho=spearman_rho(p_high, llm_norm),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
    )


def _read_holdout(scored_holdout: str) -> list[dict]:
    with StoragePath(scored_holdout).open("rb") as fh:
        t = pq.read_table(fh)
    out = []
    cols = {n: t.column(n).to_pylist() for n in ("source", "id", "text", "score_raw", "score_normalized")}
    for s, doc_id, text, raw, norm in zip(
        cols["source"], cols["id"], cols["text"], cols["score_raw"], cols["score_normalized"], strict=True
    ):
        if raw is None or int(raw) < 0:
            continue
        if norm is None or (isinstance(norm, float) and math.isnan(norm)):
            continue
        if not text:
            continue
        out.append({"source": str(s), "id": str(doc_id), "text": str(text), "llm_score": float(norm)})
    return out


def _write_report(path: str, rows: list[dict], overall: HoldoutMetrics, per_source: dict[str, HoldoutMetrics]) -> None:
    summary = {
        "overall": overall.__dict__,
        "per_source": {s: m.__dict__ for s, m in per_source.items()},
    }
    json_blob = json.dumps(summary, indent=2, sort_keys=True)

    lines = ["source\tid\tllm_score\tllm_label\tp_high"]
    for r in rows:
        lines.append(f"{r['source']}\t{r['id']}\t{r['llm_score']:.6f}\t{int(r['llm_label'])}\t{r['p_high']:.6f}")
    tsv_blob = "\n".join(lines) + "\n"

    if path == "-":
        print(json_blob)
        print()
        print(tsv_blob[:2000] + ("...[truncated]" if len(tsv_blob) > 2000 else ""))
        return

    StoragePath(path).write_bytes(tsv_blob.encode("utf-8"))
    summary_path = os.path.splitext(path)[0] + ".summary.json" if not path.endswith(".json") else path
    StoragePath(summary_path).write_bytes(json_blob.encode("utf-8"))
    logger.info("wrote %d-row TSV -> %s", len(rows), path)
    logger.info("wrote summary -> %s", summary_path)


def evaluate(
    *,
    model_bin_path: str,
    scored_holdout: str,
    threshold: float,
    report_path: str,
) -> HoldoutMetrics:
    _patch_numpy_copy_compat()
    holdout = _read_holdout(scored_holdout)
    if not holdout:
        raise RuntimeError(f"no usable rows in {scored_holdout}")
    logger.info("loaded %d holdout rows from %s", len(holdout), scored_holdout)

    model, local_path = _load_model_local(model_bin_path)
    logger.info("loaded model from %s (local=%s, labels=%s)", model_bin_path, local_path, model.get_labels())

    rows: list[dict] = []
    for r in holdout:
        p = predict_p_high(model, r["text"])
        rows.append(
            {
                "source": r["source"],
                "id": r["id"],
                "llm_score": r["llm_score"],
                "llm_label": 1 if r["llm_score"] >= threshold else 0,
                "p_high": p,
            }
        )

    overall = _compute_metrics(rows, threshold)
    logger.info(
        "OVERALL n=%d (pos=%d neg=%d) AUC=%.4f spearman=%.4f acc=%.4f P=%.4f R=%.4f F1=%.4f",
        overall.n,
        overall.n_pos,
        overall.n_neg,
        overall.auc,
        overall.spearman_rho,
        overall.accuracy,
        overall.precision,
        overall.recall,
        overall.f1,
    )

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    per_source: dict[str, HoldoutMetrics] = {}
    for s, srows in by_source.items():
        if len(srows) < 5:
            continue
        per_source[s] = _compute_metrics(srows, threshold)

    _write_report(report_path, rows, overall, per_source)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", required=True, help="GCS path to trained model.bin")
    parser.add_argument("--scored-holdout", required=True, help="Scored holdout parquet (score.py output)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--report", default="-", help="Output TSV path (use '-' for stdout summary)")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    evaluate(
        model_bin_path=args.model_bin,
        scored_holdout=args.scored_holdout,
        threshold=args.threshold,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
