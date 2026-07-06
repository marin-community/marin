# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the LLM-quality classifier to AllenAI's dolma3-fasttext-quality.

Loads both models, runs them over the same scored holdout parquet (the
output of :mod:`llm_quality.score`), and reports Spearman rank
correlations:

  * ``ours`` vs ``llm_score``  -- how well we predict the rubric.
  * ``dolma3`` vs ``llm_score`` -- how well dolma3 predicts our rubric.
  * ``ours`` vs ``dolma3``     -- how much we agree with dolma3.

Per-source breakdown identifies sources where the two classifiers
diverge -- those are the ones where the new model is adding (or losing)
signal vs. the production baseline.

The dolma3 model is consumed as a pre-staged ``model.bin`` on GCS to
avoid re-downloading the 4 GiB file. The dolma3_quality DAG already
stages it under
``gs://marin-eu-west4/datakit/dolma3-quality/_model/dolma3-quality_<hash>/model.bin``;
point ``--dolma3-bin`` at that path (or any other GCS copy).

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=2 --memory=16G \\
        --extra=cpu --region europe-west4 \\
        --job-name "llm-quality-vs-dolma3-$(date +%Y%m%d-%H%M%S)" -- \\
        python -m experiments.datakit.cluster.quality.v0.ops.eval_vs_dolma3 \\
          --model-bin       $BASE/model/sonnet46-thr05/model.bin \\
          --dolma3-bin      gs://marin-eu-west4/datakit/dolma3-quality/_model/dolma3-quality_<hash>/model.bin \\
          --scored-holdout  $BASE/scored/eval-n1000-seed43-sonnet46.parquet \\
          --report          $BASE/eval/vs-dolma3-sonnet46-thr05.tsv

where BASE=gs://marin-eu-west4/datakit/llm-quality-classifier
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.v0.ops.eval_holdout import (
    _load_model_local,
    _patch_numpy_copy_compat,
    predict_p_high,
    spearman_rho,
)

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 100_000


@dataclass
class CorrelationRow:
    n: int
    spearman_ours_vs_llm: float
    spearman_dolma3_vs_llm: float
    spearman_ours_vs_dolma3: float


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


def _correlations(rows: list[dict]) -> CorrelationRow:
    ours = [r["p_high_ours"] for r in rows]
    dolma3 = [r["p_high_dolma3"] for r in rows]
    llm = [r["llm_score"] for r in rows]
    return CorrelationRow(
        n=len(rows),
        spearman_ours_vs_llm=spearman_rho(ours, llm),
        spearman_dolma3_vs_llm=spearman_rho(dolma3, llm),
        spearman_ours_vs_dolma3=spearman_rho(ours, dolma3),
    )


def compare(
    *,
    model_bin_path: str,
    dolma3_bin_path: str,
    scored_holdout: str,
    report_path: str,
) -> CorrelationRow:
    _patch_numpy_copy_compat()
    rows = _read_holdout(scored_holdout)
    if not rows:
        raise RuntimeError(f"no usable rows in {scored_holdout}")
    logger.info("loaded %d holdout rows from %s", len(rows), scored_holdout)

    ours_model, ours_local = _load_model_local(model_bin_path)
    logger.info("loaded ours from %s (local=%s)", model_bin_path, ours_local)

    dolma3_model, dolma3_local = _load_model_local(dolma3_bin_path)
    logger.info("loaded dolma3 from %s (local=%s)", dolma3_bin_path, dolma3_local)

    enriched: list[dict] = []
    for r in rows:
        p_ours = predict_p_high(ours_model, r["text"])
        p_dolma3 = predict_p_high(dolma3_model, r["text"])
        enriched.append(
            {
                **r,
                "p_high_ours": p_ours,
                "p_high_dolma3": p_dolma3,
            }
        )

    overall = _correlations(enriched)
    logger.info(
        "OVERALL n=%d  ours~llm=%.4f  dolma3~llm=%.4f  ours~dolma3=%.4f",
        overall.n,
        overall.spearman_ours_vs_llm,
        overall.spearman_dolma3_vs_llm,
        overall.spearman_ours_vs_dolma3,
    )

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        by_source[r["source"]].append(r)
    per_source: dict[str, dict] = {}
    for s, srows in by_source.items():
        if len(srows) < 5:
            continue
        per_source[s] = _correlations(srows).__dict__

    summary = {"overall": overall.__dict__, "per_source": per_source}
    lines = ["source\tid\tllm_score\tp_high_ours\tp_high_dolma3"]
    for r in enriched:
        lines.append(
            f"{r['source']}\t{r['id']}\t{r['llm_score']:.6f}" f"\t{r['p_high_ours']:.6f}\t{r['p_high_dolma3']:.6f}"
        )
    tsv_blob = "\n".join(lines) + "\n"
    json_blob = json.dumps(summary, indent=2, sort_keys=True)

    if report_path == "-":
        print(json_blob)
        print()
        print(tsv_blob[:2000] + ("...[truncated]" if len(tsv_blob) > 2000 else ""))
        return overall

    StoragePath(report_path).write_bytes(tsv_blob.encode("utf-8"))
    summary_path = os.path.splitext(report_path)[0] + ".summary.json"
    StoragePath(summary_path).write_bytes(json_blob.encode("utf-8"))
    logger.info("wrote %d-row TSV -> %s", len(enriched), report_path)
    logger.info("wrote summary -> %s", summary_path)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", required=True, help="GCS path to our trained model.bin")
    parser.add_argument(
        "--dolma3-bin",
        required=True,
        help=(
            "GCS path to dolma3-fasttext-quality model.bin "
            "(reuse the staged copy from dolma3_quality/all_sources_quality.py)"
        ),
    )
    parser.add_argument("--scored-holdout", required=True, help="Scored holdout parquet")
    parser.add_argument("--report", default="-")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    compare(
        model_bin_path=args.model_bin,
        dolma3_bin_path=args.dolma3_bin,
        scored_holdout=args.scored_holdout,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
