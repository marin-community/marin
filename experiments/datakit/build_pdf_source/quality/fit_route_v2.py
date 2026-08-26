# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the shipped router v2 booster and calibrate its threshold onto the corpus.

The split in :mod:`~experiments.datakit.build_pdf_source.quality.analyze_route_v2` exists to
*evaluate* the method. Once the method is settled, holding data out of the shipped model only makes
it worse, so this trains on every labelled row -- for exactly the round count the domain-disjoint
evaluation stopped at, so the shipped model is never trained for longer than the measured one -- and
reports the held-out numbers alongside, so the artifact is not the only thing that was measured.

**The threshold is a quantile, not a fit.** The score is a probability of a judged preference, so its
absolute value means nothing operationally and only its rank does. The emitted threshold is the
quantile of the model's own output over the *whole* corpus -- all 100,000 documents, not just the
20,000 that carry labels -- that sends :data:`TARGET_DOCUMENT_BUDGET` of documents to the VLM.
Recalibrating for a different budget, or for a new crawl, is a quantile and not a retrain.

The sidecar records the threshold, the feature contract with each group's price, the measured
operating point in CPU core-hours, and the two arithmetic gates that run before the score is
consulted at all. A consumer that cannot reproduce the gates cannot use the threshold.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-route-v2-fit --extra pdf \\
        --cpu 16 --memory 48GB --disk 16GB --enable-extra-resources \\
        -- python -m experiments.datakit.build_pdf_source.quality.fit_route_v2
"""

import hashlib
import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import fsspec
import numpy as np
import polars as pl
import xgboost as xgb
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_LEGIBILITY_FLOOR_DPI,
    DEFAULT_MAX_VISUAL_TOKENS,
)
from experiments.datakit.build_pdf_source.quality import route_v2_features as contract
from experiments.datakit.build_pdf_source.quality.analyze_route_v2 import (
    ARMS,
    at_budget,
    clumping,
    escalation_scores,
    frontier,
    group_gains,
    joined,
    knee,
    trainable,
)
from experiments.datakit.build_pdf_source.quality.judge_preference_set import ESCALATE_COLUMN
from experiments.datakit.build_pdf_source.quality.train_route_model import BOOSTER_PARAMS, fit, matrix, split_by

logger = logging.getLogger(__name__)

MODEL_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/model/pdf_route_v2"
BOOSTER_NAME = "route_v2_classifier.ubj"
SIDECAR_NAME = "route_v2_classifier.json"

# The operating point the artifact ships calibrated to. Not a claim that this is the right budget --
# the frontier in `analyze_route_v2` is what the choice should be made from, and recalibrating is a
# quantile. This is the knee the evaluation reported, so the artifact ships somewhere defensible
# rather than somewhere round.
TARGET_DOCUMENT_BUDGET = 0.50

# The arm the artifact ships. Read from the evaluation's own arm table so the shipped feature set is
# one that was measured, rather than a list restated here and free to drift from it.
SHIPPED_ARM = ARMS[-1]


def train_final(frame: pl.DataFrame, features: list[str], rounds: int) -> xgb.Booster:
    """Train on every labelled row for a fixed number of rounds."""
    dataset = xgb.DMatrix(matrix(frame, features), label=frame[ESCALATE_COLUMN].to_numpy(), feature_names=features)
    return xgb.train(BOOSTER_PARAMS, dataset, num_boost_round=rounds)


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")

    frame = joined(fs)
    features = SHIPPED_ARM.features
    rows = trainable(frame, features)
    logger.info(
        "training on %d documents, %d domains, %d features", rows.height, rows["domain"].n_unique(), len(features)
    )

    # Evaluate first, on documents from domains the model never saw, and take the round count from
    # that fit so the shipped model is not trained for longer than the evaluated one.
    split = split_by(rows, "domain", ESCALATE_COLUMN)
    evaluated = fit(split, features)
    rounds = evaluated.best_iteration + 1
    held_out = frontier(
        escalation_scores(evaluated, split.test, features),
        split.test[ESCALATE_COLUMN].to_numpy().astype(bool),
        split.test["num_pages"].cast(pl.Float64).to_numpy(),
        SHIPPED_ARM.core_hours,
        SHIPPED_ARM.needs_inspector,
    )
    operating = at_budget(held_out, TARGET_DOCUMENT_BUDGET)
    logger.info(
        "held out (%d documents from %d unseen domains): %.1f%% of documents / %.1f%% of pages, "
        "quality loss %.4f, catches %.1f%%, %.1f core-h per million pages",
        split.test.height,
        split.test["domain"].n_unique(),
        100 * operating.document_budget,
        100 * operating.page_budget,
        operating.quality_loss_pages,
        100 * operating.recall_of_escalations,
        operating.cpu_core_hours,
    )

    booster = train_final(rows, features, rounds)

    # The threshold is calibrated over the whole corpus rather than over the labelled subset: the
    # labelled draw is capped at 15 documents per domain, so its score distribution is flatter than
    # the corpus's and a quantile taken on it would spend a different budget in production.
    corpus = frame.drop_nulls(subset=features)
    scores = escalation_scores(booster, corpus, features)
    threshold = float(np.quantile(scores, 1.0 - TARGET_DOCUMENT_BUDGET))
    logger.info(
        "routing threshold: escalate when the score is at or above %.6f (%d corpus documents scored)",
        threshold,
        corpus.height,
    )

    with tempfile.TemporaryDirectory() as staging:
        local = Path(staging) / BOOSTER_NAME
        booster.save_model(local)
        payload = local.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    sidecar = {
        "arm": SHIPPED_ARM.name,
        "features": features,
        "feature_groups": {
            group.name: {
                "source": group.source,
                "core_hours_per_million_pages": group.core_hours,
                "columns": list(group.columns),
            }
            for group in contract.GROUPS
            if group.name in SHIPPED_ARM.groups
        },
        "boost_rounds": rounds,
        "escalation_threshold": threshold,
        "target_document_budget": TARGET_DOCUMENT_BUDGET,
        "training_documents": rows.height,
        "training_domains": rows["domain"].n_unique(),
        "corpus_documents_scored": corpus.height,
        "label": {
            "target": ESCALATE_COLUMN,
            "definition": (
                "a blind judge, shown the rendered pages, preferred the VLM's transcription " "to pdf-inspector's"
            ),
            "source": "experiments.datakit.build_pdf_source.quality.judge_preference_set",
        },
        # The score is consulted only after these. A consumer that cannot reproduce them cannot use
        # the threshold, because the model was neither trained nor calibrated on the rows they take.
        "gates": {
            "inspector_failed": "escalate: pdf-inspector returned no text, so there is no cheap route to keep",
            "below_legibility_floor": (
                f"keep: every page renders under {DEFAULT_LEGIBILITY_FLOOR_DPI} DPI at the "
                f"{DEFAULT_MAX_VISUAL_TOKENS}-token budget, so the VLM cannot read it either"
            ),
        },
        "held_out": {
            "documents": split.test.height,
            "domains": split.test["domain"].n_unique(),
            **asdict(operating),
        },
        "held_out_knee": asdict(knee(held_out)),
        "corpus_score_clumping": clumping(scores),
        "gain": group_gains(booster, features),
        "cost_model": {
            "inspector_core_hours_per_million_pages": contract.INSPECTOR_CORE_HOURS,
            "router_core_hours_per_million_pages": SHIPPED_ARM.core_hours,
            "vlm_feed_core_hours_per_million_pages": contract.VLM_FEED_CORE_HOURS,
            "vlm_gpu_hours_per_million_pages": contract.VLM_GPU_HOURS,
        },
        "sha256": digest,
    }

    with fs.open(f"{MODEL_PREFIX}/{BOOSTER_NAME}", "wb") as stream:
        stream.write(payload)
    with fs.open(f"{MODEL_PREFIX}/{SIDECAR_NAME}", "w") as stream:
        json.dump(sidecar, stream, indent=2, default=float)
    print(json.dumps(sidecar, indent=2, default=float))
    logger.info("wrote %s/%s (sha256 %s)", MODEL_PREFIX, BOOSTER_NAME, digest)


if __name__ == "__main__":
    main()
