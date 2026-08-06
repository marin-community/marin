# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the shipped routing booster from the study table and calibrate its threshold.

Run this to (re)produce the weights :mod:`experiments.build_pdf_source.classify` loads. It trains on
every usable row rather than on a split -- the split exists to *evaluate* the method
(:mod:`~experiments.build_pdf_source.quality.train_route_model`), and once the method is settled,
holding data out of the shipped model only makes it worse -- then reports the held-out numbers from
a domain-disjoint refit alongside, so the shipped artifact is never the only thing that was measured.

The threshold is calibrated, not chosen: :data:`TARGET_VLM_FRACTION` of the corpus goes to the VLM,
and the emitted threshold is the quantile of the model's own output that produces it. That matters
because the score is a probability of a *proxy label*, so its absolute value has no operational
meaning -- only its rank does. Recalibrating on a new corpus is a quantile, not a retrain.

    uv run python -m experiments.build_pdf_source.quality.fit_route_booster

Writes the booster and a JSON sidecar recording the threshold, the feature contract, and the
measured operating point, then prints the SHA-256 that :mod:`classify` pins.
"""

import hashlib
import json
import logging
import pathlib

import numpy as np
import polars as pl
import xgboost as xgb

from experiments.build_pdf_source.quality import route_features
from experiments.build_pdf_source.quality.analyze_route_study import (
    RECALL_FLOOR,
    label,
    point_at_budget,
    route_frontier,
)
from experiments.build_pdf_source.quality.train_route_model import (
    BOOSTER_PARAMS,
    THRESHOLDS,
    fit,
    matrix,
    registered_domain,
    split_by,
    vlm_scores,
)

logger = logging.getLogger(__name__)

# The operating point this booster ships at. Chosen from the cost/quality frontier: below ~35% of
# the corpus nearly every additional document sent to the VLM is one Docling would have botched,
# and past ~50% the marginal document costs more than two VLM runs per document actually rescued.
# The frontier's knee sits at 47%.
TARGET_VLM_FRACTION = 0.50

# The study table :mod:`~experiments.build_pdf_source.quality.build_route_study` writes.
STUDY_PREFIX = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_route_study"
LOCAL_STUDY = pathlib.Path("/tmp/pdf_route_booster/route_study.parquet")

OUTPUT_DIR = pathlib.Path("/tmp/pdf_route_booster")
BOOSTER_NAME = "route_classifier.ubj"
SIDECAR_NAME = "route_classifier.json"


def usable_rows(study: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """The rows a router may learn from.

    Documents whose VLM extraction is itself damaged are dropped rather than labelled: on those
    rows a disagreement measures the VLM's failure, and training on them teaches the router to send
    documents to a route that will fail them too.
    """
    return (
        label(study)
        .filter(pl.col("trustworthy") & pl.col("feature_error").is_null())
        .with_columns(domain=pl.col("url").map_elements(registered_domain, return_dtype=pl.String))
        .drop_nulls(subset=features)
    )


def train_final(frame: pl.DataFrame, features: list[str], rounds: int) -> xgb.Booster:
    """Train on every usable row for a fixed number of rounds."""
    dataset = xgb.DMatrix(matrix(frame, features), label=frame["docling_ok"].to_numpy(), feature_names=features)
    return xgb.train(BOOSTER_PARAMS, dataset, num_boost_round=rounds)


def read_study() -> pl.DataFrame:
    """Read the routing study table, from a local copy if one exists and from storage otherwise."""
    if LOCAL_STUDY.exists():
        logger.info("reading study table from %s", LOCAL_STUDY)
        return pl.read_parquet(LOCAL_STUDY)

    import fsspec  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()
    filesystem = fsspec.filesystem("s3")
    shards = sorted(filesystem.glob(f"{STUDY_PREFIX}/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no study shards under {STUDY_PREFIX}; run build_route_study first")
    logger.info("reading %d study shards from %s", len(shards), STUDY_PREFIX)

    def read(path: str) -> pa.Table:
        with filesystem.open(f"s3://{path}", "rb") as stream:
            return pq.read_table(stream)

    # Permissive promotion: a shard where no document failed infers `feature_error` as null-typed
    # rather than string, and concatenating those against a shard that did have a failure errors.
    return pl.from_arrow(pa.concat_tables([read(path) for path in shards], promote_options="permissive"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study = read_study()
    features = [name for name in route_features.FEATURE_NAMES if name in study.columns]
    frame = usable_rows(study, features)
    logger.info("training on %d documents, %d features", frame.height, len(features))

    # Evaluate first, on documents from domains the model never saw, and take the round count from
    # that fit so the shipped model is not trained for longer than the evaluated one.
    split = split_by(frame, "domain")
    evaluated = fit(split, features)
    rounds = evaluated.best_iteration + 1
    held_out = route_frontier(
        vlm_scores(evaluated, split.test, features), split.test["docling_ok"].to_numpy(), THRESHOLDS
    )
    operating = point_at_budget(held_out, TARGET_VLM_FRACTION)
    logger.info(
        "held out (%d docs from unseen domains): at %.1f%% VLM, quality loss %.4f, catches %.1f%% of bad documents",
        split.test.height,
        100 * operating.vlm_fraction,
        operating.quality_loss,
        100 * operating.recall_of_bad,
    )

    booster = train_final(frame, features, rounds)
    # The score is a probability of a proxy label, so only its rank is meaningful: the threshold is
    # the quantile of its own output that sends TARGET_VLM_FRACTION of the corpus to the VLM.
    confidence = booster.predict(xgb.DMatrix(matrix(frame, features), feature_names=features))
    threshold = float(np.quantile(confidence, TARGET_VLM_FRACTION))
    logger.info("routing threshold: docling confidence < %.6f goes to the VLM", threshold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    booster_path = OUTPUT_DIR / BOOSTER_NAME
    booster.save_model(booster_path)
    digest = hashlib.sha256(booster_path.read_bytes()).hexdigest()

    (OUTPUT_DIR / SIDECAR_NAME).write_text(
        json.dumps(
            {
                "features": features,
                "boost_rounds": rounds,
                "docling_confidence_threshold": threshold,
                "target_vlm_fraction": TARGET_VLM_FRACTION,
                "training_documents": frame.height,
                "label": {"metric": "bigram_recall_mean", "floor": RECALL_FLOOR},
                "held_out": {
                    "documents": split.test.height,
                    "vlm_fraction": operating.vlm_fraction,
                    "quality_loss": operating.quality_loss,
                    "recall_of_bad": operating.recall_of_bad,
                },
                "sha256": digest,
            },
            indent=2,
        )
    )
    print(f"booster: {booster_path}\nsha256:  {digest}\nthreshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
