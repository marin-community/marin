# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a PDF-OCR educational-quality scorer and its fasttext baseline.

Two models over the same oracle labels (``build_labels.py``), reported side by
side with the metric suite from the datakit quality work (#7040): held-out AUC,
Spearman against the oracle, threshold-0.5 accuracy/precision/recall/F1, and
calibrated-bucket versus oracle-level agreement.

* **fasttext** -- a 5-way supervised classifier over the oracle levels. The
  continuous score is the probability-weighted mean level, which ranks strictly
  better than any single class probability and so gives the fairest baseline
  Spearman.
* **pooled fast-transformer** -- the deployed datakit architecture and
  hyperparameters (``fast_transformer.train.DEPLOY_CONFIG``), retrained from
  scratch on these labels.

Splits are **by document**, not by row. A document contributes up to three
segments; a row-level split would put two windows of the same PDF on either side
of the holdout boundary and score the leak as generalization. Train / val /
holdout are disjoint document sets, and fasttext is selected on the same val
documents the fast-transformer early-stops on.
"""

import argparse
import dataclasses
import json
import logging
import os
import tempfile

import fasttext
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import calibration_knots
from experiments.datakit.cluster.quality.fast_transformer.data import build_remap, encode_texts, pack
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.metrics import auc, spearman_rho
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
    count_params,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import MODEL_STEM
from experiments.datakit.cluster.quality.fast_transformer.train import (
    DEPLOY_CONFIG,
    MAX_TOKENS,
    TOKENIZER,
    EvalMetrics,
    TrainHParams,
    _binary_metrics,
    _save_scorer,
    train_regressor,
)

logger = logging.getLogger(__name__)

HOLDOUT_DOC_FRAC = 1 / 7  # matches the #7040 eval_frac
VAL_DOC_FRAC = 0.1  # internal selection split, carved from the training documents
MAX_SCORE = 4  # oracle levels 1..5 -> normalized 0..1
EDUCATIONAL_LEVEL = 4  # oracle level (raw score 3) that FineWeb-Edu treats as the keep threshold
# Calibration filename `fast_transformer/score.py` looks for inside a model dir. Duplicated
# rather than imported: importing that module would drag zephyr/fray into this training job.
MODEL_CALIB = "calib_bme.json"

# fasttext grid, selected on val Spearman. The useful capacity moves with the size of
# the training split, so this grid has to be re-centred whenever the corpus changes: on
# the 18k-row 10k-document sample the standard CC-quality recipe (dim 100, bucket 2M,
# lr 0.3+, 25 epochs) simply memorised the split and val Spearman fell monotonically in
# lr and epochs, so the sweep had to reach down to a few epochs and unigrams. The 100k
# sample gives ~170k training rows and reopens the high end, so the grid spans both
# regimes and keeps the low end only densely enough to bracket the optimum from below.
FASTTEXT_DEFAULTS = {
    "dim": 100,
    "minCount": 2,
    "bucket": 2_000_000,
    "loss": "softmax",
    "wordNgrams": 2,
    "thread": 16,
    "verbose": 0,
}
FASTTEXT_GRID = (
    # Low end, kept so a corpus that still prefers few epochs is not selected at a boundary.
    {"lr": 0.1, "epoch": 1},
    {"lr": 0.1, "epoch": 2},
    {"lr": 0.1, "epoch": 3},
    {"lr": 0.1, "epoch": 5},
    # Standard CC-quality territory, which only becomes reachable at this corpus size.
    {"lr": 0.1, "epoch": 10},
    {"lr": 0.2, "epoch": 10},
    {"lr": 0.3, "epoch": 10},
    {"lr": 0.2, "epoch": 25},
    {"lr": 0.3, "epoch": 25},
    {"lr": 0.5, "epoch": 25},
    {"lr": 0.3, "epoch": 50},
    # Capacity and representation variants at a mid-range schedule.
    {"lr": 0.2, "epoch": 25, "dim": 50},
    {"lr": 0.2, "epoch": 25, "minCount": 5},
    {"lr": 0.2, "epoch": 25, "bucket": 200_000},
    {"lr": 0.2, "epoch": 25, "wordNgrams": 1},
    {"lr": 0.2, "epoch": 25, "loss": "ova"},
)


@dataclasses.dataclass(frozen=True)
class Split:
    """One document-disjoint slice of the label rows."""

    name: str
    texts: list[str]
    levels: np.ndarray  # oracle levels 1..5
    doc_ids: list[str]
    segments: list[str]  # begin / middle / end

    @property
    def targets(self) -> np.ndarray:
        """Regression target in [0, 1]."""
        return ((self.levels - 1) / MAX_SCORE).astype(np.float32)

    @property
    def n(self) -> int:
        return len(self.texts)


def split_by_document(rows: dict, seed: int, holdout_frac: float = HOLDOUT_DOC_FRAC) -> tuple[Split, Split, Split]:
    """Partition rows into train / val / holdout with no document on two sides."""
    doc_ids = sorted(set(rows["id"]))
    order = np.random.default_rng(seed).permutation(len(doc_ids))
    n_holdout = max(1, int(len(doc_ids) * holdout_frac))
    n_val = max(1, int((len(doc_ids) - n_holdout) * VAL_DOC_FRAC))
    assignment = {}
    for rank, index in enumerate(order):
        if rank < n_holdout:
            assignment[doc_ids[index]] = "holdout"
        elif rank < n_holdout + n_val:
            assignment[doc_ids[index]] = "val"
        else:
            assignment[doc_ids[index]] = "train"

    buckets: dict[str, list[int]] = {"train": [], "val": [], "holdout": []}
    for index, doc_id in enumerate(rows["id"]):
        buckets[assignment[doc_id]].append(index)

    def build(name: str) -> Split:
        idx = buckets[name]
        return Split(
            name=name,
            texts=[rows["text"][i] for i in idx],
            levels=np.array([rows["quality"][i] for i in idx], dtype=np.int32),
            doc_ids=[rows["id"][i] for i in idx],
            segments=[rows["segment"][i] for i in idx],
        )

    splits = (build("train"), build("val"), build("holdout"))
    for split in splits:
        logger.info("%s: %d rows / %d docs", split.name, split.n, len(set(split.doc_ids)))
    return splits


def evaluate(scores: np.ndarray, levels: np.ndarray) -> EvalMetrics:
    """The #7040 metric suite: AUC + Spearman + threshold-0.5 classification."""
    targets = (levels - 1) / MAX_SCORE
    y_true = [1 if t >= 0.5 else 0 for t in targets.tolist()]
    y_pred = [1 if s >= 0.5 else 0 for s in scores.tolist()]
    accuracy, precision, recall, f1 = _binary_metrics(y_true, y_pred)
    return EvalMetrics(
        n=len(y_true),
        auc=auc(y_true, scores.tolist()),
        spearman_rho=spearman_rho(scores.tolist(), targets.tolist()),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def educational_auc(scores: np.ndarray, levels: np.ndarray) -> float:
    """AUC for the practically relevant cut: keep documents the oracle scored >= 3."""
    return auc([1 if level >= EDUCATIONAL_LEVEL else 0 for level in levels.tolist()], scores.tolist())


def bucket_agreement(scores: np.ndarray, levels: np.ndarray, knots: dict) -> dict[str, float]:
    """Calibrated-bucket versus oracle-level agreement, as reported in #7040."""
    calibrated = np.interp(scores, knots["xk"], knots["yk"])
    predicted = np.digitize(calibrated, BUCKET_EDGES)
    oracle = np.clip(levels - 1, 0, len(BUCKET_EDGES)).astype(int)
    return {
        "exact": float(np.mean(predicted == oracle)),
        "within_1": float(np.mean(np.abs(predicted - oracle) <= 1)),
    }


def _fasttext_line(text: str, level: int | None = None) -> str:
    """One fasttext record: labels are the oracle levels, text collapsed to a line."""
    body = " ".join(text.split()).lower()
    return f"__label__{level} {body}" if level is not None else body


def _fasttext_scores(model, texts: list[str]) -> np.ndarray:
    """Probability-weighted mean oracle level, normalized to [0, 1]."""
    labels, probabilities = model.predict([_fasttext_line(t) for t in texts], k=5)
    out = np.empty(len(texts), dtype=np.float32)
    for i, (row_labels, row_probs) in enumerate(zip(labels, probabilities, strict=True)):
        expected = sum(int(label.removeprefix("__label__")) * p for label, p in zip(row_labels, row_probs, strict=True))
        out[i] = (expected - 1) / MAX_SCORE
    return out


def train_fasttext(train: Split, val: Split, work_dir: str, seed: int) -> tuple[object, dict]:
    """Fit the fasttext grid, returning the model with the best val Spearman.

    The training file is written in shuffled order because fasttext streams it
    sequentially, once per epoch, on a learning rate that decays to zero -- it
    never shuffles internally. Label rows arrive grouped by segment, so writing
    them in their natural order trains the model on every ``begin`` window, then
    every ``middle``, then every ``end`` at a vanishing learning rate. That alone
    cost the baseline 0.6 Spearman and drove its ``end`` correlation negative.
    """
    train_path = os.path.join(work_dir, "train.txt")
    order = np.random.default_rng(seed).permutation(train.n)
    levels = train.levels.tolist()
    with open(train_path, "w") as stream:
        for i in order.tolist():
            stream.write(_fasttext_line(train.texts[i], levels[i]) + "\n")

    best_model, best_params, best_rho = None, None, -2.0
    for overrides in FASTTEXT_GRID:
        params = {**FASTTEXT_DEFAULTS, **overrides}
        model = fasttext.train_supervised(input=train_path, **params)
        rho = spearman_rho(_fasttext_scores(model, val.texts).tolist(), val.targets.tolist())
        logger.info("fasttext %s -> val spearman %.4f", overrides, rho)
        if np.isfinite(rho) and rho > best_rho:
            best_model, best_params, best_rho = model, params, rho
    logger.info("fasttext best: %s (val spearman %.4f)", best_params, best_rho)
    return best_model, {k: v for k, v in best_params.items() if k != "verbose"}


@dataclasses.dataclass
class TrainedScorer:
    """A fitted fast-transformer plus everything needed to score new segments."""

    model: FastTransformer
    remap: dict[int, int]
    config: FastTransformerConfig
    max_tokens: int
    train_scores: np.ndarray
    holdout_scores: np.ndarray
    info: dict

    def score(self, texts: list[str]) -> np.ndarray:
        """Sigmoid quality score in [0, 1] for arbitrary segment text."""
        raw = encode_texts(TOKENIZER, texts, self.max_tokens)
        return predict(self.model, pack(raw, self.remap, np.zeros(len(texts), dtype=np.float32), self.max_tokens).ids)


def train_fast_transformer(train: Split, val: Split, holdout: Split, max_tokens: int, hp: TrainHParams) -> TrainedScorer:
    """Train the pooled fast-transformer on the training documents."""
    train_raw = encode_texts(TOKENIZER, train.texts, max_tokens)
    val_raw = encode_texts(TOKENIZER, val.texts, max_tokens)
    holdout_raw = encode_texts(TOKENIZER, holdout.texts, max_tokens)
    lengths = np.array([len(row) for row in train_raw])
    logger.info(
        "segment tokens under %s: mean %.0f p90 %.0f truncated %.1f%%",
        TOKENIZER,
        lengths.mean(),
        np.percentile(lengths, 90),
        100 * float((lengths >= max_tokens).mean()),
    )

    remap = build_remap(train_raw, min_count=2)
    vocab = len(remap) + 2
    packed = {
        name: pack(raw, remap, split.targets, max_tokens)
        for name, raw, split in (
            ("train", train_raw, train),
            ("val", val_raw, val),
            ("holdout", holdout_raw, holdout),
        )
    }
    config = FastTransformerConfig(
        vocab_size=vocab, max_tokens=max_tokens, dropout=0.1, final_pool="mean", **DEPLOY_CONFIG
    )
    model = FastTransformer(config, key=jax.random.PRNGKey(hp.seed))
    logger.info(
        "fast-transformer: params=%.2fM flops/token=%.0fK vocab=%d",
        count_params(model) / 1e6,
        config.flops_per_token() / 1e3,
        vocab,
    )
    best_model, best_epoch, seconds = train_regressor(
        model, packed["train"].ids, packed["train"].scores, packed["val"].ids, packed["val"].scores, hp
    )
    info = {
        "params": count_params(best_model),
        "flops_per_token": config.flops_per_token(),
        "best_epoch": best_epoch,
        "train_seconds": seconds,
        "vocab_size": vocab,
        "max_tokens": max_tokens,
        "truncated_segment_frac": float((lengths >= max_tokens).mean()),
        "config": dataclasses.asdict(config),
    }
    return TrainedScorer(
        model=best_model,
        remap=remap,
        config=config,
        max_tokens=max_tokens,
        train_scores=predict(best_model, packed["train"].ids),
        holdout_scores=predict(best_model, packed["holdout"].ids),
        info=info,
    )


def per_segment_spearman(scores: np.ndarray, holdout: Split) -> dict[str, float]:
    """Spearman within each window position, so a weak tail is visible separately.

    Begin windows carry a PDF's title and abstract; middle and end windows are
    references, tables, and OCR noise, so a scorer can rank the whole set well
    while being near-useless on the parts that decide a long document's fate.
    """
    targets = (holdout.levels - 1) / MAX_SCORE
    out = {}
    for segment in ("begin", "middle", "end"):
        mask = [i for i, s in enumerate(holdout.segments) if s == segment]
        if len(mask) < 2:
            continue
        out[segment] = spearman_rho([float(scores[i]) for i in mask], [float(targets[i]) for i in mask])
        out[f"{segment}_n"] = len(mask)
    return out


def write_holdout_scores(
    scorer: TrainedScorer, rows: dict, holdout_doc_ids: set[str], knots: dict, output_path: str
) -> None:
    """Score every segment of every holdout document and write them for the browser.

    Covers all three windows, including the overlapping ones that short documents
    excluded from training -- the browser shows a chip per window, and the oracle
    columns are populated for all three there too.

    ``ft_score`` is on the oracle's own 0..4 scale: the calibrated score times
    ``MAX_SCORE``, so it is directly comparable to the ``edu_score_v2_*`` columns
    and ``round()`` reproduces the bucket ``np.digitize`` would assign.
    """
    index = [i for i, doc_id in enumerate(rows["id"]) if doc_id in holdout_doc_ids]
    texts = [rows["text"][i] for i in index]
    calibrated = np.interp(scorer.score(texts), knots["xk"], knots["yk"])
    table = pa.table(
        {
            "id": [rows["id"][i] for i in index],
            "segment": [rows["segment"][i] for i in index],
            "ft_score": pa.array((calibrated * MAX_SCORE).astype(np.float32), pa.float32()),
            "oracle_score": pa.array([rows["quality"][i] - 1 for i in index], pa.int8()),
        }
    )
    with StoragePath(output_path).open("wb") as stream:
        pq.write_table(table, stream)
    logger.info("wrote %d holdout segment scores over %d docs -> %s", table.num_rows, len(holdout_doc_ids), output_path)


def report(name: str, train_scores: np.ndarray, holdout_scores: np.ndarray, train: Split, holdout: Split) -> dict:
    """Holdout metrics plus the calibration fit on train predictions only."""
    metrics = evaluate(holdout_scores, holdout.levels)
    knots = calibration_knots(train_scores, train.levels.astype(float))
    result = {
        "n_holdout": metrics.n,
        "auc": metrics.auc,
        "spearman_rho": metrics.spearman_rho,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "educational_auc": educational_auc(holdout_scores, holdout.levels),
        "bucket_agreement": bucket_agreement(holdout_scores, holdout.levels, knots),
        "per_segment_spearman": per_segment_spearman(holdout_scores, holdout),
        "calibration": knots,
    }
    logger.info(
        "%s HOLDOUT spearman=%.4f auc=%.4f edu_auc=%.4f f1=%.4f bucket exact=%.3f within-1=%.3f",
        name,
        result["spearman_rho"],
        result["auc"],
        result["educational_auc"],
        result["f1"],
        result["bucket_agreement"]["exact"],
        result["bucket_agreement"]["within_1"],
    )
    logger.info("%s per-segment spearman: %s", name, {k: round(v, 4) for k, v in result["per_segment_spearman"].items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="label parquet from build_labels.py")
    parser.add_argument("--out", required=True, help="directory for the metrics json")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=TrainHParams.epochs, help="fast-transformer epoch cap")
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=HOLDOUT_DOC_FRAC,
        help="fraction of documents held out; raise it to leave more documents unseen for inspection",
    )
    parser.add_argument("--save-model", help="directory for the .eqx + remap + meta + calibration artifacts")
    parser.add_argument("--scores-out", help="parquet of fast-transformer scores for every holdout segment")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()  # no-op inside a CoreWeave pod; wires CW_KEY_* on a dev box
    logger.info("jax devices: %s", jax.devices())

    columns = ("id", "segment", "text", "quality", "use_for_training")
    with StoragePath(args.labels).open("rb") as stream:
        table = pq.read_table(stream, columns=list(columns))
    rows = {name: table.column(name).to_pylist() for name in columns}
    # Split and fit on trainable rows only; the overlapping windows short documents
    # excluded are still scored later for the browser.
    trainable = {
        name: [v for v, keep in zip(values, rows["use_for_training"], strict=True) if keep]
        for name, values in rows.items()
    }
    logger.info(
        "loaded %d label rows (%d trainable) over %d docs",
        table.num_rows,
        len(trainable["id"]),
        len(set(rows["id"])),
    )

    train, val, holdout = split_by_document(trainable, args.seed, args.holdout_frac)
    hp = TrainHParams(seed=args.seed, epochs=args.epochs)

    with tempfile.TemporaryDirectory() as work_dir:
        baseline, baseline_params = train_fasttext(train, val, work_dir, args.seed)
    baseline_result = report(
        "fasttext", _fasttext_scores(baseline, train.texts), _fasttext_scores(baseline, holdout.texts), train, holdout
    )
    baseline_result["hyperparameters"] = baseline_params

    scorer = train_fast_transformer(train, val, holdout, args.max_tokens, hp)
    ft_result = report("fast-transformer", scorer.train_scores, scorer.holdout_scores, train, holdout)
    ft_result.update(scorer.info)

    if args.save_model:
        _save_scorer(scorer.model, scorer.remap, TOKENIZER, scorer.config, args.save_model, MODEL_STEM)
        with StoragePath(f"{args.save_model.rstrip('/')}/{MODEL_CALIB}").open("w") as stream:
            json.dump(ft_result["calibration"], stream)
        logger.info("saved calibration -> %s/%s", args.save_model.rstrip("/"), MODEL_CALIB)

    if args.scores_out:
        write_holdout_scores(scorer, rows, set(holdout.doc_ids), ft_result["calibration"], args.scores_out)

    results = {
        "labels": args.labels,
        "seed": args.seed,
        "splits": {
            name: {"rows": split.n, "docs": len(set(split.doc_ids))}
            for name, split in (("train", train), ("val", val), ("holdout", holdout))
        },
        "level_counts": {
            int(level): int(count) for level, count in zip(*np.unique(holdout.levels, return_counts=True), strict=True)
        },
        "fasttext": baseline_result,
        "fast_transformer": ft_result,
        "hparams": dataclasses.asdict(hp),
    }
    destination = f"{args.out.rstrip('/')}/metrics.json"
    with StoragePath(destination).open("w") as stream:
        json.dump(results, stream, indent=2)
    logger.info("wrote %s", destination)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
