# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the calibration for the corpus scores ``score_corpus`` wrote.

``calibrate.py`` fits cutpoints through ``score_bme`` -- whole-document
begin/middle/end scoring over raw text. That is not the function that produced
the corpus scores. ``score_corpus`` scores the *begin window* only (the first
``max_tokens`` Nemotron ids of chunk 0) and feeds the harrier document embedding
as a super token, which ``score_bme`` does not supply at all. Cutpoints are only
valid for the scoring function they were fitted through, so this fits them
through the corpus path: the folded artifact, the begin window, the embedding.

The fit itself is ``calibrate.py``'s, unchanged -- ``calibration_knots``,
``per_type_knots`` and ``apply_calibration`` are model-agnostic and are imported
rather than restated. Only the scoring function and the row set differ.

Row set: the seed-0 id-set holdout of the 88k labels, the split every arm of
this campaign held out and the scale-up work left untouched. The model was
trained on the rest, so fitting cutpoints on the rest would fit them to
memorized scores. The training-set fit is reported alongside so the size of that
bias is visible, but it is not what gets written.

The run also applies the fitted edges to a sample of the real written score
shards, so the bucket shares are measured on the corpus rather than inferred
from the labeled sample's own distribution.
"""

import argparse
import json
import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer import domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    DEFAULT_MIN_PER_TYPE,
    apply_calibration,
    calibration_knots,
    parity_ratio,
    per_type_knots,
)
from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts, pack
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import DEFAULT_LABELS, holdout_id_set
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    DEFAULT_FOLDED_DIR,
    DEFAULT_MANIFEST,
    load_folded_scorer,
    read_manifest,
)

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_MLP = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/domain_mlp/domain_mlp.npz"
DEFAULT_OUT = f"{DEFAULT_FOLDED_DIR}/calib_bme.json"
SCORE_BATCH = 4096
# Score shards sampled to measure the corpus bucket distribution. The shards are
# hash-partitioned across every source, so a few hundred of them span the corpus
# rather than one region of it.
DEFAULT_CORPUS_SHARDS = 200
JOINED_COLUMNS = ["id", "text", "embedding", "glm52_source", "glm52_content_type", "glm52_quality"]


def score_begin_window(scorer, texts: list[str], embeddings: np.ndarray, batch_size: int) -> np.ndarray:
    """Corpus-path scores: the begin window plus the document-embedding super token.

    ``score_corpus`` takes the stored Nemotron ids of chunk 0, truncates them to
    ``max_tokens`` and adds ``NUM_RESERVED``. The corpus tokenizer and the donor
    tokenizer share one 131,072-entry vocabulary, so tokenizing the text here with
    the scorer's own tokenizer and truncating to the same ``max_tokens`` produces
    the same ids; ``pack`` applies the same full-vocab remap. What must not differ
    is the *window* and the embedding, and neither does.
    """
    raw = encode_texts(scorer.tokenizer_name, texts, scorer.max_tokens)
    packed = pack(raw, scorer.remap, np.zeros(len(texts), dtype=np.float32), scorer.max_tokens)
    return np.asarray(predict(scorer.model, packed.ids, batch_size=batch_size, doc_embed=embeddings))


def report_fit(name: str, raw: np.ndarray, levels: np.ndarray, types: np.ndarray, knots: dict) -> np.ndarray:
    """Log a fit's cutpoints, agreement with the oracle, and bucket shares."""
    calibrated = apply_calibration(raw, types if "types" in knots else None, knots)
    buckets = np.digitize(calibrated, BUCKET_EDGES)
    oracle = np.clip((levels - 1).astype(int), 0, 4)
    cuts = knots["default"]["xk"][1:-1] if "types" in knots else knots["xk"][1:-1]
    logger.info("CALIB[%s] n=%d global cutpoints %s", name, len(raw), [round(c, 4) for c in cuts])
    logger.info(
        "CALIB[%s] calibrated-bucket vs oracle-level: exact %.3f  within-1 %.3f",
        name,
        float(np.mean(buckets == oracle)),
        float(np.mean(np.abs(buckets - oracle) <= 1)),
    )
    shares = np.bincount(buckets, minlength=5) / len(buckets)
    logger.info("CALIB[%s] bucket shares %s", name, [round(float(s), 4) for s in shares])
    logger.info("CALIB[%s] parity ratio %.2fx", name, parity_ratio(buckets, types))
    for type_name in sorted(set(types.tolist())):
        mask = types == type_name
        logger.info(
            "CALIB[%s]   %-14s n=%-5d top-share=%.1f%% bottom-share=%.1f%%",
            name,
            type_name,
            int(mask.sum()),
            100 * float((buckets[mask] == 4).mean()),
            100 * float((buckets[mask] == 0).mean()),
        )
    return buckets


def sample_corpus_scores(manifest: str, num_shards: int, seed: int) -> np.ndarray:
    """Scores read from a random sample of the written score shards."""
    frame = read_manifest(manifest)
    paths = frame.get_column("output_path").to_list()
    order = np.random.default_rng(seed).permutation(len(paths))
    got: list[np.ndarray] = []
    read = 0
    missing = 0
    for index in order:
        if read >= num_shards:
            break
        path = paths[index]
        try:
            with StoragePath(path).open("rb") as fh:
                column = pq.ParquetFile(fh).read(columns=["score"]).column("score")
        except Exception:
            missing += 1
            continue
        got.append(column.to_numpy(zero_copy_only=False))
        read += 1
    logger.info("corpus sample: %d shards read, %d absent, %d scores", read, missing, sum(len(g) for g in got))
    if not got:
        raise ValueError(f"no readable score shards under the manifest {manifest}")
    return np.concatenate(got)


def report_corpus(scores: np.ndarray, cuts: list[float]) -> dict:
    """Bucket distribution of the real corpus scores under the fitted edges."""
    buckets = np.digitize(scores, cuts)
    shares = np.bincount(buckets, minlength=5) / len(buckets)
    percentiles = np.percentile(scores, [1, 5, 25, 50, 75, 95, 99])
    logger.info(
        "CORPUS raw scores n=%d mean=%.4f std=%.4f p1/p5/p25/p50/p75/p95/p99=%s",
        len(scores),
        float(scores.mean()),
        float(scores.std()),
        [round(float(p), 4) for p in percentiles],
    )
    logger.info(
        "CORPUS bucket shares under cutpoints %s: %s", [round(c, 4) for c in cuts], [round(float(s), 4) for s in shares]
    )
    return {"n": len(scores), "shares": [float(s) for s in shares], "cutpoints": [float(c) for c in cuts]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_FOLDED_DIR)
    p.add_argument("--joined", default=DEFAULT_JOINED, help="glm52-labels x harrier-embeddings join")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="88k label parquet defining the seed-0 holdout id set")
    p.add_argument("--domain-mlp", default=DEFAULT_DOMAIN_MLP, help="embedding content-type classifier npz")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--min-per-type", type=int, default=DEFAULT_MIN_PER_TYPE)
    p.add_argument("--batch-size", type=int, default=SCORE_BATCH)
    p.add_argument("--corpus-shards", type=int, default=DEFAULT_CORPUS_SHARDS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    scorer = load_folded_scorer(args.model_dir)
    logger.info(
        "scorer %s: tokenizer=%s max_tokens=%d doc_embed_dim=%d",
        args.model_dir,
        scorer.tokenizer_name,
        scorer.max_tokens,
        scorer.model.config.doc_embed_dim,
    )

    joined = load_joined(args.joined, columns=JOINED_COLUMNS)
    holdout = holdout_id_set(args.labels)
    is_holdout = np.array([doc_id in holdout for doc_id in joined["id"]])
    levels = np.array(joined["glm52_quality"], dtype=float)
    oracle_types = np.array(joined["glm52_content_type"])
    logger.info(
        "rows: %d joined, %d in the seed-0 holdout (of %d holdout ids); holdout level counts %s",
        len(is_holdout),
        int(is_holdout.sum()),
        len(holdout),
        {level: int((levels[is_holdout] == level).sum()) for level in (1, 2, 3, 4, 5)},
    )

    texts = [t or "" for t in joined["text"]]
    embeddings = embedding_matrix(joined["embedding"])
    raw = score_begin_window(scorer, texts, embeddings, args.batch_size)
    logger.info(
        "raw begin-window scores: mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(raw.mean()),
        float(raw.std()),
        float(raw.min()),
        float(raw.max()),
    )

    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    predicted_types = domain_mlp.predict(typer, typer_labels, joined["embedding"])
    logger.info(
        "predicted content types agree with the oracle on %.1f%% of rows",
        100 * float((predicted_types == oracle_types).mean()),
    )

    ho = np.flatnonzero(is_holdout)
    global_knots = calibration_knots(raw[ho], levels[ho])
    report_fit("holdout-global", raw[ho], levels[ho], predicted_types[ho], global_knots)

    type_knots = per_type_knots(raw[ho], levels[ho], predicted_types[ho], min_per_type=args.min_per_type)
    report_fit("holdout-per-type", raw[ho], levels[ho], predicted_types[ho], type_knots)
    logger.info(
        "per-type fit: %d of %d types have their own remap",
        len(type_knots["types"]),
        len(set(predicted_types[ho].tolist())),
    )

    # The training-set fit, for comparison only. The model memorized these rows,
    # so its cutpoints sit where a trained model puts its own training data --
    # reported so the bias is visible rather than assumed absent.
    train_knots = calibration_knots(raw[~is_holdout], levels[~is_holdout])
    report_fit("trainset-global", raw[~is_holdout], levels[~is_holdout], predicted_types[~is_holdout], train_knots)

    corpus = None
    if args.corpus_shards:
        scores = sample_corpus_scores(args.manifest, args.corpus_shards, args.seed)
        corpus = report_corpus(scores, list(global_knots["xk"][1:-1]))

    with open_url(args.out, "w") as fh:
        fh.write(json.dumps(type_knots))
    logger.info("wrote calibration -> %s", args.out)
    logger.info(
        "RESULT %s",
        json.dumps(
            {
                "global_cutpoints": [float(c) for c in global_knots["xk"][1:-1]],
                "per_type_default_cutpoints": [float(c) for c in type_knots["default"]["xk"][1:-1]],
                "per_type_cutpoints": {
                    name: [float(c) for c in k["xk"][1:-1]] for name, k in sorted(type_knots["types"].items())
                },
                "trainset_cutpoints": [float(c) for c in train_knots["xk"][1:-1]],
                "holdout_rows": int(is_holdout.sum()),
                "corpus": corpus,
                "out": args.out,
            }
        ),
    )


if __name__ == "__main__":
    main()
