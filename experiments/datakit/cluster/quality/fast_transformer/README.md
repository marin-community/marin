# Fast-transformer document-quality classifier

A pooled transformer that scores a document's value as LLM-pretraining data. It
replaces the domain-biased fasttext quality filter: trained against a **type-aware,
source-blind** oracle rubric ([`rubric.py`](rubric.py)) that scores each document
*as an example of its own type*, so excellent code, math, multilingual, and prose
all reach the top buckets instead of code/multilingual/wiki being dumped at ~0.

The deployed score is calibrated so its fixed 0.2-bucket quantization is
quality-coherent: a bucket means the same quality level across content types.

## Pipeline

```
rubric.py    type-aware oracle rubric — how docs are scored 1..5 (labeling itself is offline)
   │  labels: gs → merged parquet (5,578 oracle labels: consensus + junk-gate)
train.py     train the pooled FastTransformer on the labels → model.eqx + remap + meta
calibrate.py fit the monotonic bme calibration on the labels → calib_bme.json
score.py     score_normalized — the reference pipeline's per-source quality step
             (datakit/quality/<source>) → source/id/score/quality_bucket + samples
```

The stage report (single HTML page over all sources) lives in
`experiments/datakit/reports/quality.py` and runs as the pipeline's
`datakit/report/quality` step.

Retrain + recalibrate the deployed model:

```bash
python -m experiments.datakit.cluster.quality.fast_transformer.train \
    --labels s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet \
    --out-dir s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2
python -m experiments.datakit.cluster.quality.fast_transformer.calibrate \
    --model-dir s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2 \
    --out       s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2/calib_bme.json
```

## Hero data: the fusion scorer

The hero corpus is scored by a fusion variant of the model (`doc_embed_dim=1024`)
that reads a document's first 512 Nemotron tokens and its Harrier embedding. Two
steps produce the registered data, both resolved through
`experiments/datakit/hero_data.py` and driven by [`run.py`](run.py):

```
score_fusion.py  fusion_score_step — normalized text + Harrier leaf → id/score (raw sigmoid),
                 one Zephyr task per shard pair on GPU workers, tokenizing in the task
bucket.py        quality_step — fusion scores + content types + per-type calibration →
                 source/id/content_type/raw_score/score/quality_bucket, the QualityScores
                 dataset the store reads, in the normalized row order
quality_model.py QualityPin — the model directory, its digests, and the calibration
keyed_rows.py    the id-keyed match between co-partitioned shards both steps use
```

`hero_data.fusion_scores` is pinned to the completed run of the score stage, and
`hero_data.quality` recomputes from the bucket step's identity. A refit calibration
reruns only the bucket stage:

```bash
uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --job-name hero-quality-bucket \
    --no-wait -- python -m experiments.datakit.cluster.quality.fast_transformer.run --stage bucket
```

## Scoring

`score.py` scores **whole-doc (bme)**: the score is the mean over begin/middle/end
~512-token windows, so a source whose docs share a long boilerplate prefix
(agent/tool trajectories) is not scored blind by the first 512 tokens. Sources that
are genuinely uniform in quality stay near-constant — the report flags those as
`uninformative` (a variance gate) versus `homogeneous` (real spread, one bucket).

Calibration is a monotonic remap, so it does not change document ranking; it only
warps the bell-shaped raw score so the fixed cutpoints `[0.2, 0.4, 0.6, 0.8]` land
on the oracle quality levels (labeled-set bucket-vs-level agreement: within-1 ≈0.98).
The fusion scorer's calibration carries one curve per predicted content type plus a
default; `calibrate.apply_calibration` routes each document through its type's curve.

## Architecture

`embed → pool over 64-token windows → input proj + positions → N transformer
layers over the super-tokens → pool → scalar quality head`. Pooling at the window
boundary amortizes the transformer cost by ~64×, keeping inference under a
<1M FLOPs/token budget while still running real self-attention. Deployed config:
`meanmaxmin` pooling, `pool_window=64`, `embed_dim=256`, `hidden_dim=256`,
`num_layers=2`, `num_heads=4`, `max_tokens=512`, tokenizer `intfloat/multilingual-e5-small`.

The fusion scorer projects and layer-norms the document embedding to `hidden_dim`,
appends it as a gated super-token the real tokens attend to, and adds it to the head
through a zero-initialized skip. Its deployed config: `pool_window=16`,
`embed_dim=256`, `hidden_dim=384`, `num_layers=4`, `num_heads=8`, `max_tokens=512`,
Nemotron token ids over a 131,072-entry vocabulary.

## Files

Core:

- [`rubric.py`](rubric.py) — the type-aware, source-blind oracle rubric (system prompt + content types).
- [`model.py`](model.py) — the pooled `FastTransformer` regressor, with the optional document-embedding fusion.
- [`data.py`](data.py) — tokenize the oracle-scored text and pack dense padded arrays + a compact vocab.
- [`train.py`](train.py) — `train_from_labels`: train the deployed scorer from the label parquet, plus `fit`/`train_regressor` and the holdout metrics.
- [`calibrate.py`](calibrate.py) — fit the monotonic bme calibration (`calib_bme.json`) and apply it, globally or per content type.
- [`scorer.py`](scorer.py) — `PooledScorer`: load a trained model + vocab remap and score arbitrary text.
- [`score.py`](score.py) — `score_normalized`: the per-source quality step (bme + calibration → buckets + samples side output).
- [`score_fusion.py`](score_fusion.py) — `fusion_score_step`: raw fusion scores from normalized text and Harrier embeddings.
- [`bucket.py`](bucket.py) — `quality_step`: the store-ready `QualityScores` dataset from fusion scores and content types.
- [`quality_model.py`](quality_model.py) — `QualityPin` and the model and calibration digests a step checks before writing.
- [`keyed_rows.py`](keyed_rows.py) — id-keyed matching between co-partitioned shards.
- [`run.py`](run.py) — the hero fleet driver for the score and bucket stages.
- [`metrics.py`](metrics.py) — rank-based AUC / Spearman used by the training holdout.
- [`artifact.py`](artifact.py) — `QualityScores` step artifact + the fixed `BUCKET_EDGES`.

## Artifacts

- Labels: `s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet` (5,578 oracle labels; `label_batch` marks `consensus_v3` / `junkgate_web_wiki` / `junkgate_code_math`).
- Model: `s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2/` (`.eqx` + `_remap.json` + `_meta.json` + `calib_bme.json`).
- Fusion model: `s3://marin-us-east-02a/marin/datakit/models/quality/nemotron_88k/` (same layout; `calib_bme.json` holds the per-type curves). Pinned as `hero_data.NEMOTRON_88K`.
