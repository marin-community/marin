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

## Scoring

`score.py` scores **whole-doc (bme)**: the score is the mean over begin/middle/end
~512-token windows, so a source whose docs share a long boilerplate prefix
(agent/tool trajectories) is not scored blind by the first 512 tokens. Sources that
are genuinely uniform in quality stay near-constant — the report flags those as
`uninformative` (a variance gate) versus `homogeneous` (real spread, one bucket).

Calibration is a monotonic remap, so it does not change document ranking; it only
warps the bell-shaped raw score so the fixed cutpoints `[0.2, 0.4, 0.6, 0.8]` land
on the oracle quality levels (labeled-set bucket-vs-level agreement: within-1 ≈0.98).

### Calibration is per content type

The oracle grades each document as an example of its own type, but the types do not
share a ceiling: solved agent trajectories reach the top score 3.4% of the time
against prose's 26%. One global remap encodes prose's scale, so whole types are
pushed out of the top bucket however good their members are.

[`calibrate.py`](calibrate.py) therefore fits one remap per type. The remap stays
monotonic within a type, so it reorders nothing — it corrects the *offset* and
nothing else.

The alternative considered was equal-count bands within a domain, and the data
rules it out. Per-domain mean quality runs 2.13 to 4.38: forcing equal mass would
put one domain's quality-3.0 documents in the top bucket while dropping another's
quality-4.0 documents out of it. Sources differ just as much (2.13 for
`cp/wikiteam`, which is 73% junk, against 4.72 for `cp/arxiv_papers`), and that
difference is the most reliable signal the filter has. `test_calibrate.py` asserts
both halves — a compressed type recovers, and a genuinely poor type does not — so
the distinction cannot be quietly undone.

Calibration needs a type where the pipeline has only text, so
[`content_type.py`](content_type.py) predicts one: a hashed bag of tokens plus
structural features, a hash and one matrix multiply. It is fitted on *predicted*
types rather than the oracle's, so a type that absorbs some neighbouring documents
gets cutpoints fitted on the mixture it will actually be handed. `score.py` carries
the predicted type on every scored record, so per-type parity stays auditable on
the corpus rather than only on the labeled set.

## Architecture

`embed → pool over 64-token windows → input proj + positions → N transformer
layers over the super-tokens → pool → scalar quality head`. Pooling at the window
boundary amortizes the transformer cost by ~64×, keeping inference under a
<1M FLOPs/token budget while still running real self-attention. Deployed config:
`meanmaxmin` pooling, `pool_window=64`, `embed_dim=256`, `hidden_dim=256`,
`num_layers=2`, `num_heads=4`, `max_tokens=512`, tokenizer `intfloat/multilingual-e5-small`.

## Building a label set

Labels come from an offline grader (GLM-5.2, served by
[`glm52_vllm.py`](../glm52_vllm.py)) applied to a stratified draw from the corpus
sample. Two failures on this path produced label sets that looked healthy and were
not, so the pipeline is built around detecting them rather than around throughput:

```
sample_labels.py    stratified draw across all sources → label_set/ shards
label_with_glm52.py serve the grader, label with checkpointing → labels.parquet
gate_labels.py      decide whether the result is fit to train on (run this before train.py)
```

Give the driver real resources. It holds the whole label set in memory to feed
chunks — measured at 0.9 GB as an Arrow table and 1.8 GB once materialized as
Python rows — while `iris job run` defaults to 0.1 CPU and a small memory request.
Under-requesting does not fail cleanly: the task is SIGKILLed (exit 137) partway
through the read, before the module logs anything, which reads as a mysterious
startup crash rather than as an OOM. A request of 4 GB or more also needs
`--enable-extra-resources`, which the CLI rejects rather than trims.

**Serve on GB200.** The H100 fleet is not a working fallback for this model: FlashInfer
JIT-builds the SM90 fused-MoE kernel on first start and the link fails with
`cannot find -lnvrtc`, ~29 minutes in, after all 756 GB of weights have loaded. SM100
takes a path that does not build it. GB200 capacity is contended and the gang binds to
one NVLink domain, so it can sit queued for hours; `--max-retries` lets it ride out
transient Ray startup races without losing checkpointed work.

```bash
iris --cluster=marin job run --target-cluster cw-us-east-08a \
    --enable-extra-resources --cpu 4 --memory 8g --disk 64g --max-retries 6 \
    -- python -m experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 \
    --label-set  s3://.../label_set_100k \
    --out        s3://.../glm52_labels.parquet \
    --run-id <tag> --fleet gb200 --object-store-endpoint https://cwobject.com
```

Reruns resume from `<out>.chunks/`, skipping ids already labeled, so an
interrupted run costs only its startup.

**Documents are excerpted, not cut.** A hard cut at the character cap left text
ending mid-token, which the rubric correctly reads as damage — so the grader marked
those documents invalid and scored them 1. That put 85% of the bottom bucket at the
cap and made length predict quality at Spearman -0.25. `sample_labels.py` therefore
cuts on a boundary and appends an explicit `[Excerpt ends here …]` marker, and the
rubric states that the marker is the harness shortening the document rather than
damage in the source.

**The character cap is not a token budget.** CJK and code tokenize far denser than
English, so a 12k-character document can exceed 12k tokens; with output tokens
reserved on top, those prompts overflow the context and the server rejects them.
Counting such rejections as ordinary dropped documents hides the failure completely,
because what is lost is exactly the *longest* documents — the same length bias the
excerpting fix removes. Enlarging the context is the wrong lever: it spends KV cache
on every request to accommodate a handful of dense ones. The prompt text is capped
instead, narrower than the stored excerpt, and a chunk that mostly fails aborts rather
than dropping rows. A small tail of unusually dense documents still overflows (~1% on
the 88k set), which is what the long-document retention gate measures.

## Validation

`gate_labels.py` is the gate between labeling and training, because a poisoned label
set still trains a model and that model still reports plausible metrics.

It compares against the **input** set, not only the output. Selective loss is
invisible from the survivors alone: drop every long document and the remaining
length distribution is still perfectly well behaved. Only the input/output
comparison shows the hole.

The length check is **directional and measured above a 500-character floor**. The
poisoning signature is long documents scoring *worse*; quality rising with length is
ordinary signal, because short documents genuinely are junk (under 200 characters:
41.5% invalid, mean quality 1.66). Excluding that stub tail sharpens rather than
softens the test — the known-poisoned set reads -0.343 above the floor against
-0.250 overall, while the corrected set reads -0.006.

Aggregates alone are not sufficient to accept a model. A candidate that improved
every summary statistic — wider bucket spread, better cross-domain parity, higher
within-domain variance — turned out to promote gym timetables and demote worked
Fourier derivations. [`sample_disagreements.py`](sample_disagreements.py) exists to
make reading the disagreements a repeatable step rather than an ad-hoc one.

### Known limitation: per-type ceiling compression

Ranking *within* a content type is sound; absolute calibration *across* types is not.
Agentic trajectories carry an explicit outcome banner and the grader keys on it
correctly (failed 2.57 < unverified 2.72 < solved 3.36 mean quality), yet even
successfully solved trajectories reach 5 only 3.4% of the time against prose's 26%.
Multilingual's lower aggregate is a composition effect rather than a blanket penalty
— one Japanese web source is 56% of it at 7.4% top-share, while `dolma4pdfs` reaches
23.6%, on par with prose. Bucketing within domain by quantile uses only the ranking,
which is the part that is correct.

## Files

Core:

- [`rubric.py`](rubric.py) — the type-aware, source-blind oracle rubric (system prompt + content types).
- [`model.py`](model.py) — the pooled `FastTransformer` regressor.
- [`data.py`](data.py) — tokenize the oracle-scored text and pack dense padded arrays + a compact vocab.
- [`train.py`](train.py) — `train_from_labels`: train the deployed scorer from the label parquet, plus `fit`/`train_regressor` and the holdout metrics.
- [`calibrate.py`](calibrate.py) — fit the monotonic bme calibration (`calib_bme.json`).
- [`scorer.py`](scorer.py) — `PooledScorer`: load a trained model + vocab remap and score arbitrary text.
- [`score.py`](score.py) — `score_normalized`: the per-source quality step (bme + calibration → buckets + samples side output).
- [`metrics.py`](metrics.py) — rank-based AUC / Spearman used by the training holdout.
- [`artifact.py`](artifact.py) — `QualityScores` step artifact + the fixed `BUCKET_EDGES`.

Labeling:

- [`sample_labels.py`](sample_labels.py) — stratified draw across every source, excerpted on a boundary.
- [`label_with_glm52.py`](label_with_glm52.py) — run the grader with checkpointing and resume; aborts a chunk that mostly fails rather than dropping rows.
- [`gate_labels.py`](gate_labels.py) — decide whether a label set is fit to train on.
- [`content_type.py`](content_type.py) — predict a document's content type, which the per-type calibration needs at scoring time.

Evaluation:

- [`domain_eval_set.py`](domain_eval_set.py) — census → proportional allocation → draw, so no single source dominates the evaluation.
- [`score_eval_set.py`](score_eval_set.py) — score the evaluation set with a given model version.
- [`compare_by_domain.py`](compare_by_domain.py) — per-domain bucket distributions across model versions.
- [`sample_disagreements.py`](sample_disagreements.py) — surface the documents two scorers most disagree about, for reading.
- [`intruder_ab.py`](intruder_ab.py) — bucket-coherence intruder test, `--bucketing {global,domain-quantile}`.
- [`gate_model.py`](gate_model.py) — decide whether a trained model beats the deployed one: within-type ranking, cross-type parity, and per-source signal, measured on the training holdout.

## Artifacts

- Labels: `s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet` (5,578 oracle labels; `label_batch` marks `consensus_v3` / `junkgate_web_wiki` / `junkgate_code_math`).
- Model: `s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2/` (`.eqx` + `_remap.json` + `_meta.json` + `calib_bme.json`).
