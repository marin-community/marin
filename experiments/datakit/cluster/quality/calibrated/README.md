# Content-type-calibrated document quality

Makes the datakit quality **buckets coherent** — so `q4` is uniformly excellent and
`q0` uniformly junk *regardless of source* — without retraining the quality model.

## The problem

The deployed fasttext quality classifier (`../v0`) sorts documents by
**domain/modality/language, not intrinsic quality**. Straight from the store:

- clean, correct code and pristine arXiv **math abstracts** sit in **q0** (score 0.00–0.20);
- **non-English** text is uniformly bottom-bucketed;
- meanwhile mediocre English grant prose reaches q4.

Root cause: the `v0` rubric distills a generic "LLM-pretraining value" target that is
itself recoverable from source identity (source alone predicts the oracle at AUC
0.852), so any faithful distillation reproduces the domain bias. Raising oracle-AUC
does **not** fix bucket coherence — it is a *label/definition* problem.

## The insight

Evaluated against **content-type-aware, source-blind** oracle labels (see
`rubric.py`), the current classifier already **ranks reasonably well *within* each
type** (within-source pair-accuracy: multilingual 0.80, overall 0.65). Its only real
failure is a **per-type offset**: it scores multilingual at raw mean ≈ 0.18 while the
true quality mean is ≈ 0.51, so excellent multilingual can never reach the top bucket.

So the fix is not a new quality model — it is **per-type calibration** of the existing
score.

## The method (`recalibrate.py`)

1. a cheap fasttext **content-type classifier** (`prose/code/math/multilingual/structured`);
2. a per-type **affine rescale** of the raw score onto a common quality scale, fit
   against the type-aware oracle labels;
3. global cutpoints + an **absolute q4 excellence floor** → 5 comparable buckets.

All fasttext-cheap → stays under the deployed FLOPs/token budget.

```python
from experiments.datakit.cluster.quality.calibrated.recalibrate import CalibratedQuality
scorer = CalibratedQuality.load("calib_type.bin", "gs://.../calib.json")
calibrated_score, bucket = scorer.score(text, raw_score)  # bucket ∈ 0..4
```

Batch re-score a parquet:

```bash
uv run python -m experiments.datakit.cluster.quality.calibrated.recalibrate \
  --input scored.parquet --output recalibrated.parquet \
  --type-model calib_type.bin --calibration gs://.../calib.json \
  --text-col text --score-col score
```

## Validation

Type-aware source-blind labels: **2,089** documents, 38 sources, scored by Claude via
the `rubric.py` prompt. Reliability (380 double-labeled): raters never differ by >1
level (within-1 = 1.000), pair-order agreement 0.85 (gap≥1) / 0.97 (gap≥2).

Effect on a **fresh 2,660-doc store sample** (content-type classifier 90% accurate) —
share of each type across buckets, current → calibrated:

| type | current `q0..q4` | calibrated `q0..q4` |
|---|---|---|
| multilingual | `[.61, .23, .14, .01, .01]` | `[.32, .30, .15, .09, .14]` |
| code | `[.21, .03, .06, .20, .50]` | `[.21, .07, .21, .43, .07]` |
| synthetic | `[.24, .15, .17, .27, .17]` | `[.16, .21, .16, .18, .29]` |
| math | `[.05, .08, .14, .25, .49]` | `[.05, .08, .14, .14, .58]` |

- multilingual: **61%→32%** in q0, **1%→14%** in q4 (no longer dumped in junk);
- the top bucket went from **code+math only (0% multilingual)** to containing all
  types; q0 went from 47%-multilingual to a type-mixed junk bucket;
- math correctly stays high (genuinely high quality); code's over-scoring is corrected.

Source-disjoint held-out, multilingual **excellent→top-2-bucket recovery rose from
0.06 → 0.39** (0.67 with oracle content types).

Artifacts + labels:
`gs://marin-us-east5/tmp/ttl=90d/rav/quality-calibration/` (`labels_v1.parquet`,
`calib.json`, `calib_type.bin`).

## The stronger option: the pooled ranker (recommended)

Calibration fixes the per-type *offset* but inherits the current score's within-type
*ranking*. Retraining a **pooled fast-transformer** (`../fast_transformer/`) on the
type-aware labels instead of calibrating produces a better *ranker* that also removes
the bias — and it stays cheap (0.056M FLOPs/token, 18× under budget). Score arbitrary
text with `fast_transformer/scorer.py::PooledScorer`.

Results on 5,010 labels covering all 113 store sources (deployment-scenario holdout),
within-type pair-accuracy and Spearman vs the type-aware target:

| model | Spearman | pair-acc (ALL) | notes |
|---|---|---|---|
| deployed fasttext | 0.44 | 0.67 | domain-biased |
| pooled, single-rater labels | 0.59 | 0.67 | at the single-rater label ceiling |
| **pooled, 2-rater consensus** | **0.69** | **0.74** | **recommended; beats deployed on every type** |

The single-rater 0.62 ceiling is a *label-noise* ceiling, not a model limit (a 7×
larger model only reached 0.61). A **second independent rating pass → consensus (mean)
labels** broke it: Spearman 0.59→0.69, pair-acc 0.67→0.74, and it fixed the one type
(synthetic) where the deployed model previously led. Recommended deployable =
`pooled_consensus` (score → per-type calibration/global buckets + q4 floor here).

Artifacts: `gs://marin-us-east5/tmp/ttl=90d/rav/quality-calibration/`
(`labels_v3_consensus.parquet`, `pooled_consensus.eqx` + `_remap.json` + `_meta.json`).
Further upside is diminishing (3rd rater / pairwise near the top boundary).
