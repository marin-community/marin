# Per-source DF filter for decon (marin#6852 follow-on)

Precision fix for the **boilerplate false-positive** class that no geometry
change (`\n\n` split, min-matched-count) can touch: n-grams that are *ubiquitous
within a source* (a congressional enacting clause across every `cp/usgpo` bill,
a license header across a code source) match eval text but carry zero
contamination signal. Filter them per source; a genuine leak's *distinctive*
n-grams stay rare-in-source and survive, so recall is preserved.

## Why this design (validated on 100B, 04/2026)

- Ran per-source eval-n-gram document frequency over the 100B sample
  (`ops/per_source_df.py`) + a validation (`ops/validate_per_source_filter.py`).
- **Precision:** `cp/usgpo` enacting-clause overlap `1.000 → 0.000` when the
  usgpo common-set is removed. (The 100M `\n\n` run flags 6 docs; per-source
  filtering removes the source-concentrated ones.)
- **Recall:** worst-case common coverage of 420 verbatim-injected eval items
  median **0.000**, **0/420** erasable. Contrast: eval-frequency weighting fixed
  the FPs too but lost **45%** recall (it conflates boilerplate with
  benchmark-duplicated real items). Per-source DF is the only recall-safe lever.
- **Scope: per-source only** (no global DF). Per-source catches the structural
  boilerplate that is the bulk of the problem; it is simpler, estimable from a
  small sample, corpus-size-independent, and *more* recall-safe (a broadly
  scattered real leak survives because it is dense in no single source). Global
  DF is deferred — it catches diffuse boilerplate (famous quotes, standard
  identities) but needs a large-fraction scan and is less recall-safe.

Depends on the already-landed `\n\n` `paragraph_delimiter` knob.

## Mechanism

1. Build the eval bloom + index once (unchanged, shared across sources).
2. Per source: sample ~5k docs, extract `\n\n` 13-grams, count how many of the
   source's sampled docs contain each **eval** n-gram (membership via the bloom —
   the only n-grams a drop-set can contain). Threshold → per-source drop-set.
3. At **mark time** for that source's docs, exclude drop-set n-grams from *both*
   numerator and denominator of the paragraph overlap. An all-boilerplate
   paragraph collapses to zero relevant n-grams → no flag; a paragraph with a
   real leak keeps its distinctive n-grams → still flags at ~1.0.

One bloom, built once. The drop is a mark-time exclusion set, not a second bloom.

## Sample size

- **~5,000 docs/source.** Threshold is fractional (~0.5% of a source's docs);
  ~20 expected hits at threshold needs `N ≈ 20/0.005 = 4000`. Cases that matter
  are far above (enacting clause ≈ 21% of usgpo → ~1000 hits in 5k).
- Floor tiny sources to their full doc set; cap large ones at 5k.
- Data is shuffled + normalized upstream, so "sample" = a **5k-doc prefix** of
  the same per-source data being deconned (no reservoir, no re-shuffle). Cheap:
  ~single-digit GB, minutes.

## Config (centralize beside `NGRAM_LENGTH` in `decon_arm.py`)

```python
DF_SAMPLE_DOCS = 5000        # docs/source for the DF estimate
DF_COMMON_FRAC = 0.005       # n-gram common if in >= this fraction of source docs
DF_COMMON_MIN_ABS = 5        # and >= this absolute count (small-source floor)
```

## Implementation

### 1. Mark side — thread a drop-set through the overlap (lib/marin/src/marin/datakit/decon.py)

`_paragraph_overlap_and_matches` gains `drop_hashes` and removes those n-grams
from both sides (same shape as the cluster-D `_has_alpha` filter):

```python
def _paragraph_overlap_and_matches(
    paragraph, bf, ngram, drop_hashes: frozenset[int] = frozenset()
) -> tuple[float, list[int]]:
    ...
    ngrams = list(_extract_ngrams(paragraph, ngram.ngram_length, ngram.stride))
    if not ngrams:
        return 0.0, []
    hashes = [_bloom_hash(ng) for ng in ngrams]
    kept = [h for h in hashes if h not in drop_hashes]   # drop source-common n-grams
    if not kept:
        return 0.0, []
    matched = [h for h in kept if h in bf]
    return len(matched) / len(kept), matched
```

Thread `drop_hashes` through `_make_marker` → `mark_shard` (load the source's
drop-set once per shard alongside the bloom) → `decon_to_parquet` → `decon_step`.
`decon_step` folds a fingerprint of the drop-set (or the producing step's output
id) into `hash_attrs` so a changed drop-set re-addresses the mark.

### 2. Per-source drop-set step (new, in decon.py or a sibling)

A `StepSpec` per source: depends on `(prebuilt_bloom, source_normalized)`, emits a
parquet of drop hashes.

```python
def build_source_drop_set(
    *, source_normalized_path, prebuilt_bloom_dir, output_path,
    text_field, ngram, sample_docs, frac, min_abs,
) -> int:
    bf = dupekit.Bloom.load_bytes(StoragePath(bloom_paths(prebuilt_bloom_dir)[0]).read_bytes())
    counts: Counter[int] = Counter()
    n = 0
    for record in islice(_iter_normalized_docs(source_normalized_path), sample_docs):
        n += 1
        hits = {
            h for feat in _extract_features(str(record[text_field]), ngram)
            if (h := _bloom_hash(feat)) in bf
        }
        counts.update(hits)
    thr = max(min_abs, int(frac * n))
    drop = [h for h, c in counts.items() if c >= thr]
    write_parquet_file(({"hash": h} for h in drop), output_path, _DROP_SCHEMA)
    return len(drop)


def source_drop_set_step(*, name, source_normalized, prebuilt_bloom, ...) -> StepSpec:
    # hash_attrs: bloom id, ngram_length, overlap_threshold, paragraph_delimiter,
    #             sample_docs, frac, min_abs, feature_filter_version
    ...
```

Uses `_extract_features` (so it honors `paragraph_delimiter` + `_has_alpha`),
membership via the same bloom, `islice` for the prefix sample.

### 3. Wire into `decon_arm.py`

```python
bloom = build_eval_bloom_step(..., paragraph_delimiter=PARAGRAPH_DELIMITER)
for name, sample_step in sampled.items():
    drop = source_drop_set_step(
        name=f"datakit/decon_drop/{name}", source_normalized=sample_step,
        prebuilt_bloom=bloom, ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD, paragraph_delimiter=PARAGRAPH_DELIMITER,
        sample_docs=DF_SAMPLE_DOCS, frac=DF_COMMON_FRAC, min_abs=DF_COMMON_MIN_ABS,
    )
    steps.append(decon_step(
        name=f"datakit/testbed_decon/{name}", normalized=sample_step,
        prebuilt_bloom=bloom, source_drop_set=drop,   # new dep
        ngram_length=NGRAM_LENGTH, overlap_threshold=OVERLAP_THRESHOLD,
        paragraph_delimiter=PARAGRAPH_DELIMITER, ...,
    ))
```

`decon_step` loads its source's drop-set from `source_drop_set.output_path` at
mark time. Default `source_drop_set=None` → empty drop-set → other callers
(`reference_pipeline`, `all_sources_decon`) unaffected.

### 4. Content-addressing

- `source_drop_set_step.hash_attrs`: bloom output id, ngram config,
  `paragraph_delimiter`, `sample_docs`, `frac`, `min_abs`, `feature_filter_version`.
- `decon_step.hash_attrs`: add the drop-set step's output id (via the dep) — a
  changed drop-set re-addresses the mark. No new `FEATURE_FILTER_VERSION` bump
  needed; the drop-set dep carries the change.

## Testing

- Unit (`tests/datakit/test_decon.py`):
  - a synthetic n-gram planted in ≥ threshold of a source's sample lands in the
    drop-set and stops flagging a doc that only matches on it;
  - a distinctive eval n-gram (in one doc) stays out of the drop-set → still flags;
  - `drop_hashes=frozenset()` reproduces current behavior.
- End-to-end: re-run 100M via `decon_arm` with the filter on; confirm `cp/usgpo`
  drops out and the remaining flags are the diffuse-global ones (identity,
  Gettysburg, treaty) — i.e. 6 → ~3, all non-source-concentrated. Re-run the
  injection recall harness (`ops/recall_test.py`) to confirm no verbatim-recall
  regression.

## Cost

- Bloom: unchanged (~minutes over 240 MB evals).
- Drop-set steps: 115 × ~5k-doc prefix reads + n-gram extraction ≈ single-digit
  GB, minutes total (parallel across sources). Cheap and content-addressed
  (recompute only when eval corpus, source sample, or thresholds change).
- Mark: one extra set-membership per n-gram; negligible.

## Deferred / follow-ups

- **Global DF** (diffuse boilerplate: famous quotes, standard identities). Adds
  a large-fraction scan + a corpus-size-dependent threshold + more recall
  exposure. Revisit only if the diffuse FPs (identity/Gettysburg/treaty) prove
  worth it. Data already computed: `user/rav/decon_viewer/corpus_df_100b.parquet`.
- Threshold tuning (`frac`, `min_abs`) once the 100M end-to-end lands.
- Piggyback alternative: instead of a standalone 5k sample, harvest per-source DF
  from an unfiltered decon run's `matched_hashes` (group by source). Free over the
  full corpus, but needs a two-pass mark (or per-paragraph match stats in the
  output so the filtered flag is a cheap recompute). Only worth it if we later add
  global DF, which wants the full-corpus counts anyway.
```
