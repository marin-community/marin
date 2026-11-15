# Document low-value scoring

The scorer flags documents dominated by machine output, data dumps, markup,
damaged OCR, repetitive templates or embedded trajectories. It measures bounded
head, middle and tail views and runs a small binary classifier. Its score is
higher for low-value content. A calibration file supplies explicit operating
points targeting a natural-document false-drop rate on the calibration sample.
That target is an estimate, not a guarantee for a new corpus.

The implementation lives in `marin.datakit.low_value`. Models and calibration
are JSON, checked against caller-supplied SHA-256 digests before loading. Inference
uses NumPy and the existing tokenizer stack; it does not require scikit-learn or
unpickle executable model objects. The [research record](https://github.com/marin-community/marin-logbooks/issues/52)
contains the model comparison and annotation protocol. Use the
[published replay bundle and digests](../model-cards/replay-low-value-v1.md#artifacts)
for a concrete model, calibration and tokenizer pin.

## Text and token caches

Load the exact published model and its matching calibration. `model.json` records
the structural feature version, feature order, bundled tokenizer file digests and view size.
`calibration.json` names those model bytes and the available operating points.

```python
from marin.datakit.low_value.model import load_model, load_model_tokenizer, load_operating_point
from marin.datakit.low_value.scoring import score_views, text_views

model = load_model(model_uri, model_sha256)
point = load_operating_point(calibration_uri, calibration_sha256, model.sha256, 0.02)
tokenizer = load_model_tokenizer(model)
views = text_views(document_text, tokenizer, model.window_tokens)
scores = score_views([views], model)
flagged = point.drop(scores)
```

Text is encoded with Datakit's text preprocessor, including its BOS/EOS policy
and long-string handling. Existing token caches avoid whole-document decoding:
`score_token_cache` reads row offsets and at most three token windows per row.
Overlapping windows merge, so short documents are reviewed once. It yields
`DocumentScore` records in the requested index order, retaining the cache row
index as `id`.

```python
from marin.datakit.low_value.scoring import score_token_cache

rows = score_token_cache(
    cache_dir,
    cache_tokenizer_sha256=model.tokenizer_sha256,
    tokenizer=tokenizer,
    model=model,
    operating_point=point,
    indices=range(number_of_rows),
)
for row in rows:
    consume(row.id, row.low_value_score, row.is_low_value)
```

The supplied tokenizer object must match the cache producer's actual tokenizer.
Pass its `tokenizer.json` SHA-256 digest as `cache_tokenizer_sha256`. The bundled
loader verifies vocabulary, tokenizer configuration and template bytes before
staging the backend. Passing the model's expected digest for an incompatible cache does not convert
its IDs. The model JSON pins the bundled tokenizer file digests; callers bypassing
`load_model_tokenizer` remain responsible for supplying the matching object. The maintained format is the selected structural
boosted-tree head. Embedding and logistic comparisons remain in the research archive.

## Datakit quality step

`experiments.datakit.cluster.quality.low_value.pipeline.quality_step` binds a
normalized source to a co-partitioned score leaf. Supply a `LowValuePin` with a
model directory relative to `MARIN_PREFIX`, both artifact digests, an explicit
operating point, and a `ZephyrContext` with sufficient CPU and memory for each
scoring task. The task requests one CPU and 4 GiB memory per shard.

```python
from experiments.datakit.cluster.quality.low_value.pipeline import LowValuePin, quality_step

scores = quality_step(
    name="quality/low-value/my-source",
    source="my-source",
    normalized=normalized_step,
    pin=LowValuePin(model_path, model_sha256, calibration_sha256),
    target_false_drop=0.02,
    zephyr_context=context,
)
```

Every output shard keeps its normalized basename and document ID order. Columns
are `source`, `id`, `tokens`, `low_value_score` and `is_low_value`. Consumers can
join the leaf by normalized ID. The step hashes model and calibration
digests, source and operating-point choice; recalibration changes its artifact
identity. Completed output shards are skipped on resume. The artifact's
`documents_scored_this_run` counts newly scored documents, excluding skipped
shards. Model and calibration bytes are checked again before every source run.

The side leaf leaves the existing graded quality buckets available. Adopting a
cutoff on another store requires a fresh labeled calibration sample, including
protected natural code, math, multilingual documents and legible OCR. Keep the
calibration and evaluation splits separate. Bounded views can miss contamination
between windows, and exact-content grouping does not eliminate near duplicates.

The [public evaluation manifest](../model-cards/replay-low-value-v1.md#artifacts)
contains visible-view fingerprints, splits and labels.
Document text and annotation requests/responses remain private pending explicit
publication approval. Model-assisted labels require interpretation as such.
