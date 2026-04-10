Marin has most of the pieces for end-to-end data processing \- download, dedup, filtering, classification, decontamination, tokenization \- but the code is scattered across `experiments/` and `lib/marin/` with inconsistent formats, ad-hoc ID handling, and unclear provenance.

We propose consolidating this into **datakit**: a set of composable pipeline stages with standardized formats and conventions, living in `lib/marin/datakit/`. Dataset-specific wiring (e.g., "for Arxiv, apply these transforms") lives in `experiments/` or reference configurations.


Links:
 * [marin\#2355](https://github.com/marin-community/marin/issues/2355)
 * [gdoc](https://docs.google.com/document/d/1kDSzONg32zv2VnCO4FJiMP0fcjRSjgP0uTDpI4_C4O0)

# Golden Path

The canonical pipeline for getting a dataset from source to training:

`Download → Normalize → Embed → Classify/Filter → Dedup → Tokenize`

Notably, datakit in the proposed form, doesn’t include **data mixing** or **training**.

## 1\. Download

Download raw dataset from Hugging Face (or other sources). Raw downloads are preserved as-is in their original format and directory structure.

## 2\. Normalize to Standard Format

Convert raw data into the **datakit standard format**:

* **File format**: Parquet \- columnar, widely supported, supports pushdown filters and column projection.
* **Mandatory columns**:
  * `id` \- unique document identifier (see [ID Column](#id-column) below)
  * `text` \- primary text content \- we enforce UTF-8
* **Arbitrary additional columns**: any fields present in the raw data are preserved
* **Directory structure**: preserver original directory structure
* **Partition structure**: partition layout from the source does NOT need to be preserved at this point \- and in most cases it will not be
  * We may want to introduce a more efficient partitioning at this stage and preserve the new partitioning until tokenization
  * The partitions must follow `part-x-of-y` suffix naming convention
* **Sort invariant**: each partition is sorted by `id`
* **Typed output:** in the code the data has typed representation via `Artifact`

This is the "intake" step \- all downstream stages operate on normalized Parquet datasets.

## 3\. Embed

Produce vector embeddings for each document. Output is an **attributes dataset** (see [Attributes Datasets](#attributes-datasets)) with embedding vectors keyed by `id`.

## 4\. Quality Classification, Topic Assignment

Each classifier produces an **attributes dataset** containing scores/labels keyed by `id`.

## 5\. Deduplication

Produces an **attributes dataset** marking duplicate spans or documents.

## 7\. Consolidation

Join attributes datasets back to the source documents and apply filters:

* Filter by classifier thresholds (e.g., quality score \> 0.8)
* Remove duplicate spans/documents

Output is a clean, filtered Parquet dataset \- still sorted by `id`, still co-partitioned.

## 8\. Tokenize

Convert clean text into tokenized Levanter cache format.

**Tokenization is the boundary where per-document structure ends.** The tokenizer concatenates documents into fixed-size token sequences for efficient training. Partition structure from earlier stages does not carry through \- the output is sharded Levanter TreeStore caches with a `.stats.json` summary.

# Core Design Decisions

## Parquet as the Standard Format

All intermediate datasets (from normalization through consolidation) use the Parquet columnar format. Benefits:

* Column projection (only read the columns you need)
* Filter pushdown
* Efficient sorted merge joins via Zephyr
* Mature ecosystem with broad tooling support

NOTE: We initially considered Vortex for its pushdown and lookup capabilities, but encountered blocking issues with Zephyr pipeline integration (see [vortex\#6905](https://github.com/vortex-data/vortex/issues/6905)). Parquet provides the same columnar benefits with a proven ecosystem. If Vortex matures, we can revisit.

## ID Column {#id-column}

* **Preserve existing IDs** when present in the raw data (e.g., WARC-Record-ID in DCLM, HF row indices). These carry provenance meaning and aid debugging.
  * But rename column to `source_id`
* **Generate deterministic IDs** via content hash. Column named `id`. Deterministic hashing ensures reproducibility \- re-running the pipeline produces the same IDs, which preserves caching and diffing.

## Co-Partitioning Invariant

The key invariant that enables efficient joins: **Attributes datasets must have the same number of shards and the same key-range partitioning as their source dataset.**

This means:

* The normalization step determines the partition structure
* All downstream stages (embed, classify, dedup) preserve this structure \- same shard count, same ID ranges per shard
* Consolidation can use Zephyr's `sorted_merge_join` without a costly `group_by` shuffle

This is enforced by convention: each processing stage reads source partitions 1:1 and writes output partitions with matching structure.

## Attributes Datasets {#attributes-datasets}

Processing stages (embed, classify, dedup) produce **attributes datasets** \- lightweight Parquet files containing:

* `id` — matching the source document ID
* Stage-specific output columns (e.g., `quality_score`, `is_duplicate`, `topic_label`)

Attributes datasets:

* Use Parquet format
* Are co-partitioned with the source (same shard count and key ranges)
* Are sorted by `id` within each partition
* Can be joined back to source documents via `sorted_merge_join`

Multiple attribute datasets from different stages can be joined together during consolidation to apply compound filters.

## Step Orchestration via StepSpec

Datakit builds on `StepSpec` \- the pure-data step descriptor that captures identity, dependencies. Each datakit stage (normalize, classify, dedup, etc.) is a `StepSpec` with:

* **`name`**: human-readable stage name (e.g., `"fineweb/normalize"`)
* **`deps`**: upstream `StepSpec`s whose `output_path` this stage reads from
* **`hash_attrs`**: configuration values that affect output (model name, thresholds, etc.) — changes invalidate the cache
* **`fn`**: the callable that performs the work, receiving `output_path` as its argument

`StepSpec` gives us automatic cache invalidation (via `hash_id` derived from name \+ attrs \+ dep paths), dependency tracking, and deterministic output paths. The step runner handles locking, heartbeats, and status \- datakit stages just describe what to run.

Example wiring:

```py
download = StepSpec(
    name="fineweb/download",
    fn=lambda output_path: download_hf(output_path=output_path, dataset_id="HuggingFaceFW/fineweb"),
    hash_attrs={"dataset_id": "HuggingFaceFW/fineweb", "revision": "abc1234"},
)

normalize = StepSpec(
    name="fineweb/normalize",
    deps=[download],
    fn=lambda output_path: normalize_to_parquet(
        input_path=download.output_path, output_path=output_path, text_field="text",
    ),
    hash_attrs={"text_field": "text"},
)

quality = StepSpec(
    name="fineweb/quality",
    deps=[normalize],
    fn=lambda output_path: classify(
        input_path=normalize.output_path, output_path=output_path, model="fasttext-quality-v1",
    ),
    hash_attrs={"model": "fasttext-quality-v1"},
)

dedup = StepSpec(
    name="fineweb/dedup",
    deps=[normalize],
    fn=lambda output_path: deduplicate(
        input_path=normalize.output_path, output_path=output_path, mode="fuzzy_document",
    ),
    hash_attrs={"mode": "fuzzy_document"},
)

consolidated = StepSpec(
    name="fineweb/consolidated",
    deps=[normalize, quality, dedup],
    fn=lambda output_path: consolidate(
        source_path=normalize.output_path,
        attribute_paths=[quality.output_path, dedup.output_path],
        output_path=output_path,
        quality_threshold=0.8,
    ),
    hash_attrs={"quality_threshold": 0.8},
)

tokenized = StepSpec(
    name="fineweb/tokenized",
    deps=[consolidated],
    fn=lambda output_path: tokenize(
        input_path=consolidated.output_path, output_path=output_path,
        tokenizer="meta-llama/Llama-3.1-8B",
    ),
    hash_attrs={"tokenizer": "meta-llama/Llama-3.1-8B"},
)
```

# API Surface

## `lib/marin/datakit/`

Core primitives — the reusable building blocks:

```
lib/marin/datakit/
  normalize       # Raw format -> standard Parquet (id, text, ...)
  embed           # Document embedding
  classify        # Quality/topic classification
  dedup           # Deduplication (exact + fuzzy)
  consolidate     # Join attributes + apply filters
```

## `experiments/` (or reference configurations)

Dataset-specific wiring \- which transforms to apply for a given dataset, expressed as `StepSpec` DAGs.

# Execution Plan

* Implement `datakit/normalize.py` \- standard schema definitions, ID generation, raw format to Parquet conversion with mandatory columns
* Integration tests for the normalize step
* Integration tests covering download, normalize, dedup and tokenize at reasonable scale
* Update Grug/ferry experiment definitions to consume datakit pipeline outputs directly

# Non-Goals

* **Replacing the mixing or training APIs** \- datakit standardizes everything upstream of tokenization.
* **Supporting non-text modalities** \- the initial scope is text datasets with a mandatory `text` field. Multimodal support can be added later by relaxing this constraint.

# Open Questions

1. **ID uniqueness enforcement**: Per-partition validation is cheap and will be the default. Should we also support global uniqueness checks? What's the failure mode — warn or error?
2. **Non-text datasets**: Code datasets, structured data \- do we need a configurable primary field, or is `text` always sufficient?
3. **Versioning**: How do we version datakit outputs so that downstream consumers (Grug) can pin to a specific processing run? `StepSpec.hash_id` provides content-based versioning, but do we need human-readable version tags as well?
