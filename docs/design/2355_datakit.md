Marin has most of the pieces for end-to-end data processing \- download, dedup, filtering, classification, decontamination, tokenization \- but the code is scattered across `experiments/` and `lib/marin/` with inconsistent formats, ad-hoc ID handling, and unclear provenance.

We propose consolidating this into **datakit**: a set of composable pipeline stages with standardized formats and conventions, living in `lib/marin/src/marin/datakit/`. Dataset-specific wiring (e.g., "for Arxiv, apply these transforms") lives in `experiments/` or reference configurations.


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
* **Partition identity**: derived, not stored on rows. Normalize writes `part-x-of-y` files and downstream stages recover the source partition index from the sorted file order (zephyr's `shard.shard_idx`). Stages that emit attributes materialize it as a `partition_id` column on their own output — see `marin.datakit.decon`.
* **Arbitrary additional columns**: any fields present in the raw data are preserved
* **Directory structure**: preserver original directory structure
* **Partition structure**: partition layout from the source does NOT need to be preserved at this point \- and in most cases it will not be
  * We may want to introduce a more efficient partitioning at this stage and preserve the new partitioning until tokenization
  * The partitions must follow `part-x-of-y` suffix naming convention
* **Sort invariant**: each partition is sorted by `id`
* **Typed output:** in the code the data has typed representation via `Artifact`

This is the "intake" step \- all downstream stages operate on normalized Parquet datasets.

### Structured chat normalization

SFT sources registered by `marin.datakit.sft_sources.all_sft_sources()` normalize
into Parquet with one conversation per row. Their `messages` column follows the
text subset of [OpenAI Harmony's message schema](https://github.com/openai/harmony):

```json
{
  "messages": [
    {"role": "user", "name": null, "content": [{"type": "text", "text": "What is 2 + 2?"}]},
    {"role": "assistant", "name": null, "channel": "analysis", "content": [{"type": "text", "text": "Add the numbers."}]},
    {"role": "assistant", "name": null, "channel": "final", "content": [{"type": "text", "text": "4"}]}
  ]
}
```

`messages` is a nested Parquet list of structs. Each message has `role`, optional
`name`, a list of text content parts, and optional `channel` and `recipient`.
Each item in `row["messages"]` can be loaded with
`openai_harmony.Message.from_dict`. There is no
conversation-level `content` column.

Source processing and normalization both write Parquet using the explicit Arrow schema
`marin.datakit.chat.CHAT_SCHEMA`. Optional message fields are nullable, so tool
recipients survive even when the first rows contain only ordinary conversation.
Sources extend this schema with their metadata columns and pass the same schema
to `write_parquet(schema=...)` and `normalize_chat_step(output_schema=...)`.
The `id` and `source_id` columns are strings; numeric source IDs are converted
to strings by `chat_document`.

Source adapters construct `openai_harmony.Message` objects directly. The shared
OpenAI-style source helper, `openai_chat_messages`, handles role aliases,
`reasoning_content`, tagged reasoning, and function-call dictionaries. It emits
`analysis` for reasoning, `commentary` for tool calls and their text preambles,
and `final` for other assistant text, using `ChatChannel`. Dataset-specific
adapters repair protocols such as SWE-ZERO's `THOUGHT:` and GLM's missing opening
reasoning delimiter before calling that helper. `chat_document` serializes Harmony
messages into the processed source artifact; there is no intermediate canonical
OpenAI-message artifact.

Protocol parsers live in `download/terminus.py` (JSON command batches) and
`download/opencode.py` (inline tool calls). Source adapters choose the parser
and handle dataset-specific prompt fields, such as recovering an empty first
user message from an `instruction` field. The parsers return OpenAI-style message
dictionaries and tool definitions for the source helper; they do not define
the shared Harmony contract or run the normalization pipeline.

The Parquet normalizer accepts only serialized Harmony messages. It validates
conversation structure, channels, and function handoffs, then hashes and deduplicates
records. It does not interpret reasoning delimiters or inline tool-call syntax.
Literal text such as `<think>` in a Harmony final answer stays literal text.

A function call is an assistant `commentary` message addressed to
`functions.<name>`, with JSON-object arguments in its text content. Its response
is a `tool` message named `functions.<name>`, on `commentary`, addressed to
`assistant`. The source helper resolves call IDs before discarding them. Parallel
calls retain source call order. All calls in a batch precede its observations,
which are reordered to the same call order, including repeated calls to the
same function.

The entire `chat_template_kwargs` column is a JSON string. Tool definitions are
available as `json.loads(row["chat_template_kwargs"])["tools"]`; normalization
fills in missing definitions from observed Harmony calls. Consumers supply these
definitions when rendering. Definition names must be unique and parameter schemas
must be JSON objects; normalization does not validate arguments against JSON Schema.

Conversations start with an optional system/developer prefix, then a nonempty
user message, and end with an assistant message. Consecutive user messages are
rejected. Adjacent assistant messages are expected: analysis, commentary, calls,
and final answers are separate messages within a turn. After `final`, the next
message must be a user message. Outstanding calls must receive observations
before conversation continues, but an entire final call batch may remain
unanswered. A reasoning-only assistant ending is also allowed.

The row ID hashes the normalized Harmony messages and template kwargs. Exact
deduplication is enabled by default. Metadata stays outside the conversation.
The normalization version changes with this schema so old SFT caches are not
reused as Harmony data. Rendering, tokenization, packing, and loss masks are
separate stages; this path does not change the existing Marin chat template.
When rendering SFT data with Harmony, explicitly disable automatic analysis
removal (`RenderConversationConfig(auto_drop_analysis=False)`).

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

For per-document stages (embed, per-doc classify) this falls out of reading source partitions 1:1 and reusing the input shard index in the output filename. For stages that shuffle globally (fuzzy/exact dedup, anything graph-structured), the shuffle key determines the output partition: normalize's exact dedup groups by `id`, so the post-shuffle shard index is itself the partition number.

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

## `lib/marin/src/marin/datakit/`

Core primitives — the reusable building blocks:

```
lib/marin/src/marin/datakit/
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
