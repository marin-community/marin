## Overview

This package implements three deduplication modes for large document corpora, all built on the Zephyr distributed dataflow engine. The pipeline always produces **attribute sidecar files** (Vortex format) marking which documents/paragraphs are duplicates — it does **not** filter the source data in place.

**Exact dedup** hashes full document text (`dedup_exact_document`) or individual paragraphs (`dedup_exact_paragraph`) using xxHash-128 and groups identical hashes, keeping the lexicographically first document ID as canonical. **Fuzzy dedup** (`dedup_fuzzy_document`) uses MinHash LSH: documents are shingled into 5-grams (small enough to catch near-duplicates despite minor edits), hashed into 286 permutations split across 26 bands (~11 rows/band). This band/row configuration targets a Jaccard similarity threshold of roughly 0.8 — 26 bands maximizes sensitivity at that threshold given the 286-perm budget. The seed=42 default ensures reproducible signatures. After LSH, `connected_components` runs the Hash-to-Min algorithm (iterative label propagation) to find duplicate clusters; documents whose `component_id != id_norm` are marked as duplicates.

Call `dedup_exact_document` / `dedup_exact_paragraph` / `dedup_fuzzy_document` directly — they are the entry points. Each returns a stats dict. `connected_components` and the helpers (`group_files`, `make_document_dedup_aggregator`, `finalize_dedup`) are building blocks used internally but exposed for custom pipelines.

---

## API

### Entry Points

```python
from marin.processing.classification.deduplication.exact import dedup_exact_document

def dedup_exact_document(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> dict: ...
```
Full-document exact dedup via xxHash-128; writes `{output_path}/data/` Vortex sidecars with `dup_doc: True` attributes. `exact.py:186`

```python
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

def dedup_exact_paragraph(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> dict: ...
```
Paragraph-level exact dedup; sidecars carry `dup_spans` lists instead of a boolean flag. `exact.py:50`

```python
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

def dedup_fuzzy_document(
    *,
    input_paths: str | list[str],
    output_path: str,
    text_field: str = "text",
    filetypes: list[str] | None = None,
    fuzzy_minhash_num_perms: int = 286,
    fuzzy_minhash_num_bands: int = 26,
    fuzzy_minhash_ngram_size: int = 5,
    fuzzy_minhash_seed: int = 42,
    max_parallelism: int,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
) -> dict: ...
```
Near-duplicate document dedup via MinHash LSH + connected components; writes CC metadata under `{output_path}/metadata/cc/` and sidecars under `{output_path}/data/`. `fuzzy.py:27`

### Graph / CC Primitives

```python
from marin.processing.classification.deduplication.connected_components import connected_components

def connected_components(
    ds: Dataset[CCInput],
    ctx: ZephyrContext,
    *,
    output_dir: str,
    max_iterations: int = 10,
    preserve_singletons: bool = True,
) -> tuple[bool, Sequence[str]]: ...
```
Hash-to-Min iterative label propagation; returns `(converged, list_of_vortex_paths)`. `connected_components.py:51`

### Shared Utilities

```python
from marin.processing.classification.deduplication.dedup_commons import group_files

def group_files(files: list[str], num_groups: int) -> list[list[str]]: ...
```
Deterministic round-robin file bucketing to cap Zephyr shard count. `dedup_commons.py:53`

```python
from marin.processing.classification.deduplication.dedup_commons import make_document_dedup_aggregator

def make_document_dedup_aggregator(
    *,
    idx_to_path: dict[int, str],
    input_paths: str | list[str],
    output_path: str,
    counter_prefix: str,
) -> Callable[[int, Iterator[dict]], dict]: ...
```
Returns a `group_by` reducer closure; shared by exact-doc and fuzzy-doc pipelines. `dedup_commons.py:151`

```python
from marin.processing.classification.deduplication.dedup_commons import finalize_dedup

def finalize_dedup(
    shard_results: list[dict],
    mode: DedupMode,
    method: str,
    level: str,
) -> dict: ...
```
Aggregates counters, logs summary, finalizes W&B run, returns stats dict. `dedup_commons.py:203`

### Configuration

```python
from marin.processing.classification.deduplication.dedup_commons import DedupMode

class DedupMode(StrEnum):
    EXACT_PARAGRAPH = auto()
    EXACT_DOCUMENT = auto()
    FUZZY_DOCUMENT = auto()
```
`dedup_commons.py:24`

---

## Gotchas

- **`fuzzy_minhash_num_perms` must be divisible by `fuzzy_minhash_num_bands`** — `dedup_fuzzy_document` raises `ValueError` immediately if not. The defaults (286, 26) are not evenly divisible by round numbers; verify before changing either independently.
- **`max_parallelism` is required (no default)** on all three entry points. Omitting it is a hard `TypeError`; there is no safe fallback.
- **Output is attribute sidecars, not filtered data.** All three functions write files marking duplicates; they do not produce a deduplicated corpus directly. Downstream steps must read the `dup_doc`/`dup_spans` attributes to filter.
- **`connected_components` returns `(converged, paths)`, not a dataset.** If `converged=False` the algorithm hit `max_iterations=10` without stabilizing; the output is still usable but may over-retain duplicates in very large or densely connected clusters.
- **Worker `cpu` sizing for fuzzy dedup**: the `dupekit` MinHash stage spawns a native Rust thread pool. Default `ResourceConfig(cpu=1)` may starve it — set `worker_resources=ResourceConfig(cpu=5, ...)` for throughput-sensitive runs (per `fuzzy.py` docstring).
