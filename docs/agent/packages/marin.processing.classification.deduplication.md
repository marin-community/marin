## Overview

This package implements three dedup strategies—exact-paragraph, exact-document, and fuzzy-document—all sharing a common Zephyr-based distributed pipeline pattern. Each entry point collects input files, assigns integer file indices, runs a Zephyr `Dataset` pipeline to hash/signature documents, groups by hash/bucket, annotates duplicates, then writes **attribute sidecar files** (not filtered corpora) in Vortex format marking which records are duplicates.

Fuzzy dedup adds a two-phase flow: first compute MinHash LSH buckets (yielding `CCInput` records), then run `connected_components` using the Hash-to-Min algorithm to find near-duplicate clusters. The CC output files are then loaded to annotate `is_dup` where `component_id != id_norm` (i.e., not the canonical node). Default MinHash params (`num_perms=286, num_bands=26, ngram_size=5`) encode a ~0.5 Jaccard similarity threshold with 26 bands of 11 rows each—these are tuned for web-scale text and should not be changed casually. `max_iterations=10` for CC is a safety cap; convergence is typically reached in 3–6 iterations.

## API

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
Exact full-document dedup via xxHash3-128; writes `is_dup` attribute sidecars. `exact.py:186`

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
Exact paragraph dedup; writes `dup_spans` attribute sidecars instead of per-doc booleans. `exact.py:50`

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
MinHash LSH + connected components fuzzy dedup; calls `connected_components` internally. `fuzzy.py:27`

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
Iterative Hash-to-Min CC algorithm; returns `(converged, list_of_vortex_paths)`. `connected_components.py:51`

```python
from marin.processing.classification.deduplication.dedup_commons import group_files

def group_files(files: list[str], num_groups: int) -> list[list[str]]: ...
```
Round-robin deterministic file grouping to cap Zephyr shard count. `dedup_commons.py:53`

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
Factory returning a `group_by` reducer shared by exact-doc and fuzzy-doc dedup. `dedup_commons.py:151`

```python
from marin.processing.classification.deduplication.dedup_commons import DedupMode

class DedupMode(StrEnum):
    EXACT_PARAGRAPH = auto()
    EXACT_DOCUMENT = auto()
    FUZZY_DOCUMENT = auto()
```
`dedup_commons.py:24`

## Gotchas

- **Output is sidecars, not filtered files.** All three entry points write attribute files marking duplicates; downstream steps must apply these to actually remove records.
- **`num_perms` must be divisible by `num_bands`** — `dedup_fuzzy_document` raises `ValueError` immediately if not. The default 286/26 satisfies this (26×11=286).
- **`connected_components` returns paths, not data.** The second element of the tuple is a list of Vortex file paths to load; pass them to `Dataset.from_list(...).load_vortex()`.
- **Worker RAM default is 32 GB** — appropriate for large batches but may be over-provisioned for small datasets; fuzzy dedup's Rust MinHash thread pool may consume extra CPU beyond the `cpu=1` default, so increase `cpu` (e.g. to 5) if workers are CPU-throttled.
- **Canonical record selection is the lexicographic minimum `id`** (sort order in `group_by`), not insertion order — ensure document IDs are stable across runs for reproducible dedup.
