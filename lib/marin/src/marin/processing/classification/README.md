# Classification Processing

This directory contains classification-adjacent post-processing workflows:

- Deduplication in [`deduplication/`](./deduplication)
- Attribute-driven dataset filtering in [`consolidate.py`](./consolidate.py)

Decontamination has moved to [`marin.datakit.decon`](../../datakit/decon.py) — it consumes datakit-normalized Parquet and emits a co-partitioned Parquet attributes dataset (`id`, `partition_id`, `contaminated`, `max_overlap`).

## Deduplication

Run exact or fuzzy deduplication from Python by importing the helpers under [`deduplication/`](./deduplication).

## Consolidation

[`consolidate.py`](./consolidate.py) consumes attribute files and filters or rewrites documents. Supported filter types:

- `remove_spans`: remove text spans such as duplicate paragraphs
- `remove_docs`: drop whole documents when an attribute marks them as duplicates
