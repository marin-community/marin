# Classification Processing

This directory now contains the classification-adjacent post-processing workflows that remain in Marin:

- Deduplication in [`deduplication/`](./deduplication)
- Decontamination in [`decon.py`](./decon.py)
- Attribute-driven dataset filtering in [`consolidate.py`](./consolidate.py)

## Deduplication

Run exact or fuzzy deduplication from Python by importing the helpers under [`deduplication/`](./deduplication), or use the quickstart config for decontamination-style attribute generation:

```bash
uv run python -m marin.processing.classification.decon \
  --config_path lib/marin/src/marin/processing/classification/config/quickstart_decontaminate.yaml
```

## Decontamination

[`decon.py`](./decon.py) builds bloom filters and marks duplicate spans or train-test overlap attributes. The quickstart config lives at [`config/quickstart_decontaminate.yaml`](./config/quickstart_decontaminate.yaml).

## Consolidation

[`consolidate.py`](./consolidate.py) consumes attribute files and filters or rewrites documents. Supported filter types:

- `remove_spans`: remove text spans such as duplicate paragraphs
- `remove_docs`: drop whole documents when an attribute marks them as duplicates
