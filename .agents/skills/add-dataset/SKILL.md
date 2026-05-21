---
name: add-dataset
description: Register or inspect a Hugging Face dataset for Marin pipelines.
---

# Skill: Dataset Schema Inspection and Registration

## Overview
Inspect a Hugging Face dataset schema with Marin's schema inspection tool, then
register an `ExecutorStep` so the dataset can be downloaded in Marin pipelines.
- Simple datasets: add the step to
  [`experiments/pretraining_datasets/__init__.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/__init__.py)
  (see the `fineweb_edu` entry).
- Multipart/complex datasets: create a dedicated file (e.g.
  [`experiments/pretraining_datasets/nemotron.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/nemotron.py))
  and add the step there.
- Datasets with HF-exposed subsets/splits: pattern-match on `nemotron.py`,
  defining a separate step per subset.

## Prerequisites
- Prefer repo-managed dependencies over ad hoc `pip install`.
- For repeated work in a checked-out Marin repo, install the synced environment
  with `uv sync --all-packages`, then run
  `uv run lib/marin/tools/get_hf_dataset_schema.py`.
- For one-off schema inspection without a provisioned environment, use
  ephemeral deps:
  `uv run --with datasets --with pyyaml lib/marin/tools/get_hf_dataset_schema.py ...`
- Ensure access to the dataset (Hugging Face Hub ID, local path, or other
  supported format).

## Usage

Command line:
```sh
uv run lib/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
```

Python import:
```python
from marin.tools.get_hf_dataset_schema import get_schema
schema = get_schema(dataset_name="wikitext", config_name="wikitext-103-v1")
```

## Rules
1. **Config handling**: always check whether a dataset requires a config first.
   If required, the tool returns
   `{"error": "Config name is required.", "available_configs": [...]}` — select
   an appropriate config from the list and retry with `--config_name`.
2. **Text field selection**: prioritize a field named exactly `text`; fall back
   to fields containing `text`; consider string-type fields if no obvious text
   field exists. Examine `sample_row` to verify field contents.
3. **Error handling**: handle missing config, dataset not found, and remote
   code execution required. Retry with appropriate parameters (e.g.
   `--trust_remote_code`).
4. **Performance**: the tool streams to avoid full downloads; expect quick
   responses. `sample_row` may be empty for some datasets.

## Output Format
The tool returns a JSON object:
```json
{
  "splits": ["train", "validation", ...],
  "text_field_candidates": ["text", "content", ...],
  "features": {
    "text": "string",
    "label": "int64",
    ...
  },
  "sample_row": {
    "text": "Example content...",
    ...
  }
}
```

## Example: dataset requiring a config
```sh
$ uv run lib/marin/tools/get_hf_dataset_schema.py wikitext
{
  "error": "Config name is required.",
  "available_configs": ["wikitext-103-raw-v1", "wikitext-103-v1", ...]
}

$ uv run lib/marin/tools/get_hf_dataset_schema.py wikitext --config_name wikitext-103-v1
{
  "splits": ["train", "validation", "test"],
  "text_field_candidates": ["text"],
  "features": {"text": "string"},
  "sample_row": {"text": "Article content..."}
}
```

For datasets needing remote code, add `--trust_remote_code`.

## Next Steps
Once the schema is inspected and the dataset is registered, cargo-cult existing
dataset configs for tokenization:
- Apply transformations (e.g. field mapping).
- Estimate token counts and file sizes.
- Find similar dataset configurations in Marin's existing experiments.
- Copy and adapt tokenization configs from similar datasets.
- Run ablations or trials.

## See Also
- `lib/marin/tools/get_hf_dataset_schema.py`
- [Hugging Face datasets documentation](https://huggingface.co/docs/datasets/)
</content>
