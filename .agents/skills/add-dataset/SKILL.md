---
name: add-dataset
description: Register a named Hugging Face dataset for Marin by inspecting its schema and adding the appropriate experiments/datasets module.
---

# Register a Hugging Face dataset

Inspect the schema without downloading the full dataset:

```sh
uv run lib/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
```

For programmatic inspection:

```python
from marin.tools.get_hf_dataset_schema import get_schema

schema = get_schema(dataset_name="wikitext", config_name="wikitext-103-v1")
```

Use repo-managed dependencies. For a one-off inspection without a provisioned
environment, add `--with datasets --with pyyaml` to `uv run`.

If the result says a config is required, select one of `available_configs` and
retry with `--config_name`. Add `--trust_remote_code` only after inspecting the
dataset repository and accepting its code-execution boundary. The tool streams;
do not replace it with a full dataset download.

If the dataset cannot be found, stop and report the identifier, path, or access
failure instead of guessing a replacement.

Choose the text field from the reported schema. Prefer an exact `text` field,
then a field containing `text`, then another string field. Inspect `sample_row`
to verify the content; it may be empty for some datasets. The result also
reports `splits`, `text_field_candidates`, and `features`.

Add a leaf module under `experiments/datasets/` using the lazy builders in
`marin.experiment.data`:

- expose `<name>_dataset()` for one corpus;
- expose `<name>_datasets() -> dict[str, ...]` for a keyed family;
- for Hugging Face subsets, follow `experiments/datasets/nemotron.py` and return
  one keyed handle per subset.

Validate the selected config, splits, text mapping, and one sample before adding
tokenization or downstream experiment configuration.
