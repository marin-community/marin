---
name: add-dataset
description: Register or inspect a Hugging Face dataset for Marin pipelines.
---

# Add a Dataset

Inspect a Hugging Face dataset, then add a lazy builder under
`experiments/datasets/`. A leaf exposes `<name>_dataset()` for one corpus or
`<name>_datasets() -> dict[str, ...]` for a keyed family. Copy the closest
existing module (`svg.py` for one corpus, `nemotron.py` for subsets/splits).

## Inspect

Use the synced environment when available; otherwise use ephemeral dependencies:

```sh
uv run lib/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
uv run --with datasets --with pyyaml lib/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
```

Python callers can use `get_schema(dataset_name=..., config_name=...)` from
`marin.tools.get_hf_dataset_schema`.

If the tool reports `Config name is required`, choose from
`available_configs` and retry with `--config_name`. Use `--trust_remote_code`
only when the dataset requires it. Check `splits`, `features`, and
`sample_row`; prefer a field named exactly `text`, then text-containing fields,
then a verified string field. The tool streams data and may return an empty
sample row.

## Register

- Add the smallest appropriate module in `experiments/datasets/` using
  `marin.experiment.data` lazy builders.
- For HF subsets/splits, return one handle per subset with stable keys.
- Match existing tokenization/field-mapping conventions; estimate token counts
  and size, then run the relevant local tests or trial.

Do not install ad hoc dependencies when the repo environment provides them.
See `lib/marin/tools/get_hf_dataset_schema.py` and existing dataset modules for
the API; do not duplicate their builder patterns.
