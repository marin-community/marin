# Recipe: Dataset Schema Inspection

## Overview
This recipe guides agents and humans in inspecting Hugging Face dataset schemas using Marin's schema inspection tool. The inspection is the first step toward exposing a dataset for training. After capturing the schema, register an `ExecutorStep` so the dataset can be downloaded in Marin pipelines. For simple datasets, add the step to [`experiments/pretraining_datasets/__init__.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/__init__.py) (see the `fineweb_edu` entry). If the dataset is multipart or more complex, create a dedicated file (e.g., [`experiments/pretraining_datasets/nemotron.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/nemotron.py)) and add the step there instead. Datasets with Hugging Face–exposed subsets or splits should pattern match on [`experiments/pretraining_datasets/nemotron.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/nemotron.py), defining separate steps for each subset.

## Prerequisites
- Install the [Hugging Face `datasets` library](https://huggingface.co/docs/datasets/) (e.g., `pip install datasets`).
- For YAML output, install `pyyaml` (e.g., `pip install pyyaml`).
- Ensure access to the dataset: Hugging Face Hub ID, local path, or other supported formats.
- Use `uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py` for direct execution, or install Marin with `uv sync` for CLI access.

## Guidelines for Humans

### Command Line Usage
```sh
uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
```

### Python Import
```python
from marin.tools.get_hf_dataset_schema import sample_dataset, estimate_tokens
result = sample_dataset("roneneldan/TinyStories", n=5)  # returns raw row dicts in result["samples"]
```

### Common Workflows
1. **Basic inspection**: `uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py roneneldan/TinyStories`
2. **Token estimation**: `uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py roneneldan/TinyStories --text_field text --estimate_tokens`

## Rules for Agents

### 1. Config Handling
- ALWAYS check if a dataset requires a config first
- If configs are required, the tool returns:
  ```json
  {
    "error": "Config name is required.",
    "available_configs": ["config1", "config2", ...]
  }
  ```
- Select an appropriate config based on the available options

### 2. Text Field Selection
- `sample_dataset()` returns raw row dicts — examine them to identify the right fields
- Prioritize fields named exactly 'text'
- Fall back to fields containing 'text' in their name
- Consider string-type fields if no obvious text field exists

### 3. Error Handling
- Handle common error cases:
  - Missing config
  - Dataset not found
  - Remote code execution required
- Retry with appropriate parameters (e.g., --trust_remote_code)

### 4. Performance
- Tool uses streaming to avoid downloading full datasets
- Expect quick responses for schema info
- Sample row might be empty for some datasets

## Output Format

The tool returns a JSON object with:
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

## Examples

### Sample a Dataset
```sh
$ uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py roneneldan/TinyStories --n 3
{
  "dataset": "roneneldan/TinyStories",
  "config": "default",
  "split": "train",
  "schema": ["text"],
  "num_rows": 2119719,
  "seed": 42,
  "num_samples": 3,
  "samples": [
    {"text": "Once upon a time..."},
    {"text": "There was a little girl..."},
    {"text": "One day, a boy found..."}
  ]
}
```

### Estimate Tokens

Estimate token counts by sampling random rows, tokenizing with Llama 3, and extrapolating by total row count:

```sh
$ uv run lib/marin/src/marin/tools/get_hf_dataset_schema.py common-pile/pubmed_filtered \
    --text_field text --estimate_tokens
```

## Next Steps
Once the schema is inspected, the quality is reviewed, and the dataset is registered (for example in [`experiments/pretraining_datasets/__init__.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/__init__.py) or in a dedicated file), the goal is to cargo-cult existing dataset configs for tokenization:
- Apply transformations (e.g., field mapping).
- Find similar dataset configurations in Marin's existing experiments.
- Copy and adapt tokenization configs from similar datasets.
- Run ablations or trials.

These will be detailed in expanded recipes or subsequent PRs.

## See Also
- [get_hf_dataset_schema.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/tools/get_hf_dataset_schema.py) — dataset inspection tool (sample, estimate tokens)
- [Hugging Face datasets documentation](https://huggingface.co/docs/datasets/)
