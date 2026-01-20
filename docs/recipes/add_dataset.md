# Recipe: Dataset Schema Inspection

## Overview
This recipe guides agents and humans in inspecting Hugging Face dataset schemas using Marin's schema inspection tool. The inspection is the first step toward exposing a dataset for training. After capturing the schema, register an `ExecutorStep` so the dataset can be downloaded in Marin pipelines. For simple datasets, add the step to [`experiments/pretraining_datasets/__init__.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/__init__.py) (see the `fineweb_edu` entry). If the dataset is multipart or more complex, create a dedicated file (e.g., [`experiments/pretraining_datasets/nemotron.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/nemotron.py)) and add the step there instead. Datasets with Hugging Face–exposed subsets or splits should pattern match on [`experiments/pretraining_datasets/nemotron.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/nemotron.py), defining separate steps for each subset.

## Prerequisites
- Install the [Hugging Face `datasets` library](https://huggingface.co/docs/datasets/) (e.g., `pip install datasets`).
- For YAML output, install `pyyaml` (e.g., `pip install pyyaml`).
- Ensure access to the dataset: Hugging Face Hub ID, local path, or other supported formats.
- Use `uv run lib/marin/tools/get_hf_dataset_schema.py` for direct execution, or install Marin with `uv sync` for CLI access.

## Guidelines for Humans

### Command Line Usage
```sh
uv run lib/marin/tools/get_hf_dataset_schema.py <dataset_name> [options]
```

### Python Import
```python
from marin.tools.get_hf_dataset_schema import get_schema
schema = get_schema(dataset_name="wikitext", config_name="wikitext-103-v1")
```

### Common Workflows
1. **Basic inspection**: `uv run lib/marin/tools/get_hf_dataset_schema.py roneneldan/TinyStories`
2. **Multi-config dataset**: First try without config, then use the suggested config from the error
3. **Remote code datasets**: Add `--trust_remote_code` flag when needed

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
- Prioritize fields named exactly 'text'
- Fall back to fields containing 'text' in their name
- Consider string-type fields if no obvious text field exists
- Examine sample_row to verify field contents

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

### Basic Usage
```sh
$ uv run lib/marin/tools/get_hf_dataset_schema.py roneneldan/TinyStories
{
  "splits": ["train", "validation"],
  "text_field_candidates": ["text"],
  "features": {"text": "string"},
  "sample_row": {"text": "Once upon a time..."}
}
```

### Dataset with Config
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

### Dataset with Remote Code
```sh
$ uv run lib/marin/tools/get_hf_dataset_schema.py c4 --config_name en --trust_remote_code
{
  "splits": ["train", "validation"],
  "text_field_candidates": ["text"],
  "features": {"text": "string", "url": "string", "timestamp": "string"},
  "sample_row": {"text": "Web content..."}
}
```

## Example Agent Prompt
To test with an AI agent:
> "Follow the dataset schema inspection recipe. Inspect the schema for 'roneneldan/TinyStories' using the get_hf_dataset_schema tool. Apply all rules and output the full schema as JSON."

## Testing and Validation
- **Human Testing:** Run the examples above and verify the output matches your expectations. Cross-check with Hugging Face's dataset viewer for accuracy.
- **Agent Testing:** Input this recipe into tools like Claude or Cursor, using the example prompt. Report issues or successes in the Marin Discord (#coding-roombas) or GitHub issues.
- **Integration Testing:** Ensure the schema output integrates with Marin's pipelines (e.g., feed into token estimation tools in future steps).

## Next Steps
Once the schema is inspected and the dataset is registered (for example in [`experiments/pretraining_datasets/__init__.py`](https://github.com/marin-community/marin/blob/main/experiments/pretraining_datasets/__init__.py) or in a dedicated file), the goal is to cargo-cult existing dataset configs for tokenization:
- Apply transformations (e.g., field mapping).
- Estimate token counts and file sizes.
- Find similar dataset configurations in Marin's existing experiments.
- Copy and adapt tokenization configs from similar datasets.
- Run ablations or trials.

These will be detailed in expanded recipes or subsequent PRs.

## See Also
- [get_hf_dataset_schema.py on GitHub](https://github.com/marin-community/marin/blob/main/lib/marin/tools/get_hf_dataset_schema.py)
- [Hugging Face datasets documentation](https://huggingface.co/docs/datasets/)
