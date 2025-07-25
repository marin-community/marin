# Recipe: Adding a New Pretraining Dataset to Marin

## Overview
This recipe provides a structured guide for humans and AI agents to add new pretraining datasets to Marin. It emphasizes a step-by-step approach to ensure compatibility with Marin's data pipelines, such as tokenization and training workflows. Starting with schema inspection helps identify splits, subsets, and text fields, making the process more efficient and "roomba-friendly" (i.e., amenable to automation and junior contributors). These instructions also facilitate human onboarding by enforcing clear, reproducible practices.

> **Reference:** Derived from community discussions on agent-friendly workflows, including @dlwh's checklist for dataset addition.

## Prerequisites
- Install the [Hugging Face `datasets` library](https://huggingface.co/docs/datasets/) (e.g., `pip install datasets`).
- For YAML output, install `pyyaml` (e.g., `pip install pyyaml`).
- Ensure access to the dataset: Hugging Face Hub ID, local path, or other supported formats.
- Marin installed in editable mode (`pip install -e .`) for CLI access, or run the script standalone.

## Step 1: Inspect Dataset Schema
Inspect the dataset to determine its structure, including splits, subsets, and potential text fields. This ensures alignment with Marin's pretraining requirements, such as text-based tokenization (e.g., via `default_tokenize` in experiments).

Use the `get_hf_dataset_schema` tool for automated analysis:

### Command-Line Usage
```sh
python -m marin.tools.get_hf_dataset_schema <dataset-name-or-path> [options]
```
- **Examples:**
  - Hugging Face dataset:
    ```sh
    python -m marin.tools.get_hf_dataset_schema roneneldan/TinyStories
    ```
  - Local path (e.g., directory or file):
    ```sh
    python -m marin.tools.get_hf_dataset_schema /path/to/local/dataset
    ```
  - Focus on a specific split:
    ```sh
    python -m marin.tools.get_hf_dataset_schema roneneldan/TinyStories --split train
    ```
  - Output as YAML:
    ```sh
    python -m marin.tools.get_hf_dataset_schema roneneldan/TinyStories --output yaml
    ```

### Import as a Function
For programmatic use:
```python
from marin.tools.get_hf_dataset_schema import get_schema
schema = get_schema("roneneldan/TinyStories", split="train")  # Optional: split, output_format
print(schema)  # Or json.dumps(schema) for JSON
```

### Output Explanation
The tool returns a dictionary (serialized as JSON or YAML) with:
- `splits`: List of available splits (e.g., `["train", "validation"]`).
- `subsets`: List of subsets, if applicable (often empty; e.g., language-specific variants).
- `text_field_candidates`: Potential text-containing keys, prioritized by common names like "text", "content", or "body" (based on string dtype or keyword matches).
- `sample_row`: A representative row from the first split (or specified split) for preview.
- `error`: Any loading issues (e.g., "Dataset not found").

To parse in code: Use `json.loads()` for JSON output.

#### Example Output (JSON)
```json
{
  "splits": ["train", "validation"],
  "subsets": [],
  "text_field_candidates": ["text"],
  "sample_row": {
    "text": "One day, a little girl named Lily found a toy in her room..."
  }
}
```

### Edge Cases and Advanced Handling
- **Local vs. Hugging Face Datasets:** For local paths (e.g., Parquet, CSV, or directories), pass the path directly. The tool uses `load_dataset` with appropriate builders (e.g., "parquet" for .parquet files). Convert non-standard formats (e.g., custom JSON) to supported types via libraries like `pandas` beforehand if needed.
- **Multi-Modal Datasets:** If non-text fields (e.g., images or audio) are present, they are noted in features but ignored for text-focused pretraining. Focus on `text_field_candidates` and flag for filtering if irrelevant.
- **Large or Streaming Datasets:** For massive datasets, enable streaming internally (via `streaming=True` in advanced calls) to avoid full loading. Example: Add `--streaming` flag if implemented in future updates.
- **No Text Field Found:** Review `text_field_candidates` for alternatives. If none, prepare for transformations in subsequent steps (e.g., concatenating fields).
- **Errors:** Common issues include authentication (for private datasets) or format mismatches. Retry with explicit builder (e.g., `load_dataset(path, 'csv')`).

## Rules for Agents
These rules provide explicit guidance for AI agents (e.g., Codex, Claude, Cursor) to ensure consistent, reliable execution:
- **Rule 1:** Prioritize the "text" field if present; otherwise, select the top candidate from `text_field_candidates` based on string dtype and relevance.
- **Rule 2:** Always output the schema as JSON for easy parsing and downstream tooling.
- **Rule 3:** If no text field is found or candidates are ambiguous, flag for human review and suggest a transformation (e.g., map "content" to "text").
- **Rule 4:** Use the sample row to verify data format; if it contains non-text (e.g., binary), recommend skipping or filtering.
- **Rule 5:** For datasets with subsets, evaluate each independently and merge schemas if they share structures.

## Example Agent Prompt
To test with an AI agent:
> "Follow Step 1 of the add_dataset.md recipe. Inspect the schema for 'roneneldan/TinyStories' using the get_hf_dataset_schema tool. Apply all rules and output the full schema as JSON."

## Testing and Validation
- **Human Testing:** Run the examples above and verify the output matches your expectations. Cross-check with Hugging Face's dataset viewer for accuracy.
- **Agent Testing:** Input this recipe into tools like Claude or Cursor, using the example prompt. Report issues or successes in the Marin Discord (#coding-roombas) or GitHub issues.
- **Integration Testing:** Ensure the schema output integrates with Marin's pipelines (e.g., feed into token estimation tools in future steps).

## Next Steps (Teaser)
Once the schema is inspected:
- Apply transformations (e.g., field mapping).
- Estimate token counts and file sizes.
- Cargo-cult existing dataset configs for tokenization.
- Run ablations or trials.

These will be detailed in expanded recipes or subsequent PRs.

## See Also
- [get_hf_dataset_schema.py](../marin/tools/get_hf_dataset_schema.py) (or view on [GitHub](https://github.com/marin-community/marin/blob/main/marin/tools/get_hf_dataset_schema.py)).
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/).
- Marin's [data processing tutorials](../docs/tutorials/data.md) for context.

## Version History
- **v0.1 (July 24, 2025):** Initial draft focused on schema inspection. Future updates will cover full dataset addition workflow.

Feedback on this recipe is welcome—submit issues or PRs to improve agent compatibility and clarity.