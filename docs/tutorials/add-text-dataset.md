# How to add a text dataset

## TL;DR

Inspect the source schema and license, pin an immutable revision, build a
`(download, normalize)` chain under `marin.datakit.download`, and add only
training-approved sources to `all_sources()`. Use a custom staging step when the
source is not already a text table. Preserve the source license and stable
document identifiers in every staged record.

## 1. Inspect the source without downloading it

For a Hugging Face dataset, inspect the available configs, splits, schema, and a
sample row:

```bash
uv run lib/marin/tools/get_hf_dataset_schema.py ORG/DATASET
uv run lib/marin/tools/get_hf_dataset_schema.py ORG/DATASET \
  --config_name CONFIG
```

Record the exact repository commit and license from the upstream dataset card.
Do not use a moving branch such as `main` as the Datakit revision. Confirm that
the dataset-level license covers the underlying documents. A repository license
can cover only the packaging or code while embedded papers retain separate
copyrights.

Classify the source before adding it to an active registry:

- `TRAINING_ALLOWED`: the terms permit the intended training and downstream
  model use.
- `EVAL_ONLY`: training would contaminate an evaluation or exceed the allowed
  use.
- `BLOCKED`: rights, access, or source identity remain unresolved.

Only `TRAINING_ALLOWED` sources belong in `all_sources()`.

## 2. Use the generic Hugging Face path for text tables

Use `hf_normalize_steps()` when each upstream row already contains the desired
training document:

```python
from marin.datakit.download.hf_simple_util import (
    NormalizationSchema,
    hf_normalize_steps,
)


def example_normalize_steps():
    return hf_normalize_steps(
        marin_name="example/text",
        hf_dataset_id="ORG/DATASET",
        revision="FULL_COMMIT_SHA",
        staged_path="raw/example-text",
        hf_urls_glob=("data/train-*.parquet",),
        text_field="content",
        file_extensions=(".parquet",),
        normalization_schema=NormalizationSchema.BARE,
    )
```

Set `text_field` from the inspected schema. Restrict `hf_urls_glob` to the
selected config so unrelated variants do not enter the same mixture component.
Return separate chains when variants need independent weights or ablations.

`NormalizationSchema.BARE` emits normalized IDs and text. Use `FULL` when
downstream work needs source columns that are safe and stable.

## 3. Add a staging transform for archives, books, or PDFs

A custom staging step must emit JSONL or Parquet rows with this minimum shape:

```json
{
  "id": "stable-source-document-id",
  "text": "training text",
  "source": "source-name",
  "license": "SPDX or exact upstream license",
  "provenance": {
    "source_url": "https://...",
    "revision": "immutable revision"
  }
}
```

Use `IngestionSourceManifest` for the source URL, license, usage policy,
transform name, and format-specific metadata. Write `metadata.json` with
`write_ingestion_metadata_json()` after materialization. The manifest content
fingerprint belongs in the staging `StepSpec.hash_attrs`; a source or transform
change then produces a new artifact identity.

For books, emit one record per stable chapter or section instead of one record
per file representation. For a document released as PDF, page PNGs, and JSON
sidecars, select one canonical text representation and use the upstream
document ID for all representations. This prevents the same page from entering
the corpus several times before global deduplication.

Preserve attribution fields required by the license. Do not infer training
rights from public download access.

## 4. Materialize before active registration

Keep a new pipeline outside `all_sources()` until its normalized and Harrier
artifacts have completed. A task-specific candidate catalog can expose the
pipeline to a ferry without changing the production source set. Adding an
unbuilt source to `all_sources()` breaks the checked-in hero-data manifests.

The science candidate catalog can be inspected or run through the standard
trigger script:

```bash
uv run python -m experiments.datakit.scripts.trigger_sources \
  --catalog science-candidates --list-pending
```

After the artifacts exist, import the chain factory in
`marin.datakit.sources` and add a registry row with an evidence-based rough
token count:

```python
("example/text", example_normalize_steps, 1.25),
```

The count is in billions of Marin-tokenizer tokens. Label an estimate in a
nearby comment and replace it after materializing and tokenizing the source.
Do not add a chat-only source to `all_sources()` to obtain a weight. Chat
sources and their explicit weights belong in `all_sft_sources()`.

## 5. Validate behavior

Test source-specific transformations with small in-memory fixtures. Assert on
emitted text, stable IDs, provenance, license preservation, and policy gates.
Do not add tests that only assert registry membership.

Run the narrow tests first, then the affected safe suite:

```bash
uv run pytest tests/datakit/download/test_<source>.py -q
./infra/pre-commit.py --all-files --fix
uv run pyrefly check
uv run --no-project infra/ci/run_tests.py
```

For a large source, use the ferry smoke-file cap documented in
`marin.datakit.normalize` instead of downloading the full corpus from a local
development machine.
