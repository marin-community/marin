# How to add a chat dataset

Add a chat dataset so Marin can validate and deduplicate its conversations,
preserve reasoning and tool interactions, and render them consistently for
training. You write a source converter and register its processing steps.
Marin stores conversations in Harmony, a structured message format with roles,
channels for reasoning and answers, and tool recipients.

Complete the [installation](installation.md) first. Run the commands below from
the repository root. The example assumes a Hugging Face dataset containing JSONL
files with string-valued `prompt` and `answer` fields.

## 1. Create the source module

Create `lib/marin/src/marin/datakit/download/example.py`. Use
[superior_reasoning.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/superior_reasoning.py)
as the starter: copy its imports, `HF_DATASET_ID`, `HF_REVISION`, `STAGES`,
`row_to_chat_doc`, `transform_chat`, and `superior_reasoning_chat_normalize_steps`.
Omit the functions for plain-text processing.

Adapt these parts:

- Set `HF_DATASET_ID` to your dataset and `HF_REVISION` to a fixed commit revision.
- Set `STAGES` to the JSONL filenames to download. `transform_chat` reads these
  same filenames; change its reader if your source uses another file format.
- Rename the factory to `example_chat_normalize_steps`. Replace the
  `superior-reasoning` names in the factory and `ZephyrContext` with your source name.
- Set the processed step's `hash_attrs` version to `"v1"`. Bump it whenever your
  conversion changes to rebuild cached outputs.

A `StepSpec` describes a processing step and its dependencies. The copied factory
connects the pinned download to `transform_chat`, then to chat normalization.
`transform_chat` uses Zephyr to read rows, apply your converter with `flat_map`,
and write the returned documents as Parquet using `CHAT_SCHEMA`.

## 2. Adapt the row converter

Inspect real records, including failed and incomplete conversations. Locate the
original user request, assistant response, and any reasoning or tool interactions.
Replace the copied `row_to_chat_doc` with:

```python
from zephyr import counters

from marin.datakit.download.rollout_transforms import checked_openai_chat_document


def row_to_chat_doc(row: dict) -> list[dict]:
    prompt = row.get("prompt")
    answer = row.get("answer")
    if not prompt or not answer:
        counters.pipeline.update_counter("example/chat/missing_text_filtered", 1)
        return []
    return checked_openai_chat_document(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        HF_DATASET_ID,
        counter_prefix="example/chat",
    )
```

The helper converts OpenAI-style messages to a Harmony document, assigning the
channels used by the normalizer. It returns a one-document list on success and
counts and drops conversion failures (`ValueError` or `UnicodeError`). Count
intentional source filters too, as above.

Preserve recorded text and message order. Combine adjacent user messages in your
converter. Every conversation needs its original user request and must end with
an assistant answer or tool call. Images and audio are unsupported. Normalization
checks conversation structure and removes exact duplicates; it does not establish
whether answers are correct.

For sources with reasoning or tools:

- Put separately recorded reasoning in the assistant's `reasoning_content` field.
- Preserve tool call IDs and their matching replies. Pass recorded function
  definitions through `chat_template_kwargs={"tools": tools}`. Exclude records
  whose original prompts or required tool definitions cannot be recovered.
- Use [coderforge.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/coderforge.py)
  for a tool example, or
  [terminus.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/terminus.py)
  for terminal-agent conversations that encode commands as assistant text.

The full output contract is enforced in
[chat_normalize.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/chat_normalize.py).
To retain extra fields such as reward or teacher, pass them to the document helper,
extend `CHAT_SCHEMA` with their Arrow fields, and use that schema for both the
Parquet writer and `normalize_chat_step(output_schema=...)`.

## 3. Register the source

In `lib/marin/src/marin/datakit/sft_sources.py`, add the import:

```python
from marin.datakit.download.example import example_chat_normalize_steps
```

Inside `all_sft_sources()`, add this entry to the existing `rows` list:

```python
("example", example_chat_normalize_steps),
```

For a chat-only source, add its estimated token count in billions to `token_counts`
before the function's final `return`. For example, 250 million tokens becomes:

```python
token_counts["example"] = 0.25
```

Replace `0.25` with your source's estimate. This count supplies a rough mixture
weight; it is not a sampling probability. Use a published token count, or tokenize
a representative sample and multiply the mean count by the number of records.
Record your estimate's method beside the entry. Sources already in `all_sources()`
inherit their counts. Keep chat-only sources out of that pretraining registry.

The registry creates a `DatakitChatSource` and adds rendering and text
normalization steps automatically.

## 4. Verify the output

First, put a few representative source records in local JSONL files under
`scratch/chat-fixture/`, using the filenames in `STAGES`. Include any format
quirks and intentional drops. Save this as `scratch/check_chat_source.py`:

```python
from pathlib import Path
from pprint import pprint

import pyarrow.parquet as pq

from marin.datakit.chat_normalize import normalize_chat_to_parquet
from marin.datakit.download.example import transform_chat

transform_chat("scratch/chat-fixture", "scratch/chat-processed")
artifact = normalize_chat_to_parquet(
    input_path="scratch/chat-processed",
    output_path="scratch/chat-normalized",
)
print(artifact.main_output_dir)
print(artifact.counters)
parquet_file = next(Path(artifact.main_output_dir).rglob("*.parquet"))
batch = next(pq.ParquetFile(parquet_file).iter_batches(batch_size=1))
pprint(batch.to_pylist()[0]["messages"])
```

Run `uv run python scratch/check_chat_source.py`. Use fresh output directories
when rerunning after a converter change: the copied writer skips existing files.
If you extended `CHAT_SCHEMA`, pass it as `output_schema` to the normalizer here too.

Inspect the output Parquet and compare messages with the input records. Check that
requests, reasoning, tool definitions, and replies survived. Review filter,
quarantine, and duplicate counts. Quarantine counters count dropped rows; no
separate quarantine file is saved. Reproduce unexpected drops with individual
fixture records. Normalization fails if more than 5% of its input records have
empty messages or fail validation; investigate those failures before proceeding.

Add regression tests for your source's format quirks and intentional drops under
`tests/datakit/download/`, following [TESTING.md](https://github.com/marin-community/marin/blob/main/TESTING.md).
Run the checks from the repository root:

```bash
uv run --no-project infra/ci/run_tests.py
./infra/pre-commit.py --changed-files --fix
```

To process the full registered source, use this in a separate script:

```python
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner

source = all_sft_sources()["example"]
StepRunner().run([source.normalized], max_concurrent=1)
chat = read_artifact(source.chat_normalized.output_path, NormalizedData)
text = read_artifact(source.normalized.output_path, NormalizedData)
print(chat.main_output_dir, chat.counters)
print(text.main_output_dir, text.counters)
```

This builds all dependencies, including the full download. Configure
[MARIN_PREFIX](../explanations/marin-prefix.md) for output storage before running.
For cloud data, run in the data's region using the
[Iris setup and job instructions](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).

Your source is now registered and materialized. `source.chat_normalized` describes
the validated structured-chat output; `source.normalized` describes deduplicated
text rendered with Marin's chat template. The printed paths locate those datasets.
Selecting the source for a training mixture is a separate step.
