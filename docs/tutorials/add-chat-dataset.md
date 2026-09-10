# How to add a chat dataset

Add a source converter, write its conversations to Parquet, then register it in
`all_sft_sources()`. Marin's Datakit pipeline validates and deduplicates the
conversations.

This tutorial uses a prompt/response dataset. For complete source modules, see
[`superior_reasoning.py`](../../lib/marin/src/marin/datakit/download/superior_reasoning.py),
[`coderforge.py`](../../lib/marin/src/marin/datakit/download/coderforge.py) for tool use,
and [`glm_kernelgym_rollouts.py`](../../lib/marin/src/marin/datakit/download/glm_kernelgym_rollouts.py)
for repairs to reasoning boundaries.

## 1. Inspect the source

Pin the dataset revision. Inspect successful, failed, and incomplete conversations
for user requests, assistant responses, reasoning, tool calls, results, and
definitions. Every source needs the original user requests. Tool calls also need
the recorded tool definitions. Leave a source out of the chat registry if these
inputs cannot be recovered.

Put the converter in `lib/marin/src/marin/datakit/download/`. Keep repairs specific
to the source there, such as combining adjacent user messages or restoring a
missing reasoning opener. Preserve recorded text and order; do not invent prompts
or infer tool definitions from call arguments.

## 2. Convert rows to chat documents

Datakit stores conversations as serialized Harmony messages. For OpenAI-style
messages, `checked_openai_chat_document` handles this conversion:

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
        "organization/example",
        counter_prefix="example/chat",
    )
```

Adapt the fields and checks to your source; this example assumes strings. Return
an empty list for a dropped row or a one-element list for a chat document. Count
intentional drops. The helper counts and skips conversion failures raised as
`ValueError` or `UnicodeError`; other errors propagate.

Pass separate reasoning as the assistant message's `reasoning_content`. Pass tool
definitions as `chat_template_kwargs={"tools": tools}` and preserve call IDs so
the converter can match results to calls. If you construct Harmony `Message`
objects directly, serialize them with `chat_document` instead.

### Conversation requirements

These rules apply to the converted Harmony messages. The helper supplies channels
and tool recipients from the OpenAI-style input. The validators in
`marin.datakit.chat_normalize` require:

- Text-only messages; images and audio are unsupported.
- System/developer instructions first, without a channel or recipient.
- A nonblank user request to start the conversation. User messages have no channel
  or recipient; the source must combine adjacent user messages.
- Nonblank assistant messages with an `analysis`, `commentary`, or `final` channel.
  A conversation continuing after a final answer needs a new user turn.
- Tool calls as assistant commentary addressed to `functions.<name>`, with a JSON
  object of arguments. Each called tool needs an explicit definition with a
  unique name and an object-valued `parameters` field.
- Tool replies as commentary addressed to `assistant`, named for their calls.
  Replies must match pending calls in order before the conversation resumes.
- An assistant message at the end. Reasoning-only endings and unanswered final
  batches of tool calls are allowed.

Normalization accepts serialized Harmony. Convert source fields such as
`tool_calls` and `reasoning_content` first. Validation checks conversation
structure; incorrect answers and arguments that violate a tool's parameter schema
can remain as examples of failed attempts.

## 3. Write and normalize Parquet

A source needs a download, a conversion step, and a chat normalization step.
For a Hugging Face dataset stored as Parquet, add this beside `row_to_chat_doc`.
Replace the dataset ID, revision, file pattern, and `example` names:

```python
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_chat_doc)
        .write_parquet(f"{output_path}/part-{{shard:05d}}.parquet", schema=CHAT_SCHEMA)
    )
    ZephyrContext(name="example-chat", resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)


def example_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/example",
        hf_dataset_id="organization/example",
        revision="<commit hash>",
        hf_urls_glob=["data/train-*.parquet"],
    )
    processed = StepSpec(
        name="processed-chat/example",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "v1"},
    )
    return processed, normalize_chat_step(
        name="normalized-chat/example", download=processed, output_schema=CHAT_SCHEMA
    )
```

A `StepSpec` describes a pipeline step and its dependencies. Bump the conversion
version when its output changes so cached results are rebuilt. Normalization
validates conversations, hashes messages plus template arguments, removes exact
duplicates, and preserves the incoming document ID as `source_id`.

To retain metadata, add Arrow fields to `CHAT_SCHEMA` and use that extended schema
in both `write_parquet` and `normalize_chat_step`. Pass values as keyword arguments
to the document helper, such as `upstream_id=row["id"]`. Metadata becomes top-level
columns. Reserve `source_id` for normalization. The helper stores
`chat_template_kwargs` as JSON text.

## 4. Register the source

In [`sft_sources.py`](../../lib/marin/src/marin/datakit/sft_sources.py), import
your factory:

```python
from marin.datakit.download.example import example_chat_normalize_steps
```

Add `("example", example_chat_normalize_steps)` to `rows` inside
`all_sft_sources()`. For a chat-only source, also add
`token_counts["example"] = <estimated billions of tokens>` before the return.
Existing text sources get this value from `all_sources()`; keep chat-only sources
in the SFT registry.

The registry builds a `DatakitChatSource` from the factory's steps and adds the
rendering and text-normalization steps automatically.

## 5. Verify the output

Read [`TESTING.md`](../../TESTING.md), then add conversion tests under
`tests/datakit/download/`. Cover the source's format quirks, preserved requests and
reasoning, tool associations, and intentional drops. Run from the repository root:

```bash
uv run pytest tests/datakit/download tests/datakit/test_chat.py tests/datakit/test_chat_normalize.py
uv run --no-project infra/ci/run_tests.py
./infra/pre-commit.py --changed-files --fix
```

Start with a small fixture. Before training, run the full source in an environment
configured for its data region (see [Iris job operations](../../lib/iris/OPS.md)).
Save this as `scratch/run_chat_source.py`, replace
`superior-reasoning` with your registered name, and run
`uv run python scratch/run_chat_source.py`:

```python
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner

source = all_sft_sources()["superior-reasoning"]
StepRunner().run([source.chat_normalized], max_concurrent=1)
artifact = read_artifact(source.chat_normalized.output_path, NormalizedData)
print(artifact.main_output_dir)
print(artifact.counters)
```

The runner builds dependencies, including download and conversion. Compare
normalized Parquet records in `artifact.main_output_dir` with the source. Check
roles, reasoning, tool definitions, and results. Inspect conversion counters in the
Zephyr job logs and normalization counters printed above for unexpected drops.
Normalization fails if empty or quarantined records exceed its 5% rejection
limit. When using Iris, run near the data to avoid
cross-region reads. These checks do not establish answer quality.
