# How to add a chat dataset

A Datakit chat source converts its original records into structured Harmony messages,
then normalizes them into Parquet. Add the conversion in a source module under
`lib/marin/src/marin/datakit/download/` and register its processing steps in
`lib/marin/src/marin/datakit/sft_sources.py`.

Use `superior_reasoning.py` as a small prompt/response example, `coderforge.py` for
records with tool definitions, and `glm_kernelgym_rollouts.py` for a source that
needs its own reasoning-boundary repairs. All three live in the download directory.

## 1. Inspect the original records

Pin the dataset revision and inspect examples before choosing a parser. Find the
original user request, assistant response, reasoning, tool calls, tool results,
and tool definitions. Check unsuccessful and incomplete conversations too.

Keep source-specific repairs in the source module. For example, a source may need
to combine adjacent user messages or repair a missing reasoning opener. Preserve
the recorded text and ordering; do not invent missing prompts or tool definitions.
Every source needs the original user requests; definitions are needed when tools
are called. Leave an export out of the chat registry if those required inputs
cannot be recovered.

## 2. Convert each row

Write a `row_to_chat_doc` function that returns a list of zero or one documents.
For OpenAI-style messages, use `checked_openai_chat_document` from
`marin.datakit.download.rollout_transforms`:

```python
from zephyr import counters

from marin.datakit.download.rollout_transforms import checked_openai_chat_document


def row_to_chat_doc(row: dict) -> list[dict]:
    prompt = row.get("prompt")
    answer = row.get("answer")
    if not prompt or not answer:
        counters.pipeline.update_counter("example/chat/missing_text_filtered", 1)
        return []
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return checked_openai_chat_document(
        messages,
        "organization/example",
        counter_prefix="example/chat",
    )
```

This example assumes the source fields contain strings. Adapt the field names and
checks to the actual export. The helper converts messages into Harmony, including
separate reasoning and tool calls. It counts and quarantines conversion errors
that are `ValueError` or `UnicodeError`. Other errors propagate. Keep intentional
source filters counted as well so drops can be investigated.

If the source already supplies separate reasoning, pass it as `reasoning_content`
on the assistant message. For tools, pass the recorded definitions through
`chat_template_kwargs={"tools": tools}`. Preserve call IDs in source messages so
the converter can associate results with calls before producing Harmony messages.
Do not reconstruct tool definitions from observed arguments.

For a source that constructs Harmony `Message` objects directly, use
`chat_document` to serialize them instead.

## 3. Meet the chat contract

The schema and validators live in `marin.datakit.chat_normalize`:

- Messages contain text only; images and audio are unsupported.
- System/developer instructions come first, with no channel or recipient.
- The conversation starts with a nonblank user request. User messages have no
  channel or recipient, and adjacent user messages must be combined by the source.
- Assistant messages contain nonblank text and identify `analysis`, `commentary`,
  or `final`. After a final answer, a continuing conversation needs a new user turn.
- Tool calls are assistant commentary addressed to `functions.<name>`, with a JSON
  object of arguments. Every called tool needs an explicit definition with a unique
  name and an object-valued `parameters` field.
- Tool replies are commentary addressed to `assistant`, named for their calls.
  Replies must match pending calls in order before the conversation resumes.
- Records end with an assistant message. Reasoning-only endings and unanswered
  final batches of tool calls are allowed, preserving incomplete attempts.

The normalizer expects serialized Harmony, not source fields such as `tool_calls`
or `reasoning_content`. Incorrect answers and tool arguments that violate the
parameter schema are not rejected solely for being wrong: failed attempts can
still be useful examples.


## 4. Write and normalize Parquet

Use a Zephyr pipeline to read the pinned download, apply `row_to_chat_doc` with
`flat_map`, and write Parquet with an explicit schema. Start with `CHAT_SCHEMA`
from `marin.datakit.chat_normalize`.

If retaining annotations such as reward or teacher, extend `CHAT_SCHEMA` with
explicit Arrow fields and pass those values to the document helper. Metadata is
currently stored as source-specific top-level columns; it is not automatically
packed into a catchall. Keep original dataset identifiers in `source_id`.
Normalization preserves an existing `source_id`; otherwise it uses the input
record’s `id`. `chat_template_kwargs` is stored as JSON text.

Follow `transform_chat` and `superior_reasoning_chat_normalize_steps` in
`lib/marin/src/marin/datakit/download/superior_reasoning.py` for a complete writer
and step-factory example. Create a `StepSpec` for `transform_chat`,
with the download as a dependency and a version in `hash_attrs`. Then return the
processed step and normalization step:

```python
return processed, normalize_chat_step(
    name="normalized-chat/example",
    download=processed,
    output_schema=SOURCE_CHAT_SCHEMA,
)
```

Here `processed` is the chat transformation `StepSpec`, and `SOURCE_CHAT_SCHEMA`
is the schema used by its Parquet writer. Import `normalize_chat_step` from
`marin.datakit.chat_normalize`. Use `CHAT_SCHEMA` directly if there are no extra
columns. Pass the same schema to both writing stages so optional fields survive.

Normalization validates conversations, hashes messages plus template arguments,
and removes exact duplicates. Bump the source transformation version when its
output changes so cached processed and normalized artifacts are rebuilt.

## 5. Register and verify the source

Add the chat step factory to `all_sft_sources()` in `sft_sources.py`. Its
`DatakitChatSource` records the name, ordered processing steps, and approximate
token count. Existing text sources reuse their weights from `all_sources()`.
For a chat-only source, add its explicit weight in billions to `token_counts`
inside `all_sft_sources()`. Do not add it to the pretraining registry merely to
supply this weight: that registry also defines the expected coverage of pinned
hero-data artifacts, including embeddings and tokenized outputs.

Add behavior tests under `tests/datakit/download/` for the source conversion.
Cover its real format quirks, preservation of requests and reasoning, tool
associations when present, and intentional drops. Read `TESTING.md` first.

Run the chat tests and repository checks from the repository root:

```bash
uv run pytest tests/datakit/download tests/datakit/test_chat.py tests/datakit/test_chat_normalize.py
uv run --no-project infra/ci/run_tests.py
./infra/pre-commit.py --changed-files --fix
```

Before using the source in training, run its processing chain in an environment
configured for the data's region. For example, save this as
`scratch/run_chat_source.py`, replace `superior-reasoning` with your registered
name, and run `uv run python scratch/run_chat_source.py`. This runs the full source,
not a bounded sample; use a small fixture for initial development.

```python
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner

source = all_sft_sources()["superior-reasoning"]
StepRunner().run([source.normalized], max_concurrent=1)
artifact = read_artifact(source.normalized.output_path, NormalizedData)
print(artifact.main_output_dir)
print(artifact.counters)
```

The runner builds dependencies, including download and conversion. Inspect actual
normalized Parquet records in `artifact.main_output_dir`. Check roles, reasoning, tool definitions, and results
against the source. Compare input, source-filter, normalization-quarantine, and
deduplication counts. Normalization fails if quarantined records exceed its 5%
health limit; investigate unexpected drops instead of raising the limit. Run near
the source data when using Iris to avoid cross-region reads. Formatting checks do
not establish answer quality.
