# How to add a chat dataset

Add a source converter, register it, then inspect the processed conversations
before training. Datakit handles validation, exact deduplication, and rendering
with the Marin chat template.

If your source is already in `all_sft_sources()`, skip to
[Try the source](#try-the-source).

## Add the converter

Start from the chat functions in
[superior_reasoning.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/superior_reasoning.py).
Copy them into a new module under `lib/marin/src/marin/datakit/download/`.
Pin the download revision and adapt `row_to_chat_doc` to your source's fields:

```python
from zephyr import counters
from marin.datakit.download.rollout_transforms import checked_openai_chat_document


def row_to_chat_doc(row: dict) -> list[dict]:
    if not row.get("prompt") or not row.get("answer"):
        counters.pipeline.update_counter("example/chat/missing_text", 1)
        return []
    return checked_openai_chat_document(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        "organization/example",
        counter_prefix="example/chat",
    )
```

The helper converts OpenAI-style messages into Harmony, Datakit's structured
conversation format. Keep the source's user requests, reasoning, and tool
exchanges. Put separately recorded reasoning in `reasoning_content`; pass tool
definitions as `chat_template_kwargs={"tools": tools}` and preserve tool-call IDs.
For a tool-using source, follow
[coderforge.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/coderforge.py).
Do not invent missing requests or tool definitions.

Reuse the example's Parquet writer and `normalize_chat_step`. Both use
`CHAT_SCHEMA`; if you add metadata columns, pass the extended schema to both.
Bump the transformation `StepSpec`'s `hash_attrs["version"]` when its output
changes so Datakit rebuilds the cache.

## Register and test it

In [sft_sources.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sft_sources.py),
import your chat step factory and add `(source_name, factory)` to `rows` in
`all_sft_sources()`. Add a rough token count in billions to `token_counts` if the
source is new.

Add tests under `tests/datakit/download/` using a few real source records.
Check that requests, reasoning, and tool exchanges survive conversion, and that
intentional drops are counted. Run your test file and the repository checks:

```bash
uv run pytest tests/datakit/download/test_example.py
./infra/pre-commit.py --changed-files --fix
```

Replace `test_example.py` with your test file.

## Try the source

Save this as `scratch/run_chat_source.py`, replace the source name, and run
`uv run python scratch/run_chat_source.py` from the repository root:

```python
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner

source = all_sft_sources()["superior-reasoning"]
StepRunner().run([source.normalized], max_concurrent=1)
artifact = read_artifact(source.chat_normalized.output_path, NormalizedData)
print(artifact.main_output_dir)
print(artifact.counters)
```

This downloads and processes the full source; use fixtures while developing the
converter. Run near the stored data to avoid cross-region reads. Inspect a few
conversations in the printed directory against the originals, and investigate
unexpected drops in the counters. Rendered text is at `source.normalized.output_path`.

## Run an SFT experiment

This step requires a Grug 67B/A2B checkpoint and access to TPUs. Set
`SFT_INIT_CHECKPOINT` to a compatible native Levanter checkpoint and
`SFT_TOKENIZER_COMMIT` to the matching Marin tokenizer's commit SHA. The model
configuration is in `experiments/sft/datakit.py`.
Choose your TPU allocation, zone, and global batch size in `SFT_TPU`, `SFT_ZONE`,
and `SFT_BATCH_SIZE`. Replace `superior-reasoning` with your source, then preview
the run:

```bash
uv run python -m experiments.sft.datakit \
  --source superior-reasoning \
  --init-checkpoint "$SFT_INIT_CHECKPOINT" \
  --tokenizer marin-community/marin-tokenizer \
  --tokenizer-revision "$SFT_TOKENIZER_COMMIT" \
  --run-id example-sft --steps 100 --batch-size "$SFT_BATCH_SIZE" \
  --tpu "$SFT_TPU" --zone "$SFT_ZONE"
```

Add `--run` to prepare the token store, or `--stage train --run` to launch training.
Repeat `--source` to mix sources in proportion to their retained token counts;
omitting it selects all registered sources.

The default context is 262,144 tokens with four context shards. Training packs
whole conversations and trains on prompts and responses, keeping conversations
separate. Overlength conversations are dropped and counted.
The run starts from checkpoint weights with a fresh optimizer; `--steps` counts
new SFT steps. This recipe has CPU parity tests but has not yet been validated
in a full-context TPU run.
