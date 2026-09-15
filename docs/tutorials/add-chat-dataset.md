# How to add a chat dataset

Add a source adapter under `lib/marin/src/marin/datakit/download/`, register it in
`sft_sources.py`, and validate its output before training.

## 1. Inspect the source

Pin the dataset revision. Inspect complete, failed, and incomplete conversations.
Locate user requests, assistant answers, reasoning, tool calls, tool results, and
tool definitions. Exclude exports whose original requests or required tool
definitions cannot be recovered.

Choose an adapter to follow:

- [superior_reasoning.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/superior_reasoning.py): prompt/response pairs.
- [coderforge.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/coderforge.py): conversations with tools.
- [glm_kernelgym_rollouts.py](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/glm_kernelgym_rollouts.py): source-specific reasoning repairs.

## 2. Convert and normalize

Implement `row_to_chat_doc(row) -> list[dict]`. Return one document for an accepted
row or an empty list for a rejected row. Count intentional rejections.

Use `checked_openai_chat_document` from
`marin.datakit.download.rollout_transforms` to convert OpenAI-style messages into
Harmony, the structured chat format used by Datakit.

- Preserve message text and order. Keep source-specific repairs in the adapter.
- Pass separate reasoning as `reasoning_content` on the assistant message.
- Pass recorded tool definitions as `chat_template_kwargs={"tools": tools}`.
  Preserve tool-call IDs so the converter can associate calls with results.
- Keep original dataset IDs in `source_id`.

Write the converted rows to Parquet using `CHAT_SCHEMA` from
`marin.datakit.chat_normalize`. If retaining extra fields, extend that schema.
Pass the same schema to the writer and `normalize_chat_step`.

Follow the chosen adapter to build a download → conversion → normalization chain
of `StepSpec` objects. Put a version in the conversion step's `hash_attrs`; bump
it whenever the conversion output changes.

## 3. Register the source

Add the chat step factory to `all_sft_sources()` in
`lib/marin/src/marin/datakit/sft_sources.py`. For a chat-only source, add its
estimated token count in billions to `token_counts` there. Existing text sources
reuse their registered counts. Do not add a chat-only source to `all_sources()`
just to supply a count.

## 4. Test the adapter

Read [TESTING.md](https://github.com/marin-community/marin/blob/main/TESTING.md).
Add tests under `tests/datakit/download/` for source-specific formats, preserved
requests and reasoning, tool associations, and intentional rejections. Include
failed and incomplete attempts that the source should retain.

Run from the repository root:

```bash
uv run pytest tests/datakit/download tests/datakit/test_chat.py tests/datakit/test_chat_normalize.py
uv run --no-project infra/ci/run_tests.py
./infra/pre-commit.py --changed-files --fix
```

## 5. Inspect normalized output

After testing small fixtures, select an Iris cluster in the source data’s region.
Use its default output bucket or set [MARIN_PREFIX](../explanations/marin-prefix.md).
Save this script, replace `superior-reasoning` with your source name, and run it with
`uv run python <script.py>`:

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

Compare Parquet records in `main_output_dir` with the originals. Check requests,
reasoning, tool definitions, and results. Review rejection and duplicate counts
in `artifact.counters`.
Investigate unexpected drops; do not raise the normalizer's 5% quarantine limit
to bypass a failure.

## 6. Prepare training text

Run the SFT pipeline in the data's region, with the
[eval corpus prepared](https://github.com/marin-community/marin/blob/main/experiments/datakit/decontam/prepare_eval_corpus.py).
Replace the source names below with the sources you intend to combine:

```bash
uv run python -m experiments.datakit.sft_pipeline \
    --sources superior-reasoning,numinamath-1.5 \
    --pool-workers 16 --max-concurrent 4
```

The pipeline deduplicates across the selected sources, removes eval contamination,
and logs each filtered output path. Load that path with
`read_artifact(path, NormalizedData)` and use its `main_output_dir` for tokenization.
Inspect removal counts and flagged examples in the decontamination and dedup
reports under `<MARIN_PREFIX>/datakit/sft/report/`. Investigate unexpected removals
before tokenization. Check system/developer instructions and tool definitions
separately if they may contain eval data; automated matching excludes those fields.
