# Stored benchmark prompt lengths

`prompt_lengths.json` records, for every registered chat benchmark, how long its
longest prompt is. A benchmark's prompts come from its dataset and its prompt
template, so the value is a property of the benchmark rather than of a run, and
it is measured once by a tool instead of at eval time.

## How a run uses it

`eval.contracts.prompt_length.resolve_task_max_tokens` reserves
`longest_prompt_tokens + prompt_margin_tokens` for the prompt and gives the
benchmark the rest of the context window as its response budget. A run that
passes `--max_tokens` explicitly keeps that cap; a run that passes only
`--max_length` gets a per-benchmark split. This is what stops a benchmark with
short prompts from truncating long answers at a global cap: with a
40,896-token window, OlympiadBench's 3,392-token longest prompt leaves roughly
37,000 tokens for the response instead of 8,192.

A benchmark listed under `unmeasured` gets no derived budget and keeps its own
default, so an unmeasured benchmark never silently gets a worse cap than before.

## Fields

- `reference_tokenizer` — prompt text depends on the chat template and token
  counts depend on the tokenizer, so every value is measured against this one
  tokenizer. Treat the numbers as an estimate for other tokenizers; the margin
  absorbs the difference, and over-reserving only shortens the response budget.
- `prompt_margin_tokens` — headroom added to every measured prompt length.
- `benchmarks.<Task>.longest_prompt_tokens` — the reserved prompt budget before
  the margin.
- `benchmarks.<Task>.longest_prompt_chars` — the same prompt's character length,
  which is tokenizer-independent and useful when comparing two refreshes.
- `benchmarks.<Task>.distinct_prompt_count` — how many distinct prompts the
  benchmark rendered. Repeated sampling of the same problem renders the same
  prompt, so this is a count of prompts and not of requests.
- `benchmarks.<Task>.sha256` — digest of the sorted distinct prompt texts. Any
  change to the data, the template, or the few-shot configuration changes it,
  which is how `--check` detects a stale entry.
- `unmeasured.<Task>` — why a benchmark has no stored value.

## Refreshing

Measuring renders each benchmark's prompts by driving its own
`generate_responses` with a prompt-capturing LM stand-in, so it needs the
benchmark's dataset and its extra installed:

```bash
uv sync --extra benchmarks                                              # every benchmark's deps
uv run python scripts/benchmarks/compute_prompt_lengths.py              # refresh everything
uv run python scripts/benchmarks/compute_prompt_lengths.py AIME24       # refresh one benchmark
uv run python scripts/benchmarks/compute_prompt_lengths.py --check      # report drift, write nothing
```

Refresh the entry whenever you change a benchmark's dataset, prompt template, or
few-shot configuration, and commit the updated JSON with that change.
`tests/contracts/test_prompt_length.py` requires every registered benchmark to
appear in exactly one of `benchmarks` or `unmeasured` and re-renders the
benchmarks whose data is checked into this repository, which is the staleness
check CI can run.

A full refresh downloads every benchmark's dataset, which is tens of gigabytes:
LiveCodeBenchv5 and LiveCodeBenchv5_official alone pull an 11 GB cache, and
because they pass `cache_dir="./"` it lands in the working directory rather than
the shared Hugging Face cache. Run the full sweep deliberately and prefer naming
the benchmarks you changed. `--timeout` bounds one benchmark's render, so a
dataset that never finishes downloading is recorded as unmeasured instead of
stalling the sweep; RepoBench is recorded that way today.
