# NUPA-Loose

`NUPA-Loose` evaluates numerical understanding and processing using the tasks from
[Number Cookbook: Number Understanding of Language Models and How to Improve It](https://arxiv.org/abs/2411.03766)
and permissive answer extraction. Evalchemy registers the 238,926-prompt sample as
`NUPA-Loose` and a fixed stratified 5,000-prompt panel as `NUPA5K-Loose`.

## Differences from the reference evaluation

The paper's baseline evaluation requests direct numeric answers without tools or
chain-of-thought (Section 2.4 and Appendix A.2). The
[reference text scorer](https://github.com/GraphPKU/number_cookbook/blob/46aefaafb2651d5a14853c2c475f3f390b4be78b/src/text_model/metrics.py)
extracts a numeric prefix after its answer delimiter or marker.

Evalchemy's [permissive extraction change](https://github.com/marin-community/evalchemy/pull/187)
also accepts trailing boxed numbers, inline math, equations, explicit answer
lines, and a bare number on the last line. It discards content before the last
`<|end_think|>` marker and rejects an unfinished `<|start_think|>` block.
These rules can credit a correct number after a reasoning response even when the
model did not follow the direct-answer instruction. They measure numeracy with
less dependence on output formatting and instruction following.

Numeric comparison remains representation-sensitive: rounded decimals,
additional decimal digits, and unreduced fractions still fail exact match against
an exact target. `format_valid_rate` measures whether the extractor accepts an
answer; it does not measure compliance with the direct-answer instruction.

Report scores under the Loose task names. They are not directly comparable to
the paper's direct-answer results. The former Evalchemy names `NUPA` and `NUPA5K`
were replaced by `NUPA-Loose` and `NUPA5K-Loose`; update task selections and install
extras accordingly.

## Dataset protocol

The benchmark downloads the canonical MIT-licensed
[`HaotongYang/NUPA_text`](https://huggingface.co/datasets/HaotongYang/NUPA_text)
test data at revision `01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7`.
The 688 MB source JSON contains 2,387,501 examples nested under 44 task keys and
2,391 task/digit groups.

Evalchemy reproduces the Number Cookbook text-model evaluation default: sample
100 examples from every task/digit group with Python's `random.Random(20222943)`.
The resulting benchmark contains 238,926 prompts. Two source groups contain
fewer than 100 examples; every other group contributes 100. The loader streams
one source task at a time, so it does not deserialize the complete source file
at once.

The source repository is GPL-3.0, so Evalchemy does not copy its implementation.
The dataset is separately distributed under the MIT license. The loader and
scorer here are clean-room implementations based on the paper's protocol and
observable metric definitions.

## Run the benchmark

Install the benchmark extra and evaluate an OpenAI-compatible endpoint:

```bash
uv sync --extra nupa-loose

eval --model local-completions \
  --tasks NUPA-Loose \
  --limit 1000 \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

The full protocol contains 238,926 requests. Use `--limit` for development and
small comparisons; record the limit with any reported score. `--debug` uses four
checked-in records without downloading the source dataset.

For a cheaper comparison that retains coverage of every released task/digit
stratum, run `--tasks NUPA5K-Loose`. Its checked-in identity manifest selects 5,000
unique records by deterministic round-robin over the 2,391 strata, ordered by
task name and numeric digit length. Records within each stratum are ordered by
the SHA-256 digest of their exact source text. NUPA5K-Loose reports the ordinary
unweighted mean over the fixed panel; it is not an unbiased estimator of the
full publication protocol. Reported results must name `NUPA-Loose` or `NUPA5K-Loose` and
must not present the variants as interchangeable.

## Materialize the selected rows

The production loader reads the pinned nested source directly. The same
selection can be materialized as row-oriented JSONL for inspection:

```bash
uv run --extra nupa-loose python -m eval.chat_benchmarks.NUPA-Loose.data_prep.flatten_hf_dataset \
  --output /tmp/nupa_test.jsonl
```

Each row records its task, operation, answer representation, digit length,
S/M/L/XL length bucket, prompt, and target answer. `--num-each` and
`--random-seed` override the published protocol for experiments.

## Metrics

Numeric-component alignment follows the public NUPA text evaluation protocol.
Extraction also accepts final numbers in `\boxed{}`, inline math, or a trailing
equation after a thinking response. Numeric comparison remains representation-sensitive.
The benchmark reports:

- `exact_match`: representation-sensitive equality after format-specific extraction.
- `digit_match`: aligned digit accuracy between the extracted answer and target.
- `dlength`: absolute difference in total digit count; lower is better.
- `format_valid_rate`: fraction of responses accepted by the expected answer parser.
- `no_answer_rate`: fraction of responses without an extracted answer; lower is better.

Metrics are emitted overall, by task, by length bucket, and by task/bucket pair.
`exact_match` is the primary score.
