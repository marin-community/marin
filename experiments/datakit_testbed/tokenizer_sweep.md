# Tokenizer sweep

`tokenizer_sweep.py` builds reusable tokenizer-comparison DAGs over a normalized
Datakit corpus. It is configured by `TokenizerSweepMainConfig`, whose `sweep`
field contains a `TokenizerSweepConfig`.

The default `issue_5821_default_config()` reproduces the issue #5821 GPT-OSS
and Llama sweep:

- HF families: `gpt-oss`, `llama`, `gpt-oss-place-digits`, `llama-place-digits`
- upstream tokenizers: `openai/gpt-oss-20b`, `meta-llama/Meta-Llama-3.1-8B`
- tokenizer-training sample: deterministic 50B-token-equivalent random shard sample
- derived vocab sizes: 262k, 128k, 32k, 8k
- holdout retokenization window: `[100B, 200B)`

## Numeric pretokenizer

The upstream Llama and GPT-OSS pretokenizers already isolate numeric text with
`\p{N}{1,3}`, which splits a run from the left (`1234567 -> 123|456|7`). The
`*-place-digits` variants change those boundaries to right-aligned groups
(`1234567 -> 1|234|567`) using a three-stage pipeline:

- isolate contiguous numeric runs from surrounding text;
- split each run into right-aligned groups of three digits, e.g. `1234567 -> 1|234|567`;
- cap each regex-isolated numeric run at 510 characters before triplet splitting to avoid catastrophic backtracking.

## Common commands

Submit tokenizer training plus holdout retokenization with the default #5821 recipe:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py
```

Prepare only sampled windows and trained tokenizers:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py --sweep.phase prep
```

Retokenize a train window with already-trained tokenizers:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py --sweep.phase train_tokenization
```

Run only specific families and vocab sizes:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py \
  --sweep.family_filter '[llama,llama-place-digits]' \
  --sweep.size_filter '[32768,8192]'
```

## Custom recipe

Define a new sweep by overriding the typed config fields rather than editing
the script. A custom one-family sweep can be launched with CLI overrides:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py \
  --sweep.run_id my-tokenizer-sweep \
  --sweep.corpus.normalized_base gs://my-bucket/data/datakit/sample/my-corpus \
  --sweep.corpus.total_tokenized_tokens 250000000000 \
  --sweep.tokenizer_train.tokens 50000000000 \
  --sweep.holdout.start_tokens 50000000000 \
  --sweep.holdout.tokens 50000000000 \
  --sweep.vocab_sizes '[262144,32768,8192]' \
  --sweep.hf_families '[{name: my-tokenizer, base_tokenizer: org/base-tokenizer}]'
```

For more complex sweeps, put the same fields in a Draccus config file and pass
it as the script config. The most important fields are:

- `sweep.run_id`: output/run id suffix;
- `sweep.staging_prefix`: executor prefix;
- `sweep.corpus.normalized_base`: GCS prefix containing normalized source artifacts;
- `sweep.corpus.total_tokenized_tokens`: token count used to convert token windows to corpus fractions;
- `sweep.tokenizer_train`, `sweep.holdout`, `sweep.train_retokenize`: token windows;
- `sweep.vocab_sizes`: first size is trained; later sizes are derived by BPE-rank truncation;
- `sweep.hf_families`: HF tokenizer families to train from upstream tokenizers;
- `sweep.official_truncated_families`: optional HF tokenizers to truncate without retraining;
- `sweep.family_filter` and `sweep.size_filter`: optional subsets for a particular submission.
- `sweep.sample_resource`, `sweep.hf_train_resource`, and `sweep.tokenize_worker_resource`: worker resources.
  The default recipe uses TPU-backed flexible workers over `v5p-8`, `v6e-4`, `v4-8`, and `v5litepod-4`
  so tokenizer and retokenization jobs do not launch CPU-only nodes. Fray flexible TPU requests must
  use topology-compatible alternatives; override `tpu_types` with a different compatible list for other TPU sets.

TokenMonster is intentionally not part of this default recipe; add it in a
separate focused experiment if needed.

## Matched Grug MoE comparison

`tokenizer_moe_comparison.py` trains matched token-level Grug MoE controls from
the completed tokenizer-sweep caches. It contains no chunked-MoE policies. The
rung shapes and AdamH hyperparameters are frozen to the historical controls;
the model implementation is the Grug MoE implementation on the checked-out
revision. Tokenizer family and vocabulary size are the only differences within
one invocation:

| rung | layers | expert FFN | experts / top-k | batch x sequence | steps | tokens |
|---|---:|---:|---:|---:|---:|---:|
| d512 | 6 | 256 | 64 / 4 | 512 x 4096 | 399 | 836,763,648 |
| d768 | 8 | 384 | 64 / 4 | 512 x 4096 | 1,292 | 2,709,520,384 |

`--tpu_type` may select another TPU topology with at least four chips; it is not
restricted to v6e-8. The launcher pins child compute to `--region` and rejects
cache, output, and compute-region mismatches before submitting work.

The comparison must rerun the complete 16-cell matrix: GPT-OSS, Llama, and
both place-digits variants, each at 8k and 32k vocabulary and at d512 and d768.
Do not compare newly trained digits cells against vanilla controls from a
different Grug code revision. Submit the complete matrix through Iris with:

```bash
.venv/bin/iris --cluster marin job run --no-wait \
  --job-name tokenizer-moe-all-canonical-adamh-20260710 \
  --priority interactive --region europe-west4 \
  --tpu v6e-4 --enable-extra-resources --preemptible \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -- uv run python experiments/datakit_testbed/tokenizer_moe_comparison.py \
    --cache_prefix gs://marin-eu-west4/data/datakit/tokenized/tokenizer-sweep-20260526 \
    --output_prefix gs://marin-eu-west4 \
    --tokenizer_run_id tokenizer-sweep-20260526 \
    --region europe-west4 \
    --tpu_type v6e-8 \
    --version canonical-adamh-all-20260710 \
    --families '[gpt-oss,llama,gpt-oss-place-digits,llama-place-digits]'
```

The command discovers only cache components with a completed
`train/.stats.json`, requires exactly `--expected_sources` components for both
training and holdout, weights training sources by recorded token count, and
evaluates BPB on the separately tokenized holdout window every 1,000 steps and
at the final step.

The same DAG has a report artifact that depends on all 16 checkpoints. It reads
the final `eval/byte_weighted_bpb` and `eval/byte_weighted_macro_bpb` values from
the exact W&B run names, verifies every final step, and requires equal parameter
counts across tokenizer families for each `(vocab, rung)` pair. Missing,
duplicate, partial, non-finite, or architecture-mismatched results fail the
report instead of producing a partial comparison. Successful runs write:

```text
gs://marin-eu-west4/tokenizer-comparison/results/canonical-adamh-all-20260710/results.md
gs://marin-eu-west4/tokenizer-comparison/results/canonical-adamh-all-20260710/results.json
```

The Markdown output is the authoritative 16-row results table. Its BPB is
computed as total loss in bits divided by total evaluated decoded bytes; the
historical token-weighted `eval/bpb` is retained for compatibility but is not
used in this comparison.
