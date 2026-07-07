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

The `*-place-digits` variants make three numeric-specific changes:

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
  The default recipe uses TPU-backed `v6e-8` workers so tokenizer and retokenization jobs do not launch
  CPU-only nodes.

TokenMonster is intentionally not part of this default recipe; add it in a
separate focused experiment if needed.
