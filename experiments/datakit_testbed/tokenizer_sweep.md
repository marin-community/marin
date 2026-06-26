# Tokenizer sweep

`tokenizer_sweep.py` builds reusable tokenizer-comparison DAGs over a normalized
Datakit corpus. The default config reproduces the issue #5821 GPT-OSS/Llama
sweep:

- HF families: `gpt-oss`, `llama`, `gpt-oss-place-digits`, `llama-place-digits`
- upstream tokenizers: `openai/gpt-oss-20b`, `meta-llama/Meta-Llama-3.1-8B`
- tokenizer-training sample: deterministic 50B-token-equivalent random shard sample
- derived vocab sizes: 262k, 128k, 32k, 8k
- holdout retokenization window: `[100B, 200B)`

The `*-place-digits` variants isolate numeric runs, split them into right-aligned
groups of three digits, and cap each regex-isolated numeric run at 510
characters to avoid catastrophic backtracking.

## Common commands

Submit tokenizer training plus holdout retokenization:

```bash
python experiments/datakit_testbed/tokenizer_sweep.py
```

Prepare only sampled windows and trained tokenizers:

```bash
TOKENIZER_SWEEP_PHASE=prep \
python experiments/datakit_testbed/tokenizer_sweep.py
```

Retokenize a train window with already-trained tokenizers:

```bash
TOKENIZER_SWEEP_PHASE=train_tokenization \
python experiments/datakit_testbed/tokenizer_sweep.py
```

Run only specific families and vocab sizes:

```bash
TOKENIZER_SWEEP_FAMILIES=llama,llama-place-digits \
TOKENIZER_SWEEP_SIZES=32768,8192 \
python experiments/datakit_testbed/tokenizer_sweep.py
```

## Reusable config

Use these environment variables to define a new sweep without editing code:

- `TOKENIZER_SWEEP_RUN_ID`: output/run id suffix. Defaults to `tokenizer-sweep`.
- `TOKENIZER_SWEEP_STAGING_PREFIX`: executor prefix. Defaults to `gs://marin-eu-west4`.
- `TOKENIZER_SWEEP_NORMALIZED_BASE`: GCS prefix containing normalized source artifacts.
- `TOKENIZER_SWEEP_TOTAL_TOKENIZED_TOKENS`: token count used to convert requested token windows to corpus fractions.
- `TOKENIZER_SWEEP_TOKENIZER_TRAIN_TOKENS`: tokenizer-training sample size. Defaults to 50B.
- `TOKENIZER_SWEEP_HOLDOUT_START_TOKENS`: start of holdout tokenization window. Defaults to 100B.
- `TOKENIZER_SWEEP_HOLDOUT_TOKENS`: length of holdout tokenization window. Defaults to 100B.
- `TOKENIZER_SWEEP_RETOKENIZE_TRAIN_START_TOKENS`: start of train retokenization window. Defaults to 0.
- `TOKENIZER_SWEEP_RETOKENIZE_TRAIN_TOKENS`: length of train retokenization window. Defaults to 50B.
- `TOKENIZER_SWEEP_VOCAB_SIZES`: comma-separated vocab sizes. The first size is trained; later sizes are derived.
- `TOKENIZER_SWEEP_FAMILIES`: comma-separated subset of configured families.
- `TOKENIZER_SWEEP_SIZES`: comma-separated subset of configured vocab sizes.
- `TOKENIZER_SWEEP_HF_FAMILIES_JSON`: JSON object defining HF tokenizer families.
- `TOKENIZER_SWEEP_OFFICIAL_TRUNCATED_FAMILIES_JSON`: JSON object defining official tokenizers to truncate by rank.

Custom HF families can be written in compact form:

```bash
TOKENIZER_SWEEP_HF_FAMILIES_JSON='{
  "my-tokenizer": "org/base-tokenizer",
  "my-tokenizer-place-digits": {
    "base_tokenizer": "org/base-tokenizer",
    "place_aligned_digits": true
  }
}'
```

For a non-default corpus, set the normalized path and total token count together:

```bash
TOKENIZER_SWEEP_RUN_ID=my-tokenizer-sweep \
TOKENIZER_SWEEP_NORMALIZED_BASE=gs://my-bucket/data/datakit/sample/my-corpus \
TOKENIZER_SWEEP_TOTAL_TOKENIZED_TOKENS=250000000000 \
TOKENIZER_SWEEP_TOKENIZER_TRAIN_TOKENS=50000000000 \
TOKENIZER_SWEEP_HOLDOUT_START_TOKENS=50000000000 \
TOKENIZER_SWEEP_HOLDOUT_TOKENS=50000000000 \
python experiments/datakit_testbed/tokenizer_sweep.py
```
