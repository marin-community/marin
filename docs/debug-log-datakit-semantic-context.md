# Debugging log for DataKit semantic context overflow

Keep the exhaustive fuzzy-dedup semantic review restart-safe when a document
pair fits the character cutoff but exceeds the inference model's token context.

## Initial status

Three of four semantic-review partitions stopped after about eight hours.
Partitions 0 and 1 failed with HTTP 400 errors because direct prompts contained
at least 129,025 input tokens plus a 2,048-token output allowance. Partition 2
failed independently after its inference endpoint returned HTTP 502. All
completed 128-pair checkpoints remained readable and hash-valid.

## Hypothesis 1

`MAX_DIRECT_CHARS=300_000` is not a safe proxy for the model's token limit on
source code. The first unfinished partition-0 batch contains direct prompts
with 214,156 and 158,066 tokens. The first unfinished partition-1 batch
contains a 165,243-token direct prompt.

## Changes to make

When an automatically selected direct review receives the model's context-limit
error, rerun that pair through the existing exhaustive chunk path. Preserve an
explicitly forced direct review as an error. Add a regression test at the
inference boundary.

## Results

The regression failed before the fix and passed afterward. All 11
semantic-review tests pass. Exact Qwen tokenizer inspection showed that
chunking the three blocked pairs produces maximum input sizes of 91,028,
116,973, and 90,483 tokens. With the 2,048-token output allowance, each stays
below the 131,072-token context limit.

The fallback applies only after an automatically selected direct request
receives a context-limit response. Other bad requests and explicitly forced
direct reviews still fail. Existing completed shards remain valid because the
fallback affects only pairs for which no outcome was persisted.

## Future work

- [ ] Consider selecting direct versus chunked by exact tokenizer count when
      model tokenizers are available to coordinator tasks.
