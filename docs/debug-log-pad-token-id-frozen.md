# Debugging log for pad_token_id frozen tokenizer failure

Investigate the shared `FrozenInstanceError: cannot assign to field 'pad_token_id'` failure seen in fresh
RegMix subset and GRP-no-L2 validation launches.

## Initial status

Claude Code reported the same runtime failure on both:

- `baseline_regmix_raw_optimum_k080_uncheatable_bpb`
- `baseline_genericfamily_power_family_penalty_no_l2_raw_optimum`

The failure text was:

`dataclasses.FrozenInstanceError: cannot assign to field 'pad_token_id'`

The 300M GRP run did not hit this failure because it launched from an older workspace snapshot before the
same eval-harness code path was exercised.

## Hypothesis 1

The lm-eval harness is mutating `tokenizer.pad_token_id` directly, but the tokenizer wrappers in Levanter
are frozen dataclasses.

## Changes to make

- Confirm all `pad_token_id` mutations in the codebase.
- Replace the eval-harness mutation with an "effective pad token" helper that falls back to EOS without
  mutating the tokenizer.
- Add a regression test that calls `loglikelihood([])` with a frozen tokenizer whose `pad_token_id` is
  `None`.

## Results

The only production writes to `tokenizer.pad_token_id` were in
`lib/levanter/src/levanter/eval_harness.py`, in `loglikelihood()` and `generate_until()`.

The tokenizer wrappers in `lib/levanter/src/levanter/tokenizers.py` are frozen dataclasses and expose
`pad_token_id` as a read-only property over `_pad_id`, so direct assignment is invalid.

Implemented `_effective_pad_token_id(tokenizer)` and changed the harness to use it instead of mutating the
tokenizer. Also threaded the resolved pad token into `_pack_requests()` and padding accounting.

Added regressions in `lib/levanter/tests/test_eval_harness.py` covering:

- EOS fallback without mutation
- `loglikelihood([])` on a frozen tokenizer without raising `FrozenInstanceError`

## Future Work

- [ ] Consider whether generation paths should surface the effective pad token explicitly to downstream code
      rather than just validating it early.
- [ ] Audit other runtime components for similar mutation of frozen tokenizer/config wrappers.
