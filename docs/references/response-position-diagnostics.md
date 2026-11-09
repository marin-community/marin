# Response-position and length diagnostics

Training–inference mismatch (TIM) compares trainer and sampler chosen-token log
probabilities. Each pooled or staleness bucket reports `pos_first256/` and
`pos_last256/` statistics. Only tokens with a positive loss mask contribute.
Windows count all response tokens, including masked tool output; response
padding is excluded. First and last windows overlap on short responses.

## Last-window correction

Token positions were already absolute. The old last-window cutoff used the
trainable-token count instead of full response length, so the window began too
early when a response contained masked tokens.

For 100 model tokens, 1,000 masked tool tokens, then 100 model tokens, the full
response has 1,200 tokens. The correct last-256 cutoff is position 944. The old
calculation used 200 trainable tokens and produced a cutoff of −56, incorrectly
including the opening model section. The corrected last-window statistics
contain only the 100 closing model tokens in this example.

## Added long-response measurement

`response_ge2048/` adds a separate summary for tokens from responses containing
at least 2,048 full response tokens. This measurement reports `selected_tokens`,
`finite_tokens`, `log_ratio_abs_mean` and `frac_outside_0_5_2` under each mismatch
bucket. It is independent of the last-window correction.

The ratio is learner probability divided by sampler probability.
`frac_outside_0_5_2` counts finite ratios outside [0.5, 2].

Finite statistics exclude missing or nonfinite probabilities. Check the selected
and finite counts before comparing means; an empty bucket omits its mean.
