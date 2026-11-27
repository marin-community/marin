# Matched-25 accounting checkpoint — unfinished

This preserves interrupted work in an isolated checkout. The goal is paused.
It is not ready to apply to a training run. No behavioral tests, payload-invariance
tests, distributed checks, or runtime qualification have been performed for this
patch. Publication lint and syntax checks do not establish those properties.

## Implemented in the partial patch

- Optional prompt-token, loss-mask-token and graded-sequence totals on generated
  work, captured before nonfatal content rejection discards the trajectory facts.
- Work summaries carried with buffer dispositions and aggregated into generated,
  admitted, rejected and per-disposition metrics for each admission observation
  window. Generated work can belong to a different policy version or future batch;
  it is not equated with the current batch's accepted work.
- Availability flags for missing work summaries. Missing values are omitted,
  while an observed empty collection reports zero.
- Structured RLOO-N counts from the existing estimator classifications, including
  excluded, below-minimum and zero-variance-filtered groups/sequences. The filter
  flag distinguishes filtered counts from a census of all constant-reward groups.
- A shared diagnostic for consumed tokens and loss-mask eligibility versus
  nonzero finalized policy credit, wired into driver and worker batch paths.
  Its `loss_bearing` fields mean nonzero advantage under a positive loss mask and
  nonzero correction weight. They do not prove a nonzero gradient or measure
  separate KL, entropy, FTPO or distillation losses.

The intended scope is additive observability shared by the Megatron control and
native learner. The patch has not demonstrated that training payloads or runtime
behavior remain unchanged. It changes no recipe configuration or acceptance rule.

## Unfinished and unverified

- No regression tests yet cover real objective/selection helpers, zero-variance
  and mixed-reward groups, content filters, padding, missing summaries, or exact
  equality of trained payloads with and without accounting.
- The RLOO-N classification counts describe the estimator input, which can include
  driver DP-padding rows; the new final-credit diagnostic excludes those rows.
  The step-wise estimator branch has not been wired to these RLOO-N counters.
- The new worker reductions and global group counts need real rank-ownership
  validation. Content-rejection summaries, disposition categories and observation
  window boundaries also need behavioral review before using the totals.
- No matched-run collector or acceptance changes were implemented. Missing,
  undefined and observed-zero metrics must remain distinguishable end to end.

## Optimizer application units

The preserved recipe's mini-batch size 128 is **prompts**, not sequence rows.
With four samples per prompt, a 128-prompt round and mini-batch both contain 512
rows. With update epochs equal to one and all four samples retained, this is one
optimizer application per RL round. Megatron DP16 receives 32 rows per rank and
micro-batch size one gives 32 accumulation microsteps, not 32 applications. The
native path likewise multiplies the mini-batch prompt count by four.

The matched comparison needs 25 actual optimizer applications under the verified
batch/epoch geometry, with actual `policy_update_steps` and `skipped_steps`
evidence. Twenty-five nominal rounds alone are insufficient. Do not reinterpret
128 prompts as 128 rows, alter the batch geometry, or extend the training stop
rule to compensate for skipped applications.
