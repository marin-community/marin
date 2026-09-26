# Agent Guide: experiments/grug/paper_rep

## Objective

Honest replication of arXiv 2609.19107's base-size experiment (Table 6) with a
dense model: train the three arms (`vanilla`, `op1`, `op1-vanilla-recipe`) at
d8/1B tokens and compare final validation losses against the paper targets
(3.3279 / 3.3057 / 3.3135). See the variant `README.md` for the full context
and the documented deviations from the paper.

## Autonomy

This workflow is designed to run end-to-end without human confirmation. The
agent is authorized to:

- Create branches, commit, and push without asking
- Create GitHub experiment issues and post comments
- Submit Iris jobs and kill only jobs submitted by self
- Reserve a TPU slice for the duration of each arm

Do not stop to ask for confirmation at any step. If something fails, diagnose
and retry or report the failure — do not block waiting for input.

## Procedure

1. Run the arms one at a time (see README for the submit command). The first
   arm materializes the FineWeb caches; the later arms reuse them. Screening
   arms (`screen-*`, 1000 steps, see README) follow the same flow.
2. Monitor via `iris job logs <id>` / W&B. Each arm is ~30 minutes
   (screening arms ~20).
3. Pull final metrics from W&B: `eval/loss` at the final step (the single
   tagged validation set is the held-out FineWeb file, so the micro average
   is exactly its loss; the per-tag key is `eval/fineweb-val-gpt2/loss`).
   Check the W&B run's summary keys if this guide drifts.
4. Compare against the paper targets; report in the experiment issue with
   direct W&B run links (always include the links, not just run names).
5. Record the verdict in the variant README and the experiment issue:
   replicate / partial / fail, with the observed vs paper gaps.

## Standing constraints

- Never stop, restart, or bounce an Iris cluster.
- The three arms must share the data caches and seed (0); differences between
  arms must come only from model and recipe.
- Do not change recipe values without an explicit paper citation (they are
  transcribed from paper Table 5 and verified against the source).
- If a training arm crashes, check `iris job logs` and the ray/fray
  diagnostics before resubmitting; preemptions auto-resume from checkpoints.
