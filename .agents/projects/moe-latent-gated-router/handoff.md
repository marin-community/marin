# Gated latent router — issue #9110

## Current experiment

The authorized comparison reuses Larry's finished TPU RMSNorm + GatedNorm
references from https://github.com/marin-community/marin/issues/6822.
Only the router input changes from full-width h to the existing gated latent.
The gate, expert input, MLA attention, shared expert, initialization, optimizer,
data caches, batch, schedule, and evaluation recipe are preserved.
Credit Zihan Qiu for the idea in issue #9110.

Worktree: `/Users/kaiyuew/Downloads/Project/marin-latent-gated-router`.
Branch: `codex/moe-latent-gated-router`.
Launcher: `experiments.grug.moe_latent_gated_router.larry_launch`.

| width | new identity | steps | batch | TPU |
|---|---|---|---|---|
| 512 | moe-lgr-9110-larry-d512 | 10980 | 32 | v5p-8, us-east5 |
| 768 | moe-lgr-9110-larry-d768 | 16875 | 64 | v5p-8, us-east5 |

Both use W&B `marin-community/dial_moe`, group `moe-lgr-9110-larry-router`.
Reference IDs are
`moe_may_compute_opt_mla_hd192_vd256_qc_latentmoe_half_norm_gatednorm_xsa_d{dim}`.
Their saved configs are `larry_reference_d{dim}.json` beside the launcher.
Code artifacts: `source-dial_moe-_callable_runner.py:v698` (512), `:v696` (768).
Their model and training sources are identical; current imports and the expert
API were adapted. The underlying runtime is current, not the July environment;
throughput comparisons must retain that qualification. Historical analytic FLOP
logging is not an accurate MLA/latent accounting; use step/token-matched loss.

## Submission and recovery

CPU parent `/kaiyuew/<identity>`, child `<parent>/grug-train-<identity>`.
The launcher dispatches one child; do not give the parent a TPU.
Sanitized submission (set LGR_DIM and LGR_RUN_ID for exactly one cell):

```bash
uv run --no-sync iris --cluster=marin job run --no-wait \
  --job-name "$LGR_RUN_ID" --cpu 1 --memory 2G --region us-east5 --extra cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e GIT_COMMIT "$LGR_COMMIT" \
  -e LIBTPU_INIT_ARGS '--xla_tpu_scoped_vmem_limit_kib=50000' \
  -- python -m experiments.grug.moe_latent_gated_router.larry_launch \
  --dim "$LGR_DIM" --run-id "$LGR_RUN_ID" --run
```

Checkpoints: `gs://marin-us-east5/users/kaiyuew/grug/<identity>/checkpoints`.
Fresh identities start from scratch; exact retries resume their own checkpoints.
Do not duplicate pending/running children. Allow at most two manual recoveries
per cell, diagnose before retrying, and do not mutate the cluster.
No smoke, no new baseline, no H100 ladder, no larger cells are authorized here.

## Completion

Verify terminal Iris state, finished W&B, final Paloma macro loss, final token
count, last-100-step throughput, and final checkpoint metadata. Compare only to
Larry's gated references, with the runtime qualification above. Update #9110
and the logbook. Keep quiet on unchanged healthy/pending state. Remove the
monitor after completion or a reported unrecoverable failure.

The previous September treatments (suffix `rmsgated`) are finished but were
compared to mismatched May full-width references. Their apparent causal
regression is withdrawn. Do not reuse their checkpoints or monitoring state.
