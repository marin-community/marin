# Gated latent router continuation

User confirmed the intended architecture: `down(h) -> RMSNorm -> gate -> router + experts`.
Retain the latent RMSNorm. Do not restore or compare the superseded gate-only smoke.

Issue: https://github.com/marin-community/marin/issues/9110. Idea: Zihan Qiu.
Worktree: `/Users/kaiyuew/Downloads/Project/marin-latent-gated-router`.
Branch: `codex/moe-latent-gated-router`. Exact training source is in the state file.
Owner: thread heartbeat `follow-gated-latent-moe-ablation-9110`, every 10 minutes.
State: `scratch/20260911-1105_moe-lgr-9110-rmsgated_monitoring_state.json`.

## Scope and current state

**Never submit a control training job.** Reuse existing Agent MoE baseline runs.
The control in code exists only for numerical tests and schedule calculation.
The treatment computes `z = GatedNorm(RMSNorm(h @ latent_down))`; both router and
routed experts consume z. Shared experts and outer norms retain hero behavior.

Latest user instruction: “别smoke浪费卡了，直接跑”. **Skip all smoke runs.**
The pending corrected smoke was cancelled before any attempt was allocated.
The two full Gate 1 parents were submitted from `edf9b2871`:

- `/kaiyuew/moe-lgr-9110-d512-rmsgated` at 18:53:16 UTC.
- `/kaiyuew/moe-lgr-9110-d768-rmsgated` at 18:53:58 UTC.

Each expected child is `<parent>/grug-train-<run-id>`. Read the `gate1` records in
state for exact source, current status, checkpoint paths and recovery commands.
At 21:34 UTC, both runs were pending automatic attempt 2 after scheduler
preemption. Temporary checkpoint metadata is verified: d512 step 1921 and d768
step 648, saved at 21:24 UTC. Expected retries resume these checkpoints.
Last recorded global steps were 2492/659; metrics were finite with zero overflow.
W&B may retain preemption-era values until replay catches up. Use attempt-specific
Iris logs during recovery, and state for exact checkpoint paths and current status.
Do not duplicate these submissions or restart either cancelled smoke.

Old `gated` identities are superseded: the gate-only smoke finished, and the
full d512 parent plus child were cancelled. Old d768 was never submitted.
Keep all corrected experiments under fresh `rmsgated` identities and output roots.
The original user checkout is dirty and unrelated; work only in this worktree.
All 32 pinned data caches already exist in us-central1. Do not alter production
jobs, cluster settings, or shared data caches.

## Startup, recovery and progression

Use `uv run --no-sync iris --cluster=marin job list --prefix <canonical-parent>`.
The current CLI does not accept `job list --json`. Read bounded recent logs;
avoid replaying hundreds of dependency-install lines. Use `task describe` and `attempt logs` on failure; this checkout has no `job summary`.
Pending for capacity is not a failure and must not cause a duplicate submission.
The first smoke submit was rejected because `--reserve v5p-8` conflicted with the
CPU-parent placement; removing `--reserve` resolved it. Keep the parent CPU-only.

Verify the two full Gate 1 cells directly: intended child, correct configuration,
fresh checkpoint start or intended resume, and advancing finite W&B metrics.
Before a recovery, check that its parent and child are not already pending/running.
Use the existing launcher; do not create a separate training implementation.

```bash
LGR_DIM=512
LGR_RUN_ID="moe-lgr-9110-d${LGR_DIM}-rmsgated"
LGR_COMMIT=$(git rev-parse HEAD)
uv run --no-sync iris --cluster=marin job run --no-wait \
  --job-name "$LGR_RUN_ID" --cpu 1 --memory 2G --region us-central1 --extra cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e GIT_COMMIT "$LGR_COMMIT" \
  -e LIBTPU_INIT_ARGS '--xla_tpu_scoped_vmem_limit_kib=50000' \
  -- python -m experiments.grug.moe_latent_gated_router.launch \
  --dim "$LGR_DIM" --arm gated_latent --run-id "$LGR_RUN_ID" --version dev --run
```

The child is `<parent>/grug-train-<run-id>` and requests v5p-8 in us-central1.
The full schedules are 13,642 steps at d512 and 19,378 at d768. Checkpoints live
under `gs://marin-us-central1/users/kaiyuew/grug/<run-id>/dev/checkpoints/step-<N>`.
Time-based checkpoints use separate region-local temporary storage; resume via
the exact same trainer ID and artifact path. Record identities immediately.
No secret values belong in state files, logs, or issue comments.

Exact full-run recovery commands are stored in the JSON state file. If code changes,
run appropriate checks, commit/push, and record the new source commit before recovery.
Bundles do not contain `.git`; all corrected jobs explicitly receive `GIT_COMMIT`
so W&B can log their source. At most two manual recoveries per cell; diagnose
persistent failures and report them rather than retrying indefinitely.

## Results and gates

Existing reference IDs are `moe_may_compute_opt_d{dim}_ep1` in
`marin-community/marin_moe`, as linked from `experiments/grug/moe/README.md`.
Require reference and treatment W&B state `finished`; retrieve final Paloma
macro loss, total tokens, and last 100 training steps of throughput. Validate
the actual W&B configuration, rather than relying on a run name. Record routing
metrics, checkpoint metadata, terminal Iris state and code revision.

Reference runs use v4-32 and a different recipe without the hero latent bottleneck.
They have different token counts. Report those confounds. For reference compute
C_b, losses L_b/L_v, tokens N_b/N_v and token rates T_b/T_v, use:

```text
compute_ratio = ((L_b - 1.6) / (L_v - 1.6)) ** (1 / 0.0941)
reference_time_needed = compute_ratio * N_b / T_b
variant_time = N_v / T_v
effective_speedup = reference_time_needed / variant_time
```

Report compute_ratio and actual variant analytic compute separately. Do not treat
tokens/sec as FLOPs/sec or imply the cross-hardware speedup isolates the gate change.
Gate 1 passes only if effective speedup exceeds 1 at both widths. If it fails,
publish the measured comparison and stop without launching larger cells.

On a pass, run only gated_latent at d1024 and d1280 with analogous IDs and command.
Full schedules are 16,425 and 14,473 steps. Compare to the existing references,
fit `loss(C)=1.6+A*C**(-alpha)` to the four treatment points, and project at 1e21
and 1e23. The guide references are 2.534 and 2.205. Gate 2 needs improvements at
all four widths and both projections. No fresh baseline is authorized at this gate.

Update issue #9110 and the append-only logbook at meaningful milestones. Issue
comments begin with `🤖`; retain credit to Zihan Qiu. Keep the issue's Status and
Conclusion current. Stay quiet while merely capacity-pending. After the final gate
decision or an unrecoverable blocker, report it and remove the heartbeat.
