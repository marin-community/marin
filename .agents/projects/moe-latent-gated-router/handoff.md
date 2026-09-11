# Gated latent router continuation

Issue: https://github.com/marin-community/marin/issues/9110. Idea: Zihan Qiu.
Worktree: `/Users/kaiyuew/Downloads/Project/marin-latent-gated-router`.
Branch: `codex/moe-latent-gated-router`. Training source: `da78e56d0`.
Owner: thread heartbeat `follow-gated-latent-moe-ablation-9110`, every 10 minutes.
State: `scratch/20260911-1028_moe-lgr-9110_monitoring_state.json`.

## Scope and current state

The user clarified that no new baseline is needed. **Never submit a control
training job.** Reuse existing Agent MoE baseline runs. The code retains a control
only for numerical tests and schedule calculation. The treatment is literally
`z = GatedNorm(h @ latent_down)` with no latent RMSNorm, and router plus routed
experts both consume z. Shared experts and outer norms retain the hero behavior.

At handoff the only submitted workload is the five-step smoke:

- Parent: `/kaiyuew/moe-lgr-9110-d512-gated-smoke`.
- Child: `/kaiyuew/moe-lgr-9110-d512-gated-smoke/grug-train-moe-lgr-9110-d512-gated-smoke`.
- Child is now running on v5p-8. JAX initialization and the expected W&B run are verified.
- Parent has read all cached dependencies and dispatched only the expected child.
- Model starts from scratch as intended. No advancing training loss or final checkpoint is verified yet.
- W&B ID/name: `moe-lgr-9110-d512-gated-smoke`, project `marin-community/marin_moe`.
- Expected final checkpoint:
  `gs://marin-us-central1/users/kaiyuew/grug/moe-lgr-9110-d512-gated-smoke/dev/checkpoints/step-5`.

The original user checkout is dirty and unrelated. Work only in this worktree.
Do not alter production jobs, cluster configuration, or shared data caches.
All 32 canonical data caches are already successful in us-central1. Recipe-drift
warnings concern new resource fields; pinned cached outputs are intentionally reused.

## Startup, recovery and progression

Use `uv run --no-sync iris --cluster=marin job list --prefix <canonical-parent>`.
The current CLI does not accept `job list --json`. Read bounded recent logs;
avoid replaying hundreds of dependency-install lines. Use `job summary` on failure.
Pending for capacity is not a failure and must not cause a duplicate submission.
The first smoke submit was rejected because `--reserve v5p-8` conflicted with the
CPU-parent placement; removing `--reserve` resolved it. Keep the parent CPU-only.

After a successful five-step smoke, verify finite training/evaluation, terminal
successful Iris state, W&B `finished`, and final checkpoint `metadata.json`.
Then submit exactly two full Gate 1 cells, `dim=512` and `dim=768`, with run IDs
`moe-lgr-9110-d512-gated` and `moe-lgr-9110-d768-gated`. Before each submission,
check that its parent and child do not already exist in a pending/running state.
Use the existing launcher; do not create a separate training implementation.

```bash
LGR_DIM=512
LGR_RUN_ID="moe-lgr-9110-d${LGR_DIM}-gated"
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

The exact smoke resubmit command is stored in the JSON state file. If code changes,
run appropriate checks, commit/push, and record the new source commit before recovery.
The smoke bundle did not contain `.git`; full jobs explicitly receive `GIT_COMMIT`
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
