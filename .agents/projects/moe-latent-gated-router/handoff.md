# Gated latent router continuation

User confirmed the intended architecture: `down(h) -> RMSNorm -> gate -> router + experts`.
Retain the latent RMSNorm. Do not restore or compare the superseded gate-only smoke.

Issue: https://github.com/marin-community/marin/issues/9110. Idea: Zihan Qiu.
Worktree: `/Users/kaiyuew/Downloads/Project/marin-latent-gated-router`.
Branch: `codex/moe-latent-gated-router`. Exact training source is in the state file.
Monitoring completed; heartbeat `follow-gated-latent-moe-ablation-9110` was deleted.
State: `scratch/20260911-1105_moe-lgr-9110-rmsgated_monitoring_state.json`.

## Final state

Both RMSNorm-before-gate runs in the first, small-model experiment stage (Gate 1) completed successfully on September 12: d512 Iris attempt 3 and d768 attempt 2 exited 0, and both W&B runs are finished. Permanent checkpoint metadata is verified at steps 13,642 and 19,378. Routing overflow remained zero and recorded metrics were finite. Source: `edf9b2871`; architecture: `down(h) -> RMSNorm -> gate -> router + routed experts`. Idea: **Zihan Qiu**.

| Hidden width | Treatment Paloma loss | Reference Paloma loss | Tokens, treatment / reference | Last-100 mean tokens/sec, treatment / reference | Treatment non-embedding training FLOPs | Recentered compute ratio | Effective speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3.84245491 | 3.54216671 | 1,788,084,224 / 1,439,170,560 | 340,373.87 / 431,263.12 | 3.798955e+17 | 0.217010 | 0.137854 |
| 768 | 3.55822515 | 3.22727251 | 5,079,826,432 / 4,423,680,000 | 200,154.66 / 291,668.98 | 2.798080e+18 | 0.139819 | 0.083556 |

Both widths fail the >1 effective-speedup threshold. Gate 1 fails; no d1024/d1280 runs or Gate 2 scaling fit will be launched. No additional baseline or smoke was run.

Using the [Agent MoE guide](https://github.com/marin-community/marin/blob/codex/moe-latent-gated-router/experiments/grug/moe/agent.md) scaling-law assumption (loss asymptote 1.6, exponent 0.0941), the compute ratio is `((L_reference - 1.6)/(L_treatment - 1.6))**(1/0.0941)`. Effective speedup is a model-based estimate, not a measured acceleration. It divides the estimated reference time to reach the treatment loss (`compute_ratio * reference_tokens / reference_TPS`) by treatment time (`treatment_tokens / treatment_TPS`). TPS is the mean of exactly the last 100 training steps. These times exclude queueing, preemption, compilation, evaluation and checkpoint overhead. Analytic non-embedding training FLOPs are `3 * (forward_FLOPs_per_token - 2*hidden_dim*vocab_size) * treatment_tokens`; they are distinct from the loss-derived compute ratio.

The reused May references use a different architecture, optimizer/data recipe and v4-32 hardware; treatment uses the [scaled hero latent configuration](https://github.com/marin-community/marin/tree/codex/moe-latent-gated-router/experiments/grug/moe_latent_gated_router) on v5p-8. This comparison does not isolate the causal effect of adding the gate or changing router input.

- [d512 treatment](https://wandb.ai/marin-community/marin_moe/runs/moe-lgr-9110-d512-rmsgated) / [reference](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d512_ep1)
- [d768 treatment](https://wandb.ai/marin-community/marin_moe/runs/moe-lgr-9110-d768-rmsgated) / [reference](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d768_ep1)

Final checkpoints: `gs://marin-us-central1/users/kaiyuew/grug/moe-lgr-9110-d512-rmsgated/dev/checkpoints/step-13642` and `gs://marin-us-central1/users/kaiyuew/grug/moe-lgr-9110-d768-rmsgated/dev/checkpoints/step-19378`.

The experiment is complete. Do not submit or resume any cell.

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
