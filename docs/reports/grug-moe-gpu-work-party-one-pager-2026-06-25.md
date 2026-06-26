# Grug MoE GPU Work Party One-Pager

Date: 2026-06-25 PT

## Goal

Get enough people oriented to move Grug MoE GPU throughput work in parallel. The current engineering target is the May d2560/L26 Grug MoE shape on CoreWeave H100s, with `model_axis=1`.

## Baselines To Use

| Purpose | Run | Config | Result |
|---|---|---|---|
| Current N2 fwd/bwd baseline | [May402](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-402S4096-W2048-B64-R2-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-NOFLAGS-N2-cw-20260624-020812) | N2, B4/device, ring, SGD, FA4, Pallas CE, `recompute_all`, no flags | 20.63 MFU, 190.8k tok/s, 1.374s |
| Current N4 fwd/bwd baseline | [May403](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-403S4096-W2048-B128-R4-E8M1-PALLASCE-RING-RECOMPUTEALL-FA4SGD-NOFLAGS-N4-cw-20260624-021515) | N4, B4/device, same as May402 | 19.93 MFU, 368.6k tok/s, 1.422s |
| Readable attribution profile | [May215](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-215S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4SGD-MOESCOPED-PROFILE-N1-cw-20260621-0022) | N1, B8, ring, SGD, XLA CE, scoped MoE profile | 16.07 MFU, 74.3k tok/s, 0.441s |
| Muon cost A/B | [May140](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-140S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4MUON5-SCOPED-PROFILE-N1-cw-20260618-0623) vs [May141](https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-141S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4SGD-AUTOTUNE-PROFILE-N1-cw-20260618-0637) | N1, B8, MuonH5 vs SGD | MuonH5: 1.492s; SGD: 0.438s |

Do not use O1/JAX-Toolbox flags as a default. May404 regressed May402 by about 11%. AutoPGLE without O1 was at best a small final-step improvement and emitted empty-trace warnings.

## Workstreams

| Workstream | Starting point | Before tomorrow / first owner task | Success criterion | Owner |
|---|---|---|---|---|
| Pallas/Mosaic fused MoE kernel | May215 shows ring MoE transport/layout and `w13` expert GEMM are both hot. | Draft or refine the kernel boundary: fused dispatch plus `W_gate/W_up + SiLU`, with backward requirements written down. | Tiny single-node harness beats the scoped ring `gather_inputs + w13_ragged_dot` region, or gives a clear compiler/kernel blocker. |  |
| MuonH grouped-bank path | May140 vs May141 attributes about 1.05s/step to Muon/update. Grouped-bank harness work is promising but still not production-good. | Make the grouped-bank harness avoid OOM and report stable perf; do not assume it is ready for model integration. | No OOM, stable timing, and no extra grouped-to-FSDP boundary collectives in compiled HLO. |  |
| Fwd/bwd baseline and remat | May402/May403 are the current runnable baselines. B8/device does not fit in ring-MoE `save_moe`/`recompute_all` tests. | Keep future tests one-axis-at-a-time against May402/May403. | One named phase moves by at least 10% without losing stability. |  |
| Cross entropy | #6572 landed H100 B-tiled CE. #6596 says the backward gap is mostly recompute-vs-materialize. | Verify the May path is using the intended H100 B-tiled CE where expected. | CE phase is measured in a full profile and attributed to B-tiled forward/backward rows. |  |
| FA4 attention | #6613 merged native SM90 D128 GQA backward. Old FA4 batch all-gather was fixed by shard_map. | Confirm the SM90 path is active in the May H100 profile; decide what remains for #6377. | FA4 profile rows map to the expected SM90 path and are no longer dominated by accidental batch all-gather. |  |
| Roofline/profile dashboard | #6573 tracks May208 roofline attribution. May215 already has a phase table. | Turn May215 and May402/403 into a reusable dashboard/phase-table format. | Every new profile has MoE transport, expert GEMM, CE, FA4, optimizer, collectives, and unknown buckets. |  |
| CoreWeave ops | #6637 fixed CUDA toolchain staging. #6473 added `dev_gpu.py`. N4 clique startup remains flaky. | Document the stable launch/profile serving commands and record clique/rendezvous failures. | People can reproduce a profile and know which failures are infra vs model/code. |  |

## Fill-In Experiment Table

Use this table for any result presented in the meeting.

| Run | Owner | Hypothesis | Nodes | Batch/device | Remat | MoE backend | Optimizer | Flags | MFU | tok/s | step s | Main profile movement | Decision |
|---|---|---|---:|---:|---|---|---|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Fill-In Workstream Task Table

Use this table to assign work.

| Task | Workstream | Owner | Input artifact | Command or branch | Done when | Blocker |
|---|---|---|---|---|---|---|
| Prototype fused dispatch + `W_gate/W_up + SiLU` harness | Pallas/Mosaic MoE |  | May215 phase table |  | correctness test + first H100 timing |  |
| Stabilize grouped-bank Muon harness | MuonH |  | #6493 harness |  | no OOM + stable perf row |  |
| Verify CE path in May profile | CE |  | May215 or fresh readable profile |  | B-tiled rows identified |  |
| Verify FA4 SM90 active | FA4 |  | #6613 + May profile |  | SM90 row visible or code-path proof |  |
| Build phase table | Roofline |  | May215, May402/403 |  | dashboard/table attached to #6573 |  |

## Copy/Paste Profile Commands

Run these from the repo root:

```bash
cd /Users/dlwh/.codex/worktrees/d91e/marin
```

Serve the already-downloaded May215 readable MoE profile:

```bash
uv run --with tensorboard tensorboard \
  --logdir=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profiles/issue4312_may215 \
  --port=6030
```

Download and serve May141, the SGD no-Muon single-node profile:

```bash
cd /Users/dlwh/.codex/worktrees/d91e/marin
uv run --with tensorboard python lib/levanter/scripts/wandb_tensorboard_profile.py \
  marin-community/marin_moe/GM2560-MAY-141S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4SGD-AUTOTUNE-PROFILE-N1-cw-20260618-0637 \
  --download-root=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profiles/tb-may141 \
  --port=6031
```

Download and serve May140, the MuonH5 single-node profile:

```bash
cd /Users/dlwh/.codex/worktrees/d91e/marin
uv run --with tensorboard python lib/levanter/scripts/wandb_tensorboard_profile.py \
  marin-community/marin_moe/GM2560-MAY-140S4096-W2048-B8-R1-E8M1-XLACE-RING-REPEMB-REPOUT-FA4MUON5-SCOPED-PROFILE-N1-cw-20260618-0623 \
  --download-root=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profiles/tb-may140 \
  --port=6032
```

Serve the local May390 DeepEP wrapper-tax profile if you want historical DeepEP context:

```bash
uv run --with tensorboard tensorboard \
  --logdir=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profiles/may390 \
  --port=6033
```

Generate or refresh a structured summary for May215:

```bash
cd /Users/dlwh/.codex/worktrees/d91e/marin
uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize \
  --profile-dir=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profiles/issue4312_may215 \
  --xplane-output-dir=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profile_reports/issue4312_may215/xprof_tables \
  --breakdown-mode=exclusive_global \
  --output=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profile_reports/issue4312_may215/summary.json
```

Render the May215 markdown report:

```bash
cd /Users/dlwh/.codex/worktrees/d91e/marin
uv run python lib/marin/tools/profile_summary.py report \
  --summary=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profile_reports/issue4312_may215/summary.json \
  --output=/Users/dlwh/.codex/worktrees/d91e/marin/scratch/profile_reports/issue4312_may215/report.md
```

## Meeting Opening Script

The baseline is now clear enough to split work. We have a 20 MFU runnable fwd/bwd configuration on N2/N4 at B4/device with `recompute_all`. We have a readable single-node profile showing MoE transport/layout, expert GEMM, CE, and FA4 all matter. DeepEP has been too fragile to be the main plan, so the MoE kernel lane should focus on a custom Pallas/Mosaic fused dispatch plus gate/up/SILU kernel. Muon is a separate lane: grouped bank is still an OOM/perf problem, not a solved representation.

The meeting should assign owners to phases, not just discuss "GPU perf" broadly.
