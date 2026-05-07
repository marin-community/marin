# H100 Fused CE Hybrid Benchmark Plan

## Goal

Determine whether an H100 fused CE hybrid path is worth keeping for the Grug MoE canary:

- Baseline: current XLA fused CE.
- Candidate: `pallas_gpu` entry point using XLA streaming forward plus custom streaming backward, following the GB10 lesson.

Follow `.agents/skills/agent-research/SKILL.md`: keep a logbook, record exact commands/runs, and compare apples to apples.

## Context

Issue #5510 showed the H100 Grug MoE canary falling back from `pallas_gpu` CE to XLA. That is expected for the current native tiled GPU Pallas CE kernel: the H100 large-vocab shape does not fit the shared-memory constraints.

GB10 exploration suggests the useful experiment is not native Pallas forward. The winning GB10 shape was hybrid: keep XLA forward and replace backward with a custom streaming path.

There is a local prototype in this worktree that:

- detects H100 + large vocab CE,
- skips `pallas_gpu` autotune for that internal-routing shape,
- routes forward through XLA streaming CE,
- uses an H100 custom backward vocab block policy.

## Experiment

1. Recreate or reuse the local H100 hybrid patch.
2. Run a direct H100 CE microbench at the Grug local loss shape:
   - `B=8192`, `H=1024`, `V=128256`
   - `x=bfloat16`, `w=float32`, compute dtype `float32`
   - compare XLA CE vs H100 hybrid CE
   - measure forward and value+grad; value+grad is the important result.
3. Run one Grug MoE H100 canary with the candidate path enabled.
4. Verify the logs clearly record selected CE implementation/path.

## Decision Rule

Keep/PR the hybrid only if it gives a material, reproducible H100 value+grad or canary step-time improvement with acceptable numerical parity.

If it does not move the needle, close #5510 as “XLA CE expected on H100 for this shape” and do not pursue native H100 CE unless a non-truncated profile shows CE is a real bottleneck.

## Do Not Overreach

Do not spend this pass rewriting native H100 Pallas CE. If native Pallas forward is pursued, it should be a separate kernel-design task, not a tuning-table tweak.
