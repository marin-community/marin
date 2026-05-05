# CoreWeave Grug MoE Accelerator Test Plan

## Goal

Measure the current Grug MoE model on CoreWeave H100, GH200, and B200 pools in
single-node and multi-node settings. The output should explain throughput,
MFU, memory headroom, compile/first-step overhead, and scaling behavior.

Assume #5428 lands with GPU JAX/JAXlib `0.10.0` + CUDA 13. Current #5428 signal
from the staged PR-head runtime: H100/GH200/B200 direct JAX smokes passed, and
H100x8/GH200x1/B200x8 Grug training/profiler smokes completed 50 steps.

Treat #5452 as a blocker for multi-node performance claims, not for the whole
study. Single-node H100/GH200/B200 work can proceed. Multi-node benchmark rows
must wait until the readiness gate below proves the exact runtime branch can
complete a cheap distributed Grug path.

## Operating Model

- Propose each real-cluster batch with stop criteria, then run to that stop
  condition after approval.
- Ask again only for higher cost, new hardware/cluster scope, live cluster
  mutation, stopping shared jobs, or destructive recovery.
- Keep a newest-first action log at
  `/tmp/marin-cw-grug-moe-accelerator-perf/action-log.md`.
- Redirect large command output to `/tmp/marin-cw-grug-moe-accelerator-perf/`
  and record those paths in the final report.

## Research Discipline

- Use one long-lived branch: `research/cw-grug-moe-accelerator-perf`.
- Keep an append-only research logbook:
  `.agents/logbooks/cw-grug-moe-accelerator-perf.md`.
- Use stable experiment IDs, e.g. `CW-GRUG-001`, in W&B run names, logbook
  entries, and reports.
- Keep comparable runs in the same W&B project/group so charts are easy to
  build.
- Label major claims as `exploratory`, `replicated`, or `stable`.
- When a meaningful matrix is complete, tag the commit and include a repro
  bundle: commands, commit, hardware, batch/steps, env vars, W&B links, and the
  result table.

## Current Validation Status

Status as of 2026-05-05, before #5428 has landed on `origin/main`:

- Current #5428 branch confirmation: direct JAX/CUDA 13 smokes passed on
  `research/cw-grug-moe-accelerator-perf` commit `2574d5c55`, based on #5428
  head `7b53485c`, for H100/GH200/B200. Logs are
  `/tmp/marin-cw-grug-moe-accelerator-perf/direct-*-20260505-2253.log`.
- Dependency precheck: `uv lock --check` passed. Local macOS cannot install the
  CUDA 13 JAX wheels, so exact GPU package versions were verified inside
  CoreWeave jobs.
- Direct JAX/CUDA 13 smokes passed on H100, GH200, and B200. All reported
  JAX `0.10.0`, `jax-cuda13-plugin==0.10.0`, `jax-cuda13-pjrt==0.10.0`,
  CUDA 13 NVIDIA packages, backend `gpu`, and a successful bf16 matmul.
- Short Grug smokes passed:
  - H100x8 on `coreweave-ci`: `/romain/cw-grug-smoke-h100-20260505-2157`.
  - GH200x1 on `coreweave-rno2a`: `/romain/cw-grug-smoke-gh200-20260505-2153`.
  - B200x8 on `coreweave-usw09b`: `/romain/cw-grug-smoke-b200-20260505-2157`.
- Each short smoke reached 50/50 steps, saved a checkpoint, produced a profiler
  window, and exited cleanly. Treat these as scheduling/runtime/model-plumbing
  evidence only, not benchmark throughput.
- The GH200 smoke exposed real single-device sharding issues in current Grug MoE;
  the local worktree now contains fixes in `experiments/grug/moe/model.py`.
- GH200/B200 MFU prerequisites are now covered locally by Fray GPU type and FLOP
  mappings plus focused tests.
- W&B was offline in these jobs, so profiler artifacts were produced in the job
  container but are not durable cloud artifacts yet.

## Metrics

Report for every benchmark cell:

- p50/p90 steady-state step time.
- mean and p50 tokens/sec.
- tokens/sec/GPU and tokens/sec/node.
- MFU, if device FLOP mapping is valid.
- compile time and first-step time, separate from steady state.
- final loss sanity, memory high-water/OOM boundary, loading time, hook time.
- profile bottleneck class when profiled: compute, NCCL/collectives, host/data,
  memory, compile/autotune, or profiler/logging overhead.

Before interpreting MFU, verify `NVIDIA GH200 480GB` and `NVIDIA B200` map
correctly in the Levanter/Fray device FLOP path. Tokens/sec remains valid even
if MFU mapping needs a patch.

## Test Ladder

### 1. Dependency Check

On the exact commit under test:

- `uv lock --check`.
- Confirm `gpu` resolves JAX/JAXlib `0.10.0` and CUDA 13 packages.
- Confirm `cpu`, `tpu`, and `vllm` do not pull CUDA/NVIDIA runtime packages.

### 2. Direct JAX Smokes

Run streaming direct jobs, not `--no-wait`, using #5431/#5440 CoreWeave guidance:

```bash
uv run iris --cluster=<cluster> job run \
  --enable-extra-resources --cpu=4 --memory=16G --disk=32G \
  --gpu=<gpu-request> --extra=gpu \
  -- python -c '<nvidia-smi + jax backend/devices + tiny bf16 matmul>'
```

Rows:

| Target | Cluster/config | GPU request |
| --- | --- | --- |
| H100 | `coreweave-ci` | `H100x1` |
| GH200 | `coreweave-rno2a` | `H200x1` |
| B200 | `coreweave-usw09b` | `B200x1` |

Fresh venv note: install `uv pip install 'marin-iris[controller]'` before
CoreWeave CLI use.

### 3. Short Training Smokes

Run 40-60 step current-model Grug MoE smokes with profiler enabled:

| Target | Shape | Purpose |
| --- | --- | --- |
| H100 | 1 node x 8 GPU | confirm post-merge parity with #5428 canary |
| GH200 | 1 node x 1 GPU | first real training/profiler evidence |
| B200 | 1 node x 8 GPU | first real training/profiler evidence |

Current status: complete on staged #5428 runtime. Re-run once after #5428 is on
merged main if we need merged-main evidence before spending benchmark budget.

Stop if any smoke fails. If full current Grug MoE does not fit on GH200x1,
record that as memory evidence rather than switching the main comparison to a
smaller model.

### 4. Single-Node Benchmarks

Profiler off for timing runs. Use 200-300 steps, discard warmup/compile/first
step, and measure the final 100 steps.

Initial batch sweep:

| Target | Shape | Initial batches |
| --- | --- | --- |
| H100 | H100x8 | `16, 24, 32, 48, 64` |
| GH200 | GH200x1 | `1, 2, 4, 8, 16` |
| B200 | B200x8 | `16, 24, 32, 48, 64, 96` |

Repeat the best 1-2 batch sizes per accelerator three times.

### 5. Multi-Node Readiness Gate

Do not run multi-node performance sweeps until #5452 is separated into its
component failure modes. Current evidence says CoreWeave H100 can scale to two
hosts and JAX distributed can start, but `main` has not yet produced a clean
end-to-end two-host Grug canary.

Run this gate before any multi-node benchmark matrix:

1. **Workflow/controller sanity.** Confirm the chosen launch path can submit the
   parent job and child Grug job without the parent disappearing from the fresh
   controller DB. This is a workflow/Iris reliability check, not a Grug result.
2. **Direct two-host JAX collective smoke.** Run a minimal two-replica H100 job
   that prints process count/index, sees one GPU per host, initializes JAX
   distributed, completes a tiny `psum` or all-gather, completes one bf16
   matmul, and exits cleanly.
3. **Tiny two-host Grug smoke.** Run current Grug MoE for 10-20 steps, profiler
   off, with a tight timeout. The pass condition is visible step/loss progress
   and clean terminal state; do not infer performance from this run.
4. **Escalate hardware only after H100 passes.** Start with H100 because #5452
   is already H100 evidence. Add B200/GH200 multi-node readiness checks only
   after H100 is clean.

Stop and report if any gate step fails. The report should identify the failing
layer: workflow/controller, scheduling/networking, JAX distributed/NCCL, or
Grug training.

### 6. Multi-Node Benchmarks

Only run this section after the multi-node readiness gate passes on the same
commit/runtime. Then run weak and strong scaling:

| Target | Nodes |
| --- | --- |
| H100 | 1, 2, maybe 4 |
| B200 | 1, 2, maybe 4 |
| GH200 | 1, 2, 4, maybe 8 |

Weak scaling: hold per-GPU batch constant. Strong scaling: hold global batch
constant where feasible. Report scaling efficiency as `speedup / node_count`.

### 7. Targeted Profiles

Profile only the cells needed to explain results:

- best single-node H100, GH200, and B200;
- first H100/B200 multi-node cell where scaling drops;
- best and first-bad GH200 multi-node cells.

Use short profile windows after warmup to avoid the event cap observed in #5428.
Summarize with `lib/marin/tools/profile_summary.py`.

## Final Outputs

The final output should be a durable benchmark report, not just W&B links.

- Checked-in research logbook:
  `.agents/logbooks/cw-grug-moe-accelerator-perf.md`.
- W&B report with charts/tables for single-node throughput, batch sweeps,
  tokens/sec/GPU, tokens/sec/node, MFU where valid, multi-node scaling
  efficiency, and selected profile summaries.
- Concise Markdown report titled something like
  `CoreWeave H100/GH200/B200 Grug MoE Performance Characterization`.
  Put it under `docs/reports/` if broadly useful, or `.agents/projects/` if it
  remains operational/internal.
- Repro bundle covering commit/tag, exact commands, hardware/config mapping,
  env vars, profiler windows, W&B run table, and caveats.
- Optional code/docs PR for benchmark harness changes, MFU mapping fixes, or
  new runbook sharp edges discovered during the study.

The report should make concrete recommendations for current Grug MoE:

- best accelerator for single-node training,
- best accelerator for multi-node training, only if the readiness gate passes;
  otherwise state that multi-node is blocked and report the failing layer,
- where GH200 fits, if anywhere,
- max stable batch per target,
- expected throughput and scaling efficiency,
- bottleneck diagnosis,
- residual risks and unmeasured cases.

## Completed First Approval Batch

Completed on staged #5428 runtime, before #5428 landed on `origin/main`:

1. H100/GH200/B200 direct JAX smokes passed on the staged runtime.
2. H100x8, GH200x1, and B200x8 50-step training/profiler smokes passed on the
   staged runtime.
3. GH200/B200 device FLOP mapping support was added locally.

Notes:

- The "merged main" part remains provisional until #5428 actually lands.
- Direct JAX smokes passed on all three accelerators.
- Training/profiler smokes passed on all three target single-node shapes.
- GH200 required local model sharding fixes before it passed.
- GH200/B200 MFU mapping support was added locally.

## Next Approval Batch

After #5428 lands, do a cheap merged-main confirmation first:

1. Re-run the three direct JAX smokes.
2. Re-run one short Grug smoke per accelerator only if the merged commit differs
   materially from this staged runtime.
3. Make W&B/profiler capture durable before any benchmark matrix: either provide
   online W&B credentials to the Iris task env or capture profiler outputs to
   durable storage.

Then run the single-node benchmark matrix:

- H100x8: 200-300 steps at batch sizes `16, 24, 32, 48, 64`.
- GH200x1: 200-300 steps at batch sizes `1, 2, 4, 8, 16`.
- B200x8: 200-300 steps at batch sizes `16, 24, 32, 48, 64, 96`.
- Repeat the best 1-2 batch sizes per accelerator three times.

Keep multi-node out of the benchmark matrix until the #5452 readiness gate
passes on the same runtime. It can be proposed as a separate approval batch in
parallel with, or immediately after, the single-node benchmark work.

## Guardrails

- Do not restart clusters, scale NodePools, stop shared jobs, or mutate live
  cluster state without explicit approval.
- Executor parent jobs must stay CPU-only.
- Use `--gpu` for hardware and `--extra=gpu` for dependencies.
- For GH200, request `H200x1`; verify `NVIDIA GH200 480GB` in `nvidia-smi`.
- Prefer synthetic data for accelerator characterization; avoid cross-region
  data movement in the main benchmark.
