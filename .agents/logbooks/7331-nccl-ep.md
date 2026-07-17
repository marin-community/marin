# TransformerEngine NCCL_EP expert-parallel MoE on B200 — logbook

Issue: https://github.com/marin-community/marin/issues/7331
Branch: `research/mcwitt/7331-nccl-ep` (logbook, probe scripts, build recipes).
Bench work builds on the standalone MFU benchmark branch `mcwitt/moe-standalone-ep`
(`experiments/grug/moe/standalone`); NCCL_EP bench commits go on a child branch of it.

Experiment IDs: `NCCLEP-###`. W&B tags: `nccl-ep`, `7331` (plus per-run tags as useful).

## Goal

NCCL_EP working on B200-class GPUs at **64 GPUs with EP≥8** at the reference
"prod" config from #7012/#7279 — the point is to derisk running at scale:

- Reference config (B200MFU-032): **d5120 L48 e64 top4 b1024 seq4096**, 64 GPUs
  = replica-2 × 32-GPU FSDP+EP model copy (`--replica-axis-size 2`), cf 1.0,
  MuonH, 20 steps / warmup 8, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
- Baselines to beat/compare (B200MFU-032/-033): ring_cute EP4 **20.83 %**,
  EP1 (pure FSDP) 20.22 %, a2a_cute EP8 **19.12 %**, pure-XLA raa EP8 18.75 %;
  production driver (FSDP-only) 23.32 %. EP8 is the current hard ceiling at
  b1024 (all EP16/32/64 arms fail, B200MFU-033); the EP≥8 CUBIN-load failure is
  *intermittent* (B200MFU-035) and not yet reliably worked around.
- Platform: cw-us-east-08a GB200 NVL72 (4-GPU arm64 Grace nodes); 64-GPU jobs
  = 16-node iris gangs, hard-colocated into one rack (all-NVLink/MNNVL).

## Context (from #7012 logbook, `.agents/logbooks/7012-b200-moe-mfu.md` on `research/mcwitt/7012-b200-mfu`)

- `H-te` (Blocked there; this issue unblocks it): TE 2.15.0+42b840051 wheel does
  **not** ship the NCCL_EP JAX surface (probe 2026-07-15) — building from the WIP
  branch is prerequisite #1. TE `grouped_dense` already proven QuACK-class
  (1,449 TF/s, B200MFU-011); NGC container recipe (apptainer + PYTHONPATH
  overlay + CPU-torch) works on this stack.
- Source-read facts (TE PRs #3034/#3036 + WIP branch
  `jberchtold/teddy-te-ep-integration-2026-07-08-support-quantization`):
  - "8192-token limit" = `max_tokens_per_rank` staging-buffer default at
    `ep_bootstrap`, not structural. NB the reference config is **65,536
    tokens/rank** (16 seq/GPU × 4096) — staging buffers must be sized up 8×.
  - Dispatch wire bf16-only on the main PR; fp8 wire on the quantization WIP branch.
    Quantize happens post-dispatch (`grouped_quantize` → `grouped_gemm`).
  - One-sided SM specialization: comm `max_num_sms` only; GEMM opportunistic.
  - Requires **process-per-GPU** (`local_device_count()==1`) — our gangs run one
    process per 4-GPU node, so the bench needs a 4-procs/node (64-process) launch mode.
  - Disables NCCL comm-splitting and command-buffer capture around the EP FFI ops.
- `H-smspec` falsified twice (B200MFU-016/-021): SM-capping NCCL CTAs loses
  monotonically on our stack. The part of NCCL_EP under test is the
  chunked-pipeline decomposition + fused dispatch, not the SM budget.
- XLA dispatch baselines for the microbench: `ragged_all_to_all` one-shot kernel
  ~297 ms/call (B200MFU-018); NCCL fallback via
  `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` (B200MFU-025).
- Ops gotchas inherited: GPU jobs need `--cpu 16 --memory 64g` (default OOMs
  `import jax`); `iris job logs/list` take `--cluster=cw-us-east-08a` (not
  `--target-cluster`, which is `job run`-only); `bash -lc` drops the task venv
  (use `bash -c`); check printed module `__file__`s against stale imports;
  MNNVL is the default cross-node transport inside a rack.

## Hypothesis queue

### Active
- `H-build`: TE + NCCL_EP is buildable from the WIP branch on this stack
  (arm64 Grace + sm_100a + cu13 container). Open sub-questions: is the
  NCCL_EP backend's NCCL dependency (libnccl_ep) public/vendored, or does the
  build need an NVIDIA-internal artifact? Does the JAX surface build need extra
  flags? **First derisk item — resolve by source read before spending GPU time.**
- `H-ep8-prod`: TE NCCL_EP dispatch/combine runs at the reference config with
  EP≥8 across 64 GPUs and is competitive with (or beats) a2a_cute EP8 19.12 %.
  Sub-risks: 65k tokens/rank staging memory; process-per-GPU × 26–48-layer scan
  compile behavior; no-command-buffer + no-comm-split interaction with the rest
  of the step; the intermittent EP≥8 CUBIN failure (B200MFU-035) — NCCL_EP
  changes the dispatch path but not XLA's per-kernel-module loading.
- `H-microbench`: at matched shapes, NCCL_EP dispatch+combine ≪ XLA raa NCCL
  fallback per call (it should pipeline staging chunks and skip host syncs).

### Blocked / later
- `H-fp8wire`: fp8 dispatch wire (quantization WIP branch) halves dispatch
  bytes; ties into MXFP8 work (#7282). After bf16 wire works.

## Decision log
- 2026-07-17: kick off from #7331; branch `research/mcwitt/7331-nccl-ep`;
  reference config + baselines frozen to B200MFU-032 values.

## Entries
