---
topic: grug-context-parallel-attention
issue: 8141
description: Evaluate and implement context-parallel attention for the Grug hero shape through sequence length 262144.
author: dlwh
---

# Grug Context-Parallel Attention: Task Logbook

## Current TL;DR

NVIDIA Transformer Engine JAX remains the first implementation candidate, but the current Marin GPU
image cannot qualify it. TE 2.17.1 fails during cuDNN backward graph construction for both the exact
262144-token Grug cases and NVIDIA's official 65536-token CP4 control. The newer source-pinned TE
commit cannot build because the staged CUDA toolkit lacks `cicc`. No XLA timing or HBM result exists,
and no AllGather or 64-GPU job was submitted. The WIP branch contains the exact-shape benchmark and
distributed launcher; [#8141](https://github.com/marin-community/marin/issues/8141) is the public
coordination record. The next TE attempt requires a complete CUDA development toolkit. The fallback
is a global ring plus a bounded 511-token halo for the 40 SWA512 layers.

## Scope

- Goal: Train the Grug hero attention shape through sequence length 262144 without replicating the
  full KV sequence on every device, while preserving value and gradient parity.
- Primary metrics: attention forward/backward step time, effective TFLOP/s, communication bytes,
  peak HBM, and max/mean value and gradient error against a reference.
- Constraints: BF16 compute; causal packed-document masks; 48 query heads; 12 local and 6 global KV
  heads; head dimension 128; 512-token local windows; every sixth layer global; existing GB200 hero
  layout is the first target.
- Coordinating issue/PR: [#8141](https://github.com/marin-community/marin/issues/8141). No direct
  duplicate was found on 2026-08-10.

## Baseline

- Date: 2026-08-10
- Code ref: `a5f0269edc35a3766958adb494cef7d371632ebd`
- Branch: `codex/research/grug-context-parallel-attention`
- Baseline shape: batch 1024, sequence 4096, 64-way FSDP per rack, GPU FA4/CuTe attention.
- Target shape ladder: 4096, 16384, 65536, 131072, and 262144 tokens. A fixed-token comparison
  uses global batches 1024, 256, 64, 32, and 16 respectively.
- Baseline numbers: not measured in this thread yet.

## Hypothesis Queue

### Active

- `GCPA-002`: The existing Grug FA4/CuTe local kernel can be reused inside a ring protocol by
  exposing output log-sum-exp and defining a distributed custom VJP. Next test: inspect FA4 forward
  and backward residuals and identify the minimum local-kernel API change.
- `GCPA-004`: The 40 SWA512 layers may need a bounded-halo path if TE CP communication dominates
  their small attention workload. Next test: compare TE Ring/AllGather with a local halo oracle and
  retain full CP only for the eight global layers if the result warrants the integration cost.

### Blocked

- `GCPA-003`: TE AllGather may beat TE Ring for CP4 on a single NVLink domain. Blocker: TE 2.17.1
  fails backward graph construction before XLA, and the newer source cannot build on the current
  image. Resume when a current TE build passes NVIDIA's official CP4 control.
- `GCPA-005`: The newer source-pinned TE commit can run the exact Grug shape on GB200. Blocker: the
  Iris CUDA environment lacks nvcc's `cicc` device compiler. Resume when the task image exposes a
  complete CUDA development toolkit.

### Falsified / Dead End

- `GCPA-001`: TE 2.17.1 can run the exact Grug attention shapes on the current JAX 0.11/CUDA 13/
  cuDNN 9.19 image. The exact cases and NVIDIA's official control reproduce the same
  `CUDNN_STATUS_BAD_PARAM`; see [#8141](https://github.com/marin-community/marin/issues/8141).

### Promoted

- None.

## Decision Log

- 2026-08-10: Target the GB200 FSDP hero first. The checked-in hero explicitly selects
  `gpu_fa4_cute`; TPU Splash is relevant prior art and a fallback comparison, not the current hero
  backend.
- 2026-08-10: Keep the initial work in a local logbook. Create a coordinating public experiment
  issue only after the requested scout establishes a concrete experiment plan or the user requests
  public coordination.
- 2026-08-10: Evaluate TE before modifying Grug FA4. TE already implements the missing distributed
  protocol and VJP; a custom FA4 ring remains the fallback if TE cannot meet compatibility,
  correctness, or performance gates.
- 2026-08-10: Start with mesh `(data=16, context=4)` on one 64-GPU rack, with `context` innermost so
  each CP group stays within one four-GPU Iris task. Keep parameter FSDP over composite
  `(data, context)` so CP does not increase parameter and optimizer memory fourfold.
- 2026-08-10: Keep hidden states in contiguous sequence shards and apply TE's causal striping only to
  Q/K/V and sequence descriptors at the attention boundary. Grug's causal SConv, fused RoPE, labels,
  loss, and packed-segment semantics require natural token order outside attention.
- 2026-08-10: Open [#8141](https://github.com/marin-community/marin/issues/8141) as the coordinating
  experiment issue after the TE release and source gates failed. Keep the branch as WIP until a
  four-GPU forward/backward control passes.

## Negative Results Index

- No existing Marin issue or PR directly covers Grug context-parallel or ring attention at sequence
  length 262144. Exact GitHub searches are recorded in the entry below.
- The generic Levanter FA4 path and Grug FA4/CuTe path shard batch or heads but explicitly reject
  sequence sharding.
- The old Levanter 131K optimization issue recorded a benchmark plan but no winning result.
- No JAX distributed Pallas/Splash training kernel was found in Marin.

## Entry Log

### 2026-08-10 14:28 PDT - GCPA-000 research prologue

- Hypothesis: Existing Marin or external attention work may provide the distributed sequence
  protocol, leaving only Grug integration and tuning.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd`
- Command: repository `rg` over Grug, Levanter kernels, reports, projects, and logbooks; narrow GitHub
  issue and PR searches for context parallelism, ring attention, and 262144-token attention.
- Config: Grug FSDP hero at d6144, 48 layers, QH48, local KVH12, global KVH6, head dim 128, BF16,
  sliding window 512, global attention every six layers, 64 GB200 GPUs per rack.
- Result: The current TPU Splash wrapper supports sharded query positions but rejects KV-sequence
  sharding. The GPU FA4 wrapper has no distributed sequence protocol. No direct coordinating issue
  or PR was found.
- Interpretation: The remaining scout should prioritize distributed orchestration and backward
  semantics, not a new single-device flash-attention tile kernel.
- Next action: Rank external reuse candidates and inspect the existing FA4 local-kernel residuals.

### 2026-08-10 15:32 PDT - GCPA-001 high-effort kernel scout

- Hypothesis: An existing maintained JAX kernel can supply the distributed attention protocol and
  backward pass for the exact Grug shape.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd`
- Effort: high. The scout covered current Marin code/history, GitHub issues and PRs, NVIDIA TE source,
  release notes, API docs, examples, tests, and adversarial failure reports. Stop rule: additional
  sources no longer changed the ranked candidate or the minimum experiment.
- Result: NVIDIA Transformer Engine JAX is the only found GPU implementation that covers the required
  feature intersection: distributed sequence sharding, Ring and AllGather, forward/backward, BF16,
  GQA, THD packed causal attention, sliding windows, and causal load balancing. Stable tag 2.17.1
  contains the CP API and tests. Use exact commit `598b9eacbe9fc34ec105cf8c12f303108ca434ca`
  for the initial spike because it also contains the source-qualified example. That example uses CP4
  on four GB200 GPUs with batch 2, sequence
  65536, QH128, KVH8, D128, four padded THD segments, causal SWA8192, and BF16. Its published output
  is 126.687 ms without CP, 57.167 ms Ring, and 53.792 ms AllGather for forward/backward.
- Interpretation: The missing first-order work is Grug/mesh/packed-data integration and runtime
  qualification, not a new attention tile or distributed VJP. Exact 262144 support remains an
  experimental claim until the harness runs on the hero image.
- Artifact: `lib/levanter/scripts/bench/bench_grug_context_parallel_attention.py` exercises exact
  local Q48/KV12/SWA512 and global Q48/KV6/full-causal shapes over the fixed-token sequence ladder.
  It emits compile time, steady-state time, estimated TFLOP/s and communication, environment, and
  unsupported-shape/runtime failures as JSON lines.
- Validation: script help completed under the shared Marin environment; Pyrefly reported zero
  errors; the required file-scoped pre-commit checks passed. No GPU kernel was compiled because TE is
  absent from the current environment.
- Caveats: Ring requires stripe size 1. Ring plus SWA requires the non-scan implementation selected
  by `NVTE_FUSED_RING_ATTENTION_USE_SCAN=0`. THD metadata and segment positions must be striped with
  Q/K/V, and `max_segments_per_seq` is static. Recent TE releases fixed JAX ring crashes and packed
  sequence sharding bugs, so older versions are poor candidates. PyTorch TE issue #2186 reported NaN
  gradients for heavily padded THD+CP inputs in an older release; although it was not JAX, the
  minimum experiment must include padding and gradient parity.
- Supported CP layouts are BSHD with packed KV, separate BSHD Q/K/V, THD with packed KV, and separate
  THD Q/K/V; fully packed T3HD is not supported. THD CP accepts only the padding-causal mask, no
  bias, no attention dropout, and vanilla softmax. These match the current hero, provided attention
  sinks or other mask semantics are not enabled.
- Grug integration impact: the current mesh is fixed to `(replica_dcn, data, expert, model)`; current
  FA4 requires equal local Q/K/V sequence lengths and rejects sequence sharding; current global
  layers expand logical KV6 back to physical KV12 before FA4. A separate TE global branch should
  preserve KV6 and halve its communicated KV volume. Four causal SConv sites need a three-token left
  halo under contiguous CP sharding. Loss and MoE routing need distinct batch axes (`data`) and token
  axes (`data`, `context`) with reductions over both.
- Minimum experiment: on four GB200 GPUs, install the pinned CUDA-13-compatible TE build in an isolated
  image, run local and global 4096-token CP4 forward/backward, and compare outputs and Q/K/V gradients
  to unsharded attention with one, many, and heavily padded segments. Then run 65536 to compare Ring
  and AllGather against the official scale. Only after those pass, use the 64-GPU rack with
  `(data=16, context=4)` for 131072 and 262144, recording compile time, steady-state time, HBM, and
  communication. Reject TE if it cannot compile the exact shapes, violates parity, or loses to the
  FA4 baseline after communication is included.
- Fallback experiment: extend Grug FA4 to rectangular local-Q/remote-KV blocks with global query
  offsets, merge per-block FP32 log-sum-exp across a ring, and implement the matching distributed
  custom VJP. Use a bounded three-token/SWA halo for local layers. This is materially larger than an
  outer `shard_map` and should begin only after a concrete TE failure.
- TPU alternative: MaxText's vendored Tokamax ring attention is credible prior art for the global
  layers. MaxText PR #4537 reports Llama 3.1 8B at sequence 262144, QH32/KVH8/D128, CP16 on v5p-128
  at 41.24 seconds and 199.6 TFLOP/s/device with load balancing versus 63.32 seconds unbalanced; later
  PRs validate packed gradients and improve long-sequence backward. It does not support local SWA,
  and the requested hero currently targets GB200, so porting it is below the TE GPU probe.
- Sources:
  - [TE JAX CP example](https://github.com/NVIDIA/TransformerEngine/blob/598b9eacbe9fc34ec105cf8c12f303108ca434ca/docs/examples/jax/attention_context_parallel.py)
  - [TE JAX attention API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html)
  - [TE 2.17.1 source tag](https://github.com/NVIDIA/TransformerEngine/tree/v2.17.1)
  - [MarinSkyRL 131K CP campaign](https://github.com/marin-community/marin/issues/6656)
  - [MarinSkyRL CP implementation](https://github.com/marin-community/MarinSkyRL/blob/7695402c829ce2efaa96e1c7a3bc8b9778c1de4a/skyrl-train/skyrl_train/distributed/cp_utils.py)
  - [Levanter context mesh PR](https://github.com/marin-community/marin/pull/2187)
  - [Levanter context attention fix](https://github.com/marin-community/marin/pull/2188)
  - [Grug segmented FA4 PR](https://github.com/marin-community/marin/pull/5880)
  - [Generic FA4 GPU attention PR](https://github.com/marin-community/marin/pull/7015)
  - [TE THD+CP padded-gradient issue](https://github.com/NVIDIA/TransformerEngine/issues/2186)
  - [MaxText Tokamax ring adapter](https://github.com/AI-Hypercomputer/maxtext/blob/ef71a22420af7ca526a51c783907a3fcbf7cf506/src/maxtext/kernels/attention/tokamax_ring_attention.py)
  - [MaxText 262144-token ring results](https://github.com/AI-Hypercomputer/maxtext/pull/4537)
- Next action: qualify the TE build on the current GB200 image and run the minimum four-GPU
  correctness/performance gate before changing the hero model.

### 2026-08-10 14:48 PDT - GCPA-005 full 262144-token launch preparation

- Hypothesis: TE Ring or AllGather can compile and execute the exact fixed-token hero attention shape
  at global batch 16 on a `(data=16, context=4)` mesh across 64 GB200 GPUs.
- Commit Hash: `e93589c5d8684744a5c666c11c2db9067b22654f`
- Config: sequence 262144; batch 16; BF16; QH48; D128; local KVH12/SWA512 and global
  KVH6/full-causal; THD padding-causal mask; CP4; one JAX process per GPU; one data shard per
  four-GPU Iris task; Ring stripe 1 and AllGather stripe 512.
- Live capacity: `cw-us-east-08a` reported 12 of 804 GB200s free, with 640 held by production, 136 by
  interactive, and 16 by batch workloads. A 64-GPU interactive gang cannot admit immediately.
- Implementation: add an Iris/Fray launcher that requests 16 four-GPU GB200 tasks, installs pinned
  TE CUDA 13 wheels, starts four supervised JAX processes per task, calls `initialize_jax()` before
  importing the benchmark, and records JSON only from process zero. The generic Grug dispatcher now
  accepts additional pinned pip packages instead of duplicating its environment construction.
- Gate order: use one currently available four-GPU node to validate the TE wheel tuple and API, then
  submit the 16-node gang. Do not occupy 64 GPUs for dependency resolution or a known API mismatch.
- Source-build probe: job `/dlwh/grug-cp-te-build-bench-20260810-1454` tried TE commit
  `598b9eacbe9fc34ec105cf8c12f303108ca434ca` on one four-GB200 node. CMake selected
  `/app/.venv/bin/nvcc` but failed compiler identification because `cuda_runtime.h` was absent. The
  Iris GPU setup stages CUDA 13 compiler binaries and runtime libraries, not CUDA development
  headers. The job failed before kernel compilation and released the node.
- Qualified base tuple from that task: Python 3.12.13; JAX, jaxlib, and CUDA 13 plugin 0.11.0; CUDA
  runtime 13.0.96; cuDNN 9.19.0.56; NCCL 2.30.7; four visible GB200 GPUs. TE was absent before the
  probe.
- Decision: retain commit `598b9eacbe9fc34ec105cf8c12f303108ca434ca` as source/API provenance.
  Run the published TE 2.17.1 CUDA 13 wheels next; do not add CUDA headers to Marin's task image
  unless the release wheel lacks the required CP API.
- Release-build probe: parent `/dlwh/grug-cp-te217-s262k-ring1-coord`, child
  `/dlwh/grug-cp-te217-s262k-ring1-coord/grug-train-grug-cp-te217-s262k-ring1`. The aarch64
  `transformer-engine-jax==2.17.1` package is an sdist. Its isolated build omitted
  `nvtx3/nvToolsExt.h`, pulled both CUDA 12 and explicit CUDA 13 core packages, and failed before JAX
  initialization. The child and its four-GB200 allocation terminated cleanly.
- Environment fix: install `transformer_engine[core_cu13,jax]==2.17.1` with `uv pip install
  --no-build-isolation`. The non-isolated setup can see the existing CUDA 13 compiler and
  `nvidia-nvtx-cu12` header package, allowing TE's setup logic to select its matching core package.
- CUDA 13 retry: parent `/dlwh/grug-cp-te217-s262k-ring2-coord`, child
  `/dlwh/grug-cp-te217-s262k-ring2-coord/grug-train-grug-cp-te217-s262k-ring2`. The non-isolated
  build found NVTX and reached C++ compilation, then failed because CUDA 13's `cuda_fp16.h` includes
  `<nv/target>` from a nested CCCL directory that TE did not add. JAX 0.11 also lacks the private
  runtime-version accessor TE checks, so TE fell back to CUDA 12 dependency selection. The job
  terminated and released its node.
- Next environment fix: request the metapackage extras `core_cu13,jax`; set `CUDA_VERSION=13.0`;
  force NVIDIA-wheel include discovery; add `/app/.venv/lib/python3.12/site-packages/nvidia/cu13/include/cccl`
  to `CPLUS_INCLUDE_PATH`; compile only SM100; and disable the unused NCCL expert-parallel extension.
- Path probe: job `/dlwh/grug-cp-cuda-path-probe-20260810-1506` used one GB200 for 11.47 seconds
  and succeeded. The child received `CUDA_VERSION=13.0` and `CPLUS_INCLUDE_PATH`, but no `nv/target`
  file existed anywhere in the environment. NVTX existed at
  `/app/.venv/lib/python3.12/site-packages/nvidia/nvtx/include/nvtx3/nvToolsExt.h`; NVCC was
  `/app/.venv/bin/nvcc`. JAX, jaxlib, the CUDA plugin, and PJRT were all 0.11.0;
  `cuda_runtime_get_version()` returned 13000; runtime CUDA was 13.0.96 and NVCC was 13.2.78. The
  node was released.
- Path-probe interpretation: `nvidia-cuda-cccl` is absent from Marin's GPU environment. PyPI resolves
  a Linux aarch64 wheel for `nvidia-cuda-cccl==13.3.3.4.1`. Install CCCL and the CUDA 13 TE core in a
  completed setup phase before building the JAX extension without isolation; installing them in the
  same resolver transaction does not guarantee the headers exist before the sdist build.
- Successful import gate: `/dlwh/grug-cp-te-staged-import4-20260810-1520` succeeded in 2m27.94s on
  one GB200 and released it. Install `nvidia-cuda-cccl==13.3.3.4.1`,
  `nvidia-cudnn-frontend==1.25.0`, and `transformer_engine_cu13==2.17.1` first. Create a task-local
  `libnccl.so -> libnccl.so.2` symlink in the NVIDIA NCCL wheel directory, expose that directory via
  `LIBRARY_PATH` and `LD_LIBRARY_PATH`, then install `transformer_engine_jax==2.17.1` with
  `--no-build-isolation --no-deps` and the `transformer_engine==2.17.1` metapackage with `--no-deps`.
  TE, TE-JAX, and TE-cu13 reported version 2.17.1; all ten benchmark attention symbols,
  `MeshResource`, and TE autocast imported successfully. The only warning was the expected missing
  optional PyTorch extension.
- First 262144-token Ring gate: parent `/dlwh/grug-cp-te217-s262k-ring5-coord`, child
  `/dlwh/grug-cp-te217-s262k-ring5-coord/grug-train-grug-cp-te217-s262k-ring5`, commit
  `7f53a82da5`. One four-GB200 node ran the exact fixed-token batch 16 local KV12/SWA512 and global
  KV6/full-causal cases. Both reached TE/cuDNN before XLA compilation, then returned
  `CUDNN_STATUS_BAD_PARAM` from `fused_attn_f16_arbitrary_seqlen.cu:934`. cuDNN frontend reported
  both `libcudart.so.12` and `libcudart.so.13` and selected CUDA 12. The benchmark recorded both
  failures as JSON and exited cleanly; no HBM allocation or timing occurred.
- Runtime fix: set `CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13`, the documented selector for
  environments containing more than one CUDA runtime, and rerun the same four-GB200 gate.
- Corrected 262144-token Ring gate: parent `/dlwh/grug-cp-te217-s262k-ring6-coord`, child
  `/dlwh/grug-cp-te217-s262k-ring6-coord/grug-train-grug-cp-te217-s262k-ring6`, commit
  `5ee751bb85`. The CUDA 12 selection warning disappeared, but both exact Grug cases failed at the
  same TE/cuDNN graph-construction call with `CUDNN_STATUS_BAD_PARAM`. The failure occurs while
  querying the backward workspace, before XLA compilation, device HBM allocation, or timing. The
  child and coordinator completed, and the four GB200s were released. This rules out mixed CUDA
  runtime selection as the root cause.
- Official control: job `/dlwh/grug-cp-te217-official-ring-control2-20260810-1535` ran NVIDIA's
  source-pinned CP4 Ring example from commit `598b9eacbe9fc34ec105cf8c12f303108ca434ca` with verified
  Git blob `1557a30b7ccc216a7b52225f74a5cea26ed91fe6`. TE recognized the published B2/S65536/QH128/KVH8/
  D128/SWA8192/four-segment shape as supported, then the first backward warmup failed in
  `get_fused_attn_bwd_workspace_sizes` with the identical reshape-mode `CUDNN_STATUS_BAD_PARAM`.
  No warmup, XLA compilation, timing, or HBM measurement completed. The task failed and released
  all four GB200s. A preceding job with suffix `-1533` never imported TE because its ephemeral
  `python -c` wrapper had invalid quoting; it was corrected and is not a kernel result.
- Interpretation: TE 2.17.1's JAX CP backward is not qualified on Marin's JAX 0.11/CUDA 13/cuDNN
  9.19 GB200 image. Because NVIDIA's unmodified 65536-token control reproduces the error, the current
  blocker is not the 262144-token Grug shape or its descriptor sharding. Do not submit the 64-GPU
  gang against this release build.
- Exact-source build gate: four bounded one-GB200 tasks attempted to build commit
  `598b9eacbe9fc34ec105cf8c12f303108ca434ca`; every allocation terminated and was released. Job
  `/dlwh/grug-cp-te-source598-import-20260810-1541` failed before compilation because CMake was not
  installed. Job suffix `-1542` installed the declared build requirements and reached nvcc, then
  failed because the split CUDA package had not exposed `crt/host_config.h`. Job suffix `-1543`
  stopped before the build because the ephemeral probe used the wrong wheel namespace. Final job
  `/dlwh/grug-cp-te-source598-import4-20260810-1544` discovered the installed header, staged a
  task-local `include/crt` symlink, and advanced through preprocessing. nvcc 13.2.78 then exited 127
  because `/app/.venv/nvvm/bin/cicc` was absent. CMake inferred the CUDA root from the staged
  `/app/.venv/bin/nvcc`, but the Iris split-wheel environment does not contain the internal device
  compiler at the path nvcc requires. No TE source object, Python extension, or API imported.
- Exact source-build tuple: Python 3.12.13; JAX, jaxlib, CUDA 13 plugin, and PJRT 0.11.0; CUDA runtime
  13.0.96; nvcc, CUDA CRT, and NVVM packages 13.2.78; CCCL 13.3.3.4.1; cuDNN 9.19.0.56;
  cuDNN frontend 1.25.0; NCCL 2.30.7; CMake 4.4.2; Ninja 1.13.0. The build used
  `NVTE_FRAMEWORK=jax`, SM100, NVIDIA wheel headers, and the task-local NCCL link path.
- Decision: stop the bounded TE gate. The 2.17.1 release build has a reproducible backward graph
  failure even on NVIDIA's official control, while the newer source cannot be built with the CUDA
  components staged by the current Iris image. No AllGather or 64-GPU data16-by-CP4 gang was
  submitted. The next TE attempt requires a deliberate image/toolchain change that supplies a
  complete CUDA development toolkit, not another benchmark retry. Otherwise proceed to the custom
  Grug design: TE-or-ring global attention plus a bounded 511-token halo path for SWA512, with the
  mesh, SConv halo, packed metadata, and loss/MoE sharding changes identified above.
