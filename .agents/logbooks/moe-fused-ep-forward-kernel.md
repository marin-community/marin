# MoE Fused EP Forward Kernel: Research Logbook

## Scope
- Goal: determine whether a benchmark-only fused EP forward kernel can beat the current production `grug_moe` EP forward path on the matched EP4 workload.
- Primary metric(s): forward-only steady-state tokens/s on `v5p-8`; secondary metrics are compile time and profile shape.
- Constraints:
  - benchmark-only first; no production integration unless the benchmark wins
  - matched geometry should mirror the real EP4 profile case from `grug-moe-qwen3-32b-a4b-v5p64-bs320-ep4-cf1p0-topk4-matched-active-pf32-buf64-synthetic-profile-iris-main-r1`
  - out of scope for kickoff: backward, SparseCore, and final production API changes

## Baseline
- Date: 2026-03-08
- Code refs:
  - production EP path: [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/grug_moe.py)
  - matched benchmark harness: [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_sparsecore_dispatch.py)
  - prior production-scatter investigation: [/Users/dlwh/.codex/worktrees/4f0b/marin/.agents/logbooks/moe-collect-scatter-production.md](/Users/dlwh/.codex/worktrees/4f0b/marin/.agents/logbooks/moe-collect-scatter-production.md)
- Baseline numbers:
  - matched EP4 benchmark, `implementation=xla`: `compile_s=13.438`, `steady_s=0.079190`, `2.069M tok/s`
  - same shape, failed return-path alternatives:
    - `sorted_scatter_psum_scatter`: `1.739M tok/s`
    - `owner_bucket_all_to_all_local_scatter`: `1.545M tok/s`

## Experiment Log
### 2026-03-08 21:30 - Kickoff
- Hypothesis:
  - the remaining production EP MoE overhead is no longer likely to move with Python/JAX decomposition tweaks, but a fused forward kernel that owns routing, exchange, expert compute, and collect inside one Pallas program may still win.
  - the best external reference is the forward-only TPU fused MoE kernel in vLLM TPU inference:
    - [kernel.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/fused_moe/v1/kernel.py)
- Command:
  - source inspection only
- Config:
  - vLLM reference kernel uses one fused EP kernel with:
    - on-device top-k routing
    - metadata reduction across EP ranks
    - explicit token scatter/gather across ranks
    - fused expert up/down matmuls
    - final output accumulation
- Result:
  - the reference kernel is targeting a materially different decomposition than current `grug_moe`; it is not just a better local gather/scatter kernel.
- Interpretation:
  - this is a plausible next step because it attacks the exact boundary where our production-profile hotspot lives.
  - success should be judged against forward-only throughput first, before thinking about backward.
- Next action:
  - open a dedicated experiment issue
  - build a minimal benchmark-only forward harness that fixes shape and routing assumptions as aggressively as needed for a first comparison

### 2026-03-08 21:45 - Added a forward-only matched-geometry harness
- Hypothesis:
  - before porting any fused kernel logic, the cleanest first step is a forward-only benchmark harness that shares the current production EP path and matched geometry but removes backward from the measurement.
- Command:
  - local validation:
    - `python3 -m compileall lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py`
    - `uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py --batch 1 --seq 8 --hidden 128 --intermediate 64 --experts 4 --topk 2 --expert-axis-size 1 --capacity-factor 1.25 --warmup 0 --iters 1`
- Config:
  - new harness: [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py)
  - initial implementation surface:
    - `xla` baseline only
    - same matched EP4 preset geometry as the earlier production-style benchmark
    - forward-only `jax.jit` of `moe_mlp(...)`
- Result:
  - harness compiles and runs locally on CPU
  - it gives a clean place to add a fused-kernel implementation without carrying forward the older SparseCore benchmark paths
- Interpretation:
  - this is the right benchmark boundary for the next step: compare current forward EP MoE against a benchmark-only fused forward kernel on identical routing/shape inputs
- Next action:
  - benchmark this harness on `v5p-8` to establish a forward-only baseline
  - add the first non-XLA implementation hook for a benchmark-only fused forward kernel

### 2026-03-08 22:00 - Forward-only TPU baseline is strong; direct `tpu-inference` import path is not lightweight enough
- Hypothesis:
  - the new harness should give a materially faster forward-only baseline than the earlier training-style benchmark, and the fastest way to test a real non-XLA candidate might be a direct optional import of the vLLM TPU fused kernel.
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation xla --warmup 1 --iters 2 \
     > /tmp/moe_fused_forward_baseline.out 2>&1'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 && \
   uv run --with git+https://github.com/vllm-project/tpu-inference.git --package levanter \
     python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference --warmup 1 --iters 2 \
     > /tmp/moe_fused_forward_vllm_ref.out 2>&1'
```
- Config:
  - baseline harness: [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py)
  - optional `vllm_reference` implementation hook constructs:
    - `gating_logits` from the benchmark’s fixed `selected_experts` and `combine_weights`
    - `w1` from `w_up_gate` by splitting gate/up projections
- Result:
  - forward-only baseline on matched EP4 geometry (`batch=40, seq=4096, hidden=2048, intermediate=1536, experts=128, topk=4, EP=4`) is:
    - `compile_s=12.628`
    - `steady_s=0.035624`
    - `4.599M tok/s`
  - the direct `uv run --with git+https://github.com/vllm-project/tpu-inference.git ...` path is not viable as a lightweight import route:
    - it downloaded and installed the package
    - but benchmark execution failed with `ModuleNotFoundError: No module named 'vllm'` via `tpu_inference.logger -> vllm.logger`
- Interpretation:
  - the forward-only baseline is now locked in
  - the easiest “just import the reference kernel” route is blocked by an additional vLLM runtime dependency chain, so the next viable path is to vendor or adapt only the TPU fused-kernel pieces we actually need
- Next action:
  - stop trying to use `tpu-inference` as a lightweight optional package dependency
  - vendor a benchmark-only local copy of the fused EP forward kernel and its tuned-block-size helper with minimal dependency surface

### 2026-03-08 22:20 - Vendored kernel reaches lowering; current blocker is scoped-VMEM on our shape
- Hypothesis:
  - vendoring the TPU fused-kernel source directly should be enough to get past the package dependency issue and reveal the next real compatibility constraint on our stack.
- Commands:
  - vendored:
    - [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py)
    - [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/tuned_block_sizes.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/tuned_block_sizes.py)
  - TPU benchmark retry:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference --warmup 1 --iters 2 \
     > /tmp/moe_fused_forward_vllm_ref_local.out 2>&1'
```
- Config:
  - minimal vendoring edits:
    - replace the upstream tuned-block-size logger with a tiny local `warning_once` shim
    - point the kernel import at the local vendored tuned-block-size module
    - JAX compatibility fix: replace `jax._src.dtypes.itemsize_bits` with `jnp.dtype(dtype).itemsize * 8`
    - prepare synthetic `gating_logits` outside the compiled path
    - keep `gating_logits` in `bf16` so the kernel’s DMA source/target dtypes match
- Result:
  - the vendored kernel now lowers far enough to emit its own block-size choice:
    - key `(2048, 1536, 128, 4, 2, 2, 163840, 4)`
    - default block sizes `(128, 768, 1024, 1024, 64, 768, 1024, 1024)`
  - current blocker on `v5p-8`:
    - `104857600 bytes of scoped Vmem requested`
    - max valid bytes is `67043328`
    - XLA reports it is lowering the scoped Vmem request down to the limit
  - I stopped the compile after capturing the message so the dev TPU was not left occupied.
- Interpretation:
  - this is the first real signal from the fused-kernel path on our stack
  - the current failure mode is no longer “missing package” or “obvious JAX incompatibility”; it is that the upstream default block sizes over-allocate scoped VMEM for our target shape on `v5p-8`
  - that makes block-size retuning the next concrete task, not more import/plumbing work
- Next action:
  - add explicit block-size override controls to the harness for `vllm_reference`
  - start by shrinking `bt`, `bf`, `bd1`, and/or `bd2` until the fused kernel fits within `v5p-8` scoped-VMEM limits

### 2026-03-08 22:55 - Apples-to-apples comparison is settled; the router is not the gap
- Hypothesis:
  - the vendored fused kernel may be paying a large hidden cost because it still computes routing internally from dense `gating_logits`, while the current `xla` baseline starts from precomputed `selected_experts` and `combine_weights`.
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && \
   export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=50000 TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation xla_router --warmup 1 --iters 2 \
     > /tmp/xla_router.out 2>&1; tail -n 80 /tmp/xla_router.out'
```
- Config:
  - added `implementation="xla_router"` to the forward harness at
    [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py)
  - `xla_router` reconstructs `topk_idx` and `topk_weights` from the same synthetic `gating_logits` representation used for `vllm_reference`, then calls the existing `moe_mlp` path
- Result:
  - `xla`: `4.599M tok/s`
  - `xla_router`: `4.546M tok/s`
  - `vllm_reference` vendored default: `0.536M tok/s`
- Interpretation:
  - the gap is not explained by the benchmark feeding precomputed routing to `xla` while the fused kernel performs router work internally
  - the fused kernel is slow on its own end-to-end path even when the baseline is forced through the same routing interface
- Next action:
  - stop treating the router mismatch as the main explanation
  - keep tuning the vendored kernel’s schedule and memory shape directly

### 2026-03-08 23:40 - First real tuning win: larger token block helps; VMEM cap itself does not
- Hypothesis:
  - the upstream tuned block sizes are not good for `v5p-8`; the first likely win is to amortize token-exchange overhead with a larger token block rather than to keep shrinking the kernel.
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference \
     --warmup 1 --iters 1 --vllm-vmem-limit-kib 65536 \
     > /tmp/vllm_def64.out 2>&1; tail -n 60 /tmp/vllm_def64.out'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference \
     --warmup 1 --iters 1 --vllm-vmem-limit-kib 65536 \
     --vllm-bt 256 --vllm-bf 768 --vllm-bd1 1024 --vllm-bd2 1024 \
     --vllm-btc 64 --vllm-bfc 768 --vllm-bd1c 1024 --vllm-bd2c 1024 \
     > /tmp/vllm_bt256.out 2>&1; tail -n 60 /tmp/vllm_bt256.out'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference \
     --warmup 1 --iters 1 --vllm-vmem-limit-kib 65536 \
     --vllm-bt 256 --vllm-bf 768 --vllm-bd1 1024 --vllm-bd2 1024 \
     --vllm-btc 128 --vllm-bfc 768 --vllm-bd1c 1024 --vllm-bd2c 1024 \
     > /tmp/vllm_bt256c128.out 2>&1; tail -n 60 /tmp/vllm_bt256c128.out'
```
- Config:
  - exposed the vendored kernel compiler cap as `--vllm-vmem-limit-kib`
  - added `vmem_limit_bytes` as a static arg to
    [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py)
- Result:
  - default vendored block sizes with a `64 MiB` cap:
    - `0.5357M tok/s`
    - still emits the same clamp to `67043328` bytes, so lowering the user-facing cap to `64 MiB` does not change steady-state behavior on its own
  - `bt=256, bf=768, bd1=1024, bd2=1024, btc=64, bfc=768, bd1c=1024, bd2c=1024`:
    - `0.6016M tok/s`
    - about `+12.3%` versus the vendored default
  - `bt=256` but `btc=128`:
    - `0.4653M tok/s`
    - clearly worse
- Interpretation:
  - the first real gain comes from a larger token block, which supports the idea that the kernel is under-amortizing its token-exchange overhead on `v5p-8`
  - increasing the active-expert compute token block (`btc`) at the same `bt` hurts, so the extra win is not from coarser inner FFN compute
  - even the tuned `bt=256` result is still far behind the `xla` baseline (`0.602M` vs `4.546M-4.599M tok/s`)
- Next action:
  - if continuing on this thread, prioritize either:
    - a small additional sweep around `bt=256` that preserves the successful inner compute sizes, or
    - a deeper patch that bypasses the vendored kernel’s in-kernel router/metadata path with precomputed routing

### 2026-03-09 00:05 - Best tuned config uses `bt=256` and a full-width down-projection tile
- Hypothesis:
  - once `bt=256` is established as the only clear win, the cheapest remaining loop-reduction experiment is to widen only one hidden-dimension tile at a time instead of making all expert tiles larger.
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference \
     --warmup 1 --iters 1 --vllm-vmem-limit-kib 65536 \
     --vllm-bt 256 --vllm-bf 768 --vllm-bd1 1024 --vllm-bd2 2048 \
     --vllm-btc 64 --vllm-bfc 768 --vllm-bd1c 1024 --vllm-bd2c 2048 \
     > /tmp/vllm_bt256_d2full.out 2>&1; tail -n 80 /tmp/vllm_bt256_d2full.out'

ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_reference \
     --warmup 1 --iters 1 --vllm-vmem-limit-kib 65536 \
     --vllm-bt 256 --vllm-bf 768 --vllm-bd1 2048 --vllm-bd2 1024 \
     --vllm-btc 64 --vllm-bfc 768 --vllm-bd1c 2048 --vllm-bd2c 1024 \
     > /tmp/vllm_bt256_d1full.out 2>&1; tail -n 80 /tmp/vllm_bt256_d1full.out'
```
- Result:
  - `bt=256, bd1=1024, bd2=2048`:
    - `compile_s=114.784`
    - `steady_s=0.267731`
    - `0.61196M tok/s`
  - `bt=256, bd1=2048, bd2=1024`:
    - `compile_s=111.777`
    - `steady_s=0.270764`
    - `0.60510M tok/s`
  - best tuned configuration so far is therefore:
    - `bt=256`
    - `bf=768`
    - `bd1=1024`
    - `bd2=2048`
    - `btc=64`
    - `bfc=768`
    - `bd1c=1024`
    - `bd2c=2048`
- Integration:
  - added an exact-key tune for `(2048, 1536, 128, 4, 2, 2, 163840, 4)` to
    [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/tuned_block_sizes.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/tuned_block_sizes.py)
  - verified the plain `vllm_reference` benchmark with no manual block overrides now defaults to the tuned shape and reproduces the same best result:
    - `compile_s=114.909`
    - `steady_s=0.267739`
    - `0.61194M tok/s`
- Interpretation:
  - we found a real `v5p-8`-specific tuning improvement and captured it in the vendored table
  - even after tuning, the fused forward kernel is still about `7.5x` slower than the matched `xla` forward baseline (`0.612M` vs `4.546M-4.599M tok/s`)
  - this is now beyond what looks likely to flip with more surface-level tiling
- Next action:
  - if continuing, stop spending time on routine block sweeps
  - the next serious experiment should be a deeper kernel change, most plausibly a precomputed-routing / metadata-bypass path or a more direct adaptation of the reference kernel to our fixed-routing benchmark interface

### 2026-03-09 00:20 - Precomputed-routing path does not materially change throughput
- Hypothesis:
  - the vendored fused kernel may still be paying a significant cost in its in-kernel top-k / local metadata path, so a benchmark-only precomputed-routing mode that bypasses dense gating could materially shrink the gap.
- Implementation:
  - added `use_precomputed_routing` to
    [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/src/levanter/grug/vendor/fused_moe_v1/kernel.py)
  - benchmark-only encoding:
    - reuse the existing `gating_hbm` buffer
    - store `combine_weights[:, k]` in columns `[0:top_k]`
    - store `selected_experts[:, k]` as exact `bf16` integers in columns `[top_k:2*top_k]`
  - inside the kernel, `decode_precomputed_top_k(...)` reconstructs:
    - `top_k_logits_lst`
    - `t2e_routing`
    - local `expert_sizes`
    - zero `expert_starts`
  - added `implementation="vllm_precomputed"` to
    [/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py](/Users/dlwh/.codex/worktrees/4f0b/marin/lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py)
- Commands:
```bash
ssh dev-tpu-dlwh-scdispatch-134559 \
  'source $HOME/.local/bin/env && cd ~/marin && export TPU_STDERR_LOG_LEVEL=2 && \
   uv run --package levanter python lib/levanter/scripts/bench/bench_grug_moe_fused_ep_forward.py \
     --preset qwen3-32b-ep4-profile --implementation vllm_precomputed \
     --warmup 1 --iters 1 > /tmp/vllm_precomputed.out 2>&1; tail -n 120 /tmp/vllm_precomputed.out'
```
- Result:
  - tuned vendored dense-routing path:
    - `compile_s=114.909`
    - `steady_s=0.267739`
    - `0.61194M tok/s`
  - benchmark-only precomputed-routing path:
    - `compile_s=116.376`
    - `steady_s=0.267048`
    - `0.61352M tok/s`
- Interpretation:
  - bypassing the vendored kernel’s internal top-k path is effectively a wash on the matched EP4 forward benchmark
  - the main performance gap is therefore not in dense gating / top-k extraction
  - given:
    - `xla_router`: `4.546M tok/s`
    - tuned fused dense-routing kernel: `0.61194M tok/s`
    - tuned fused precomputed-routing kernel: `0.61352M tok/s`
    the remaining gap is in the fused kernel’s main exchange / expert / accumulation schedule itself, not in the interface mismatch
- Consequence:
  - there is no evidence yet that this forward kernel helps on our `v5p-8` target
  - without a forward win, there is no current justification to start a backward port
- Next action:
  - if this thread continues, it should only be on a materially new kernel idea
  - examples:
    - deeper surgery on the reference kernel’s exchange/metadata path
    - a different fused-forward design
    - testing on different hardware/compiler behavior

### 2026-03-09 00:35 - Final conclusion / seal
- Conclusion:
  - On `v5p-8`, the vendored/reference-style TPU fused forward kernel does not beat the public XLA path for the matched EP4 Grug MoE workload.
  - Best tuned fused forward result: `0.61194M tok/s`
  - Best precomputed-routing fused forward result: `0.61352M tok/s`
  - Matched XLA baselines:
    - `xla`: `4.599M tok/s`
    - `xla_router`: `4.546M tok/s`
- Confidence:
  - `stable` for the scoped claim above: we tested the reference kernel directly, tuned block sizes for `v5p-8`, and removed the router/top-k mismatch as a confounder.
- Decision:
  - Do not start a backward port of this kernel design on the public `v5p` stack.
  - Treat this thread as having reached the kernel wall unless we get materially new ideas, real TPU kernel/compiler help, or different hardware/compiler behavior.
- Ordered next steps:
  1. Use the public XLA/Megablox path for TPU MoE experimentation.
  2. Spend systems effort on TPU-friendly architecture/regime choices rather than more kernel tuning of this design.
  3. Revisit fused TPU kernels only if we get deeper kernel/compiler expertise or access to better low-level machinery.
