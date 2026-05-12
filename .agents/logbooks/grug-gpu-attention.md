# Grug GPU Attention: Research Logbook

## Scope
- Goal: cover the Grug attention API on GPU with native attention execution, especially `segment_ids`.
- Primary metrics: value/gradient parity vs `reference_attention`; compile-including and steady-state timings on GH200x1.
- Constraints: keep edits attention-specific; avoid MoE files touched by the SonicMoE worker; one short GH200x1 Iris job before any tuning sweep.
- References:
  - Tokamax flex attention: https://github.com/openxla/tokamax/tree/main/tokamax/_src/ops/flex_attention
  - Tokamax Mosaic GPU attention: https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/attention/pallas_mosaic_gpu.py
  - Local Tokamax refs under `/tmp/tokamax-attn-refs/`

## Baseline
- Date: 2026-05-10
- Code refs:
  - `lib/levanter/src/levanter/grug/attention.py`
  - `/tmp/tokamax-attn-refs/pallas_mosaic_gpu.py`
  - `/tmp/tokamax-attn-refs/flex_pallas_triton.py`
- Baseline numbers: pending GH200 benchmark.

## Gap Matrix
| Feature | Grug/Splash need | Tokamax Mosaic GPU | Tokamax flex Pallas-Triton | JAX cuDNN SDPA prototype |
| --- | --- | --- | --- | --- |
| GQA/MQA | KV heads can divide Q heads | Native in Tokamax base API | Native in Tokamax flex base API | Native in `jax.nn.dot_product_attention` |
| causal | Required | Native mask decomposition | Native via `mask_mod` | Native `is_causal=True` |
| sliding-window causal | Required | Native through decomposed `k_start/k_end`/causal windows | Native via `mask_mod` | Native with `local_window_size=(W - 1, 0)` plus causal |
| 1D segment IDs | Required | Not a first-class API in fetched Mosaic wrapper; would need mask decomposition/index support | Natural via captured `mask_mod` values | Supported by bool mask `[1,1,Q,K]` |
| 2D/batch segment IDs | Required | Same blocker as 1D plus batched mask decomposition | Natural via captured `mask_mod` values | Supported by bool mask `[B,1,Q,K]` |
| dense bool mask | Nice-to-have fallback okay | Supported if decomposable; arbitrary masks uncertain | Natural via `mask_mod` | Supported by bool mask |
| dense additive mask | Fallback okay | Bias supported | `score_mod`/bias path possible | fallback to reference for now |
| bf16/f16/f32 | Required | Explicitly supported | Supported by Triton path | Supported by JAX API subject to cuDNN constraints |
| gradients/training | Required | VJP module required | Tokamax op path supports residuals/VJP machinery | JAX API provides autodiff |

## Recommendation
- Short-term implementation PR: use the JAX cuDNN SDPA API as the native GPU bridge for Grug attention. It can exercise real cuDNN attention on GH200 while preserving the existing Grug/Splash semantic API and explicit fallback.
- Longer-term kernel work: Tokamax flex is the better adaptation target for arbitrary Grug masks because `mask_mod` can express `segment_ids`, causal, and sliding-window semantics without teaching Mosaic's mask decomposition about packed segments. Tokamax Mosaic is likely faster for decomposable masks but needs a design spike for native segment-aware decomposition and VJP integration before it should be ported.

## Experiment Log
### 2026-05-10 00:00 - kickoff and local inspection
- Hypothesis: JAX 0.10 cuDNN SDPA can provide the smallest correct native GPU path for Grug `segment_ids`; Tokamax flex is the best future Pallas adaptation target.
- Commands:
  - `sed -n '1,220p' /Users/dlwh/.codex/worktrees/3957/marin/.agents/skills/add-pallas-kernel/SKILL.md`
  - `sed -n '1,220p' /Users/dlwh/.codex/worktrees/3957/marin/.agents/skills/agent-research/SKILL.md`
  - `sed -n '1,260p' lib/levanter/src/levanter/grug/attention.py`
  - `sed -n '260,460p' lib/levanter/src/levanter/grug/attention.py`
  - `sed -n '1,240p' /tmp/tokamax-attn-refs/flex_pallas_triton.py`
  - `sed -n '1,220p' /tmp/tokamax-attn-refs/pallas_mosaic_gpu.py`
  - `uv run --package marin-levanter python - <<'PY' ... inspect jax.nn.dot_product_attention ... PY`
- Result:
  - Current Grug GPU path is pure `reference_attention`.
  - `jax.nn.dot_product_attention` in JAX 0.10 has `implementation='cudnn'`, accepts GQA/MQA shapes, bool `mask`, additive `bias`, `is_causal`, `local_window_size`, seq lengths, and residuals.
  - cuDNN local-window support maps causal left-window to `sliding_window_length=l_window + 1`.
- Interpretation: prototype a small Grug mask-to-JAX-SDPA bridge, keep it explicit (`implementation="gpu_cudnn"`), and validate on GH200 before changing defaults.
- Next action: add conversion helpers, correctness tests, and a short GH200 benchmark harness.

### 2026-05-10 00:20 - local prototype and CPU parity tests
- Hypothesis: the mask conversion into JAX SDPA args preserves Grug/Splash semantics before we try cuDNN on GH200.
- Commands:
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
  - `uv run ruff check --fix .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
- Config:
  - CPU-local tests exercise `jax.nn.dot_product_attention(..., implementation="xla")` against `reference_attention`.
  - Cases: no mask, causal, causal sliding window, non-causal Grug sliding-window semantics, 1D segment IDs, 2D/batch segment IDs, dense bool mask, GQA/MQA-style KV head expansion, and gradients.
- Result:
  - Pytest: `9 passed, 2 skipped in 13.44s`.
  - Ruff: `All checks passed!`.
- Interpretation: segment IDs are covered by the conversion tests locally; the remaining question is whether cuDNN accepts the same boolean segment mask path and whether it is faster on GH200.
- Next action: launch one short GH200x1 Iris job with `--max-retries 0`.

### 2026-05-10 06:31 UTC - GH200 correctness/benchmark launch
- Hypothesis: cuDNN SDPA accepts Grug segment masks represented as boolean `[B,1,Q,K]` masks and beats the materialized reference path on the representative shape.
- RUN_ID: `grug-gpu-attn-20260510-063122`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin-levanter --group dev python .agents/scripts/grug_gpu_attention_bench.py --iters 3 --warmup 1`
- Config:
  - Correctness shape: `B=2, S=64, Hq=4, Hkv=2, D=32`, causal sliding window, 2D segment IDs.
  - Benchmark shape: `B=8, S=1024, Hq=16, Hkv=4, D=128`, dtype bf16, 4 packed segments, sliding window 256.
- Result:
  - Submitted as `/dlwh/grug-gpu-attn-20260510-063122`.
  - Pod `iris-dlwh-grug-gpu-attn-20260510-063122-0-d29e1386-0` failed before the benchmark ran.
  - Logs: `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.` Then the harness raised `RuntimeError: This benchmark requires a JAX GPU backend.`
- Interpretation: the Iris command selected CPU JAX because it omitted the GPU extra; this is not a cuDNN/attention result.
- Next action: relaunch once with `--extra gpu`.

### 2026-05-10 06:32 UTC - GH200 relaunch with GPU JAX extra
- Hypothesis: using the Marin GPU extra installs CUDA-enabled JAX on the GH200 node.
- RUN_ID: `grug-gpu-attn-gpujax-20260510-063222`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin --extra gpu --group dev python .agents/scripts/grug_gpu_attention_bench.py --iters 3 --warmup 1`
- Result:
  - Submitted as `/dlwh/grug-gpu-attn-gpujax-20260510-063222`.
  - Reached `backend=gpu devices=['cuda:0']`, so `--extra gpu` fixed CUDA JAX.
  - Failed in `_check_correctness` on `jax.jit(gpu_cudnn_attention)(q, k, v, mask)` with small `B=2, S=64, Hq=4, Hkv=2, D=32`, bf16, causal sliding-window, and 2D segment IDs.
  - Error: `jax.errors.JaxRuntimeError: INTERNAL: [cudnn_frontend] Error: No valid execution plans built.`
- Interpretation: cuDNN reached GPU but could not build a plan for the first segment-mask case. Need a feature matrix before any broad benchmark.
- Next action: add cuDNN feature matrix harness and launch one short diagnostic job.

### 2026-05-10 06:36 UTC - cuDNN feature matrix harness
- Hypothesis: cuDNN SDPA may work for causal/no-mask subsets but fail when Grug `segment_ids` require a boolean `[B,1,Q,K]` mask.
- Commands:
  - `uv run ruff check --fix .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
- Config:
  - Added `--mode cudnn-matrix` to `.agents/scripts/grug_gpu_attention_bench.py`.
  - Matrix cases: no mask, causal, non-causal sliding, segment-only 1D, segment-only 2D, causal+2D segment, causal sliding+2D segment; each run with no-GQA (`Hq=2,Hkv=2`) and GQA (`Hq=4,Hkv=2`).
- Result:
  - Local tests: `10 passed, 3 skipped in 13.11s`.
  - Ruff: `All checks passed!`.
  - Pyrefly: `INFO 0 errors`.
- Interpretation: ready for short GH200 diagnostic; do not run broad benchmark until matrix result is known.
- Next action: launch `--mode cudnn-matrix` on GH200x1 with `--max-retries 0`.

### 2026-05-10 06:35 UTC - GH200 cuDNN feature matrix launch
- Hypothesis: cuDNN support will split by mask feature; segment masks are the expected failure mode.
- RUN_ID: `grug-gpu-attn-cudnn-matrix-20260510-063539`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin --extra gpu --group dev python .agents/scripts/grug_gpu_attention_bench.py --mode cudnn-matrix --dtype bf16`
- Result: launched; awaiting logs.
- Next action: inspect pod logs with `kubectl -n iris get pods` / `kubectl -n iris logs`.

### 2026-05-10 06:36 UTC - GH200 cuDNN feature matrix result
- RUN_ID: `grug-gpu-attn-cudnn-matrix-20260510-063539`
- Pod: `iris-dlwh-grug-gpu-attn-cudnn-matrix-20260510-063539-859bc07d-0`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris get pods | rg "grug-gpu-attn-cudnn-matrix-20260510-063539"`
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris logs iris-dlwh-grug-gpu-attn-cudnn-matrix-20260510-063539-859bc07d-0 --tail=400`
- Result:
  - Pod completed successfully after printing the matrix.
  - Backend: `backend=gpu devices=['cuda:0']`.
  - All bf16 cuDNN cases failed with `JaxRuntimeError: INTERNAL: [cudnn_frontend] Error: No valid execution plans built.`
  - Failed cases:
    - `no_gqa:no_mask`
    - `no_gqa:causal`
    - `no_gqa:sliding`
    - `no_gqa:segment_1d`
    - `no_gqa:segment_2d`
    - `no_gqa:causal_segment_2d`
    - `no_gqa:sliding_segment_2d`
    - `gqa:no_mask`
    - `gqa:causal`
    - `gqa:sliding`
    - `gqa:segment_1d`
    - `gqa:segment_2d`
    - `gqa:causal_segment_2d`
    - `gqa:sliding_segment_2d`
- Interpretation:
  - On this GH200/JAX 0.10/CUDA setup, JAX cuDNN SDPA cannot currently be trusted even as a no-mask fast path for the tiny diagnostic shape.
  - This does not isolate segment masks as the only issue, but it confirms cuDNN is not the implementation path for Grug segment IDs right now.
  - Keep `gpu_cudnn` explicit/experimental only; do not make it default.
  - Prioritize a Tokamax flex/Pallas adaptation for real segment-ID flash attention because its `mask_mod` hook is the natural fit for Grug/Splash segment semantics.
- Next action: keep the prototype fallback path explicit, document unsupported cuDNN status in final report, and avoid broad cuDNN benchmark relaunch.

### 2026-05-10 06:40 UTC - direct cuDNN no-mask sanity harness
- Hypothesis: direct `jax.nn.dot_product_attention(..., implementation="cudnn")` with no mask and simple no-GQA shapes distinguishes a wrapper/layout bug from a backend/image limitation.
- Source/layout check:
  - JAX 0.10 source documents query shape `(B,T,N,H)` and key/value shape `(B,S,K,H)`.
  - Grug passes q/k/v as `[B,Q,Hq,D]`, `[B,K,Hkv,D]`, `[B,K,Hkv,D]`; these match the JAX source contract directly.
  - The wrapper passes these arrays unchanged to `jax.nn.dot_product_attention`.
- Commands:
  - `uv run ruff check --fix .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run pyrefly check .agents/scripts/grug_gpu_attention_bench.py lib/levanter/src/levanter/grug/attention.py lib/levanter/tests/grug/test_attention.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
- Config:
  - Added `--mode cudnn-direct` with direct no-mask calls for fp16 and bf16, no GQA, simple shapes: `B=1,S=128,H=1,D=64`, `B=2,S=128,H=2,D=64`, and `B=2,S=256,H=2,D=128`.
- Result:
  - Ruff: `All checks passed!`
  - Pyrefly: `INFO 0 errors`
  - Tests: `10 passed, 3 skipped in 19.87s`
- Next action: launch one short GH200 direct no-mask diagnostic with `--max-retries 0`.

### 2026-05-10 06:38 UTC - GH200 direct cuDNN no-mask launch
- Hypothesis: if direct no-mask cuDNN fails for fp16/bf16 simple shapes, the issue is the GH200 image/backend rather than the Grug wrapper or segment masks.
- RUN_ID: `grug-gpu-attn-cudnn-direct-20260510-063842`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin --extra gpu --group dev python .agents/scripts/grug_gpu_attention_bench.py --mode cudnn-direct`
- Result: launched; awaiting logs.
- Next action: inspect pod logs.

### 2026-05-10 06:39 UTC - GH200 direct cuDNN no-mask result
- RUN_ID: `grug-gpu-attn-cudnn-direct-20260510-063842`
- Pod: `iris-dlwh-grug-gpu-attn-cudnn-direct-20260510-063842-cddc9f4a-0`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris get pod iris-dlwh-grug-gpu-attn-cudnn-direct-20260510-063842-cddc9f4a-0`
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris logs iris-dlwh-grug-gpu-attn-cudnn-direct-20260510-063842-cddc9f4a-0 --tail=400`
- Result:
  - Pod completed successfully after printing direct no-mask results.
  - Backend: `backend=gpu devices=['cuda:0']`.
  - Direct `jax.nn.dot_product_attention(..., implementation="cudnn")` failed for every no-mask case:
    - `fp16_B1_S128_H1_D64`
    - `bf16_B1_S128_H1_D64`
    - `fp16_B2_S128_H2_D64`
    - `bf16_B2_S128_H2_D64`
    - `fp16_B2_S256_H2_D128`
    - `bf16_B2_S256_H2_D128`
  - All failures were `JaxRuntimeError: INTERNAL: [cudnn_frontend] Error: No valid execution plans built.`
- Interpretation:
  - The Grug wrapper shape/layout is not the cause: direct no-mask JAX cuDNN calls fail too.
  - Treat cuDNN as non-viable in this GH200 image as tested.
  - Stop cluster work on cuDNN.
  - Pivot to Tokamax flex/Pallas for segment-aware attention. The minimal adapter boundary should be a Grug `AttentionMask` to Tokamax `mask_mod` closure that returns a bool mask broadcastable to scores shape `[*B, Hq, Q, K]`.
- Next action: add a local Tokamax-flex-style mask adapter scaffold and tests for causal, sliding-window, and 1D/2D segment IDs.

### 2026-05-10 06:45 UTC - Tokamax flex adapter boundary scaffold
- Hypothesis: the smallest useful Tokamax flex/Pallas boundary is independent of the kernel port: convert Grug `AttentionMask` into a `mask_mod(scores_shape)` closure with the same semantics as Splash/reference.
- Commands:
  - `uv run ruff check --fix lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
- Result:
  - Added `lib/levanter/src/levanter/grug/flex_attention.py`.
  - Added `grug_attention_mask_for_scores(mask, scores_shape)`, returning a bool mask broadcastable to Tokamax flex scores shape `[*B, H, Q, K]`.
  - Added `grug_flex_mask_mod(mask)`, returning `None` for no mask or a `mask_mod(scores_shape)` closure for structured Grug masks.
  - Local tests: `16 passed, 3 skipped in 12.59s`.
  - Pre-commit targeted files: `OK`.
- Concrete implementation boundary:
  - Input contract remains Grug q/k/v `[B,Q,Hq,D]`, `[B,K,Hkv,D]`, `[B,K,Hkv,D]` and `AttentionMask`.
  - Tokamax flex adapter should pass q/k/v unchanged to a flex implementation with `mask_mod=grug_flex_mask_mod(mask)`.
  - The mask closure returns:
    - causal masks as `[1,1,Q,K]`;
    - sliding-window masks as `[1,1,Q,K]`;
    - 1D segment masks as `[1,1,Q,K]`;
    - 2D/batch segment masks as `[B,1,Q,K]`;
    - combined causal/sliding/segment masks through the existing `AttentionMask.materialize_mask` semantics.
  - Dense additive masks are not part of the first flex adapter; keep them on the reference path.
  - Dense bool masks can be supported later either by a captured mask array or by reference fallback.
- Critical missing features for Grug's current attention signature:
  - Actual Pallas/Triton flex kernel port or dependency wiring is not yet in-tree.
  - Backward/gradient parity for the Pallas path remains untested until the kernel is wired.
  - Runtime block-size config/tuning is not implemented.
  - cuDNN is non-viable in this GH200 image as tested, even for direct no-mask fp16/bf16 calls.
  - `gpu_xla` supports the mask semantics but is not a flash/streaming implementation and was not made default.
- Recommendation:
  - Proceed with Tokamax flex/Pallas as the implementation path for #5610-style segment-aware GPU attention.
  - Treat this branch as a prototype/scaffold unless the next step ports/wires the flex Pallas implementation and validates value + gradient parity on GH200.

### 2026-05-10 07:05 UTC - Tokamax flex callable backend wiring
- Hypothesis: wrapping Tokamax `PallasTritonFlexAttention` directly is the smallest runnable implementation spike; the broken copied/installed `flex_attention.api` should be bypassed.
- Commands:
  - `sed -n '1,620p' /tmp/tokamax-attn-refs/flex_pallas_triton.py`
  - `sed -n '1,260p' /tmp/tokamax-attn-refs/flex_base.py`
  - `uv run --package marin-levanter python - <<'PY' ... import tokamax._src.ops.flex_attention.pallas_triton ... PY`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
- Result:
  - Installed/copied Tokamax `flex_attention.api` is not usable as-is: it references `PallasTritonGatedLinearUnit`, but the module exposes `PallasTritonFlexAttention`.
  - `PallasTritonFlexAttention` imports directly, so the wrapper bypasses `api.py`.
  - Added explicit Grug backend name: `gpu_flex_pallas`.
  - `gpu_flex_pallas` is non-default and raises:
    - `NotImplementedError("gpu_flex_pallas requires the JAX GPU backend.")` off GPU;
    - `NotImplementedError("gpu_flex_pallas does not support dense masks yet.")` for dense masks;
    - explicit Tokamax import/load errors if the kernels extra is absent or the installed package shape changes.
  - Added `tokamax_flex_attention(..., implementation="xla" | "pallas_triton")` to validate the same Grug mask boundary through Tokamax locally.
  - Local Tokamax-XLA value parity and gradient parity pass for causal, sliding-window, and causal+sliding+2D segment IDs.
  - Tests: `21 passed, 3 skipped in 13.98s`.
  - Pyrefly: `INFO 0 errors`.
- Caveat:
  - The current `grug_flex_mask_mod` returns a broadcastable full logical mask. Tokamax Pallas uses `fuser.pull_block_spec` to tile `mask_mod`, but this spike still starts from Grug materialization semantics. If fuser cannot decompose this efficiently for segments, the next patch should replace it with an index-derived mask expression rather than a full-mask expression.
- Next action: launch one short GH200x1 correctness run for `gpu_flex_pallas`; do not run broad tuning.

### 2026-05-10 06:50 UTC - GH200 Tokamax flex Pallas launch
- Hypothesis: the direct `PallasTritonFlexAttention` wrapper lowers/runs on GH200 for the small Grug causal sliding-window + 2D segment-ID correctness case.
- RUN_ID: `grug-gpu-attn-flex-pallas-20260510-065029`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin-levanter --extra gpu --extra kernels --group dev python .agents/scripts/grug_gpu_attention_bench.py --implementation gpu_flex_pallas --batch 2 --seq 64 --q-heads 4 --kv-heads 2 --head-dim 32 --sliding-window 16 --iters 1 --warmup 0 --skip-reference-timing`
- Result: launched; awaiting logs.
- Next action: inspect pod logs with kubectl.

### 2026-05-10 06:51 UTC - GH200 Tokamax flex Pallas result
- RUN_ID: `grug-gpu-attn-flex-pallas-20260510-065029`
- Pod: `iris-dlwh-grug-gpu-attn-flex-pallas-20260510-065029-a16f8bb8-0`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris get pod iris-dlwh-grug-gpu-attn-flex-pallas-20260510-065029-a16f8bb8-0`
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris logs iris-dlwh-grug-gpu-attn-flex-pallas-20260510-065029-a16f8bb8-0 --tail=500`
- Result:
  - Backend: `backend=gpu devices=['cuda:0']`.
  - Job failed before kernel execution in Tokamax Pallas lowering.
  - Error:
    - `TypeError: pull_block_spec() got an unexpected keyword argument 'grid'`
    - Origin: `tokamax/_src/ops/flex_attention/pallas_triton.py`, `_tile_score_mod`, calling `fuser.pull_block_spec(fn, spec, grid=grid)`.
  - JAX 0.10 local API exposes `fuser.pull_block_spec(..., grid_len=...)`, not `grid=...`.
- Interpretation:
  - This is a concrete Tokamax/JAX bridge mismatch, not a Grug mask semantic failure.
  - The copied `/tmp/tokamax-attn-refs/flex_pallas_triton.py` already uses `grid_len=len(grid)`, so upstream/ref code appears ahead of installed `tokamax==0.0.6`.
  - Added an experimental compatibility shim at the Grug wrapper boundary that lets Tokamax's old `grid=` call map to JAX 0.10 `grid_len=`.
  - No additional GH200 launch in this pass; next short run should test the shim and then expose the next blocker, if any.
- Commands after patch:
  - `uv run ruff check --fix lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
- Result after patch:
  - Ruff: `All checks passed!`
  - Pyrefly: `INFO 0 errors`
  - Tests: `21 passed, 3 skipped in 14.16s`
- Next patch boundary:
  - Run one more short GH200 job only after deciding whether the temporary fuser shim is acceptable, or replace installed Tokamax `pallas_triton.py` usage with the copied newer reference module.
  - If the shim passes, the next likely risk is whether `grug_flex_mask_mod`'s full-mask expression is decomposed by Tokamax fuser into blockwise index loads for 2D segment IDs; if not, replace it with an index-derived expression.

### 2026-05-10 06:52 UTC - GH200 Tokamax flex Pallas fuser-shim launch
- Hypothesis: the local fuser compatibility shim unblocks installed Tokamax 0.0.6 against JAX 0.10's `pull_block_spec(..., grid_len=...)` API.
- Compatibility mismatch:
  - Installed JAX 0.10 signature: `fuser.pull_block_spec(f, out_block_specs, *, scalar_prefetch_handler=None, grid_len=None)`.
  - Installed Tokamax 0.0.6 call site: `fuser.pull_block_spec(fn, spec, grid=grid)`.
  - Copied reference `/tmp/tokamax-attn-refs/flex_pallas_triton.py` uses `grid_len=len(grid)`, matching JAX 0.10.
  - Wrapper shim maps `grid` to `grid_len=len(grid)` without vendoring Tokamax.
- RUN_ID: `grug-gpu-attn-flex-pallas-shim-20260510-065254`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name "$RUN_ID" --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g -e RUN_ID "$RUN_ID" -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.90 -- uv run --package marin-levanter --extra gpu --extra kernels --group dev python .agents/scripts/grug_gpu_attention_bench.py --implementation gpu_flex_pallas --batch 2 --seq 64 --q-heads 4 --kv-heads 2 --head-dim 32 --sliding-window 16 --iters 1 --warmup 0 --skip-reference-timing`
- Result: launched; awaiting logs.
- Next action: inspect pod logs with kubectl.

### 2026-05-10 06:54 UTC - GH200 Tokamax flex Pallas fuser-shim result
- RUN_ID: `grug-gpu-attn-flex-pallas-shim-20260510-065254`
- Pod: `iris-dlwh-grug-gpu-attn-flex-pallas-shim-20260510-06-4fd4b419-0`
- Command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris get pods | rg "grug-gpu-attn-flex-pallas|065254"`
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml kubectl -n iris logs iris-dlwh-grug-gpu-attn-flex-pallas-shim-20260510-06-4fd4b419-0 --tail=700`
- Result:
  - Backend: `backend=gpu devices=['cuda:0']`.
  - The `pull_block_spec(grid=...)` compatibility shim worked; execution advanced to Tokamax's `block.pallas_call`.
  - New failure:
    - `TypeError: pallas_call() got an unexpected keyword argument 'backend'`
    - Origin: `tokamax/_src/pallas/block.py`, forwarding `backend=backend` to `jax.experimental.pallas.pallas_call`.
  - Installed JAX 0.10 signature for `pl.pallas_call` has no `backend` kwarg:
    - `pallas_call(kernel, out_shape, *, grid_spec=None, grid=(), in_specs=..., out_specs=..., ..., compiler_params=None, ...)`.
- Interpretation:
  - This is another Tokamax/JAX dependency-version mismatch, not a Grug mask failure.
  - Added a second small compatibility shim at the Grug wrapper boundary: if JAX `pl.pallas_call` lacks `backend`, drop Tokamax's legacy `backend` kwarg before forwarding.
  - No further cluster run in this pass.
- Commands after second shim:
  - `uv run ruff check --fix lib/levanter/src/levanter/grug/flex_attention.py`
  - `uv run ruff check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_attention.py -q`
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/attention.py lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py .agents/scripts/grug_gpu_attention_bench.py`
- Result after second shim:
  - Ruff: `All checks passed!`
  - Pyrefly: `INFO 0 errors`
  - Tests: `21 passed, 3 skipped in 13.70s`
  - Targeted pre-commit: `OK`
- Recommendation:
  - #5610 should not depend on installed `tokamax==0.0.6` as-is with this JAX 0.10 image.
  - Best patch boundary is to vendor/snapshot the small Tokamax flex/Pallas modules at the copied-ref version, or wait for an upstream Tokamax release aligned with JAX 0.10 Pallas/fuser APIs.
  - The wrapper shims prove the mismatch is small so far, but each shim exposes the next installed-Tokamax compatibility issue; vendoring a known-good snapshot is safer than accumulating monkey patches.

### 2026-05-10 09:30 PDT - Tokamax git-head pin and GH200 smoke
- Hypothesis: pinning `tokamax` to current upstream main resolves the JAX 0.10 Pallas/fuser API mismatch without vendoring or monkey patches.
- Upstream revision:
  - `openxla/tokamax@253673862d83e67659f2e449d9c6f7b3b94f10c4`
- Code/config changes:
  - Added a workspace `tool.uv.sources` git pin for `tokamax`.
  - Added `kernels` extra conflicts with CPU/TPU/vLLM/Fray-TPU split extras because Tokamax head resolves as `0.0.12` and requires `jax>=0.10.0`.
  - Removed the temporary `pull_block_spec(grid=...)` and `pallas_call(backend=...)` compatibility shims from `levanter.grug.flex_attention`; Tokamax head already uses `grid_len` and no legacy `backend` kwarg.
- Local validation:
  - `uv run --package marin-levanter --extra kernels pytest lib/levanter/tests/grug/test_attention.py -q`
  - Result: `21 passed, 3 skipped in 14.13s`.
  - `uv run ruff check lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py pyproject.toml`
  - Result: `All checks passed!`.
- GH200 RUN_ID: `/dlwh/grug-gpu-attn-flex-pallas-githead-20260510-093021`
- GH200 result:
  - Pod completed successfully.
  - Backend: `backend=gpu devices=['cuda:0']`.
  - Tokamax built from `git+https://github.com/openxla/tokamax.git@253673862d83e67659f2e449d9c6f7b3b94f10c4`.
  - Correctness small case: `implementation=gpu_flex_pallas`, causal sliding-window + 2D segment IDs, `max_abs=0.015625`, `mean_abs=0.000802851`, BF16.
  - Tiny benchmark shape `B=2 S=64 Hq=4 Hkv=2 D=32`, `segments=4`, `sliding_window=16`: compile plus first `0.174088s`, steady `0.000105s` for one iteration.
- Interpretation:
  - The git-head Tokamax pin fixes the dependency-version mismatch observed with `tokamax==0.0.6`.
  - `gpu_flex_pallas` now runs on GH200 for the small Grug causal sliding-window + 2D segment-ID case.
  - Next attention work should replace the full-mask scaffold with an index-derived `mask_mod` if needed and then benchmark representative Grug attention shapes; the dependency blocker is cleared.

### 2026-05-10 09:35 PDT - Index-derived Grug flex mask
- Hypothesis: Grug should express causal, sliding-window, and segment-ID constraints as index-derived broadcast expressions so Tokamax can tile `mask_mod`, instead of calling `AttentionMask.materialize_mask` and starting from a full logical mask.
- Code changes:
  - Rewrote `grug_attention_mask_for_scores` to build constraints from query/key indices and captured segment-id arrays.
  - `grug_flex_mask_mod(AttentionMask())` now returns `None`, matching a genuinely empty mask.
  - Added a regression test that monkeypatches `AttentionMask.materialize_mask` to raise; `grug_flex_mask_mod` still produces the expected causal sliding-window + 2D segment mask.
- Local validation:
  - `uv run ruff check --fix lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - `uv run pyrefly check lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - `uv run --package marin-levanter --extra kernels pytest lib/levanter/tests/grug/test_attention.py -q`
  - Result: `23 passed, 3 skipped in 15.00s`.
- GH200 RUN_ID: `/dlwh/grug-gpu-attn-flex-pallas-indexmask-20260510-093504`
- GH200 result:
  - Pod completed successfully.
  - Backend: `backend=gpu devices=['cuda:0']`.
  - Correctness small case: `implementation=gpu_flex_pallas`, causal sliding-window + 2D segment IDs, `max_abs=0.015625`, `mean_abs=0.000802851`, BF16.
  - Tiny benchmark shape `B=2 S=64 Hq=4 Hkv=2 D=32`, `segments=4`, `sliding_window=16`: compile plus first `0.174465s`, steady `0.000104s` for one iteration.
- Interpretation:
  - The full-mask materialization scaffold is removed from the Tokamax flex path.
  - The GH200 smoke still passes after the mask expression rewrite.
  - Next work should benchmark representative Grug attention windows and inspect whether Tokamax fuser pulls only the needed index/segment blocks.

### 2026-05-11 22:00 PDT - Training-callable custom VJP fallback
- Hypothesis: Tokamax flex/Pallas can make Grug GPU attention training-callable if Grug wraps the Pallas forward in `jax.custom_vjp` and computes q/k/v gradients through the existing reference attention derivative.
- Background:
  - Tokamax git head still marks `PallasTritonFlexAttention` as `supports_vjp=False` in `tokamax/_src/ops/flex_attention/pallas_triton_test.py`.
  - Direct GH200 `jax.grad(gpu_flex_pallas_attention)` previously failed with `NotImplementedError: vjp not implemented`.
- Code changes:
  - Added `tokamax_flex_attention_with_reference_vjp`, which uses Tokamax flex for the forward pass and `reference_attention` inside the custom-VJP pullback.
  - `gpu_flex_pallas_attention(..., use_reference_vjp=True)` now defaults to the training-callable wrapper; `use_reference_vjp=False` preserves the raw Tokamax Pallas path for backend debugging.
  - Added a CPU-local Tokamax-XLA custom-VJP gradient regression with causal sliding-window + 2D segment IDs.
- Local validation:
  - `uv run --package marin-levanter --extra kernels pytest lib/levanter/tests/grug/test_attention.py -q`
  - Result: `24 passed, 3 skipped in 16.46s`.
  - `uv run ruff check --fix lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - Result: `All checks passed!`.
  - `uv run pyrefly check lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - Result: `INFO 0 errors`.
  - `./infra/pre-commit.py --fix lib/levanter/src/levanter/grug/flex_attention.py lib/levanter/tests/grug/test_attention.py`
  - Result: `OK`.
- GH200 RUN_ID: `/dlwh/grug-gpu-attn-flex-custom-vjp-20260512-045541`
- GH200 command:
  - `KUBECONFIG=$HOME/.kube/cw-rno2a.yaml UV_NO_SYNC=1 uv run --package marin --group dev iris --cluster=coreweave-rno2a job run --job-name grug-gpu-attn-flex-custom-vjp-20260512-045541 --no-wait --max-retries 0 --enable-extra-resources --gpu GH200x1 --cpu 32 --memory 256g --disk 128g --timeout 3600 ...`
- GH200 result:
  - Backend: `backend=gpu devices=['cuda:0']`.
  - Small correctness shape `B=2 S=64 Hq=4 Hkv=2 D=32`, BF16, 4 packed segments, causal sliding window 16:
    - value: `max_abs=0.015625`, `mean_abs=0.000825811`.
    - q/k/v gradients: `max_abs=0`, `mean_abs=0`.
  - Representative shape `B=8 S=1024 Hq=16 Hkv=4 D=128`, BF16, 4 packed segments, causal sliding window 256:
    - `reference_xla_fwd`: `0.922 ms`.
    - `gpu_flex_pallas_fwd`: `0.374 ms`.
    - `reference_xla_fwd_bwd`: `2.031 ms`.
    - `gpu_flex_pallas_custom_vjp_fwd_bwd`: `2.030 ms`.
    - Speedups: forward `2.466x`, forward+backward `1.000x`.
- Interpretation:
  - The Grug `gpu_flex_pallas` path is now training-callable and preserves segment/sliding-window correctness.
  - Forward remains accelerated and avoids full-mask materialization through the index-derived flex `mask_mod`.
  - Backward is correct but not accelerated; the reference-derived pullback materializes the mask and erases the forward-only speedup at fwd+bwd level.
  - Need a follow-up fast-backward issue with acceptance criteria for a no-full-mask VJP kernel.
