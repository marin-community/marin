# Issue 5510 H100 CE Fallback: Research Logbook

## Scope
- Goal: Determine the H100 performance impact of the fused cross-entropy path falling back from `pallas_gpu` to `xla` for the Grug MoE CoreWeave canary in issue 5510.
- Primary metric(s): H100 canary step throughput or profile share attributable to fused CE; kernel-level XLA-vs-Pallas steady-state timing if available.
- Constraints: Do not treat GB10 Pallas measurements as H100 evidence. Same-shape H100 Pallas CE does not currently lower, so there is no direct H100 apples-to-apples runtime for `pallas_gpu` at the issue shape.

## Baseline
- Date: 2026-05-07
- Issue: https://github.com/marin-community/marin/issues/5510
- Code refs:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  - `experiments/ferries/canary_ferry.py`
- Baseline case:
  - CoreWeave H100, 8 GPUs, 1 host
  - JAX 0.10.0, CUDA 13
  - Grug MoE canary local CE shape reported in issue comment: `B=8192`, `H=1024`, `V=128256`, `x=bfloat16`, `w=float32`, `dtype=float32`

## Experiment Log

### 2026-05-07 01:00 - Issue and workflow evidence
- Hypothesis: The issue is a deterministic H100 fallback, not a sporadic warning.
- Command:
  - `gh run view 25468812595 --repo marin-community/marin --json databaseId,status,conclusion,displayTitle,createdAt,updatedAt,workflowName,headSha,jobs`
  - `gh run download 25468812595 --repo marin-community/marin --dir /tmp/marin-issue5510/run-25468812595`
  - `rg -n "Fused CE autotune|Pallas fused cross-entropy|Fused cross-entropy selected|JAX version" /tmp/marin-issue5510/run-25468812595/canary-diag-25468812595/pod-*.log -S -C 3`
- Config:
  - Cited run `25468812595`, Iris job `/runner/iris-run-job-20260507-002941/grug-train-canary-gpu-25468812595-1`
- Result:
  - Both captured task attempts log `JAX version 0.10.0`.
  - Both attempts log `Fused CE autotune miss for pallas_gpu. Sweeping 7 block-size candidates.`
  - Both attempts warn that Pallas CE is unavailable and fall back to XLA.
  - Both attempts log `Fused cross-entropy selected implementation: xla`.
- Interpretation:
  - This validates that the cited H100 canary used XLA CE, not Pallas CE.
  - The cited run was cancelled/killed and one attempt hit a profiler `stop_trace` segfault, so it cannot provide a clean training throughput measurement.
- Next action:
  - Use successful neighboring H100 canary profiles to inspect whether CE-related XLA/Triton markers dominate profile time.

### 2026-05-07 01:20 - H100 profile-summary evidence
- Hypothesis: If XLA CE fallback is a major end-to-end H100 problem, CE-related XLA/Triton markers should be visible as dominant hot ops in successful canary profiles.
- Command:
  - `gh run view 25458065200 --repo marin-community/marin --log`
  - `gh run view 25450733652 --repo marin-community/marin --log`
  - Extracted the `profile_summary.v1` JSON object from each log and inspected `run_metadata`, `time_breakdown`, `hot_ops`, `gap_before_ops`, and `gap_region_contexts`.
- Config:
  - Successful run `25458065200`: `NVIDIA H100 80GB HBM3`, 8 devices, 1 host, W&B profile artifact `jax-profile-step-5-30:v16`
  - Successful run `25450733652`: `NVIDIA H100 80GB HBM3`, 8 devices, 1 host, W&B profile artifact `jax-profile-step-5-30:v15`
- Result:
  - Both profile summaries report suspected truncation at exactly 1,000,000 complete events.
  - Both summaries have empty `step_time` stats, so they cannot quantify steady-state step-time impact.
  - Run `25458065200` time breakdown: compute `32.2%`, communication `7.7%`, host `25.8%`, stall `34.3%`.
  - Run `25450733652` time breakdown: compute `25.1%`, communication `6.4%`, host `20.6%`, stall `47.9%`.
  - Both summaries contain `triton_softmax_87` and `fusion_3129`, consistent with XLA/Triton CE-related work, and no Pallas CE custom-call marker.
  - `fusion_3129` is a top-25 hot op but not dominant:
    - run `25458065200`: `222,385.304` total duration over `55,723,771.113` total exclusive duration (`~0.40%` of the summary basis)
    - run `25450733652`: `224,743.140` total duration over `70,339,796.615` total exclusive duration (`~0.32%` of the summary basis)
  - `triton_softmax_87` appears in gap context but is not in top-25 hot ops.
- Interpretation:
  - Current H100 canary profiles are XLA CE profiles.
  - The available profile summaries do not show CE markers dominating the canary profile; stalls/host time and broader compute/communication dominate.
  - Confidence is limited by trace truncation and absent step-time data.
- Next action:
  - Inspect tuned-block selection and launch feasibility to determine whether direct Pallas comparison is possible.

### 2026-05-07 01:35 - H100 tuned-table and launch feasibility
- Hypothesis: The issue shape is unsupported by current H100 Pallas GPU constraints, so direct H100 Pallas-vs-XLA timing is unavailable.
- Command:
  - `sed -n '1,260p' lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  - `sed -n '1,240p' lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - `.venv/bin/python - <<'PY' ... infer_block_sizes_with_tuned_match(...) ... PY`
- Config:
  - `B=8192`, `H=1024`, `V=128256`, `dtype=float32`, `x_dtype=bfloat16`, `w_dtype=float32`
  - Device kinds checked: `NVIDIA H100 80GB HBM3`, `NVIDIA GB10`, `NVIDIA A100-SXM4-80GB`, `NVIDIA`
- Result:
  - H100 maps to bucket `mid-h-large-vocab` but has no tuned entry, so it returns default `BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)` with `matched=False`.
  - The default float32 weight tile is `512 * 1024 * 4 = 2,097,152` bytes.
  - The H100/GPU launch guard allows only `101,376` bytes for the weight tile, so the inferred default and all larger `v_block` candidates fail immediately.
  - The issue comment additionally tested smaller `h` candidates that pass the first guard but still fail Mosaic GPU lowering around `2.4-3.2 MB` shared memory, above H100's `232,448` byte limit.
- Interpretation:
  - The direct H100 performance delta between XLA CE and current `pallas_gpu` CE at this shape is not measurable because current Pallas CE does not produce a viable H100 kernel.
  - This is not just a missing tuned-table row; it is a kernel/runtime support problem for large-vocab + float32 LM-head weights on H100.
- Next action:
  - Recommend labeling H100 canary perf as XLA CE and opening a narrower H100 Pallas CE kernel/runtime task only if CE is proven to matter after higher-quality profiling.

### 2026-05-07 02:15 - Related PR context: #4297, #5350, and #3160
- Hypothesis: The remembered "H100 Pallas kernel" may refer to MoE `ragged_dot`, not fused CE.
- Command:
  - `gh pr view 4297 --repo marin-community/marin --json title,body,comments`
  - `gh pr view 5350 --repo marin-community/marin --json title,body,comments`
  - `git show --stat --oneline ce77573c0`
  - `git show --stat --oneline e984831fd`
  - `git show ae19f0fcc -- lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - `git show b599bc375:.agents/projects/fused_cross_entropy_gpu.md`
- Result:
  - #4297 added a Pallas-Triton GPU kernel for Haliax `ragged_dot`, i.e. MoE grouped matmul, not fused CE.
  - #4297 H100x8 validation reported Grug MoE end-to-end examples/sec and tokens/sec up `~8.8%` for Triton forward + XLA backward; the PR body also reported a forward-only kernel microbench speedup up to `5.2x` and a smaller model end-to-end gain around `20%`.
  - #5350 then moved `ragged_dot` backward contractions to explicit Pallas-Triton kernels. Its H100x8 Grug MoE report was `0.530359s` main vs `0.171189s` PR average median train step, or `3.098x`, with the CoreWeave GPU canary passing.
  - The JAX 0.10/CUDA 13 update is #5428 (`e984831fd`) and is separate from #5350 (`ce77573c0`); current `ragged_dot.py` still defaults GPU auto dispatch to `("triton", "xla")` when Pallas Triton imports.
  - Fused CE GPU history is different. #2877 was "GB10-safe GPU Pallas fused cross-entropy"; the project log explicitly says the target-like large-shape wins were primarily GB10 hybrid routing, and that for GB10 the pallas-gpu route reused XLA forward and only overrode backward.
  - #3160 intentionally extended the CE launch guard to all NVIDIA GPUs after H100 shared-memory failures, causing Pallas CE to fail fast and the API to fall back to XLA at canary scale.
- Interpretation:
  - The large H100 performance story in #4297/#5350 is real, but it is MoE `ragged_dot`, not fused CE.
  - Issue 5510 does not show that the #4297/#5350 H100 MoE Pallas-Triton kernel was disabled by JAX 0.10. It shows that the separate fused CE `pallas_gpu` path cannot launch for the H100 CE shape and falls back to XLA.
  - The H100 CE fallback predates #5428 as an intentional safety guard from #3160; JAX 0.10 may have made the warning newly visible in this canary, but it is not the same as turning off the MoE Triton path.

### 2026-05-07 02:45 - Experimental H100 hybrid patch
- Hypothesis: A small code change can make `implementation="pallas_gpu"` run on the Grug H100 CE shape, but only by using a hybrid route rather than the native tiled Pallas forward.
- Code changed:
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`
  - `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  - `lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
- Patch behavior:
  - Detects H100 + BF16 activations + BF16/FP32 LM head + `V >= 65536`.
  - Skips fused CE autotune for that H100 internal-routing shape.
  - Routes forward through the existing XLA streaming CE implementation with `block_sizes=None`, matching the current H100 XLA forward policy.
  - Keeps a pallas-gpu custom backward tile policy of `v_block=8192` for the H100 large-vocab hybrid route.
  - Adds `LEVANTER_PALLAS_GPU_H100_NATIVE_FORWARD=1` as an explicit experiment switch; native H100 forward still requires explicit block sizes and is expected to hit the same design limits unless the kernel algorithm changes.
- Validation:
  - `.venv/bin/python -m py_compile lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
  - `uv run --project lib/levanter --group test python -m pytest -o addopts='' lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py -k 'h100_large_vocab or h100_internal_route'`
    - `3 passed, 75 deselected`
  - `uv run --project lib/levanter --group test python -m pytest -o addopts='' lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py`
    - `66 passed, 12 skipped, 1 warning`
  - `uv run --project lib/haliax --group dev python -m pytest -o addopts='' lib/haliax/tests/test_ragged_dot_dispatch.py`
    - `4 passed, 1 warning`
- Interpretation:
  - This patch should stop the H100 Grug CE path from falling back out of `pallas_gpu`, but it is not proof of a native Pallas CE speedup.
  - The only honest performance question after this patch is whether the H100 custom backward route beats the current XLA backward in an H100 microbench/canary. That still requires H100 measurement.

### 2026-05-07 15:47 - H100 CE hybrid microbench
- Hypothesis: The H100 large-vocab hybrid route improves value+grad timing by replacing XLA autodiff backward with the pallas_gpu custom streaming backward while keeping forward on XLA.
- Code changed:
  - `lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py`
  - Added `--weight-dtype`, `--value-and-grad`, and `selected_route` logging so the benchmark matches the Grug H100 shape and records the route under test.
- Command:
  - `uv run iris --cluster=coreweave-ci job run --no-wait --job-name h100-ce-hybrid-microbench-20260507-1548 --enable-extra-resources --gpu H100x1 --cpu 16 --memory=96G --disk=96G --extra gpu --timeout 3600 -- python -u lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py --batch 8192 --pos 1 --embed 1024 --vocab 128256 --input-dtype bfloat16 --weight-dtype float32 --accum-dtype float32 --implementation pallas_gpu --block-sizes infer --variant-sweep --compare-xla --steps 5 --warmup 2 --value-and-grad`
  - `uv run iris --cluster=coreweave-ci job run --no-wait --job-name h100-ce-hybrid-microbench-rep2-20260507-1549 --enable-extra-resources --gpu H100x1 --cpu 16 --memory=96G --disk=96G --extra gpu --timeout 3600 -- python -u lib/levanter/scripts/bench/bench_fused_cross_entropy_loss_pallas.py --batch 8192 --pos 1 --embed 1024 --vocab 128256 --input-dtype bfloat16 --weight-dtype float32 --accum-dtype float32 --implementation pallas_gpu --block-sizes infer --variant-sweep --compare-xla --steps 10 --warmup 3 --value-and-grad`
- Config:
  - CoreWeave H100 single-GPU direct benchmark.
  - Shape `B=8192`, `H=1024`, `V=128256`, `x=bfloat16`, `w=float32`, accumulator `float32`.
  - Candidate logged `selected_route=pallas_gpu_h100_xla_forward_custom_backward_hybrid`.
- Result:
  - Run `/romain/h100-ce-hybrid-microbench-20260507-1548`:
    - hybrid forward `0.0103037598s`, value+grad `0.0300512428s`, combined `202,998 tok/s`.
    - XLA forward `0.0103253166s`, value+grad `0.0346417618s`, combined `182,178 tok/s`.
  - Replicate `/romain/h100-ce-hybrid-microbench-rep2-20260507-1549`:
    - hybrid forward `0.0102937542s`, value+grad `0.0301139942s`, combined `202,733 tok/s`.
    - XLA forward `0.0103623766s`, value+grad `0.0346346931s`, combined `182,056 tok/s`.
- Interpretation:
  - Forward is tied, as expected because the hybrid uses the XLA streaming forward.
  - Value+grad is replicated at roughly `15%` higher throughput for the hybrid route.
  - The script's forward+value_and_grad combined metric is replicated at roughly `11.4%` higher throughput.
- Next action:
  - Run one CoreWeave Grug MoE canary from this branch to check whether the kernel-level win moves end-to-end canary step time materially.

## Findings
- Confidence: replicated for fallback behavior, exploratory for end-to-end performance impact.
- The cited H100 canary and successful neighboring canary profile summaries are XLA CE runs, not Pallas CE runs.
- Same-shape H100 `pallas_gpu` CE does not currently lower, so there is no direct apples-to-apples H100 speedup/slowdown number for Pallas versus XLA at the issue shape.
- #4297/#5350 are about the H100 MoE `ragged_dot` Pallas-Triton path, not fused CE. The `~8.8%` to `3.10x` H100 gains from those PRs should not be attributed to fused CE.
- The H100 CE fallback is consistent with the intentional NVIDIA-wide CE launch guard from #3160; the JAX 0.10/CUDA 13 update is not evidence that the #5350 MoE Triton kernel was disabled.
- The best available H100 profile summaries do not show CE-related XLA/Triton markers as the dominant bottleneck, but the summaries are trace-truncated and lack step-time stats.
- GB10 historical Pallas wins are not valid H100 evidence: they use GB10-specific routing and include a hybrid XLA-forward/custom-backward path.
- A local experimental patch now makes the H100 large-vocab CE `pallas_gpu` entry point run via an XLA-forward/custom-backward hybrid route. It still needs H100 performance validation.

## Recommendations
- Keep treating current published H100 Grug MoE canary numbers as XLA CE numbers unless rerun with the experimental H100 hybrid patch.
- Make canary/profile outputs record the selected CE implementation and include it in perf labels; current H100 canary numbers should be described as XLA CE numbers.
- Do not add a simple H100 tuned-table row for this shape. The launch/mosaic failures indicate the current Pallas GPU kernel design cannot fit the H100 shared-memory regime for float32 weights.
- Run an H100 CE microbench and one Grug MoE canary with the local hybrid patch before merging it as a performance change. Compare against current XLA CE on the same JAX/CUDA image.
- If the hybrid patch does not move step time materially, prioritize the #5350-style `ragged_dot` path and broader host/stall/communication bottlenecks over native CE kernel work.
- If the team wants true native H100 Pallas CE, start a separate kernel rewrite task. The current tiled forward likely needs a different Triton/Mosaic algorithm that avoids per-program full-H weight/input tiles and keeps `(B,V)` accumulators within shared-memory/register limits.
- Before prioritizing that work, capture a non-truncated H100 profile or microbench report with step-time stats and CE-specific timings; current evidence suggests stalls/host/communication are likely larger canary-level bottlenecks than the CE fallback.
