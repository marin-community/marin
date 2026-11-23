# Diagnose vLLM performance

Use this reference when serving measurements leave a bottleneck unexplained or
you need to investigate custom model code or kernels. Choose the cheapest test
that can distinguish the plausible causes. Check commands and features against
the installed vLLM, model runner, PyTorch and Triton versions.

## Choose a diagnostic

| Question | Tools and evidence |
| --- | --- |
| Where does each iteration spend time? | Capture a short, warmed-up CPU/GPU timeline with PyTorch Profiler or Nsight Systems. Inspect worker processes, relevant ranks, communication overlap and gaps between kernels. vLLM supports `--profiler-config` and `/start_profile` and `/stop_profile`. Read the [profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling/). |
| Is the compiler producing the expected work? | Use `TORCH_TRACE` with `tlparse` to inspect compilation artifacts and shape guards. `TORCH_LOGS=output_code` exposes generated kernels. An unpacked compile cache allows targeted instrumentation of generated code. Read [compiler debugging](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/). |
| Why is a GPU kernel slow? | After locating it on the timeline, use a small benchmark with representative shapes, dtypes, strides and cache conditions. Nsight Compute can measure memory bandwidth, compute use, occupancy and instruction stalls. Read the [kernel-microbenchmark skill](https://github.com/vllm-project/vllm/blob/main/.agents/skills/kernel-microbenchmark/SKILL.md) and [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). |
| Does custom code handle real inputs? | Compare with a plain PyTorch implementation of the same operation, such as `CustomOp.forward_native` where available, and reuse existing kernel tests. Exercise padding, offset views, boundary sizes, accumulation precision and repeated graph replay with changing inputs or state. For Triton changes, inspect generated code, launch geometry and unintended JIT specialization. Read [Triton kernel guidance](https://github.com/vllm-project/vllm/blob/main/.agents/skills/triton-kernel-writing/SKILL.md) and the [CustomOp reference](https://docs.vllm.ai/en/latest/api/vllm/model_executor/custom_op/). |
| Where does a failure or hang start? | Use targeted logging, subprocess-compatible breakpoints or distributed breakpoints. `CUDA_LAUNCH_BLOCKING=1` can locate a failing CUDA operation; `NCCL_DEBUG` helps investigate communication. Confine these to isolated diagnostics. Read [troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/) and [compiler debugging](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/). |

For host-side gaps, the profiling guide also covers `cProfile` and garbage
collection diagnostics. For Nsight traces, reuse vLLM's
[analysis tools](https://github.com/vllm-project/vllm/tree/main/tools/profiler/nsys_profile_tools)
when they fit the captured trace.

## Keep the comparison meaningful

Start with a few representative iterations. Detailed stack, shape and memory
tracing increases overhead and trace size. Save raw traces with the trial's
runtime, settings, load and rank identities. Recheck final serving performance
with profiling and debug instrumentation disabled.

Separate raw kernel time from the production operation's cost. Include required
staging, padding, copies and synchronization when assessing the production
wrapper. For distributed benchmarks, measure all ranks and use the slowest rank
per sample. Inspect peer timelines before attributing a long collective to the
network. The upstream microbenchmark skill covers graph timing and cache effects.

Verify effective graph execution for the measured shapes. CUDA-graph metrics can
report padding and dispatch modes where the runner supports them. Layerwise NVTX
tracing has runner-specific limits and does not work with CUDA graphs. See
[observability configuration](https://docs.vllm.ai/en/latest/api/vllm/config/observability/).
For `vllm serve`, follow Marin's
[compilation and CUDA-graph rules](../../../../AGENTS.md#development), including
for smoke tests. Disabling them for a serving diagnostic needs explicit user
authorization. Plain reference arithmetic in an isolated component test remains
available.

Preserve the observations that support or reject each hypothesis. A faster
isolated kernel still needs matching correctness checks and a serving comparison
to establish a useful improvement.

For controlled prefill comparisons, inspect prompt tokens per rank as well as
request counts. Equal counts can hide unequal work. Record cache hits,
preemptions and whether earlier requests drained before the next wave.
[DA7's measured example](https://github.com/marin-community/marin/pull/11320#issuecomment-7092800820)
shows why these controls can change the conclusion.

If iteration logs are missing with several API workers, inspect the installed
logger factory. The [published `47a8e415bff1` factory](https://github.com/marin-community/vllm/blob/47a8e415bff17b9376f02d9b92edcc67e962eaee/vllm/v1/metrics/loggers.py#L1230-L1243)
disables the default text logger in that case. A `vllm.stat_logger_plugins` hook
can retain step details; use a logger in the `vllm` namespace under its
[logging policy](https://github.com/marin-community/vllm/blob/47a8e415bff17b9376f02d9b92edcc67e962eaee/vllm/logger.py#L63-L78).
Check emitted step coverage and reconcile token counts before interpreting it.
