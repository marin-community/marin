# Shuttle XLA FFI command-buffer boundary

## TL;DR

The first command-buffer candidate is the generated normalized-exp forward and
reverse pair in the thirteen-call Grug train-step path. Each handler launches
one fixed-shape CUDA kernel into the XLA stream. The capture-safe variant omits
CUDA launch-status polling and registers
`ffi::Traits::kCmdBufferCompatible`. The existing error-checking path remains
the default.

This checkpoint contains source generation, eligibility checks, benchmark
selection, and a measurement plan. It has not run on H100 yet.

## Motivation

The sealed thirteen-call H100 profile reports nine CUDA graph segments, 28
direct kernel launches, and 44.224 us of inter-segment or unattributed gap. The
generated path takes 0.731302 ms versus 0.591416 ms for stock XLA. The two
device-to-device copies account for 2.112 us, so restoring one continuous XLA
command graph is a more direct next experiment than copy tuning.

XLA converts contiguous compatible thunks into command buffers. At OpenXLA
revision [`11b39a84c9d7`](https://github.com/openxla/xla/blob/11b39a84c9d7bc9a425520f7868c8afc6bb5e36a/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc#L282-L293),
typed-FFI custom calls enter that conversion only when their registered handler
metadata is command-buffer compatible. `CUSTOM_CALL` is in the default command
set at the same revision in
[`debug_options_flags.cc`](https://github.com/openxla/xla/blob/11b39a84c9d7bc9a425520f7868c8afc6bb5e36a/xla/debug_options_flags.cc#L304-L310).
The benchmark does not mutate `XLA_FLAGS` after importing JAX. It relies on the
pinned XLA default and fails before device access if a startup flag explicitly
disables `CUSTOM_CALL` or replaces the default command list without it.

## Candidate boundary

The transformed HLO contains this dataflow:

```text
compact Contract + normalized-exp Fold + indexed selection
    -> generated normalized-exp forward
        output: row loss
        saved output: FP32 log normalizer
    -> generated normalized-exp reverse
        output: input cotangent, operand cotangent
```

The exact typed-FFI targets are:

```text
shuttle.routed_training.normalized_exp_contract_forward.v1
shuttle.routed_training.normalized_exp_contract_reverse.v1
```

Both sources are generated from the same generic Contract/Map/Fold semantics as
the existing path. Each has fixed launch dimensions, compile-time dynamic
shared-memory size, destination-passed output buffers, and one CUDA kernel
launch. Neither uses runtime workspace allocation, autotuning, a library
handle, or semantic atomics.

The ordinary source checks `cudaPeekAtLastError` after launch. The command-buffer
candidate removes that status query so the marked host path only enqueues the
kernel and updates non-semantic capture instrumentation. A host status query
would run while XLA records the graph and would not validate later graph
replays. Asynchronous execution errors must surface through the XLA stream in
this mode. The default handler retains the immediate launch check.

## Eligibility contract

`tile_lifetime.ffi_command_buffer` applies a conservative source audit before
adding the trait. It rejects:

- `ffi::ScratchAllocator`;
- runtime CUDA allocation or free;
- lazy cuBLAS, cuBLASLt, or cuDNN handle creation;
- `std::call_once` and `std::once_flag`;
- runtime autotuning or algorithm selection;
- CUDA launch-status queries;
- device or stream synchronization.

The current routed, shared-Map, and rank-two Contract handlers are not included
because several create cuBLAS handles lazily. Axis Fold, source Fold, and fused
Contract/Relation/Fold handlers still query CUDA launch status. They can receive
their own capture-safe variants later, after this two-handler experiment shows
that XLA graph segmentation is worth pursuing.

The audit is a bounded code-generation guard, not a proof system for arbitrary
C++. New generated families require their own source-lineage and capture-safety
review before setting the trait.

## Handler-count audit

The generated host handlers increment non-semantic counters. Without command
buffers, the benchmark requires at least `warmup + repeats + 1` handler calls.
During CUDA graph replay, XLA invokes a compatible host handler while recording
the graph and replays the recorded GPU command without invoking the handler
again. The two candidate counters therefore measure capture invocations, not
logical train-step executions.

The candidate benchmark requires at least one capture invocation for each
compatible target. It continues to require the old logical-execution lower
bound for every non-compatible handler. Exact HLO target occurrence remains one
per generated target. Nsight graph attribution, not a host counter, must prove
that replay occurred.

## Candidate invocation

Run the candidate in a fresh process. The pinned XLA default already enables
`CUSTOM_CALL` command buffers:

```bash
uv run python lib/tile_lifetime/benchmarks/xla_grug_routed_combined_gpu_custom_call.py \
  --nvcc /usr/local/cuda/bin/nvcc \
  --architecture sm_90a \
  --composition-mode shared_map_fused_reverses \
  --command-buffer-candidate normalized_exp_pair \
  --repository "$PWD" \
  --warmup 4 \
  --repeats 30
```

Omitting `--command-buffer-candidate` selects the existing source and launch

If a job overrides `xla_gpu_enable_command_buffer`, include `CUSTOM_CALL` in
that startup setting before Python imports JAX. The benchmark checks the
startup value and never changes it in process.

## One-H100 measurement plan

Use one fixed H100 allocation and the sealed Grug shape.

1. Run `disabled` and `normalized_exp_pair` in separate fresh processes. Reverse
   their process order in a second capture. Retain all 30 samples from each
   benchmark process.
2. Require identical output-tree structure, the existing ordered-FP tolerance,
   stable output hashes, one exact HLO occurrence per target, and the expected
   source/semantic digests.
3. Trace one warm execution of each candidate with Nsight Systems. Record CUDA
   graph segments, direct generated kernel launches, device copies, graph GPU
   activity, total GPU span, and inter-segment gap using the same attribution
   method as the sealed profile.
4. Confirm that the normalized-exp forward and reverse kernels move from direct
   launches into an XLA command graph. A lower graph-segment count than nine and
   a smaller 44.224 us inter-segment gap are the structural outcomes. Compare
   end-to-end medians only after those outcomes hold.
5. Reject the candidate if correctness changes, the pair remains direct, or the
   graph boundary expands across an ineligible handler. A neutral latency result
   is still useful if it shows that this pair is too small to explain the gap.

No tile, block-size, or semantic-body tuning belongs in this run. The experiment
isolates XLA command-buffer continuity.
