# Debugging log for Shuttle generic expert parallelism

Identify and recover from first-execution failures in the generic four-GB200 JAX expert-parallel benchmark without invoking the fused Mixture-of-Kittens kernel.

## Initial status

The low-priority `shuttle-generic` reservation reached `RUNNING` with one ready pod and four visible NVIDIA GB200 GPUs. The exact Mixture-of-Kittens route fixture was generated with Torch 2.10.0+cu130 from per-rank CUDA seeds `1234 + rank`; selected-expert and FP32 router-weight payloads were bit-identical under Torch 2.11.0+cu128.

The first exact-shape `ragged_all_to_all` run compiled, then exited with signal 11 / status 139 on its first executable call. Python fault handling placed both crashing threads in `jax/_src/interpreters/pxla.py::__call__`, reached from `_timings`; no benchmark JSON was emitted.

The active `/app` image was built from Marin commit `1373230331f63a0a388a7b9944b48672ad844cdc` and requires JAX/JAXlib 0.11.0 for the GPU extra. The older local worktree still specifies 0.10.1, but its relevant expert-parallel and `ragged_dot` source files are byte-identical to the active checkout. The benchmark will use the active checkout's locked 0.11.0 environment rather than mixing JAX versions with that checkout.

## Hypothesis 1: failure is in collective execution, not compilation

The absence of output did not distinguish allocation, lowering, compilation, or execution. Re-run one iteration with zero warmups, Python fault handling, unbuffered output, and full JAX tracebacks.

## Results

The one-iteration reproduction again exited 139. The stack reached the first `compiled(*inputs)` call in `_timings`, proving that input construction and explicit compilation completed. This narrows the failure to executable launch or device execution.

## Hypothesis 2: distinguish shape-independent collective failure from exact-shape resource pressure

Run compile-only and first-execution probes at a small valid four-rank shape, then repeat the exact shape with execution omitted. Record pod memory, GPU memory, kernel logs, and NVIDIA Xid evidence before changing XLA collective modes.

## Changes to make

- Add no production-code changes.
- Use benchmark CLI shape overrides for the tiny probe.
- Capture Kubernetes resource status, `nvidia-smi`, and available kernel-log evidence.

## Results

Exact-shape compile-only succeeded in 9.658 seconds. A tiny four-rank shape (`T/rank=64`, `E=8/2`, `top-k=2`, `H=256`, `I=256`) compiled in 2.187 seconds, then segfaulted identically on its first execution. The failure is therefore independent of the primary tensor sizes and their memory footprint.

After each crash, every GPU reported zero allocated MiB, zero utilization, and zero volatile uncorrected ECC errors. `/proc/meminfo` reported approximately 826 GiB available host memory. The Kubernetes Metrics API was unavailable, and the task container lacks permission to read `dmesg`, so node-level Xid evidence could not be obtained through these interfaces.

## Hypothesis 3: the enabled one-shot ragged-all-to-all runtime is incompatible with this GB200/JAX build

The pinned JAX 0.11 XLA enables the one-shot ragged-all-to-all kernel and NCCL barrier by default. Disable the one-shot path explicitly to force its ordinary fallback. If that runs, compare it with the requested symmetric-memory mode and retain the viable path; if it also fails, isolate the collective from the segmented GEMMs with a minimal `jax.lax.ragged_all_to_all` program.

## Changes to make

- No source changes.
- First run the tiny diagnostic with `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`.
- Then run the exact fixture under any viable collective mode.

## Results

Disabling one-shot without enabling the decomposer still segfaulted, with both Triton and XLA `ragged_dot`. Inspecting the active JAX libraries with `strings` confirmed that the wheel accepts the one-shot, decomposer, local-barrier, NCCL-barrier, zero-copy, collective-mode, and symmetric-buffer-filter flags.

Enabling `--xla_gpu_unsupported_enable_ragged_all_to_all_decomposer=true` while disabling one-shot made the tiny graph execute with finite output and zero drops. The native ragged thunk paths are therefore unusable on this toolchain; the decomposed collective is the viable fallback.

The exact MoK fixture with decomposed transport and Triton segmented GEMMs compiled in 8.858 seconds and measured 9.460 ms median over 50 iterations after 10 warmups, or 200.2 logical TFLOP/s per rank. Output was finite and no assignments were dropped. The tuned MoK correctness oracle is 3.613 ms, making this generic fallback 2.62 times slower.

Holding transport fixed and switching only `ragged_dot` to XLA produced a 95.100 ms median and a 60.800-second compile. Triton is 10.05 times faster and is the only viable segmented-GEMM backend in this comparison.

## Hypothesis 4: separate transport, routed contraction, and shared-expert costs

Run four exact-fixture executable probes: shared expert only; complete routed path; dispatch plus inverse-dispatch with an identity payload and fixed route-slot reduction; and already-routed local segmented W13/SwiGLU/W2 using exact expert counts and the production static capacity.

## Changes to make

- Add component selection to the standalone benchmark.
- Keep the phase probes algebraically generic and avoid the MoK fused kernel.
- Preserve fixed route-slot order for the transport probe's FP32 weighted reduction.

## Results

With 10 warmups and 50 iterations, the median latencies were:

- Shared expert: 0.382 ms.
- Complete routed path: 9.289 ms.
- Dispatch plus inverse-dispatch identity round trip: 4.875 ms.
- Already-routed padded segmented MLP: 4.696 ms.

Transport and segmented expert computation are both first-order costs and nearly account for routed latency independently. The exact seeded routes touch 3.301 destination ranks per token on average. Coalescing by `(token, owner)` would reduce H-wide dispatch rows from 6.0 to 3.301 per token, approximately 45% fewer rows, while retaining explicit route slots for deterministic combine.

## Hypothesis 5: specialize static capacity to the exact fixture bound

The current capacity factor of 1.25 pads each rank to 15,360 rows although the seeded fixture may need much less. Compute per-owner receive counts and sweep the exact safe bound, 1% and 5% headroom, and the current factor for full execution and isolated local compute.

## Results

The receiver assignment counts were `[12281, 12281, 12349, 12241]`. The minimum safe factor is `12349 / 12288 = 1.0049641927083333`.

All candidates had zero dropped assignments. Full-region medians were 8.445 ms at 12,349 rows, 8.492 ms at 12,473 rows, 8.660 ms at 12,967 rows, and 9.456 ms at 15,360 rows. Local segmented medians were 4.183, 4.184, 4.277, and 4.676 ms respectively. Exact capacity specialization improves the full generic baseline by 10.7% and local compute by 10.6%, yielding a best compliant baseline 2.34 times slower than the 3.613-ms MoK oracle.

## Hypothesis 6: DeepEP token coalescing offsets its oversized local assignment batch

DeepEP dispatches each token once per destination rank, reducing activation transport for this fixture from six route rows to 3.301 destination rows per token. Its current local packing nevertheless expands the fixed receive buffer of 8,192 tokens into 49,152 assignment rows, four times the nominal 12,288 assignments per rank. Benchmark the unmodified static assignment shape first with the deterministic fixed-route-slot FP32 collapse, then cap only the local assignment batch while retaining explicit overflow accounting.

## Changes to make

- Pin DeepEP at commit `7febc6e25660af0f54d95dd781ecdcd62265ecca` and build for `sm_100`.
- Preserve the production transport and deterministic route-slot merge.
- Report the static receive and assignment capacities in benchmark output.

## Results

The CUDA 13 build needed CCCL headers from the environment and an explicit unversioned `libcudart.so` search path; after that correction the extension built and loaded. The exact-fixture DeepEP path compiled in 13.538 seconds and measured 11.354 ms median over 50 iterations after 10 warmups, or 166.8 logical TFLOP/s per rank. The routed path alone measured 11.195 ms. Output was finite and no assignments were dropped.

Despite destination-rank coalescing, the uncapped DeepEP path was 34.5% slower than the 8.445-ms exact-capacity decomposed ragged baseline and 3.14 times slower than the 3.613-ms MoK oracle. A capped local-assignment experiment is required to distinguish DeepEP transport cost from the 49,152-row segmented-GEMM padding cost.

An initial remote-only prototype split the 8,192 receive-token buffer into four batches whose per-batch assignment shape fit the requested capacity. That retained 49,392 padded assignment rows in total and launched four pairs of segmented GEMMs rather than selecting the globally valid assignments once. Its routed median was 14.832 ms at the exact factor, 32.5% slower than the 11.195-ms uncapped routed path, so this batching design was rejected. The next prototype must form one bounded assignment batch across the entire receive buffer.

The corrected implementation performs one global compact selection from the 49,152 candidate route slots into the static assignment capacity. It reports `max(total_valid - capacity, 0)` explicitly, restores semantic route positions with sort and search, and accumulates route slots in ascending order in FP32. The pinned DeepEP intranode combine does not use atomic numerical accumulation: it enumerates contributing owner ranks in ascending rank order and sequentially adds their BF16 buffers into FP32 accumulators before the BF16 output cast. The path therefore has fixed local route-slot and cross-owner accumulation orders. At the exact safe capacity of 12,349 rows, the full benchmark measured 6.128 ms on a clean rerun, with finite output and zero overflow. The +1% and 1.25-capacity runs measured 6.113 and 6.499 ms respectively; the small difference between exact and +1% is measurement noise, while the larger 1.25 batch costs 6.1%.

The exact-capacity DeepEP result is 45.8% faster than uncapped DeepEP, 27.4% faster than the 8.445-ms exact-capacity decomposed ragged path, and 1.70 times slower than the 3.613-ms MoK oracle. Phase probes measured 1.340 ms for raw DeepEP dispatch plus identity combine and 4.185 ms for the isolated 12,349-row segmented W13/SwiGLU/W2 compute. The complete routed-only graph measured 6.727 ms, unexpectedly slower than the 6.128-ms graph that also contains the shared expert. These component measurements are therefore schedule-sensitive and must not be summed as an exact latency decomposition.

## Standalone MoK grouped-GEMM primitive probe

The standalone primitive wrapper built against MoK commit `3e1cf43ab93ad040afed52a45ab03cb490ffe4be` and ThunderKittens commit `1c3920d993404dd49a6d4c7267ea11d583bd5c68`. The isolated toolchain required all CUDA compiler packages to be pinned together: NVCC 13.0.88, CCCL 13.0.85, CUDA CRT 13.0.88, and NVVM 13.0.88. Mixing NVCC 13.2 with newer headers failed the CCCL version check; allowing CUDA CRT/NVVM 13.3 beside NVCC 13.0 emitted PTX 9.3 that its PTXAS 9.0 could not assemble.

The quick two-expert correctness check passed for W2 and both W13 projections with no NaNs or infinities. PTXAS reported 255 registers, five barriers, 224 bytes of static shared memory, and no spills for the primitive kernel. At the full 96-local-expert shape with every segment padded to 256 rows, standalone W2 measured 0.943 ms median (1,148 logical TFLOP/s), while the two separate W13 projection launches measured 2.036 ms total (1,063 logical TFLOP/s). These numbers cover only the grouped-GEMM primitive; they exclude dispatch, communication, SwiGLU, combine, and the complete MoK megakernel schedule.

## Already-dispatched DeepEP/MoK local composition

An ordinary Torch launch sequence composed the exact owner-rank-0 coalesced receiver relation with 256-row expert padding, the standalone MoK W13 and W2 primitives, standalone SwiGLU, fixed route-slot FP32 merge, inverse coalesced return, and a precomputed shared-output add. The route contained 6,755 coalesced receive tokens, 12,281 local assignments, and 24,576 padded expert rows. Repeated output was bitwise equal and finite.

The full already-dispatched local composition measured 6.890 ms. Individual stage medians were 0.419 ms for receiver gather and padding, 1.771 ms for the two W13 projections, 0.755 ms for standalone SwiGLU, 0.929 ms for W2, and 3.181 ms for fixed-slot return merge plus shared-output add. The ordinary Torch merge path is the dominant cost even before real communication and shared-expert computation are included. Recovering the fast GEMM primitives alone is therefore insufficient: the return merge, epilogue work, and launch/event schedule must move into the same physical composition.

## Future work

## Four-rank generated physical runtime

The physical runtime now consumes the compiler's `RelationPlan` and maps the
official DeepEP receive order through source-rank prefixes plus `recv_src_idx`.
All four ranks matched received payloads, local expert indices, valid router
weights, expert counts, and compiler row metadata exactly. The runtime uses the
standalone MoK grouped-GEMM primitive only, plus generated packing, SwiGLU, and
fixed-slot merge kernels. It does not call the MoK forward path or event graph.

The rank-max median over 10 warmups and 50 iterations was 4.4782 ms for the
sequential plan and 4.2682 ms when the generated shared expert ran concurrently
with asynchronous DeepEP dispatch. The latter passes the shared result to
DeepEP combine as its bias, preserving the fixed-rank combine order and avoiding
a separate shared-add launch. It is 18.1% slower than the 3.613-ms tuned MoK
oracle and 30.2% faster than the prior 6.113-ms compact DeepEP plan.

Exact-shape sequential and overlap outputs were bitwise equal and
repeat-deterministic. Separately, an independent small four-rank source-ordered
Torch reference passed with maximum absolute error 0.00012207 and mean absolute
error around 4.7e-6; this check avoids treating schedule parity alone as a
semantic oracle. The complete artifact is
`scratch/shuttle-generic-results/deepep-mok-distributed-rankmax-reference.json`.

## Final worker and gate/up layout selection

The DeepEP worker sweep continued through 56, 64, 80, and 96 communication
SMs. Rank-maximum overlap medians were 4.0154, 4.0179, 4.0148, and 4.0395 ms.
The increase at 96 SMs closed the worker-count sweep.

A fresh four-run comparison used 10 warmups and 50 rank-maximum samples for
separate and concatenated W13 at 56 and 80 SMs. Concatenation measured 3.9760
versus 4.0797 ms at 56 SMs and 4.0305 versus 4.1298 ms at 80 SMs. A repeated
56-SM concatenated run measured 3.9910 ms. The selected plan is therefore the
56-SM concatenated layout, with a two-run median of medians of 3.9835 ms.

All ranks retained exact compiler-to-DeepEP mappings, finite output, bitwise
sequential/overlap equality, repeat-bitwise equality, and a passing independent
small semantic reference. The reference maximum absolute error was
0.0001220703125.

The replacement tray reserialized the exact MoK-seeded route arrays. The NPZ
container SHA changed from `6ffd9d42...` to `c143b12f...`; this reflects archive
metadata, not different route tensors. The container-independent tensor-content
SHA256 is
`f1b5d8b3a53372eca228261b48b7ad9cfe925f1f8083f9cae07f9a24713f6908`.
It hashes each tensor name, dtype, shape, and C-order bytes for
`selected_experts` followed by `combine_weights`. Receiver assignment counts
remained `[12281, 12281, 12349, 12241]`.

The first low-priority holder expired only after its JSON artifacts were copied
locally. The replacement job `/dlwh/dev-gpu-shuttle-generic-final` reproduced
the environment and final A/B, then was released. Iris reports the job as
`killed` with reason `Terminated by user`; the Kubernetes pod is gone.

- [ ] Compare the default JAX 0.11 ragged collective mode with its valid NCCL-barrier/symmetric-memory mode only after the basic failure is localized.
- [x] Measure DeepEP with an explicit safe local assignment capacity and overflow reporting.
- [ ] Explain why the routed-only capped DeepEP graph is slower than the full routed-plus-shared graph.
