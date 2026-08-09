# Generated gradient skeletons on GB200

This checkpoint measures the fixed generic row-Fold and streaming-attention
backward schedules at Shuttle revision
`14f2bbc3f9871ab9ccf255f0b6b8b0983c2f9b83`. It also records an attempted GPU
XLA FFI execution proof at revision `36e2a52aad`. No schedule parameter was
tuned during this replay.

The run used one batch-priority NVIDIA GB200. The JSON files contain every raw
sample, counterbalanced execution order, deterministic output hashes, numerical
errors, exact revision, and device telemetry. The environment was Torch
2.10.0+cu130, Triton 3.6.0, driver 595.71.05, and a coherent CUDA 13.0 compiler,
runtime, CCCL, and NVVM stack. The GPU reported 1,950 MHz SM and 3,996 MHz memory
clocks during each completed benchmark.

## Row-normalization backward

Both measurements use 2,048 rows, hidden size 4,096, 256 threads, and the fixed
generic 32-column-groups-per-block schedule. Each result has 30 counterbalanced
samples with ten iterations per sample after ten warmups.

| Statistic | Generated | `torch.compile` | Ratio | Previous generated | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMS | 0.039704 ms | 0.051835 ms | 0.766x | 0.087160 ms | 2.20x |
| Centered LayerNorm | 0.041109 ms | 0.050499 ms | 0.814x | 0.088042 ms | 2.14x |

Both generated paths are deterministic. Maximum dX error is 1.90735e-6 for
both. Maximum feature-scale-gradient error is 7.62939e-5. These results show
that the generic corrected Fold schedule closes the prior 1.597x RMS and 1.660x
LayerNorm gaps on this shape.

## Streaming-attention backward

The generated path uses causal BF16 GQA with 32 query heads, eight KV heads,
head dimension 128, 32x32 tiles, deterministic query-major dQ, and deterministic
key-major dK/dV. It does not use atomics or an opaque attention-backward call.

At sequence length 128, all numerical gates pass and the generated median is
0.080550 ms versus 0.143165 ms for Torch SDPA backward. This is only a compile,
execution, and correctness proof because the workload is too small to evaluate
the physical schedule.

At the primary sequence length 2,048, all numerical gates still pass:

- forward maximum/mean absolute error: 0.0078125 / 7.42370e-5;
- dQ: 0.0078125 / 2.86192e-5;
- dK: 0.03125 / 1.51095e-4;
- dV: 0.03125 / 1.42324e-4.

The generated median is 2.200320 ms versus 0.155450 ms for Torch SDPA
backward, or 14.155x slower. The result is a correct generic backward proof but
not a performance acceptance result. The main physical gap is the simple
Triton schedule: its key-major dK/dV path serially revisits query heads and
query tiles, and neither reverse path has the shared-memory/TMA/WGMMA overlap of
an expert streaming-attention backward.

## GPU XLA FFI attempt

The exact `36e2a52aad` smoke did not reach CUDA handler generation. After a
separately preserved Torch/torchvision import mismatch was corrected, JAX
0.11.0 compiled the baseline natural Grug step but failed inside the registered
PRE_SCHEDULER transformation:

```text
ValueError: expected one multi-output pair-Map region, found ()
```

The callback therefore saw a different GPU pre-scheduler form than its assumed
region. No generated shared library, transformed HLO, or success summary was
emitted. `xla-gpu-ffi-recovery-failure.txt` is the authoritative source
diagnostic. This checkpoint does not claim a successful GPU XLA replacement.

## Commands

The row-normalization invocations used:

```text
python lib/tile_lifetime/benchmarks/h100_generated_row_normalization_backward.py \
  --rows 2048 --hidden 4096 --threads 256 --column-groups-per-block 32 \
  --statistic {rms,layer} --warmups 10 --repeats 30 --iterations 10 \
  --shuttle-revision 14f2bbc3f9871ab9ccf255f0b6b8b0983c2f9b83 \
  --json-output <output.json>
```

The primary attention invocation used:

```text
python lib/tile_lifetime/benchmarks/h100_generated_streaming_attention_backward.py \
  --sequence 2048 --mutation causal --block-m 32 --block-n 32 \
  --warmups 5 --repeats 10 --iterations 5 \
  --shuttle-revision 14f2bbc3f9871ab9ccf255f0b6b8b0983c2f9b83 \
  --json-output <output.json>
```

The exact source archives staged on the worker had SHA-256 values
`b79578975b28ac6031e6bc48ead224c685f4fe97234b4a429a07c4c5a548031f`
for `14f2bbc3f9` and
`69dd664aa47a630fd4dbfcc2c9ed6e3e330ecba914fbbeea9bc511657f8b34da`
for `36e2a52aad`.
