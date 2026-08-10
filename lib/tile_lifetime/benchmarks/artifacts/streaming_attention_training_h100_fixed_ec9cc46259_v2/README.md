# Streaming-attention training H100 fixed run

This artifact records the single fixed H100 replay authorized after
`nvidia-cuda-cccl==13.3.3.4.1` was added to the Linux GPU dependency lock at
Shuttle base revision `ec9cc46259df3ce29e7ebe95675e5f1bb63d854e`.

The component boundary is the existing ordinary JAX function from Q, K, V, and
output cotangent to forward output plus Q, K, and V cotangents. The generated
candidate is recovered from Contract, Map, normalized-exponential Fold, and
DomainRestriction semantics. The matched expert is Torch SDPA forced to its
Flash backend; Torch is benchmark-only.

The fixed configuration is BF16 causal GQA with batch 1, sequence 2,048, 32
query heads, 8 K/V heads, head dimension 128, block M/N 32, eight warps, and
three pipeline stages. One process uses five warmups followed by 30
counterbalanced samples of five iterations. No tuning or automatic retry is
permitted.

The earlier CPU-only dependency gate
`/dlwh/shuttle-attention-cccl-preflight-20260809` compiled CUDA FP16/BF16 and
`<nv/target>` headers with the corrected lock before this H100 experiment was
submitted. The H100 reproduction repeats that header compile before the only
benchmark invocation.

## Result

This is a negative layout-contract artifact, not a performance result. The
sole actual benchmark invocation ran the generated and Flash-SDPA boundaries,
then the pre-timing physical-layout guard rejected their result strides:

```text
Flash-SDPA:
((8388608, 128, 4096, 1),
 (8388608, 128, 4096, 1),
 (2097152, 128, 1024, 1),
 (2097152, 128, 1024, 1))

generated expectation:
((8388608, 262144, 128, 1),
 (8388608, 262144, 128, 1),
 (2097152, 262144, 128, 1),
 (2097152, 262144, 128, 1))
```

The first tuple in each group is forward output, followed by dQ, dK, and dV.
The observed Flash-SDPA storage is contiguous BSHD viewed as BHSD. The
generated boundary instead requested the wrong axis order. The process exited
before warmup or timing, so this artifact makes no latency, parity,
correctness, or determinism claim. No GB200 run followed.

The failed actual invocation was Iris task
`/dlwh/shuttle-attn-training-h100-actual-20260809`. It used one H100 80GB HBM3
(UUID `GPU-87333bd6-8e3c-d187-e433-a290b2da8e1b`), driver 595.71.05, CUDA
runtime 13.0.96, NVCC/PTXAS 13.2.78, CCCL 13.3.3.4.1, JAX/JAXLIB 0.10.1,
Triton 3.6.0, and Torch 2.11.0+cu128 as oracle only. The CUDA header smoke
succeeded and produced a 10,288-byte SM90 object before the benchmark was
invoked.

An earlier task, `/dlwh/shuttle-attn-training-h100-cccl-20260809`, failed
before invoking the benchmark because the Iris source bundle intentionally
contains no `.git` directory. The reproduction script was changed to carry the
pinned revision explicitly before the actual invocation.

## Preserved evidence

`raw/iris.log` is the complete Iris output from the actual invocation.
`raw/preserved.tgz` is the exact job-produced archive containing the generated
AOT C sources and headers, typed-FFI CUDA handler, shared object, source
StableHLO bytecode, copied generator sources, Triton metadata, header-smoke
source/object, and toolchain records. Its SHA-256 is
`97944bd4f6cf29ad6d9bc29604f8419e0971417c43799af9f8210ad48b0ed3cb`.

After terminal failure, the H100 release query returned `No resources found`
for label
`iris.task_id=dlwh-shuttle-attn-training-h100-actual-20260809-0`.
