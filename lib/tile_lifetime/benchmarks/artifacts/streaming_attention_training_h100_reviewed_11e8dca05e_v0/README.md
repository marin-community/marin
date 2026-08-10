# Streaming-attention training H100 reviewed replay

This artifact pins the single fixed H100 replay authorized after the physical
and numerical boundary review at Shuttle revision
`11e8dca05ed4e2207cc02312c9d6265ea1c32c58`.

The component boundary takes ordinary JAX Q, K, V, and output-cotangent BSHD
buffers and returns forward output plus Q, K, and V cotangents. The generated
candidate comes from recovered Contract, Map, normalized-exponential Fold, and
DomainRestriction semantics. The expert is Torch SDPA forced to its Flash
backend; Torch remains benchmark-only.

The fixed configuration is BF16 causal GQA with batch 1, sequence 2,048, 32
query heads, 8 K/V heads, head dimension 128, block M/N 32, eight warps, and
three pipeline stages. The timed Torch wrapper includes zero-copy BSHD/BHSD
input and result views. Both implementations include forward and backward.

Before warmup, the benchmark must establish exact dtype, shape, stride, and
minor-to-major equivalence for Q/K/V/dO and O/dQ/dK/dV. It must also establish
repeat determinism, finite outputs/errors, maximum absolute error at most
0.125, and mean absolute error at most 0.01 for generated and expert outputs
against the independent semantic reference.

The reproduction performed the locked dependency and CUDA header preflight,
then invoked the fixed benchmark command exactly once. It permitted no tuning,
retry, or GB200 follow-up.

## Result

The H100 run is a sealed negative result. The generated Shuttle path passed its
repeat-determinism and numerical-acceptance gates. The logical input and output
contracts also matched exactly across the generated and Flash-SDPA paths. The
Flash-SDPA oracle then produced different bitwise hashes on its two pre-timing
executions. The fail-closed gate raised
`Flash-SDPA oracle violates the declared deterministic execution contract`, so
the run recorded no timing samples and does not establish a parity ratio.

The failure occurred after compilation and after both logical-boundary checks,
but before warmup. The timed boundary was therefore never entered. The result
does not show a generated-kernel correctness, layout, or performance failure;
it shows that the forced Flash-SDPA training oracle did not satisfy the
benchmark's declared bitwise repeatability requirement on this H100 software
stack.

The generated and expert boundaries used dense BSHD storage with
minor-to-major `(3, 2, 1, 0)`. Q, dO, O, and dQ used shape
`(1, 2048, 32, 128)` and strides `(8388608, 4096, 128, 1)`. K, V, dK, and dV
used shape `(1, 2048, 8, 128)` and strides `(2097152, 1024, 128, 1)`. The
Torch call created zero-copy BHSD views inside its callable and returned
zero-copy BSHD views; no explicit layout copy was timed.

The run used one H100 80 GB HBM3 with driver 595.71.05, CCCL 13.3.3.4.1,
CUDA compiler 13.2.78, JAX/JAXLIB 0.10.1, Torch 2.11.0+cu128, and Triton 3.6.0.
Iris reports the single task terminal after 31.66 seconds, and a running-job
query for the job prefix returned no jobs after collection.

`results/negative-result.json` contains the concise gate and build audit.
`results/iris-job.log` is the unedited Iris log. `results/preserved.tgz` is the
worker-emitted evidence archive and contains the generated CUDA handler,
Triton AOT sources with embedded CUBIN, registered DSO, StableHLO fixture,
environment, and benchmark sources. Its SHA-256 is
`0d30440bce61d26947f45cd917af0731c61644a61816ec481d23b9158498ce9a`.
