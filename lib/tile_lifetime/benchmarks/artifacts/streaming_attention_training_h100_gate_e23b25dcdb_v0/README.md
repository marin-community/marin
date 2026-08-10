# Streaming-attention training H100 gate

This artifact preserves the sole authorized H100 invocation of the matched
forward-plus-backward training boundary at Shuttle revision `e23b25dcdb`. It is
a negative environment-resolution result. It contains no device compilation,
correctness, latency, or throughput evidence.

## Requested comparison

The fixed generated and Flash-SDPA boundaries were:

- `Q`, `K`, `V`, and `dO` to `O`, `dQ`, `dK`, and `dV`;
- BF16 causal GQA at `B=1`, `S=2048`, `Hq=32`, `Hkv=8`, and `D=128`;
- scale `D**-0.5`;
- generated tile configuration `BM=32`, `BN=32`, eight warps, and three
  stages;
- 30 counterbalanced samples with five calls per sample;
- generated physical output layout matched to contiguous Flash-SDPA BHSD
  results.

The job requested one preemptible H100, one CPU, 16 GB host memory, 40 GB disk,
batch priority, a one-hour timeout, and zero retries. Iris reported terminal
failure after 5.26 seconds with no preemption or retry.

## Failure

The minimal 0.5 MB Iris bundle launched and created a Python 3.12.13 virtual
environment. Dependency resolution then rejected this explicit requirement:

```text
nvidia-cuda-cccl-cu13>=13.0
```

The configured package index exposes only `nvidia-cuda-cccl-cu13<=0.0.1` under
that name. The CUDA 13 JAX extra had not yet been resolved, the generated AOT
sources were not compiled, and neither Shuttle nor Flash-SDPA executed. This is
a launch-envelope dependency-name/version error, not a limitation of the
generic streaming physical family.

The no-retry instruction was honored. `release-proof.txt` records the terminal
Iris state and an exact Kubernetes task-label query returning no pod.

## Static checkpoint

The pre-device checkpoint remains:

- implementation commits `aa67df5400` and `e23b25dcdb` on
  `codex/attention-training-parity`;
- 18 focused recovery/generation tests passing;
- changed-files pre-commit passing;
- natural JAX `jax.vjp` ownership;
- compiler-owned generic Contract, Map, Fold, and DomainRestriction forward
  and backward;
- no opaque FlashAttention call in the generated path.

No H100 or GB200 performance claim follows from this artifact. The GB200 replay
was conditional on successful H100 compilation and correctness, so it was not
submitted.
