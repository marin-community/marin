# SM100 grouped-Contract Event Tensor attachment

This artifact records the first GPU build and correctness proof for the
generated grouped-Contract synchronization ABI.

The source revision is `30c0ba6bfc123f81ec5d4c67cbc2bc91fc162866`.
The external generic grouped-GEMM primitive is pinned to Mixture-of-Kittens
`3e1cf43ab93ad040afed52a45ab03cb490ffe4be` and ThunderKittens
`1c3920d993404dd49a6d4c7267ea11d583bd5c68`; both checkouts were clean.

The run used one physical NVIDIA GB200 reported by `nvidia-smi`, driver
595.71.05, PyTorch 2.10.0+cu130, and NVCC 13.0.88. The scheduler requested one
GPU with four host CPUs and released the allocation immediately after the
proof.

Command shape:

```text
gb200_mok_gmm_probe.py
  --component w2
  --experts 1
  --hidden-size 256
  --intermediate-size 256
  --rows-per-expert 256
  --warmup 2
  --iterations 5
```

The emitted ABI fingerprint matched the extension at load time. Correctness
passed against the Torch FP32 grouped-GEMM reference. The timing distribution
is retained only as a smoke measurement; this artifact is not a performance
comparison.

Scope boundary: Shuttle owns the generated synchronization ABI/counts at the
wrapper boundary. The external primitive still owns the internal placement of
barrier arrival/wait instructions, phase advancement, TMA issue, and
accumulator release.
