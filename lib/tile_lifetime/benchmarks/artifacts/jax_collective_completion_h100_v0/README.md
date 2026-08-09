# JAX-owned collective completion on two H100 GPUs

This artifact records a real two-GPU execution of Shuttle's generic
`CollectiveFoldPlan` and Event Tensor completion boundary at revision
`e7b37a07f098caa7d8e433061b08ecfa75f68080`.

The run used two NVIDIA H100 80GB HBM3 GPUs, compute capability 9.0, driver
595.71.05, JAX/JAXlib 0.10.1, and BF16 inputs. The allocation requested two CPU
cores, 32 GB of host memory, 50 GB of disk, and batch priority. Exact device
UUIDs, sampled clocks, and power limits are retained in `results.json`.

## Boundary

Shuttle recovered the collective Fold and derived one system-visible Event
Tensor completion with initial count two. JAX emitted and differentiated the
collective; XLA selected and executed its physical transport. Shuttle did not
register a custom adjoint or communication kernel.

The full-group sum, grouped maximum mutation, and differentiated sum all ran
on both GPUs. Their generated StableHLO contains ordinary
`stablehlo.all_reduce` operations and no semantic custom call.

## Results

- Sum maximum absolute error: 0.
- Maximum mutation maximum absolute error: 0.
- Gradient maximum absolute error: 0.
- Sum, maximum, and gradient outputs were bitwise deterministic across
  repeated executions.
- Sum and maximum forward StableHLO each contain one all-reduce.
- Differentiated StableHLO contains two all-reduces.
- All three StableHLO modules contain zero custom calls.

This is correctness and ownership-boundary evidence, not a collective latency
comparison. The VMM warnings in the live process reported that fabric handle
creation was unavailable and that JAX retried with simpler handles; execution
completed successfully.

## Files

- `results.json`: exact device, toolchain, correctness, determinism, operation,
  and Event Tensor metadata.
- `sum-forward-stablehlo.txt`: full-group sum.
- `maximum-forward-stablehlo.txt`: grouped reducer mutation.
- `sum-gradient-stablehlo.txt`: JAX-owned reverse computation.
- `SHA256SUMS`: sealed manifest.

