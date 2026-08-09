# JAX collective completion CPU replay

This artifact records a backend-neutral execution check for Shuttle's recovered
collective Fold and Event Tensor completion boundary. Four virtual CPU devices
execute one full-group sum and two two-device maximum groups through JAX
named-axis collectives. JAX differentiates the sum path.

Reproduce from the repository root:

```bash
PYTHONPATH=lib/tile_lifetime/src \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run python lib/tile_lifetime/benchmarks/jax_collective_completion_cpu.py
```

The forward StableHLO contains one `stablehlo.all_reduce` and no custom call.
The differentiated program contains two all-reduces: the primal collective and
the JAX-generated adjoint collective. Forward, maximum-mutation, and gradient
results have zero error against direct references.

The replay used JAX 0.10.1, JAXlib 0.10.1, Python 3.12.11, and
macOS 15.5 on arm64. `environment.json` records the source hash and base Shuttle
revision; `SHA256SUMS` seals the machine-readable evidence.

This is not GPU or multi-host evidence. The physical transport remains owned by
JAX/XLA. Shuttle owns recovery of the reducer and numerical policy, global
device-ID to logical-axis mapping, replica grouping, and the system-visible
Event Tensor completion contract.
