# SM90 Event Tensor replay compile blocker

This artifact preserves the attempted H100 replay of the structural Event
Tensor attachment. It is a compile-blocker record, not a benchmark result.

## Sources and environment

- canonical comparison source: `31e673a1a9`
- Event Tensor source: `e57397b0dd0dc2760a502a1214eb2df69c5df88e`
- hardware: one NVIDIA H100 80GB HBM3, compute capability 9.0
- driver: 595.71.05
- PyTorch: 2.11.0+cu128
- NVIDIA CUTLASS DSL: 4.5.2
- QuACK kernels: 0.5.0
- FlashAttention: 4.0.0b16
- JAX and jaxlib: 0.11.0

The two source trees used the identical Shuttle-owned normalized-exponential
helper:

```text
sha256 5adc1aa6ab6ad38d4b803bc4a1fa7c3ad645396470c89ec942560fd41391cb97
lib/tile_lifetime/backends/h100/cute_normalized_exp.py
```

No private cluster, pod, or GPU identifiers are retained.

## Exact benchmark commands

Canonical comparison:

```bash
export PYTHONPATH=/tmp/shuttle-base/lib/tile_lifetime:/tmp/shuttle-base/lib/tile_lifetime/src
/tmp/event-env/bin/python \
  /tmp/shuttle-base/lib/tile_lifetime/benchmarks/h100_cute_generated_streaming_attention.py \
  --sequence-length 2048 \
  --warmups 2 \
  --repeats 3 \
  --json-output /tmp/baseline-smoke.json
```

Event Tensor attachment:

```bash
export PYTHONPATH=/tmp/shuttle/lib/tile_lifetime:/tmp/shuttle/lib/tile_lifetime/src
/tmp/event-env/bin/python \
  /tmp/shuttle/lib/tile_lifetime/benchmarks/h100_cute_generated_streaming_attention.py \
  --sequence-length 2048 \
  --warmups 2 \
  --repeats 3 \
  --json-output /tmp/event-smoke.json
```

Both commands fail during CuTe IR verification in
`NormalizedExpFoldState.finalize`, before either command can emit benchmark
samples. The compiler reports that the row-sum register tensor does not
dominate the generated `cute.get_layout` use. `event_tensor_stderr.txt`
preserves the complete terminal traceback. The canonical command was not
redirected during the live reservation, so `canonical_stderr.txt` preserves
the exact compiler diagnostic and call site rather than pretending to be a raw
complete log.

The same failure and identical helper hash show that the attempted replay is
blocked by the current canonical normalized-exponential helper, not by the
Event Tensor schedule attachment. They do not establish performance neutrality:
there are no latency samples to compare.
