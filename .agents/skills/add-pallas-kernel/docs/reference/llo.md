# TPU LLO Diagnostics Reference

Use this when TPU kernel performance is unclear and wall-clock timing alone is not enough.

This guide explains how to collect and compare HLO/LLO/Mosaic dumps so you can answer:

- is the issue structural (algorithm/stage decomposition), or
- is the issue local tuning (tile/block sizes), or
- is the issue runtime/resource pressure (VMEM/register pressure/schedule packing).

## When to use this reference

Use LLO diagnostics when any of the following hold:

- Pallas is slower than XLA on a fixed shape and you do not know why.
- block-size sweeps are noisy or fail to improve throughput.
- compiler/runtime errors appear shape-dependent.
- decomposition variants (for example matmul-only) behave very differently from full kernels.

## Prerequisites

- Keep one fixed benchmark shape/config while diagnosing.
- Compare variants in the same environment and device type.
- Set dump flags before JAX initializes TPU.
- Record exact `XLA_FLAGS` and `LIBTPU_INIT_ARGS` with each run.

## Dump setup

Create separate directories per variant so artifacts do not mix.

Example layout:

- `${ROOT}/xla/hlo`, `${ROOT}/xla/llo`, `${ROOT}/xla/mosaic`
- `${ROOT}/pallas/hlo`, `${ROOT}/pallas/llo`, `${ROOT}/pallas/mosaic`
- `${ROOT}/decomp/hlo`, `${ROOT}/decomp/llo`, `${ROOT}/decomp/mosaic`

Set these before starting Python:

```bash
export HLO_DIR="${ROOT}/${VARIANT}/hlo"
export LLO_DIR="${ROOT}/${VARIANT}/llo"
export MOSAIC_DIR="${ROOT}/${VARIANT}/mosaic"

export XLA_FLAGS="\
  --xla_dump_to=${HLO_DIR} \
  --xla_dump_hlo_as_text"

export LIBTPU_INIT_ARGS="\
  --xla_jf_dump_to=${LLO_DIR} \
  --xla_jf_dump_hlo_text=true \
  --xla_jf_dump_llo_text=true \
  --xla_jf_dump_llo_html=false \
  --xla_jf_dump_llo_static_gaps=true \
  --xla_jf_emit_annotations=true \
  --xla_jf_debug_level=2 \
  --xla_mosaic_dump_to=${MOSAIC_DIR} \
  --xla_mosaic_enable_dump_debug_info=true \
  --xla_mosaic_enable_llo_source_annotations=true"
```

If you use scoped VMEM flags during kernel tuning, keep them constant while comparing variants.

## Variant matrix

For a high-signal comparison, run at least:

1. XLA/reference implementation.
2. Full Pallas implementation.
3. One decomposition variant (temporary debug variant that removes a suspected stage).

Keep input shapes, dtype, and benchmark harness identical across runs.

## What to inspect

### HLO

- Verify expected path is used (custom calls/fusions where expected).
- Confirm decomposition variant actually removed the intended stage.
- Check aliasing/update patterns that hint at writeback strategy.

### LLO schedule summaries

Look for files matching `*schedule-analysis_final_bundles.txt`.

Capture and compare:

- total bundle count,
- non-empty bundle count,
- kernel-specific schedule sections for your target op.

Large schedule inflation vs XLA often indicates structural inefficiency.

### LLO op pressure signals

Track relative changes in:

- lane-rotation-heavy ops (`vrot.*`),
- select/mask-heavy ops (`vsel`),
- expensive transcendental ops (`vpow2.*`, etc.),
- spill slots and register pressure.

A sharp increase with throughput drop is usually a structure/schedule issue, not just tile size.

### Mosaic/source annotations

Use emitted source metadata to map expensive kernels back to exact source locations.

## Analysis workflow

1. Record steady-state throughput first.
2. Compare XLA vs full Pallas schedule summaries.
3. Compare full Pallas vs decomposition variant.
4. If decomposition is fast but full is slow: focus on removed stage(s).
5. If decomposition is still slow: focus on core layout/tiling/scheduling.
6. Apply one bounded change and rerun the same fixed-shape matrix.

## XLA->Pallas replication loop

When trying to match XLA performance, follow this order:

1. Freeze one representative shape.
2. Dump XLA and current Pallas first.
3. Infer stage boundaries from XLA fusion structure.
4. Replicate the dominant stage first (usually GEMM-heavy stage).
5. Add missing stages incrementally; after each addition, rerun correctness + throughput + dumps.
6. Use LLO pressure counters to find the first regressing step.
7. Retune block sizes only after structure converges.

## Reporting template

Include this in logbook/PR notes:

- shape/dtype/hardware,
- throughput table for all variants,
- dump paths,
- schedule bundle comparison,
- notable pressure-counter deltas,
- next hypothesis.

## Common pitfalls

- Mixed dump directories across runs.
- Comparing different effective shapes/device counts.
- Changing VMEM flags between variants.
- Retuning before structural issues are resolved.
