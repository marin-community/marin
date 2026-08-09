# Provenance

## Source

- Base Shuttle revision: `348befaad13d130bf6f4dfc7b88b9ee19c17aa08`.
- Measured revision: `9de6770953d131efacccfd3348d72ed2cbec05c1`.
- The measured change makes benchmark arrays runtime arguments; it does not
  change the generated Fold semantics or schedule.

## Hardware and toolchain

- GPU: NVIDIA H100 80GB HBM3.
- GPU UUID: `GPU-7cf3cc97-9a2b-6f82-aaa0-35a9b1d41f0e`.
- NVIDIA driver: 595.71.05.
- Power limit: 700 W.
- Captured active clocks: 1830 MHz SM, 2619 MHz memory.
- JAX, JAXlib, CUDA plugin, and CUDA PJRT: 0.11.0.
- NVCC, PTXAS, NVVM, NVRTC, and NvJitLink: 13.0.88.
- CUDA runtime: 13.0.96.
- cuBLAS: 13.6.1.10.

The low-priority allocation requested one H100, one CPU, 16 GB host memory,
and 50 GB ephemeral disk. It was released immediately after artifact capture.

## Command

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH CUDA_HOME
PYTHONPATH=/tmp/shuttle-rms-components-348befaad1/lib/tile_lifetime/src \
  /app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/jax_generated_row_normalization_backward.py \
  --rows 2048 \
  --hidden 4096 \
  --threads 256 \
  --column-groups-per-block 32 \
  --warmups 10 \
  --repeats 30 \
  --iterations 100 \
  --profile-components \
  --seed 20260809 \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --artifact-directory /tmp/shuttle-rms-h100-components/corrected \
  --xla-dump-directory /tmp/shuttle-rms-h100-components/corrected/xla \
  --json-output /tmp/shuttle-rms-h100-components/corrected/summary.json \
  --shuttle-revision 9de6770953d131efacccfd3348d72ed2cbec05c1
```

The benchmark records 30 alternating generated-first and XLA-first samples for
the full reverse and each isolated component. Each sample contains 100
iterations followed by `jax.block_until_ready`.
