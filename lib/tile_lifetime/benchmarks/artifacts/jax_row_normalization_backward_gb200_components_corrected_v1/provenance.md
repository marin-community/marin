# Provenance

## Source

- Measured Shuttle revision: `07bbabb1843155e105f3579075667ae8983e314a`.
- Allocation-control client revision: `86f20400aec5c174d756de441060632b6bd664ca`.
- The abbreviated `07bbabb184c` stored in the raw JSON resolves to the measured
  Shuttle revision above.

The allocation client is recorded separately because the GB200 controller
requires a newer Iris protocol than the pinned Shuttle branch. It did not
provide benchmark source or execute in the measured process.

## Hardware and toolchain

- GPU: one NVIDIA GB200 from a four-GPU GB200 node pool.
- GPU UUID: `GPU-c03e8225-9621-ac19-aa1e-1c1c1b3e4caa`.
- NVIDIA driver: 595.71.05.
- Power limit: 1200 W.
- Clock policy: cluster-default, unpinned.
- Captured active clocks: 1950 MHz SM, 3996 MHz memory.
- JAX, JAXlib, CUDA plugin, and CUDA PJRT: 0.11.0.
- NVCC, CUDA CRT, and NVVM: 13.3.73.
- CUDA runtime: 13.3.29.
- cuBLAS: 13.6.1.10.
- Generated CUDA architecture: `sm_100a`.

No B200 device or B200 result is part of this artifact.

The successful batch-priority holder requested one GB200, one CPU, 32 GB host
memory, and 50 GB ephemeral disk. A preceding 16 GB holder was released after
the JAX CUDA wheel installation was killed for host-memory pressure; no
benchmark ran in that holder. The successful holder was released immediately
after both captures and artifact extraction. No matching active session or pod
remained.

`environment.txt`, `runtime-packages.txt`, `post-telemetry.txt`, and
`generated-library-ldd.txt` preserve the detailed environment.

## Primary command

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH CUDA_HOME
PYTHONPATH=/tmp/shuttle-rms-source/lib/tile_lifetime/src \
  /app/.venv/bin/python \
  /tmp/shuttle-rms-source/lib/tile_lifetime/benchmarks/jax_generated_row_normalization_backward.py \
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
  --architecture sm_100a \
  --artifact-directory /tmp/shuttle-rms-gb200-components/corrected \
  --xla-dump-directory /tmp/shuttle-rms-gb200-components/corrected/xla \
  --json-output /tmp/shuttle-rms-gb200-components/corrected/summary.json \
  --shuttle-revision 07bbabb184c
```

The confirmation command is identical except that `corrected` is replaced by
`confirmation` in the three output paths. Each capture alternates
generated-first and XLA-first order across 30 samples, with 100 iterations and
`jax.block_until_ready` per sample.
