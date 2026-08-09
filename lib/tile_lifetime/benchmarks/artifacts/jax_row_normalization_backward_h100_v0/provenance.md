# Provenance

## Source

- Shuttle revision under test:
  `9e1a556477a38a4b73922a83aa8514539939e58a`
- Exact source archive SHA-256:
  `28859f8ca22e27777ae625498d2867aa613767e4a3030f8a197c7527d9e089f2`
- The committed patch adds only generic versioned CUDA-library linker flags,
  the corresponding benchmark call-site updates, and focused unit coverage.

## Hardware and toolchain

- GPU: NVIDIA H100 80GB HBM3
- GPU UUID: `GPU-7cf3cc97-9a2b-6f82-aaa0-35a9b1d41f0e`
- NVIDIA driver: 595.71.05
- Power limit: 700 W
- Captured clocks: SM 1830 MHz, memory 2619 MHz
- JAX/JAXlib/CUDA plugin: 0.11.0
- NVCC, PTXAS, NVVM, NVRTC, NvJitLink: 13.0.88
- CUDA runtime: 13.0.96
- cuBLAS: 13.6.1.10

The allocation used one low-priority H100 with one CPU, 16 GB host memory,
and 50 GB ephemeral disk. It was released after copying and verifying the raw
artifact.

## Command

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH CUDA_HOME
cd /tmp/shuttle-9e1a556477
PYTHONPATH=lib/tile_lifetime/src \
  /app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/jax_generated_row_normalization_backward.py \
  --rows 2048 \
  --hidden 4096 \
  --threads 256 \
  --column-groups-per-block 32 \
  --warmups 10 \
  --repeats 30 \
  --iterations 10 \
  --seed 20260809 \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --artifact-directory /tmp/shuttle-rms-h100-replay/final2 \
  --json-output /tmp/shuttle-rms-h100-replay/final2/summary.json \
  --shuttle-revision 9e1a556477a38a4b73922a83aa8514539939e58a
```

The benchmark performed ten warmups, then 30 alternating generated-first and
XLA-first measurements with ten iterations per sample.

