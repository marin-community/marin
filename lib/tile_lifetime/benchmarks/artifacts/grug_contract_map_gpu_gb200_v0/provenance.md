# Grug GPU typed-FFI Contract+Map replay

- Shuttle source revision: `2ed4741e134d89ceff918d6b6b3fbfc929c93254`
- Source archive SHA256: `5d8f854beae0efbb15d3fb2e652bf812d932956e743dff3f3dfa29deb822a23f`
- Iris job: `/dlwh/dev-gpu-dlwh-shuttle-grug-gb200-replay`
- Pod: `iris-dlwh-dev-gpu-dlwh-shuttle-grug-d03df6d7-0-1b32c1372fd530cd`
- Allocation: one GB200, batch priority, 1 CPU, 16 GB memory, 50 GB disk
- Clock policy: cluster-default, unpinned; telemetry captured before and after

## Final toolchain

- JAX/JAXlib/CUDA plugin/PJRT: 0.11.0
- NVIDIA driver: 595.71.05
- CUDA NVCC/CRT/NVVM/NVRTC/NvJitLink: 13.0.88
- CUDA runtime: 13.0.96
- cuBLAS: 13.0.2.14
- GPU architecture: `sm_100a`

The pip CUDA wheel contained only versioned shared objects:

- `libcublas.so.13`
- `libcublasLt.so.13`
- `libcudart.so.13`

The replay temporarily added `libcublas.so -> libcublas.so.13` and
`libcudart.so -> libcudart.so.13` inside the disposable virtual environment.
This is an environment repair, not part of the Shuttle source result.

`nvidia-nvptxcompiler==13.3.73` remained installed after a failed exploratory
attempt, but the successful XLA path used the coherent 13.0.88 NVVM/PTXAS
toolchain. The failures are preserved in `run.log`, `run2.log`, and `run3.log`.

## Final command

```bash
export PATH=/app/.venv/bin:$PATH
export PYTHONPATH=/tmp/shuttle-2ed4741e13/lib/tile_lifetime/src:/tmp/shuttle-2ed4741e13
export CUDA_PIP_ROOT=/app/.venv/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_PIP_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LIBRARY_PATH=$CUDA_PIP_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}

/app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/xla_grug_backward_multi_output_gpu_custom_call_smoke.py \
  --nvcc $CUDA_PIP_ROOT/bin/nvcc \
  --cuda-architecture sm_100a \
  --artifact-directory /tmp/shuttle-grug-gb200-replay/final2 \
  --output /tmp/shuttle-grug-gb200-replay/final2/summary.json \
  --warmup 4 \
  --repeats 30
```

The timing extension used for this replay is preserved as
`benchmark-timing.patch` and was subsequently committed. The compiler and
replacement logic are otherwise the exact pinned Shuttle revision.
