# DeepEP for Grug MoE on NVIDIA GPUs

Levanter has an experimental JAX FFI integration for DeepEP intranode expert-parallel dispatch and combine. The
integration is optional: ordinary installs keep using the built-in Grug MoE backends, and DeepEP is only loaded when a
DeepEP-backed backend or benchmark path is selected.

## Install

DeepEP is treated as an external source checkout rather than a vendored package.

```bash
git clone https://github.com/deepseek-ai/DeepEP.git /path/to/DeepEP
export DEEPEP_SRC_ROOT=/path/to/DeepEP
export DEEPEP_CUDA_ARCH=sm_100  # B200/GB200. Use sm_90 or sm_90a for H100-class systems.
```

The raw FFI build path needs `nvcc` on `PATH`. The compiled objects are cached under `~/.cache/marin` by default. To
put the cache somewhere else:

```bash
export MARIN_DEEPEP_CACHE_DIR=/path/to/cache
```

For environments where raw `nvcc` linking does not find the CUDA/PyTorch runtime libraries cleanly, the transport FFI
also supports a Torch extension build:

```bash
export DEEPEP_BUILD_WITH_TORCH_EXTENSION=1
```

Keep `DEEPEP_LOAD_AS_PYTHON_MODULE=0` with the Torch extension build path. The Python-module loader is for the raw FFI
artifact path.

## Preflight

Run the preflight before launching a GPU job:

```bash
uv run python -m levanter.kernels.deepep.preflight
```

The preflight checks:

- `DEEPEP_SRC_ROOT` is set and contains the DeepEP transport sources.
- `nvcc` is available.
- `DEEPEP_CUDA_ARCH` is one of `sm_90`, `sm_90a`, or `sm_100`.
- incompatible FFI loader/build flags are not set together.

The first real use compiles the layout and transport FFI libraries into the cache. On B200, use `DEEPEP_CUDA_ARCH=sm_100`
so the cached object targets the right device.

## Backend Selection

For a Grug MoE block with an expert mesh axis, set the config object field:

```python
config = dataclasses.replace(config, moe_implementation="deepep")
```

The backend currently targets intranode expert parallelism: the expert group must span all visible local GPUs in the
process, and the hidden dimension must be divisible by 8. If those constraints do not hold, use `moe_implementation="ring"`
as the built-in fallback.

The benchmark harness also contains capped DeepEP variants that precompute tighter receive capacities for comparing
transport performance. The production backend uses conservative static capacities for correctness and integration
simplicity; capacity tightening is the next optimization once the runtime config path settles.
