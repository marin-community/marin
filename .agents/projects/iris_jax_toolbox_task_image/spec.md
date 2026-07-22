# JAX-Toolbox-backed Iris GPU task image contracts

## Image artifacts

`lib/iris/Dockerfile.gpu` builds `ghcr.io/marin-community/iris-task-gpu` from a
pinned NVIDIA JAX-Toolbox multi-architecture digest. It installs no Marin,
Levanter, Haliax, or experiment source. CI publishes immutable `git-<short-sha>`
tags for `linux/amd64` and `linux/arm64`.

Every compatible image provides these UTF-8 artifacts:

- `/etc/iris/preserved-python-packages` contains one PEP 503-normalized Python
  distribution name per line. Blank lines and lines beginning with `#` are
  ignored. Every entry is excluded from overlay installation.
- `/etc/iris/required-python-packages` uses the same syntax. Every entry must
  also occur in the preserved manifest and must resolve in the image. Preserved
  names omitted here are explicit absent-package exclusions.
- `/etc/iris/system-cuda-toolchain` is an empty capability marker. Its presence
  means Iris must not stage a venv-owned CUDA toolkit or cuDNN runtime.

`lib/iris/Dockerfile.gpu` contains the canonical initial manifest tuple. It
includes JAX, jaxlib, CUDA plugin/PJRT packages, TransformerEngine, every
CUTLASS DSL distribution, cuda-tile, `cuda-toolkit`, CUDA 12/13 library wheel
names, and matching CPU-only Torch/Torchvision. Adding or removing a name is a
reviewed image contract change, not runtime discovery.

## Python overlay

The public setup entry point retains its existing signature:

```python
def default_setup_script(
    *,
    extras: Sequence[str] = (),
    pip_packages: Sequence[str] = (),
    python_version: str | None = None,
    packages: Sequence[str] | None = None,
) -> str:
    """Render the standard uv-based setup script as a Bash string."""
```

When the manifests are absent, the rendered script keeps the existing isolated
venv behavior. When they are present, the script:

1. rejects an empty preserved manifest, invalid names, extras, version
   specifiers, URLs, duplicates after PEP 503 normalization, and required names
   absent from the preserved manifest;
2. rejects a requested Python major/minor that differs from the image Python;
3. requires every required entry to resolve through the image interpreter;
4. creates `$IRIS_VENV` with `uv venv --system-site-packages`;
5. passes one `--no-install-package <name>` argument per preserved entry to the
   existing frozen `uv sync` command;
6. verifies that each required entry still resolves outside `$IRIS_VENV`; and
7. verifies that each explicit exclusion remains absent from `$IRIS_VENV`.

A failure exits setup nonzero with an `[iris setup]` prefix and names the
offending distribution. Distribution roots are computed with
`importlib.metadata.distribution(name).locate_file("").resolve()`; a
symlink-resolved path under `$IRIS_VENV` is a shadowing failure. Import-origin
probes are mandatory for JAX, CUDA plugin/PJRT, TransformerEngine, CUTLASS,
cuda-tile, Torch, and Torchvision. `pip_packages` retain their post-sync install
order and receive the same final audit. `packages` retains its scoped-sync
semantics.

When `/etc/iris/system-cuda-toolchain` exists, both Python manifests must also
exist or setup fails. CUDA staging logs `using image CUDA toolchain` and stages
no CUDA executable or cuDNN wheel. Python preservation without the CUDA marker
remains valid.

The task log records the image digest and protected package versions and paths.
The environment intentionally differs from `uv.lock`. Later `uv sync` or
`pip install` commands that replace protected packages are unsupported.

## Kubernetes execution

Phase one uses the existing `RunTaskRequest.task_image` override. It adds no
provider configuration and does not alter `KubernetesProviderConfig.default_image`.
Jobs without an explicit override, including CUDA-PyTorch and vLLM GPU jobs,
retain their current image. The workdir init container and log shipper are
unchanged.

For the main task container Iris supplies
`args: ["bash", "-lc", generated_script]` and omits `command`. Compatible image
entrypoints must ultimately `exec` these arguments. Tests pin exit-code and
SIGTERM propagation for the current no-entrypoint image and NVIDIA's entrypoint.
Iris contains no NVIDIA-specific entrypoint path.

## Benchmark contract

The decision sequence is control→treatment→control using one benchmark patch
derived from source commit `31b221e1db02bf553488c903ed36d8ae1f424b63`.
All model, data, placement, profiler, and source settings are identical; only
the immutable `SCALE_TASK_IMAGE` digest changes. Selected samples are steps
5–11 and 15–19; warmup steps 0–4 and profiler steps 12–14 are excluded. Compare
the treatment median with the mean of the two control medians. Reported MFU is
Levanter's `throughput/mfu` metric. Control median drift must be below 1%, and
treatment throughput regression must not exceed 2%.

The L48 no-scan treatment sets `SCALE_SCAN_LAYERS=0` and
`SCALE_MAX_RETRIES_FAILURE=0`. Its terminal outcome is classified as temporary
arena OOM, null-module `cuModuleLoadData` failure, other compiler/runtime
failure, or success. A successful run supports only that exact configuration;
two further successes are required before reporting a changed CUBIN failure
rate.

## Errors

- `Invalid preserved package manifest`: a manifest is empty after filtering,
  contains a duplicate after normalization, contains invalid syntax, or a
  required name is not preserved.
- `Python version mismatch`: the requested overlay major/minor differs from the
  image interpreter.
- `Required package missing`: a required distribution does not resolve before
  sync.
- `Preserved package shadowed`: a required distribution resolves from
  `$IRIS_VENV` after sync.
- `Excluded package installed`: an explicit exclusion is present in
  `$IRIS_VENV` after sync.

These are setup failures using the existing Iris task-failure path. No RPC or
persisted database shape changes.

## File locations

| Contract | Location |
|---|---|
| GPU task image | `lib/iris/Dockerfile.gpu` |
| Overlay rendering | `lib/iris/src/iris/cluster/setup_scripts.py` |
| Kubernetes entrypoint behavior | `lib/iris/src/iris/cluster/backends/k8s/tasks.py` |
| GPU image CI | `.github/workflows/ops-docker-images.yaml` |

## Out of scope

- Installing Marin, Levanter, Haliax, or experiment source in the image.
- Resource-derived GPU image selection or changing any cluster default.
- Selecting a GPU image in the worker/Docker backend.
- Cross-rack GPUDirect RDMA validation.
- Enabling O1, PGLE, fusion, or collective-threshold tuning in the image
  comparison.
- Claiming that one successful no-scan run eliminates every CUBIN failure mode.
