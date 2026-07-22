# JAX-Toolbox-backed Iris GPU task images

Iris should use NVIDIA JAX-Toolbox as the authoritative GPU runtime while continuing to synchronize Marin, Levanter, and Haliax from each submitted workspace. A separate `iris-task-gpu` image will supply JAX/XLA, CUDA, NCCL, TransformerEngine, and kernel tooling. Iris will install workspace source and non-GPU dependencies into an overlay venv without replacing the image-owned accelerator stack. [Research](./research.md) records the current code paths, image evidence, and #7507 baseline.

## Challenges

The current setup creates an isolated venv and installs the locked `gpu` extra. That venv shadows system packages, so a JAX-Toolbox base has no effect unless Iris both exposes system site-packages and prevents uv from installing selected distributions. The protected boundary is larger than `jax` and `jaxlib`: JAX CUDA plugins, PJRT, TransformerEngine, CUTLASS DSL, cuda-tile, and Python-packaged CUDA libraries can form one binary compatibility set.

The task image is currently independent of requested resources. `run_req.task_image` overrides one cluster default, and GPU resources are parsed only after that choice. Kubernetes also replaces OCI `ENTRYPOINT`; NVIDIA's initialization ran in the successful canary only because the command invoked it explicitly. These behaviors need generic contracts rather than NVIDIA-specific branches.

## Costs / Risks

- JAX-Toolbox is roughly 10 GB compressed. Pulling it for CPU coordinators would waste startup time and node cache, so GPU image selection adds configuration and provider logic.
- The Toolbox digest becomes a second dependency lock alongside `uv.lock`. Its nightly JAX may not satisfy Marin's declared version even when runtime behavior works.
- An incomplete preserved-package list can create a mixed binary stack that imports successfully and fails later during compilation or collectives.
- JAX-Toolbox uses Ubuntu 24.04; the current task image uses Debian slim. Shared task tooling must avoid distribution-specific assumptions.
- A full #7507 control→treatment→control sequence occupies 64 GB200 GPUs for about 33 GPU-hours.
- Routing every GPU task to this image would break CUDA-PyTorch and vLLM jobs because the image intentionally preserves CPU-only Torch for JAX kernel imports.

## Design

### GPU task image

Add an `iris-task-gpu` build target based on an immutable multi-architecture JAX-Toolbox digest. It contains no Marin, Levanter, Haliax, or experiment source. It adds only the generic task tools missing from JAX-Toolbox: uv, Node, Rust, ffmpeg, Git/SSH, and any required Iris profiling/debug binaries. JAX-Toolbox already owns CUDA, NCCL, HPC-X, and Nsight, so the GPU target does not install Iris's standalone Nsight package.

The image declares its Python ownership boundary in `/etc/iris/preserved-python-packages`. The file contains one PEP 503-normalized distribution name per line; blank lines and `#` comments are allowed. `/etc/iris/required-python-packages` lists the subset that must be installed in the image. Names present only in the preserved file are explicit exclusions that must remain absent from the overlay. Extras, version specifiers, URLs, duplicates after normalization, invalid names, and an empty effective list fail setup. The Dockerfile owns the complete initial name list, covering JAX/JAXLIB/PJRT/plugins, TransformerEngine, CUTLASS DSL, cuda-tile, matching CPU-only Torch/Torchvision, `cuda-toolkit`, and CUDA/cuDNN/NCCL wheel variants.

The image also declares `/etc/iris/system-cuda-toolchain`. This marker tells the default GPU setup not to stage CUDA executables or reinstall cuDNN from venv wheels. A preserved-package file without the CUDA marker is valid for non-CUDA system packages. The CUDA marker without both preserved and required manifests is invalid, because skipping staging while allowing CUDA wheels into the overlay creates a mixed runtime.

### Runtime overlay

`default_setup_script` detects the image files at task runtime. Setup scripts are rendered by the client, so detection must remain shell logic in the generated script. When the preserved-package file exists, setup:

1. validates both manifests and requires every name in the required subset to resolve through the image interpreter;
2. fails if the requested Python major/minor differs from the image interpreter;
3. creates `$IRIS_VENV` with `uv venv --system-site-packages`;
4. builds a Bash array containing one `--no-install-package` argument per distribution;
5. runs the existing frozen workspace sync with those arguments; and
6. verifies with `importlib.metadata` that required distributions resolve outside `$IRIS_VENV`, and that explicit exclusions are absent from it.

The audit resolves each distribution root with `distribution(name).locate_file("").resolve()` and runs import-origin probes for JAX, its CUDA plugin/PJRT, TransformerEngine, CUTLASS, cuda-tile, Torch, and Torchvision. The task log prints the image digest, package versions, and paths. The result intentionally differs from `uv.lock`; later `uv sync` or `pip install` commands that replace protected packages are unsupported. Normal workspace distributions remain editable in `$IRIS_VENV`. Custom setup scripts keep their current contract.

### Image selection and entrypoints

Phase one selects `iris-task-gpu` through the existing explicit `RunTaskRequest.task_image` field. It does not change the cluster-wide Kubernetes default: CUDA-PyTorch, vLLM, CPU, and TPU tasks keep their current images. Promoting JAX-Toolbox to a resource-derived default requires an inventory and migration rule for non-JAX GPU jobs and is outside this draft.

Kubernetes passes `args: ["bash", "-lc", <generated script>]` and omits `command`, preserving an OCI entrypoint. Compatible entrypoints must ultimately `exec` their arguments and preserve exit codes and signals. Tests cover the current image with no entrypoint and NVIDIA's entrypoint. Iris does not hardcode `/opt/nvidia/nvidia_entrypoint.sh`.

### Versioning and rollout

CI builds native ARM64 and AMD64 `iris-task-gpu` manifests and tags them `git-<short-sha>`. The Dockerfile pins JAX-Toolbox by digest. Both architecture manifests must contain the expected Python ABI and required distributions. Updating the base digest requires the overlay import test, kernel ladder, and #7507 performance gate. Rollout begins with explicit canary jobs pinned by digest; `latest` is not a benchmark or rollout input.

## Testing

Unit tests cover both manifest parsers, PEP 503 normalization, repeated uv exclusion arguments, Python mismatch, duplicate and missing required distributions, post-sync location checks, the marker combination matrix, and OCI entrypoint exit/signal propagation.

The live ladder runs before the 64-GPU comparison:

1. normal Iris workspace sync in `iris-task-gpu`, followed by package-version and import-location checks;
2. Sonic-CuTe and FA4 forward/backward on one `GB200x4` node;
3. two-node distributed initialization and NCCL transport inspection; and
4. #7507 full-MoE control→treatment→control runs using the same benchmark patch derived from `31b221e1`, with immutable images and only `SCALE_TASK_IMAGE` differing; and
5. the same L48 treatment with `SCALE_SCAN_LAYERS=0` and `SCALE_MAX_RETRIES_FAILURE=0`.

The primary metric is median step time and reported `throughput/mfu` over steps 5–11 and 15–19, excluding warmup steps 0–4 and profiler steps 12–14. Compare treatment with the mean of the two control medians; control drift must be below 1%. Adoption requires correct completion and no more than 2% throughput regression. This is an as-shipped stack comparison across JAX/XLA, CUDA, NCCL, CUTLASS, and the base OS.

The no-scan probe classifies planned XLA temporary-arena OOM, null-module `cuModuleLoadData` CUBIN failure, another compiler/runtime failure, or success separately. One success shows that exact workload completed; it does not establish that all CUBIN failures are fixed. Reaching the historical CUBIN failure surface requires repeated treatment runs before claiming a change in failure rate.

## Open Questions

- Which non-JAX GPU workloads must pass before `iris-task-gpu` can become a cluster default instead of an explicit task image?
- Should a later worker/Docker image profile use the same manifests, or keep its existing per-scale-group image contract?
- Should Iris prevent later package mutation, or is a provenance warning sufficient after the initial audited sync?
