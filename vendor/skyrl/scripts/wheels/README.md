# CUDA 13.2 native wheels

`build_native.sh` builds the existing FlashAttention, causal-conv1d, Mamba, and
Transformer Engine Torch releases against Torch 2.13.0+cu132. It checks each
upstream source commit and builds locally instead of downloading an upstream
wheel compiled for a different Torch version. `native-cu132.txt` pins the build
environment, including CUDA 13.2.1's NVCC and CCCL packages.

This checks in and tightens the build process first published with the
[original CUDA 13.2 wheels](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-torch2.13-20260910-6d12d7a0).
Reproducible here means that the source, dependencies, and build commands are
pinned. Independent builds are not guaranteed to produce identical bytes, so
runtime dependencies pin each published wheel by URL and SHA-256.

`build_native.sh` is a manual release tool. CI and runtime installation do not
invoke it; they consume the published, hash-pinned wheels.

The Transformer Engine target builds official 2.19 source on x86_64 and aarch64.
Published wheels used the [x86_64 recipe](https://github.com/marin-community/MarinSkyRL/blob/543a0ba8012604104f99c3113632aa8e6c3d9ad0/scripts/wheels/build_native.sh)
and [aarch64 recipe](https://github.com/marin-community/MarinSkyRL/blob/fa4-te219-cu132-20260920-694f3adf/scripts/wheels/build_native.sh).
Wheel URLs and hashes are in `uv.lock`.

Every source is fetched by exact commit. The pinned 2.8.3 FlashAttention wheel
was built with an unused package excluded; its
[release record](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
preserves that provenance. This script now builds unmodified upstream source,
so a new wheel needs its own qualification and release.

Use CPython 3.12.14 on Linux with git, a C++ compiler, and uv. The
qualified FlashAttention 2.8.3 build used the Iris task image
`ghcr.io/marin-community/iris-task@sha256:ecdb2f7f90f8760a7e74c49b49b67d7ecf44557298860411c148c186706067f2`,
GCC/G++ `14.2.0-19`, glibc `2.41-12+deb13u3`, git `1:2.47.3-0+deb13u1`, and
uv `0.10.3`. Install the compiler and git inside the build container. These are
Linux wheels for that task environment; they do not claim manylinux portability.

For CPU-only TE builds on either architecture, use the Iris task image
`ghcr.io/marin-community/iris-task@sha256:28a807a676b0b0ae155a80b1c6de75ae1b2932e25cef9bb8462b9ed7582c6990`.
Builds passed with two CPU cores, 16 GiB memory and 64 GiB disk, using the Python,
GCC and uv versions above and glibc `2.41-12+deb13u4`. Pinned packages supply
NVCC `13.2.78` and setuptools `80.10.2`.

Invoke through `bash` because Iris bundles do not retain executable modes:

```bash
bash scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
bash scripts/wheels/build_native.sh causal-conv1d /tmp/build-causal-conv1d
bash scripts/wheels/build_native.sh mamba-ssm /tmp/build-mamba
bash scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te
```

Each directory retains its environment and source build cache. Wheels are written
to `dist/`, with their SHA-256 digests in `SHA256SUMS`. `BUILD_INFO` records source,
toolchain and ABI; `BUILD_REQUIREMENTS.txt` records installed build packages.
Preserve the task image digest alongside these records. FlashAttention targets
SM90; other projects retain their upstream GPU architecture choices. Only
Transformer Engine supports aarch64 here.

The source checkout and recursive submodules must be clean. The script rejects
tracked changes and unexpected untracked files; Git-ignored build outputs remain
available for cache reuse.

Before publishing a wheel, install its exact bytes in the proposed runtime and
run its native forward and backward checks on each target architecture. Megatron also needs a
context-parallel forward and backward with the FlashAttention backend selected;
native attention checks alone cannot detect a Transformer Engine version cap.
Publish the source commits, build environment, and checksums with the wheel
assets. Adoption URLs and wheel hashes belong in the root dependency manifest
and lock.
