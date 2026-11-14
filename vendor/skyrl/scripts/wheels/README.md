# Native CUDA wheels

`build_native.sh` builds native extensions from exact upstream commits. It uses
ordinary source builds and does not rewrite wheel contents after compilation.
The [SkyRL lockfile](../../uv.lock) records their URLs and SHA-256 hashes.
The [TE 2.21 native release](https://github.com/marin-community/marin/releases/tag/skyrl-native-te221-cu132torch2141-20261113-0821e3af)
contains the qualified wheels and their original build records.

| Target | Source | Torch / CUDA toolkit | Linux architectures |
| --- | --- | --- | --- |
| FlashAttention 2.8.4 | `f87af617fab71c29c11e7f9bc74746ae7534a59d` | 2.14.1+cu132 / 13.2.2 | x86_64 SM90, aarch64 SM100 |
| Transformer Engine Torch 2.21.0 | `7457e47d26d9d02dfb783007335dfbe75e1b90a8` | 2.14.1+cu132 / 13.2.2 | x86_64, aarch64 |
| causal-conv1d and Mamba | Commits selected in the script | 2.13.0+cu132 / 13.2.1 | x86_64 |

TE's Python package and GPU library use NVIDIA's official wheels. NVIDIA does
not publish a CUDA 13 ARM wheel for TE 2.21's PyTorch binding. The
[published CUDA 13 x86 binding](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.21.0)
targets NVIDIA's PyTorch container release 26.08 and has not been qualified with
our Torch 2.14.1/CUDA 13.2 stack. This recipe builds the binding from unchanged
source for that stack on both architectures.
[Upstream issue #3310](https://github.com/NVIDIA/TransformerEngine/issues/3310)
and [PR #3311](https://github.com/NVIDIA/TransformerEngine/pull/3311) track
expanding CUDA 13 wheel coverage; that PR currently targets Torch 2.11.

[native-cu132-torch214.txt](native-cu132-torch214.txt) pins the packages used to
compile the TE binding, including NVCC/CCCL 13.2.86. RL jobs install the published
wheel through `uv.lock`; they do not use this file or compile TE. A qualified
official binding wheel could replace our build for each architecture.

The FA2 build also uses this requirements file. FA2's upstream source selects
C++20 for Torch 2.14 and excludes its optional Cute packages. Its supported
`FLASH_ATTN_LOCAL_VERSION` option adds the Marin build label. Core and Bridge
remain stock PyPI 0.18.0 and 0.6.0.

TE 2.21 accepts FA2 2.8.4. TE 2.19 requires FA2 below 2.8.4, so it cannot consume
this FA2 build without a different source selection. Do not change a package's
reported version to bypass that check.

Use Python 3.12.14, uv 0.13.5, git and GCC/G++ 14.2.0 on Linux. The CPU-only
builds use two cores, 32 GiB memory and 96 GiB disk, with the Iris task image
`ghcr.io/marin-community/iris-task@sha256:28a807a676b0b0ae155a80b1c6de75ae1b2932e25cef9bb8462b9ed7582c6990`
(Debian 13, glibc 2.41). The pinned Python packages supply the CUDA compiler;
a GPU is not needed to build. These wheels target that environment and do not
claim general manylinux portability.

From `vendor/skyrl/`, run:

```bash
bash scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
bash scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te
```

Use a separate build directory for each source or environment. The source and
recursive submodules must be clean; ignored compiler outputs may remain for
cache reuse. Each directory retains its build environment and source checkout.
Wheels go to `BUILD_DIRECTORY/dist/`; `SHA256SUMS`, `BUILD_INFO` and
`BUILD_REQUIREMENTS.txt` stay in `BUILD_DIRECTORY/` and
record their hashes, source, toolchain and actual build dependencies. Independent
builds need not produce identical bytes. Publish those records and the image
digest alongside each wheel.

Before adoption, install the exact wheel bytes and check attention values and
gradients, packed GQA/masks, and
[`test_megatron_flash_attention_cp2_forward_backward`](../../skyrl-train/tests/gpu/test_megatron_worker.py)
on H100 and GB200. Also check Grug train/eval agreement, a nonzero trainer
update, native save/restore, and publication to vLLM. Run the peer-batch check
with `trainer.algorithm.batch_invariant=true`. Native
attention tests alone cannot detect TE's FA2 version cap. Update runtime URLs
and hashes only after these checks pass.

The causal-conv1d and Mamba targets retain their historical Torch 2.13 build
environment in [native-cu132.txt](native-cu132.txt). They cannot rebuild the
Torch 2.14 causal-conv1d and Mamba wheels. Those two packages use the
[frozen MarinSkyRL recipes](https://github.com/marin-community/MarinSkyRL/tree/604f2c1463032ffc1f47da114f43cf1d550dd473/scripts/wheels),
including C++20 build flags and Mamba's TileLang/TVM-FFI dependency adjustments.
Core and Bridge patches in that archive are unused.
