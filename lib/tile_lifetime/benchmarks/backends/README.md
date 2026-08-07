# GB200 physical-runtime backends

The four-rank physical-runtime benchmark uses the official DeepEP intranode
dispatch and combine implementation. It does not vendor DeepEP. The verified
source revision is:

```text
DeepEP 7febc6e25660af0f54d95dd781ecdcd62265ecca
```

## Build the intranode Torch extension

The pinned checkout builds two extensions by default. The second one requires
the HybridEP/NVSHMEM development environment, which is unnecessary for the
single-tray benchmark. The adjacent `prepare_deepep_intranode_torch.py` helper
creates a detached worktree and changes only Python packaging metadata so the
official intranode `deep_ep_cpp` sources can be built independently. It also
adds the pip CUDA toolkit's CCCL include directory. No DeepEP C++ or CUDA source
is changed.

Starting from the Torch 2.10.0+cu130 environment used for the MoK oracle:

```bash
export DEEPEP_SOURCE_ROOT=/tmp/DeepEP
export DEEPEP_PATCHED_ROOT=/tmp/DeepEP-torch-intranode
export DEEPEP_BUILD_ROOT=/tmp/deepep-torch-intranode-build
export CUDA_HOME=/tmp/mok-route-env/lib/python3.12/site-packages/nvidia/cu13
export CUDA_CCCL_INCLUDE="$CUDA_HOME/include/cccl"
export TORCH_CUDA_ARCH_LIST=10.0a
export DEEPEP_DISABLE_NVSHMEM=1
export DEEPEP_BUILD_INTRANODE_ONLY=1
export MAX_JOBS=8
export PATH="/tmp/mok-route-env/bin:$CUDA_HOME/bin:$PATH"

/tmp/mok-route-env/bin/python \
  /app/lib/tile_lifetime/benchmarks/backends/prepare_deepep_intranode_torch.py \
  --source-root "$DEEPEP_SOURCE_ROOT" \
  --output-root "$DEEPEP_PATCHED_ROOT"

mkdir -p /tmp/deepep-torch-link
ln -sf "$CUDA_HOME/lib/libcudart.so.13" /tmp/deepep-torch-link/libcudart.so
ln -sf "$CUDA_HOME/lib/libnvtx3interop.so.1" /tmp/deepep-torch-link/libnvtx3interop.so
export LIBRARY_PATH="/tmp/deepep-torch-link:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/tmp/deepep-torch-link:$CUDA_HOME/lib:/tmp/mok-route-env/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

cd "$DEEPEP_PATCHED_ROOT"
/tmp/mok-route-env/bin/python setup.py build_ext \
  --build-lib "$DEEPEP_BUILD_ROOT/lib" \
  --build-temp "$DEEPEP_BUILD_ROOT/temp"
```

Load the extension and the minimally patched Python package with:

```bash
export PYTHONPATH="$DEEPEP_BUILD_ROOT/lib:$DEEPEP_PATCHED_ROOT:${PYTHONPATH:-}"
/tmp/mok-route-env/bin/python -c \
  'import deep_ep, deep_ep_cpp; print(deep_ep.Buffer, deep_ep_cpp.Config)'
```

The verified build used NVCC 13.0.88, CCCL 13.0.85, CUDA CRT 13.0.88, and
NVVM 13.0.88. The extension smoke test used four ranks and exercised
`Buffer.get_dispatch_layout`, `Buffer.dispatch`, and `Buffer.combine` with an
exact output check and explicit buffer destruction.
