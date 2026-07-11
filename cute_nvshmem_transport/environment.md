# Environment

The validated Stage 1 environment is one CoreWeave `h100-8x` node in
`cw-us-east-02a`. `nvidia-smi topo -m` reported `NV18` between every pair of
H100 80GB GPUs.

Install and validate the isolated transport environment:

```bash
uv sync --package marin-levanter --extra nvshmem-transport --no-dev
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.environment
```

Validated versions are Python 3.12.13, CUDA Python 13.2.0, cuda-core 1.1.0,
CUTLASS DSL 4.4.2, NVSHMEM 3.7.0, and NVSHMEM4Py 0.3.1.

CUTLASS DSL 4.5.2 is deliberately not used in this extra. NVSHMEM4Py 0.3.1
imports `cutlass.cute.typing.Constexpr`, which was removed in CUTLASS DSL 4.5.
The normal Marin GPU extra remains on 4.5.2 for FA4. The transport extra also
conflicts with the normal GPU and Torch-test extras because the CUDA 12.8 Torch
wheel requires `cuda-bindings<13`, while NVSHMEM4Py-CUDA13 requires CUDA
Python 13.

Run the peer-addressing probe with:

```bash
NVTP_NUM_PES=8 uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.launch
```
