# CuTe NVSHMEM Transport Research Harness

This directory contains the reproducible harness for Marin experiment
[#7114](https://github.com/marin-community/marin/issues/7114). It compares
NVSHMEM push, NVSHMEM pull, direct peer-tensor stores, and direct peer-tensor
loads on CoreWeave H100s. It is intentionally separate from the production MoE
kernel while transport and JAX interoperability remain under evaluation.

Install the isolated transport environment on Linux:

```bash
uv sync --package marin-levanter --extra nvshmem-transport
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.environment
```

The NVSHMEM transport extra conflicts with the normal GPU and Torch-test extras
because the CUDA 12.8 Torch wheel requires `cuda-bindings<13`, while
NVSHMEM4Py-CUDA13 requires CUDA Python 13.

All remote endpoints are offsets within a collectively allocated symmetric
arena. Peer tensor aliases are used only for direct CuTe loads and stores; RMA
operations retain the original symmetric address plus the remote PE.
