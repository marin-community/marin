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

JAX interoperability probes:

```bash
# Local pointer identity, visibility, and endpoint acceptance.
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.jax_interop

# Two-PE JAX source to remote symmetric inbox to JAX consumer.
NVTP_JAX_REMOTE=1 uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.jax_interop
```

All remote endpoints are offsets within a collectively allocated symmetric
arena. Peer tensor aliases are used only for direct CuTe loads and stores; RMA
operations retain the original symmetric address plus the remote PE.

The final result is documented in [report.md](report.md). The prototype is not
recommended for production: warp push leads the transport microbenchmarks, but
the JAX/XLA stream path does not compile and concurrent communication reduces
GEMM throughput by more than half.

Additional correctness and performance commands:

```bash
# Pair, ring, or all-to-all; push or pull.
NVTP_NUM_PES=8 NVTP_DIRECTION=push NVTP_PATTERN=all_to_all \
  uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.correctness_patterns

# Select individual transport variants.
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.benchmark_transport \
  --push-operations put_signal_warp peer_store_warp_signal \
  --pull-operations blocking_warp peer_load_warp nbi_batched_quiet

# Start-gated communication/GEMM overlap.
uv run --package marin-levanter --extra nvshmem-transport \
  python -m cute_nvshmem_transport.benchmark_overlap \
  --protocol push --operation put_signal_nbi_warp_quiet
```
