# Routed sparse attention H100 oracle smoke

Date: 2026-08-07

This artifact records a bounded executable-oracle bring-up for the first Shuttle
routed sparse-attention experiment. The primary MIT Block-Sparse-Attention
oracle could not be compiled in the reserved Iris task image because the image
does not contain a CUDA toolkit or the `nvcc` driver. A pinned standalone
SeerAttention Triton kernel provided the query-major JIT fallback.

## Hardware and software

- GPU: NVIDIA H100 80GB HBM3, one GPU used on an 8-GPU holder node.
- Driver: 595.71.05.
- Maximum SM clock: 1980 MHz.
- Power limit: 700 W.
- Python: 3.12.13.
- PyTorch: 2.7.1+cu128.
- PyTorch CUDA runtime: 12.8.
- Triton: 3.3.1.
- SeerAttention: `aba03e3f2caefd0ccd21e576670aa830b748c84e`.
- Block-Sparse-Attention: `49d6c39e4dc0303442cda3bb758b3925d4399c49`.
- Block-Sparse-Attention CUTLASS: `a75b4ac483166189a45290783cb0a18af5ff0ea5`.

The exact per-run clock telemetry is embedded in the JSON files. It was sampled
before tensor allocation, so the 2K record captures an idle 345 MHz clock and
the 16K record captures 1830 MHz. CUDA event timings were collected after
warm-up.

## Workload and adapter

Both runs use BF16 causal GQA with `Hq=32`, `Hkv=8`, `D=128`, query and KV
blocks of 128, and at most eight selected historical blocks per query block.
The relation is deterministic, block-shared, and always includes the current
block. The 16K relation has 996 edges and SHA-256
`b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`.

Seer's kernel does not implement GQA indexing. The adapter repeats K and V from
8 to 32 heads once outside the timed kernel. At 16K this adds 201,326,592 bytes;
the recorded cold expansion took 52.05 ms and is excluded from steady-state
kernel timing. The compiler workload and relation remain GQA/block-shared.

The tested kernel entry point is:

```python
block_sparse_triton_fn(
    q,
    expanded_key,
    expanded_value,
    mask,
    1.0 / math.sqrt(128),
    BLOCK_M=128,
    BLOCK_N=128,
    layout="bhsd",
)
```

`mask` has shape `[1, 32, S/128, S/128]` and is expanded from the prerecorded
block-shared relation only at this external adapter boundary.

## Results

At 2K, the pinned kernel matches an independent source-ordered FP32 softmax
reference with maximum absolute error 0.0078125, mean absolute error
0.00008281, and p99 absolute error 0.0009766. The 30-sample steady-state median
is 0.316752 ms. Output SHA-256 is
`d5a37270582c15d89c0fa543caf5d3af281fba9e5055ff23bc3c789903c805f2`.

At 16K, the 50-sample median is 2.388208 ms (2.384032–2.401760 ms), or 111.95
TFLOP/s over the selected block QK/PV work. Output SHA-256 is
`91972fce5061fde100dd022584692b6fc356e5e3e8fda0b06e77936af1445555`.
PyTorch causal GQA SDPA on the same logical Q/K/V has a 50-sample median of
6.282496 ms (6.210304–6.357888 ms). Neither benchmark script materializes the
dense score tensor.

The JSON files contain every raw latency sample, relation-plan time, JIT/first
run time, hashes, and correctness metrics.

## Structural limitation

Seer's kernel is an executable query-major sparse baseline, not the intended
Block-Sparse-Attention or FlashMoBA oracle. It loops over every causal KV block
and checks a dense binary mask inside that loop. Its metadata traversal is
therefore proportional to the dense causal block count rather than the number
of selected edges. It also requires the GQA expansion described above.

## Primary-oracle negative result

The holder image is Debian 13 with an NVIDIA driver but no CUDA toolkit.
Installing PyTorch's CUDA 12.8 wheels and `nvidia-cuda-nvcc-cu12==12.8.61`
provided runtime libraries, headers, and `ptxas`, but not the `nvcc` driver.
The pinned Block-Sparse-Attention build stops before compilation with:

```text
UserWarning: block_sparse_attn was requested, but nvcc was not found.
OSError: CUDA_HOME environment variable is not set.
```

The complete traceback is in `block_sparse_attention_build.log`. Repairing the
holder into a CUDA devel image was intentionally out of scope for this smoke.

## Reproduction commands

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git block_sparse_attention
git -C block_sparse_attention checkout 49d6c39e4dc0303442cda3bb758b3925d4399c49
git -C block_sparse_attention submodule update --init --recursive
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install packaging ninja psutil numpy nvidia-cuda-nvcc-cu12==12.8.61
python block_sparse_attention/setup.py build_ext --inplace

git clone https://github.com/microsoft/SeerAttention.git seer_attention
git -C seer_attention checkout aba03e3f2caefd0ccd21e576670aa830b748c84e
cp benchmark_seer_query_major.py seer_attention/benchmark_shuttle.py
cd seer_attention
CUDA_VISIBLE_DEVICES=0 python benchmark_shuttle.py --sequence-length 2048 --correctness --warmups 10 --repeats 30
CUDA_VISIBLE_DEVICES=0 python benchmark_shuttle.py --sequence-length 16384 --warmups 20 --repeats 50
```
