# Grug MoE MFU benchmark (standalone)

`grug_moe_mfu.py` is a self-contained MFU (Model FLOPs Utilization) benchmark for the
grug MoE model from [issue #7012](https://github.com/marin-community/marin/issues/7012)
(row-13 of #6979). It builds the model, runs a fixed number of train steps on
**deterministic synthetic tokens**, and reports steady-state MFU on B200 GPUs.

It imports only the `levanter.grug` kernels and `levanter.optim` — the model, optimizer
(MuonH/AdamH), and train step are inlined, so there is no dependency on the marin
pipeline, data loaders, checkpointing, or HF conversion.

## Results

Measured on 8×B200 (`gpu_fa4_cute` attention, batch 128, seq 4096, 26 layers, 64 experts
top-4 + shared expert):

| config | `--hidden-dim` | `--moe-implementation` | `--expert-parallelism` | MFU (B200) |
|---|---|---|---|---|
| row-13 | 2560 | `sonic` (XLA ragged-dot) | 1 | ~12.5% |
| d5120 variant | 5120 | `sonic_cute` (QuACK SM100 CuTeDSL) | 1 | **~17.9%** |
| row-13 | 2560 | `sonic_cute` | 1 | ~15.1% |
| row-13, EP | 2560 | `ring` | 2 / 4 / 8 | ~13.3 / 13.4 / 12.9% |
| row-13, EP | 2560 | `ragged_all_to_all` | 2 / 4 / 8 | ~1.0 / 1.8 / 3.2% |

The d5120/`sonic_cute` number was validated clean-room (fresh clone + fresh venv) at
**17.88% MFU** — 3218 TFLOP/s, 55.9k tokens/s, 9.39 s/step. All row-13 numbers were
measured sequentially on one 8×B200 node with this script's methodology (steady-state
median over steps 8–19).

## Requirements

This benchmark needs the **grug-Blackwell levanter stack**. This branch is current
`main` plus the `sonic_cute` QuACK grouped-GEMM backend (not yet upstreamed); the FA4
CuTeDSL attention backend and the expert-parallel MoE backends are already on `main`.
Run it from this branch — a plain `main` checkout lacks `sonic_cute`.

- **GPU:** NVIDIA SM100 (B200). `sonic_cute` and `gpu_fa4_cute` are SM100 kernels.
- **Environment:**
  ```bash
  uv sync --frozen --all-packages --extra=gpu
  uv pip install "quack-kernels==0.5.0"   # QuACK SM100 kernels — not in any lockfile
  ```
  `nvidia-cutlass-dsl==4.5.2` comes in via the `gpu` extra.

## Running

```bash
# row-13 (d2560, sonic) — ~12.5% MFU
python grug_moe_mfu.py --run-id row13 --output-dir runs/row13

# d5120 / sonic_cute — ~17.9% MFU
python grug_moe_mfu.py --run-id d5120 --output-dir runs/d5120 \
  --hidden-dim 5120 --moe-implementation sonic_cute --num-gpus 8

# row-13 with expert parallelism (ring backend) — ~13% MFU
python grug_moe_mfu.py --run-id row13-ep8 --output-dir runs/row13-ep8 \
  --moe-implementation ring --expert-parallelism 8 --num-gpus 8
```

Compilation dominates the first run (~10–15 min cold for d5120/`sonic_cute`). Set a
persistent XLA cache to amortize it across runs:

```bash
export JAX_COMPILATION_CACHE_DIR=$PWD/jaxcache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
```

### Flags

| flag | default | meaning |
|---|---|---|
| `--run-id` | *required* | run label |
| `--output-dir` | *required* | where `metrics_summary.json` is written |
| `--hidden-dim` | 2560 | model width (2560 = row-13, 5120 = Will's variant) |
| `--moe-implementation` | `sonic` | local: `sonic` (XLA ragged-dot), `sonic_cute` (QuACK SM100), `scatter`; expert-parallel: `ring`, `ragged_all_to_all` |
| `--expert-parallelism` | 1 | expert mesh axis size; >1 requires an expert-parallel `--moe-implementation` |
| `--attention-implementation` | `gpu_fa4_cute` | FlashAttention-4 CuTeDSL backend |
| `--num-gpus` | 8 | GPUs to shard across |
| `--num-layers` | 26 | decoder layers |
| `--num-experts` / `--num-experts-per-token` | 64 / 4 | MoE experts, top-k |
| `--batch-size` / `--seq-len` | 128 / 4096 | global batch, sequence length |
| `--steps` / `--warmup-steps` | 20 / 8 | total steps, warmup excluded from the median |

## What it measures

MFU is the honest, active-expert accounting:

```
flops_per_token   = lm_flops_per_token(..., num_experts_per_tok=num_experts_per_token)  # active experts, not all 64
flops_per_example = 3 * flops_per_token * seq_len                                        # ×3 for fwd + bwd
mfu_b200          = (flops_per_example * examples_per_second) / (num_gpus * 2.25e15)     # 2.25e15 = dense B200 bf16 peak/GPU
```

Each step prints a JSON line (`mfu_b200`, `tokens_per_second`, `duration`, `loss`, …);
the final `SUMMARY {...}` line and `metrics_summary.json` report the **steady-state
median** over the post-warmup steps.

## Data

No real data. Tokens are generated deterministically so results are reproducible and
data-independent:

```python
tokens[i, t] = (t + (step * global_batch + i) * 9973) % vocab_size
```

## Model

26-layer decoder transformer: `gpu_fa4_cute` FlashAttention-4 (half-RoPE, NoPE on long
layers, gated norm), 64 experts with top-4 routing plus a shared expert, QB-bias router,
sharded EP1 + FSDP across the GPUs, MuonH optimizer. See the inlined config in
`grug_moe_mfu.py` for the exact hyperparameters.

## Expert parallelism

`--expert-parallelism N` sets the `expert` axis of the device mesh to N (the `data`
axis absorbs the rest, so tokens still shard over all GPUs). Expert weights and their
optimizer state shard over the expert axis; the router runs data-parallel and the
routed-expert MLP dispatches tokens to the shards that own their experts.

Two structural notes for reviewers:

- The QuACK grouped GEMM (`sonic_cute`) is currently a **local-only** backend: with
  `--expert-parallelism > 1` the model requires an expert-parallel backend (`ring` or
  `ragged_all_to_all`), and those run their local grouped GEMMs through a Pallas-Triton
  kernel instead (measured 397–859 TF/s at these shapes vs 1,470–1,560 TF/s for tuned
  QuACK). That GEMM gap plus dispatch overhead accounts for most of the EP tax
  (15.1% → ~13%).
- **`ring` is the backend that works** (all-gather dispatch + psum-scatter combine over
  NVLink). `ragged_all_to_all` is currently pathological on GPU: XLA lowers
  `jax.lax.ragged_all_to_all` to `stream_executor::gpu::RaggedAllToAllKernelImpl`
  (a peer-copy kernel bracketed by multi-GPU barriers) rather than NCCL, and that one
  kernel is ~90% of device time (~297 ms/call, 104 calls/step at 26 layers) — a
  ~15× step-time regression. It is included for completeness and as a reproducer of
  the upstream XLA issue.
