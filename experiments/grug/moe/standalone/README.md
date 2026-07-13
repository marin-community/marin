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

| config | `--hidden-dim` | `--moe-implementation` | MFU (B200) |
|---|---|---|---|
| row-13 | 2560 | `sonic` (XLA ragged-dot) | ~12.5% |
| d5120 variant | 5120 | `sonic_cute` (QuACK SM100 CuTeDSL) | **~17.9%** |

The d5120/`sonic_cute` number was validated clean-room (fresh clone + fresh venv) at
**17.88% MFU** — 3218 TFLOP/s, 55.9k tokens/s, 9.39 s/step.

## Requirements

This benchmark needs the **grug-Blackwell levanter stack** (the `sonic_cute` QuACK kernel
and the FA4 CuTeDSL attention backend), which lives on the `sonic-cute-moe-b200` branch —
it is **not on `main`**. Run it from a branch based on `sonic-cute-moe-b200` (e.g.
`rav-moe-repro-v1`); a `main`-based checkout will fail to import the kernels.

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
| `--moe-implementation` | `sonic` | `sonic` (XLA ragged-dot) or `sonic_cute` (QuACK SM100) |
| `--attention-implementation` | `gpu_fa4_cute` | FlashAttention-4 CuTeDSL backend |
| `--num-gpus` | 8 | GPUs to shard across (EP1 + FSDP) |
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
