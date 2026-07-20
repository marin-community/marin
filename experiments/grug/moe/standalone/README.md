# Grug MoE MFU benchmark (standalone)

`grug_moe_mfu.py` is a self-contained MFU (Model FLOPs Utilization) benchmark for the
grug MoE model from [issue #7012](https://github.com/marin-community/marin/issues/7012)
(the "row-13" configuration of #6979 — spelled out in full below, no need to chase the
issues). It builds the model, runs a fixed number of train steps on **deterministic
synthetic tokens**, and reports steady-state MFU on B200 GPUs.

## The row-13 configuration

Everything below is the script's defaults — running with only `--run-id`/`--output-dir`
gives exactly this model. (`--hidden-dim 5120` is the only change behind the "d5120"
rows.)

| | |
|---|---|
| hidden dim | 2560 |
| layers | 26 (decoder-only, pre-norm RMSNorm + gated norm) |
| attention | MHA, 20 heads × head_dim 128 (no GQA), qk_mult 1.3 |
| positional | half-RoPE; sliding window 2048 with NoPE on the long (global) layers |
| MoE | every layer: 64 routed experts, top-4, + 1 always-on shared expert |
| expert MLP | SwiGLU, intermediate dim 1280 (= hidden/2); shared expert identical |
| router | linear (fp32 logits), sigmoid combine weights renormalized over top-k, QB bias (aux-loss-free load balancing, DeepSeek-v3-style) |
| vocab | 128,256 (Llama-3 size), tied nothing (separate embed + lm_head) |
| params | ≈18B total, ≈2.0B active per token (excl. embeddings) |
| batch | 128 sequences × 4096 tokens = 524,288 tokens/step |
| precision | params fp32, compute bf16 |
| loss | next-token CE + z-loss 1e-4 |
| optimizer | MuonH (Muon + AdamH hybrid; lr 1e-3, adam lr 1e-4, warmup 10%) |
| remat | full recompute (`recompute_all`) |
| analytic FLOPs | 5.68 GFLOP/token fwd (active-expert accounting), ×3 for fwd+bwd |
| sharding | FSDP over the `data` axis; `--expert-parallelism N` moves experts to an `expert` axis (see below) |

It imports only the `levanter.grug` kernels and `levanter.optim` — the model, optimizer
(MuonH/AdamH), and train step are inlined, so there is no dependency on the marin
pipeline, data loaders, checkpointing, or HF conversion. (One optional hook: when
launched as a multi-task iris gang job it imports `iris.runtime.jax_init` for
`jax.distributed` coordinator discovery; outside iris that path is never touched.)

## Results

Measured on 8×B200 (`gpu_fa4_cute` attention, batch 128, seq 4096, 26 layers, 64 experts
top-4 + shared expert):

| config | `--hidden-dim` | `--moe-implementation` | `--expert-parallelism` | MFU (B200) |
|---|---|---|---|---|
| row-13 | 2560 | `sonic` (XLA ragged-dot) | 1 | ~12.5% |
| d5120 variant | 5120 | `sonic_cute` (QuACK SM100 CuTeDSL) | 1 | **~17.9%** |
| row-13 | 2560 | `sonic_cute` | 1 | ~15.1% |
| row-13, EP | 2560 | `ring` | 2 / 4 / 8 | ~13.3 / 13.4 / 12.9% |
| row-13, EP | 2560 | `ring_cute` (ring dispatch + QuACK GEMMs) | 8 | **~13.9%** |
| row-13, EP | 2560 | `ragged_all_to_all` | 2 / 4 / 8 | ~1.0 / 1.8 / 3.2% |

The d5120/`sonic_cute` number was validated clean-room (fresh clone + fresh venv) at
**17.88% MFU** — 3218 TFLOP/s, 55.9k tokens/s, 9.39 s/step. All row-13 numbers were
measured sequentially on one 8×B200 node with this script's methodology (steady-state
median over steps 8–19).

The same ladder on a **4×GB200** node (per-GPU load is 2× the 8-GPU runs; MFU below is
still the script's B200-peak convention — multiply by 0.9 for GB200's 2.5 PFLOP/s peak;
per-GPU absolute throughput is +11% over B200):

| config (4×GB200) | `--moe-implementation` | EP | MFU (B200 conv.) | step |
|---|---|---|---|---|
| row-13 | `sonic_cute` | 1 | **17.0%** | 5.84 s |
| row-13, EP | `ring` | 2 / 4 | 14.9 / 14.9% | 6.66 s |
| row-13, EP | `ring_cute` | 2 / 4 | **16.8 / 16.3%** | 5.92 / 6.09 s |
| row-13, EP | `ragged_all_to_all` (XLA defaults) | 4 | 1.9% | 52.8 s |
| row-13, EP | `ragged_all_to_all` + NCCL-path flag | 4 | 14.4–14.5% | 6.9 s |
| row-13, EP | `ragged_all_to_all_cute` + NCCL-path flag | 4 | **15.8%** | 6.28 s |

"NCCL-path flag" = `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`
(see the expert-parallelism notes below). EP arms use the script's default
`--capacity-factor 1.0`; raising it costs ≈0.7pp per +0.25 (padding) and buys drop
headroom.

**32 GPUs** (8 nodes × 4 GB200, one gang / one NVLink domain, multi-node entry
point below; sharding spans all 32 GPUs — no replica axis — so per-GPU batch is
4 sequences, the memory-realistic production operating point):

| config (32 GPUs) | `--moe-implementation` | EP | MFU (B200 conv.) | step |
|---|---|---|---|---|
| row-13 | `sonic_cute` | 1 | 13.2% | 0.94 s |
| row-13, EP | `ring_cute` | 4 / 8 / 16 / 32 | **14.3** / 13.5 / 11.8 / 9.5% | 0.87–1.30 s |
| row-13, EP | `ring` | 8 | 12.8% | 0.97 s |
| row-13, EP | `ragged_all_to_all_cute` + NCCL flag | 4 / 8 / 16 / 32 | 12.1 / 12.2 / 12.3 / **12.3%** | ~1.01 s |
| row-13, EP | `ragged_all_to_all` + NCCL flag | 8 | 11.7% | 1.06 s |
| d5120 | `sonic_cute` | 1 | 13.9% | 3.02 s |
| d5120, EP | `ring_cute` | 8 | **16.2%** | 2.60 s |

Two regime changes vs the single-node numbers: at 32-way sharding **EP is a net
win** (ring_cute EP4 beats the EP1 control by +1.2pp; +2.3pp at d5120/EP8) because
dispatching tokens is cheaper than FSDP-all-gathering expert weights every layer,
and **`ragged_all_to_all_cute` is flat in EP degree** (dispatch volume is
independent of EP size), crossing above `ring_cute` at EP16.

The 32-way memory headroom also cashes the remat tax (`--remat-mode`, with the
slim `_expert_mlp` residuals): at d2560, `none` fits and is the best number —
`ring_cute` EP4 **15.0%** (vs 14.8% `all_but_moe`, 14.3% `recompute_all`); at
d5120, `none` OOMs and `all_but_moe` gives **16.9%** (`ring_cute` EP8, vs 16.2%
`recompute_all`). The slim residuals alone cost ~0.3pp under `recompute_all`
(backward re-gathers with no memory dividend), so pair them with
`all_but_moe`/`none`. All EP arms above use the default `recompute_all` unless
stated.

At the 64-GPU reference config (d5120, 48 layers, batch 1024, seq 4096;
`--replica-axis-size 2` so each 32-GPU replica group holds one FSDP+EP model
copy, 16 seq/GPU): `sonic_cute` EP1 20.2%, **`ring_cute` EP4 20.8%** (best),
`ragged_all_to_all_cute` EP8 19.1%, `ragged_all_to_all` EP8 18.8%. EP's margin
over EP1 compresses to +0.6pp — the larger per-GPU batch amortizes the FSDP
weight all-gathers. `all_but_moe`/`none` OOM at this scale (439 GiB–1.6 TiB step
temporaries); `recompute_all` is the only viable remat mode. Known limit: **EP8
is the ceiling at batch 1024 — every backend fails above it.** Ring backends
fail from EP8 up with `Failed to load in-memory CUBIN …
CUDA_ERROR_INVALID_VALUE` (the EP-group token all-gather, `x_global` in
`ep_ring.py`, crossing 2³¹ elements). Both a2a backends hit the same CUBIN
failure at EP16 — including pure-XLA `ragged_all_to_all`, so it is upstream XLA
codegen, not the CuTe adapter — and at EP32 (data axis = 1) OOM instead on a
~104 GiB temporary from the SPMD partitioner's replicate-then-repartition
fallback ("Involuntary full rematerialization", XLA b/433785288); EP64 at
`--replica-axis-size 1` fails likewise. The allocator choice
(`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` vs default `bfc`) is an exact tie on
this benchmark at the reference config.

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

### Issue #7279 single-rack matrix

These arguments reproduce the fastest 64-GB200 architecture from
[#7201](https://github.com/marin-community/marin/issues/7201#issuecomment-5016389653):

```bash
reference_args=(
  --steps 30 --warmup-steps 8
  --batch-size 1024 --seq-len 4096
  --hidden-dim 6144 --num-layers 48
  --num-experts 128 --num-experts-per-token 4
  --num-heads 48 --num-kv-heads 8 --head-dim 128
  --shared-expert-intermediate-dim 6144
  --sliding-window 512 --global-every 6
  --attention-implementation gpu_fa4_cute
  --num-gpus 64 --replica-axis-size 2
  --capacity-factor 1.0 --remat-mode recompute_all
)

baseline_args=("${reference_args[@]}" --expert-intermediate-dim 3072)
latent_args=(
  "${reference_args[@]}"
  --expert-intermediate-dim 6144
  --moe-latent-dim 3072 --moe-latent-norm
)
```

Run each arm in a fresh 16-node Iris gang job. The EP1 controls isolate local
projection and GEMM-shape costs:

```bash
# B200LMOE-001-A / 002-A: EP1 controls
python grug_moe_mfu.py "${baseline_args[@]}" \
  --run-id b200lmoe-001-a --output-dir runs/b200lmoe-001-a \
  --moe-implementation sonic_cute --expert-parallelism 1
python grug_moe_mfu.py "${latent_args[@]}" \
  --run-id b200lmoe-002-a --output-dir runs/b200lmoe-002-a \
  --moe-implementation sonic_cute --expert-parallelism 1

# B200LMOE-001-B / 002-B: ring_cute EP4
python grug_moe_mfu.py "${baseline_args[@]}" \
  --run-id b200lmoe-001-b --output-dir runs/b200lmoe-001-b \
  --moe-implementation ring_cute --expert-parallelism 4
python grug_moe_mfu.py "${latent_args[@]}" \
  --run-id b200lmoe-002-b --output-dir runs/b200lmoe-002-b \
  --moe-implementation ring_cute --expert-parallelism 4

# B200LMOE-001-C / 002-C: ring_cute EP8
python grug_moe_mfu.py "${baseline_args[@]}" \
  --run-id b200lmoe-001-c --output-dir runs/b200lmoe-001-c \
  --moe-implementation ring_cute --expert-parallelism 8
python grug_moe_mfu.py "${latent_args[@]}" \
  --run-id b200lmoe-002-c --output-dir runs/b200lmoe-002-c \
  --moe-implementation ring_cute --expert-parallelism 8

# B200LMOE-001-D / 002-D: ragged_all_to_all_cute EP8
XLA_FLAGS=--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false \
  python grug_moe_mfu.py "${baseline_args[@]}" \
  --run-id b200lmoe-001-d --output-dir runs/b200lmoe-001-d \
  --moe-implementation ragged_all_to_all_cute --expert-parallelism 8
XLA_FLAGS=--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false \
  python grug_moe_mfu.py "${latent_args[@]}" \
  --run-id b200lmoe-002-d --output-dir runs/b200lmoe-002-d \
  --moe-implementation ragged_all_to_all_cute --expert-parallelism 8
```

The matched-work latent arms change routed experts from
`6144 -> 3072 -> 6144` to `3072 -> 6144 -> 3072`. Routing and the shared expert
remain at width 6144. This preserves routed-expert parameter count and active
expert GEMM work, halves the EP activation payload, and adds one shared
`6144 -> 3072 -> 6144` projection pair per layer.

### Multi-node

The script joins one JAX mesh across a multi-node gang automatically. Under an iris
gang job (`iris job run --replicas N`, one task per node), task 0 registers its
`jax.distributed` coordinator address in the iris endpoint registry and the other
tasks poll for it — no hand-rolled rendezvous. Pass `--num-gpus` = total GPUs across
all nodes; the script fails fast on a mismatch. Sharding spans the whole fleet by
default: FSDP (the `data` axis) covers all GPUs, and `--expert-parallelism` may
exceed the per-node GPU count (e.g. EP16 on 8 nodes × 4 GPUs). For fleets larger
than one model copy, `--replica-axis-size R` adds pure data parallelism outside
the FSDP+EP copy (parameters replicated across `R` replica groups; e.g. 64 GPUs
with `R=2` = two 32-GPU copies). Only process 0 prints per-step metrics and
writes `metrics_summary.json`.

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
| `--moe-implementation` | `sonic` | local: `sonic` (XLA ragged-dot), `sonic_cute` (QuACK SM100), `scatter`; expert-parallel: `ring`, `ring_cute`, `ragged_all_to_all`, `ragged_all_to_all_cute` (`_cute` = QuACK GEMMs under the same dispatch) |
| `--expert-parallelism` | 1 | expert mesh axis size; >1 requires an expert-parallel `--moe-implementation` |
| `--replica-axis-size` | 1 | pure-DP replica groups outside the FSDP+EP model copy (`num_gpus / R` GPUs per copy) |
| `--capacity-factor` | 1.0 | EP per-shard capacity multiplier; 1.0 = exact average (drops on imbalance, zero padding) |
| `--attention-implementation` | `gpu_fa4_cute` | FlashAttention-4 CuTeDSL backend |
| `--num-gpus` | 8 | total GPUs to shard across (all nodes; asserted against the joined mesh) |
| `--num-layers` | 26 | decoder layers |
| `--num-experts` / `--num-experts-per-token` | 64 / 4 | MoE experts, top-k |
| `--expert-intermediate-dim` | hidden / 2 | routed-expert intermediate width |
| `--shared-expert-intermediate-dim` | expert intermediate | always-on shared-expert intermediate width; 0 disables it |
| `--moe-latent-dim` / `--moe-latent-norm` | off / off | compress only the routed path to this width and optionally apply latent RMSNorm |
| `--num-heads` / `--num-kv-heads` | hidden / head dim | explicit query and KV head counts |
| `--sliding-window` / `--global-every` | 2048 / 4 | local attention window and full-causal layer frequency |
| `--batch-size` / `--seq-len` | 128 / 4096 | global batch, sequence length |
| `--steps` / `--warmup-steps` | 20 / 8 | total steps, warmup excluded from the median |
| `--profile-dir` | off | write a `jax.profiler` trace of steady-state steps 10–12 (process 0 only) |
| `--remat-mode` | `recompute_all` | `recompute_all` / `save_moe` / `all_but_moe` (attention+norms checkpointed, MoE live) / `none` |

## What it measures

MFU is the honest, active-expert accounting:

```
flops_per_token   = lm_flops_per_token(..., num_experts_per_tok=num_experts_per_token)  # active experts, not all 64
flops_per_example = 3 * flops_per_token * seq_len                                        # ×3 for fwd + bwd
mfu_b200          = (flops_per_example * examples_per_second) / (num_gpus * 2.25e15)     # B200 convention
mfu_gb200         = (flops_per_example * examples_per_second) / (num_gpus * 2.50e15)     # #7201 GB200 convention
```

Each step prints a JSON line (`mfu_gb200`, `mfu_b200`, `tokens_per_second`, `duration`, `loss`, …);
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

- The plain EP backends (`ring`, `ragged_all_to_all`) run their local grouped GEMMs
  through a Pallas-Triton kernel (measured 397–859 TF/s at these shapes vs 1,470–1,560
  TF/s for tuned QuACK). The `_cute` variants (`ring_cute`, `ragged_all_to_all_cute`)
  close most of that gap by running `sonic_cute`'s QuACK expert MLP (fused-SwiGLU gated
  GEMM + down GEMM, custom_vjp with QuACK dh/dx backward) under the same dispatch — the
  seam is a plain `expert_mlp_fn(x_dispatch, group_sizes)` callable
  (`_quack_expert_mlp_fn` in `lib/levanter/src/levanter/grug/_moe/ep_common.py`). The
  remaining EP tax is dispatch machinery: all-gather / a2a legs, capacity `top_k`,
  scatter-add, and (at `--capacity-factor` > 1) padding.
- **`ring` / `ring_cute` are the best-measured backends on one NVLink domain**
  (all-gather dispatch + psum-scatter combine; `_cute` adds the QuACK GEMMs under the
  same dispatch via an `expert_mlp_fn` seam — +0.4pp at EP8/8×B200, +1.4pp at
  EP4/4×GB200 where local GEMMs are bigger).
- `ragged_all_to_all` is pathological *at XLA's defaults only*: the default
  single-host lowering of `jax.lax.ragged_all_to_all` is XLA's own one-shot peer-copy
  kernel (`stream_executor::gpu::RaggedAllToAllKernelImpl`, bracketed by multi-GPU
  barriers), and that one kernel is ~90% of device time in our profile (~297 ms/call,
  104 calls/step at 26 layers) — a ~15–28× step-time regression, corroborated by
  [openxla/xla#33386](https://github.com/openxla/xla/issues/33386) (~2% of NVLink
  bandwidth on this path). The same thunk has an NCCL send/recv path with true ragged
  sizes behind `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`:
  measured, it recovers a2a from 1.9% to 14.4%, and `ragged_all_to_all_cute` (QuACK
  GEMMs under a2a) reaches 15.8% — still behind `ring_cute` single-node because the
  NCCL path pays a host sync + index all-to-all per call, but the natural multi-node
  vehicle (multi-host a2a always takes the NCCL path, and unlike ring it never
  all-gathers unrouted tokens). The zero-copy variant flag
  (`--xla_gpu_experimental_ragged_all_to_all_zero_copy`) is a measured no-op on the
  pathological path.
