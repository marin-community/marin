# June 67B-A2B on AMD MI355X

This directory is a copy of the June 67B-A2B model and training loop
(`experiments/june_tpu_67b_a2b/moe/`) with the kernels and loop choices we
selected on AMD MI355X written as straight code. It imports June's
`GrugModelConfig` unchanged, and its training module (and through it June's
optimizer and heuristic), checkpointing and dispatch code. The parameter tree and checkpoint layout are
June's, so checkpoints move between the two.

## What differs from June

| Piece | June | This copy | Why |
|---|---|---|---|
| q/k head norm, half-RoPE, q scale | Separate XLA ops | One Pallas-Triton kernel per direction (`kernels/qk_norm_rope.py`) | XLA splits them into several kernels per tensor and direction on GPU |
| RMSNorm + GatedNorm (per block, embedding, final) | Separate modules | One custom VJP over the residual add, the norm and the gate, with Pallas-Triton kernels (`kernels/add_rms_norm.py`, `kernels/gated_norm.py`) | One pass over the rows in each direction instead of several |
| MoE backend | `moe_implementation` | Always Levanter's `ring_filtered` | The MI355X selected backend; Levanter picks its kernels by device |
| Routing statistics, QB thresholds, capacity drops | Reduced across devices in every layer | Kept per batch shard and reduced once after the layer scan | With expert parallelism, one collective per step instead of several per layer |
| Router diagnostics | Entropy, counts, load balance and z-loss in every layer | Only the z-loss, always logged; added to the loss only when `router_z_loss_coef != 0` | June does not train on the others |
| Remat policy | `remat_mode` | MoE dispatch and routing tensors, attention residuals (`context`), GatedNorm's down projection and the head-gate logits | The backward skips the MoE dispatch and two GEMMs. Saving `context` only takes effect with jax-aiter attention (our fork), the one backend that tags it; upstream backends ignore the name |
| Layer scan | Stacked or unstacked; checkpoint with `prevent_cse=True` | Always stacked; checkpoint with `prevent_cse=False` | The scan already stops CSE; without the barriers XLA fuses the backward's slices of the saved residuals (about 100 ms per step) |
| Train loop | Finishes each step before the next | Dispatches step N+1 before it checks and logs step N; the loss's finiteness is computed in the step | The host work overlaps device compute |

The kernels are Pallas-Triton: they run on GPU, in the Pallas interpreter on
CPU (the tests), and raise on TPU. `Transformer.init` raises on June config
values this copy does not implement: another `moe_implementation`, unstacked
blocks, a `ce_implementation`, a non-default `remat_mode` or `scan_unroll`, no
`pooled_transport_capacity_factor`, and
widths the kernels cannot tile (`hidden_dim` a multiple of 128, `head_dim // 4`
a power of two of at least 16). `attention_implementation` is honoured.

## Synthetic benchmark

`synthetic_benchmark.py` times the copy on random tokens; `run_config.py` holds
its command line, which covers the model size, mesh, batch, attention,
profiling and logging only. On two 8-GPU MI355X nodes, expert parallelism
within each node and data parallelism across them, with
`experiments/amd/hpcfund/run_gpu.sh` (it sets the ROCm workarounds):

```bash
sbatch -N2 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_gpu.sh --overlap-collectives \
    experiments/amd/june_67b_a2b/synthetic_benchmark.py --size full --steps 30 --gpus-per-node 8 \
    --expert-axis 8 --replica-axis 2 --batch-size 32 --attention gpu_pallas_triton_flash --watch-interval 0
```

`--size tiny` runs on CPU in a few seconds:

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 uv run python experiments/amd/june_67b_a2b/synthetic_benchmark.py \
    --size tiny --steps 5 --expert-axis 2 --batch-size 8 --watch-interval 0
```

The report ends with the median step time from step 2 on, tokens/s, model
TFLOP/s and MFU, the median wall step, then each device's peak memory.
