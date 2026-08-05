# Grug MoE EP Hero

This self-contained variant is the one-rack EP64 baseline for GB200 NVL72.

## Configuration

- Model: d6144, 48 layers, 128 routed experts, top-4 routing, and two shared experts of width 3072.
  This is 359.6 B total parameters and 20.9 B active per token.
- Attention: 48 heads, 12 local and 6 global KV heads, head dimension 128, sequence length 4096,
  sliding window 512, and every sixth layer full-causal. SConv and fused RoPE are on.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each. Two whole experts
  land on each device.
- Batch: 1024 sequences.
- Router: top-4 quantile balancing with next-step, stop-gradient expert biases and no auxiliary
  balancing loss.
- MoE backend: `fixed_all_to_all` with gather dispatch, structured custom VJPs, and capacity
  factor 1.0.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Runtime: GPU command buffers off, `cuda_async`, PGLE off, and collective overlap limit 4.
- Output: Metrics only. This throughput run does not write a checkpoint.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

## Result

A 200-step gate on one rack measures 26.2835% median MFU, 309,091 tokens/s, 7.2280% MoE drops, and
3.2971 final loss over 201 samples with a 2.3460 deviation. A 25-step gate at the same shape
measures 26 samples with a 6.5623 deviation, thus its 27.5501% median runs about 1.3 points high.
Use the 200-step number.

The same model under FSDP-64 on one rack measures 19.3951% median MFU and 235,125 tokens/s. Both
arms share one analytic FLOP count of 44.491 GFLOP per token, because that count depends only on the
model config, so their MFU values share a denominator. They do not do equal work at capacity 1.0:
EP discards 9.97% of assignments against 1.88%. At capacity factor 1.5 the EP drop rate falls to
1.5719%, below the FSDP rate, and EP still measures 22.9037%.

## Sweeps

Four launcher options move the shape from the hero spec. They keep the hidden dimension, so the
compute-scaled optimizer values stay constant across a sweep.

| option | effect |
| --- | --- |
| `--num-experts` | routed expert count. Must divide the 64-way expert axis. |
| `--intermediate-dim` | routed expert width |
| `--num-experts-per-token` | routed top-k |
| `--capacity-factor` | fixed all-to-all capacity factor |

Three quantities move independently, which sets what a sweep can afford on one rack:

- Active routed neurons are top-k multiplied by width.
- Parameters are expert count multiplied by width.
- The all-to-all buffers are tokens multiplied by top-k.

Width appears in the first two and not the third, thus width is the cheap way to buy active compute
and top-k is the expensive way. Six buffers scale with top-k, and one of them is float32: top-6
costs 30.75 GiB against 20.50 GiB for top-4 at this shape.

Measured limits on one rack: 591.6 B total parameters runs at 24.0032% median MFU (256 experts of
width 2560), and 641.4 B fails with `NCCL operation ncclAlltoAll` and a CUDA out-of-memory. Lower
top-k does not lift that limit, so parameters bind rather than the communicator. A capacity sweep at
the hero shape measures 27.5501% MFU at 9.97% drops, 26.1065% at 5.3984%, 25.2283% at 3.2571%, and
22.9037% at 1.5719%, or about 0.9 points of MFU for each point of drops recovered.

## Launch

Print the plan without a GPU run:

```bash
python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-017-200 \
  --num-steps 200 \
  --version 2026.08.05

python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-011-e256-i2560 \
  --num-steps 25 \
  --num-experts 256 --intermediate-dim 2560 \
  --version 2026.08.05
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="mhep-017-200"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 21600 \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32575 \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id "$run_id" --num-steps 200 --version 2026.08.05 --run
```

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
