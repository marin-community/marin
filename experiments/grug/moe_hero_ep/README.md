# Grug MoE EP Hero

This self-contained variant is the selected EP64 configuration for GB200 NVL72. Each
data-parallel rack uses one 64-device expert mesh.

## Configuration

- Model: d6144, 48 layers, 192 routed experts of width 6144 (hidden-wide), top-4 routing, latent
  width 3072, and two shared experts of width 3072. This is 535.420 B total parameters and 24.454 B
  active per token. Depth rounds up to the nearest even count.
- Attention: 48 heads, 12 local and 6 global KV heads, head dimension 128, sequence length 4096,
  sliding window 2048, and every fourth layer full-causal with the final layer also global. SConv
  and fused RoPE are on.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each. Additional racks use
  the `replica_dcn` axis. Three whole experts land on each device in each rack.
- Batch: 1024 global sequences. The launcher does not scale the batch with the rack count.
- Router: top-4 quantile balancing with next-step, stop-gradient expert biases and no auxiliary
  balancing loss.
- MoE backend: `fixed_all_to_all` by default, with `ragged_all_to_all` as the `ep-ragged` flavor.
  The default backend uses gather dispatch, structured custom VJPs, and capacity factor 1.33.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Runtime: one JAX process per GPU, GPU command buffers at the XLA default capture set, `cuda_async`, PGLE off (the
  per-process CUPTI sessions cannot profile concurrently; an explicit `JAX_ENABLE_PGLE` env
  setting still wins), and collective overlap limit 4.
- Output: Metrics only by default. `--save-checkpoints` writes checkpoints, but restore with the
  pinned-host optimizer state has a known memory-kind mismatch. Do not use these checkpoints to
  restart a run.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

## Result

The capacity factor 1.33 default passed a five-step, full-watch gate with all 76 norm fields finite
on each step. The highest confirmed routing cell capacity is 1,829, and 1.33 keeps a small margin
below it; this cell load scales with the expert count and top-k, not the expert width, so it holds
as the width moves. Capacity factor 1.34 ran out of CUDA memory at the prior expert width 6272.

The matched 200-step throughput and loss were measured at the prior expert width 6272 with capacity
factor 1.30 and automatic PGLE: a last-50 mean of 262,683 tokens/s at 3.9642% drops, a drop-adjusted
252,271 tokens/s, and a mean loss of 3.2417. The narrower 6144 width lowers per-expert compute and
memory below these figures.

### EP ablation ladder (4k context)

The default EP configuration — histogram QB, standard init, latent MoE — trained across the
downsized d768–d2048 ladder at 4096 sequence length and 750 tokens per active parameter. Final
Paloma macro loss, both as trained (with capacity drops) and re-scored dropless
(`sonic_cute` at one chunk), against issue [#8062](https://github.com/marin-community/marin/issues/8062):

| size | drop % (last 50) | Paloma (with drop) | Paloma (dropless) |
| --- | --- | --- | --- |
| d768 | 5.50% | [3.2326](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d768) | [3.0331](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d768-dropless-eval) |
| d1024 | 5.94% | [2.9849](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d1024) | [2.7930](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d1024-dropless-eval) |
| d1536 | 6.61% | [2.7487](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d1536) | [2.5710](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d1536-dropless-eval) |
| d2048 | 7.11% | [2.5858](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d2048) | [2.4106](https://wandb.ai/marin-community/marin_moe/runs/mhep-ladder-hist-noinit-20260808c-ep64-d2048-dropless-eval) |

The drop-free re-eval is the fair comparison to a dropless FSDP run; the training-time drops grow
with width and are recovered by scoring dropless.

## Sweeps

Five launcher options move the shape from the hero spec. They keep the hidden dimension, so the
compute-scaled optimizer values stay constant across a sweep.

| option | effect |
| --- | --- |
| `--num-experts` | routed expert count. Must divide the 64-way expert axis. |
| `--intermediate-dim` | routed expert width |
| `--num-experts-per-token` | routed top-k |
| `--latent-dim` | routed input and output width |
| `--capacity-factor` | fixed all-to-all capacity factor |

Three quantities move independently, which sets what a sweep can afford on one rack:

- Active routed neurons are top-k multiplied by width.
- Parameters are expert count multiplied by width.
- The all-to-all buffers are tokens multiplied by top-k.

Width appears in the first two and not the third, thus width is the cheap way to buy active compute
and top-k is the expensive way. Six buffers scale with top-k, and one of them is float32: top-6
costs 30.75 GiB against 20.50 GiB for top-4 at this shape.

The selected E192 model runs at expert width 6144 and capacity factor 1.33. Width 6400 failed at
capacity factor 1.30 in the size search. The
[experiment record](../../../.agents/logbooks/7279-moe-hero-ep.md) contains the size and capacity
searches.

## Run Controls

| option | effect |
| --- | --- |
| `--dp-racks` | sets the data-parallel rack count; `--batch-size` stays global |
| `--batch-size` | sets global sequences per step and the optimizer token budget |
| `--schedule-steps` | sizes the learning-rate schedule while `--num-steps` bounds the run |
| `--flavor` | selects `ep` or `ep-ragged` |
| `--eval-every` | adds Paloma evaluation at the selected interval |
| `--save-checkpoints` | writes checkpoints with the restore limitation above |
| `--watch-interval`, `--watch-mode` | select inline or diagnostic norm collection |
| `--profile-start-step`, `--profile-steps` | select the rank-0 XProf window |
| `--seed` | sets the trainer seed |

## Launch

### Hero

Print the plan without a GPU run:

```bash
python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-017-200 \
  --dp-racks 1 \
  --num-steps 200 \
  --version 2026.08.05

python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-011-e256-i2560 \
  --dp-racks 1 \
  --num-steps 25 \
  --num-experts 256 --intermediate-dim 2560 \
  --version 2026.08.05
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="mhep-017-200"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32575 \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id "$run_id" --dp-racks 1 --num-steps 200 --version 2026.08.05 --run
```

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

### Small-scale ablations

`small_scale_abl_launch.py` runs a downsized hero shape (`--size` in `d768`…`d2048`) on one GB200
rack. It fixes the batch at ~4M tokens per step to hold the fixed-all-to-all drop dynamics, and
sizes the step count from the model's active-parameter count: `num_steps` trains
`--tokens-per-active-param` (default 750) tokens per active parameter. `--flavor ep` keeps the
64-way expert axis; `--flavor fsdp-nodrop` runs the same shape dropless, and `--flavor fsdp-chunk4`
runs it with four-chunk capacity. Print the plan without a GPU run:

```bash
python -m experiments.grug.moe_hero_ep.small_scale_abl_launch \
  --run-id mhep-abl-d1024-ep \
  --size d1024 \
  --flavor ep \
  --version 2026.08.10
```

Submit one rung through the Marin Iris controller:

```bash
run_id="mhep-abl-d1024-ep"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32576 \
  -- python -m experiments.grug.moe_hero_ep.small_scale_abl_launch \
    --run-id "$run_id" --size d1024 --flavor ep --version 2026.08.10 --run
```

The wider rungs need more than one rack to hold their batch: `--dp-racks N` replicates the run
across `N` racks, and the launcher sizes the fleet request accordingly. Ablation runs report to W&B
group `moe-hero-ep-small-abl` and carry Paloma and uncheatable evaluation at `--steps-per-eval`.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
