# Grug MoE EP Hero

This self-contained variant is the selected EP64 configuration for GB200 NVL72. Each
data-parallel rack uses one 64-device expert mesh.

## Configuration

- Model: d6144, 48 layers, 384 routed experts of width 3072, top-8 routing, latent width 3072, and
  two shared experts of width 3072. Depth rounds up to the nearest even count.
- Attention: 48 heads, 12 local and 6 global KV heads, head dimension 128, sequence length 4096,
  sliding window 2048, and every fourth layer full-causal with the final layer also global. SConv
  and fused RoPE are on.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each. Additional racks use
  the `replica_dcn` axis. Six experts are on each GPU in each rack.
- Batch: 1024 global sequences. The launcher does not scale the batch with the rack count.
- Router: top-8 quantile balancing with next-step, stop-gradient expert biases and no auxiliary
  balancing loss.
- MoE backend: `fixed_pooled_wave_all_to_all`. Each sender uses one fixed pool per
  destination and stripes it over three static waves. The receiver runs all six local experts in
  each wave and drops rows above the fixed expert capacity. Expert IDs travel in the activation
  payload, so the method does not use a metadata collective. The receiver capacity factor is 1.33,
  and the sender capacity factor is 1.05. Each wave has receiver capacity equal to two full expert
  buffers.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Runtime: one JAX process per four-GPU worker, BF16 parameters and compute, GPU command buffers
  off, `cuda_async`, PGLE off, and collective overlap limit 4.
- Output: Metrics only by default. `--save-checkpoints` writes checkpoints, but restore with the
  pinned-host optimizer state has a known memory-kind mismatch. Do not use these checkpoints to
  restart a run.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

## Result

The selected configuration completed 200 steps on one rack. Over steps 150 through 199, median
throughput was 256,818 tokens/s and median MFU was 24.03%. Median routing drop rate was 2.41%, and
the final drop rate was 2.21%. The final loss was 3.2510. All 16 workers completed without an OOM,
nonfinite value, failure, or preemption. See the
[W&B run](https://wandb.ai/marin-community/rav_moe/runs/mhep-103-bf16params-pooled-striped-wave2-send105-recv133-200-20260814)
and the [XProf trace](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmhep-101-bf16params-pooled-striped-wave2-send105-recv133-profile-20260814&tool=trace_viewer).

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
| `--capacity-factor` | pooled receiver capacity factor |

Three quantities set what a sweep can fit on one rack:

- Active routed neurons are top-k multiplied by width.
- Parameters are expert count multiplied by width.
- The sender pool is token assignments multiplied by the sender capacity factor and divided across
  three waves.

The selected E384 model runs at expert width 3072 and capacity factor 1.33.

## Run Controls

| option | effect |
| --- | --- |
| `--dp-racks` | sets the data-parallel rack count; `--batch-size` stays global |
| `--batch-size` | sets global sequences per step and the optimizer token budget |
| `--schedule-steps` | sizes the learning-rate schedule while `--num-steps` bounds the run |
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
  --run-id mhep-pooled-wave \
  --num-steps 200 \
  --version 2026.08.14
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="mhep-pooled-wave"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32575 \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id "$run_id" --num-steps 200 --version 2026.08.14 --run
```

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

### Legacy small-scale ablations

`small_scale_abl_launch.py` runs the earlier E192, top-4 shape (`--size` in `d768`…`d2048`) on one
GB200 rack. It fixes the batch at ~4M tokens per step to hold the fixed-all-to-all drop dynamics, and
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
