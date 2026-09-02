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
- Batch: The d6144 production run uses 11264 global sequences across 11 racks. A one-rack
  diagnostic uses 1024 sequences.
- Router: top-8 quantile balancing uses a global histogram with 10,000 bins. It has next-step,
  stop-gradient expert biases and no auxiliary balancing loss.
- MoE backend: `ragged_all_to_all`. One update carries each (peer, local expert) pair, so rows
  arrive grouped by expert, and local experts run in two chunks that share the 1.15 receiver
  capacity. The transport reaches XLA's device-initiated (NCCL LSA) kernel, which needs Marin's patched
  PJRT build, installed on GB200 through the `gpu` extra (`lib/marin/pyproject.toml`); a run that
  reaches the stock plugin fails at startup.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Weights: fp32 on device with bf16 compute. A checkpoint written with a pinned-host fp32 master
  migrates in process on restore: its stored fp32 master is read directly into the run's params
  (the bf16 compute copy goes unread), and the next save writes the new layout. The reverse
  (synthesizing a master) is refused.
- Runtime: Each GPU has one JAX process. The recipe uses `cuda_async`, no PGLE, and no GPU
  command buffers. The ragged transport stages each layer's residual carry on pinned host, which
  frees the HBM the latency-hiding scheduler needs to run. Collective overlap stays at 1: the
  offload, the scheduler, and a higher limit corrupt training together, though no pair of them
  does.
- Resources: Each four-GPU worker requests 120 CPU, 890 GB of RAM, and 1 TB of disk.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

Bounded diagnostics write metrics only by default. `--save-checkpoints` writes checkpoints below
`--checkpoint-path` and resumes from the newest complete checkpoint. PR
[#8480](https://github.com/marin-community/marin/pull/8480) bounded pinned-host restore memory. Its
d6144 run restored step 164 with a 735 GiB fleet peak against a 940 GiB request.

## Results

### Transport

[#8549](https://github.com/marin-community/marin/pull/8549) compared the ragged and pooled-wave
transports head to head: both runs restored the live hero's step-6000 checkpoint on one NVL72
rack and ran back to back, with the transport the only variable and fp32 weights on device in
both:

| | ragged | pooled-wave |
| --- | --- | --- |
| MFU | 22.87% | 22.71% |
| assignments dropped | 0.018% | 2.67% |
| loss at the scored step | 1.4727 | 1.4777 |
| runtime device peak | 137.9 GiB | 149.9 GiB |

The throughput gap is inside the run-to-run spread (standard deviation over the scored steps was
0.11 for ragged and 0.59 for pooled-wave, and three earlier ragged runs ranged 22.34–22.58), so
this buys the drop rate and the headroom at parity rather than a speedup. At d768 over 10.8k steps
ragged also finished ahead on train loss (1.939 vs 1.956) and eval bpb (0.975 vs 1.033).

Dropping the pinned-host fp32 master is worth about 0.4 MFU on the ragged path, measured on a
paired hero. Pooled-wave needed the master to fit at all.

### Earlier pooled-wave gates

These ran before the ragged transport, at capacity and process settings the recipe no longer uses.

The 1.10 sender and 1.15 receiver configuration completed a 20-step, one-rack gate. Median
throughput over steps 2 through 19 was 250,691 tokens/s, and final throughput was 246,947 tokens/s.
The final loss was 6.3224. The final total drop rate was 19.33%: 7.14% at the sender and 12.19% at
the receiver. The receiver dropped 13.12% of assignments that reached it. This short gate validates
memory use and metric reporting. It does not estimate the steady drop rate. All 16 workers completed
without an OOM, nonfinite value, failure, or preemption. See the
[W&B run](https://wandb.ai/marin-community/rav_moe/runs/mhep-118-recv-metrics-send110-recv115-smoke).

The prior 1.05 sender and 1.33 receiver configuration completed 200 steps on one rack. Over steps
150 through 199, median throughput was 256,818 tokens/s and median MFU was 24.03%. Median routing
drop rate was 2.41%, and the final drop rate was 2.21%. The final loss was 3.2510. All 16 workers
completed without an OOM, nonfinite value, failure, or preemption. See the
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

## Diagnostic sweeps

Five launcher options move the shape from the hero spec. They keep the hidden dimension, so the
compute-scaled optimizer values stay constant across a sweep.

| option | effect |
| --- | --- |
| `--num-experts` | routed expert count. Must divide the 64-way expert axis. |
| `--intermediate-dim` | routed expert width |
| `--num-experts-per-token` | routed top-k |
| `--latent-dim` | routed input and output width |
| `--capacity-factor` | receiver capacity factor |

Three quantities set what a sweep can fit on one rack:

- Active routed neurons are top-k multiplied by width.
- Parameters are expert count multiplied by width.
- The receiver buffer is token assignments multiplied by the receiver capacity factor, split
  across the transport's two expert chunks.

The selected E384 model runs at expert width 3072 and receiver capacity factor 1.15.

## Diagnostic controls

| option | effect |
| --- | --- |
| `--dp-racks` | sets the data-parallel rack count; `--batch-size` stays global |
| `--batch-size` | sets global sequences per step and the optimizer token budget |
| `--schedule-steps` | sizes the learning-rate schedule while `--num-steps` bounds the run |
| `--eval-every` | adds Paloma evaluation at the selected interval |
| `--save-checkpoints` | writes periodic and final checkpoints |
| `--checkpoint-minutes` | sets the wall-clock checkpoint interval |
| `--checkpoint-path` | places checkpoints at an explicit storage prefix |
| `--checkpoint-debug` | publishes checkpoint phase and memory telemetry |
| `--training-data synthetic` | reuses a deterministic batch without opening TensorStore |
| `--watch-interval`, `--watch-mode` | select inline or diagnostic norm collection |
| `--profile-start-step`, `--profile-steps` | select the rank-0 XProf window |
| `--seed` | sets the trainer seed |

## Launch

### Bounded diagnostics

`launch_diagnostics.py` uses the d6144 model, Harrier 2026.08.18 data, process layout, watch config,
and TensorStore cache from the production recipe. Its stop step, evaluation, and checkpoint policy
stay independent from the production run.

The default 25-step diagnostic uses simulated epoching. Set `--schedule-steps 390251` to use the raw
production mixture. The diagnostic default now matches the production watch and 890 GB RAM request.

Print the plan without a GPU run:

```bash
python -m experiments.grug.moe_hero_ep.launch_diagnostics \
  --run-id mhep-ragged \
  --num-steps 200 \
  --version 2026.08.14
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="mhep-ragged"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32575 \
  -- python -m experiments.grug.moe_hero_ep.launch_diagnostics \
    --run-id "$run_id" --num-steps 200 --version 2026.08.14 --run
```

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

Submit a rack-local XProf trace through the Marin Iris controller:

```bash
run_id="mhep-rack-profile"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e IRIS_PORT_JAX 32576 \
  -- python -m experiments.grug.moe_hero_ep.launch_diagnostics \
    --run-id "$run_id" --num-steps 8 --schedule-steps 390251 --batch-size 1024 \
    --profile-start-step 5 --profile-steps 2 --training-data synthetic \
    --version dev --run
```

Batch 1024 keeps the production local batch of 16 sequences per GPU. The trace does not include
the 11-rack `replica_dcn` collectives or their global histogram reduction.

### Small-scale hero-shape ablations

`small_scale_abl_launch.py` runs the hero shape — 384 experts / top-8, hidden/2-wide experts in a
hidden/2 latent, capacity 1.15 — at a downsized width (`--size` in `d768`…`d2048`) on one GB200 rack.
It fixes the batch at ~4M tokens per step per rack to hold the drop dynamics, and sizes the step
count from the model's active-parameter count: `num_steps` trains `--tokens-per-active-param`
(default 750) tokens per active parameter. Each flavor names its own transport rather than
following the hero default, because comparing them is what this launcher is for: `--flavor ragged`
is the hero's and needs a GB200 fleet, `--flavor ep` is the pooled-wave arm it replaced at a 1.15
sender capacity over 3 waves, and `--flavor fsdp-nodrop` / `--flavor fsdp-chunk4` run the same
shape dropless and at four-chunk capacity. The pooled gates are tunable with `--capacity-factor`
(receiver) and `--transport-capacity-factor` (sender). Print the plan without a GPU run:

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

### Scaling ladder

`launch_scaling_ladder.py` trains one uniform hero recipe at five widths so a narrow rung predicts
the `d6144` hero (which is the hero itself). Every rung shares the hero data (the Harrier
2026.08.18 two-phase mixture on the Marin tokenizer, simulated against the 18.75T target budget),
the offloaded MuonH optimizer, the hero mixed precision, 384 experts / top-8, the ragged
all-to-all transport, the QB histogram estimator at 10k bins, and a dropless held-out eval. Only
the width and the rack count vary; the rack count, batch, step budget, eval cadence, and
checkpoint policy all follow `--size`:

| size | racks | batch | steps | eval | checkpoints |
|---|---|---|---|---|---|
| d768 | 1 | 1024 | 11,420 | every 5% | final only |
| d1024 | 2 | 2048 | 15,276 | every 5% | final only |
| d1536 | 6 | 6144 | 15,128 | every 5% | final only |
| d2048 | 11 | 11264 | 20,072 | every 5% | final only |
| d6144 | 11 | 11264 | 390,251 | every 3000 | every 6k |

Train batch is 1024 × racks; eval batch is 64 × racks (one sequence per device). The step budget is
791 tokens per active parameter (18T at d6144); pass `--num-steps` to override.

A rung resumes from the newest checkpoint it finds. The permanent checkpoints above go to the
durable output root, and a rolling temporary checkpoint every hour goes to region-local temp
storage with the shared 14-day lifecycle TTL. One temporary checkpoint is kept. A hardware fault, a
host out-of-memory, or a preemption thus costs at most one hour of training. The training job
retries 1000 times on failure and 100 times on preemption.

Launch or resume the production d6144 hero with `trigger_hero.sh`. The trigger records the full
`HEAD` commit and whether the tree has staged, unstaged, or untracked changes in the Iris system
reason and coordinator environment. The coordinator job name carries the short commit and
`clean`/`dirty` state. These values describe the source bundle submitted by that invocation; the
run ID continues to identify the checkpoint and output lineage across resumptions.

```bash
WANDB_API_KEY=... ./experiments/grug/moe_hero_ep/trigger_hero.sh
```

```bash
python -m experiments.grug.moe_hero_ep.launch_scaling_ladder \
  --run-id ladder-d768 --size d768 --version dev
```

Submit a rung through the Marin Iris controller as with the launchers above, swapping the module and
passing `--size`. Runs report to W&B group `moe-hero-ep-scaling-ladder`.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
