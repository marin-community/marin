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
- MoE backend: `fixed_pooled_wave_all_to_all`, as a temporary fallback. Each sender uses one fixed
  pool per destination and stripes it over three static waves. The receiver runs all six local
  experts in each wave and drops rows above the fixed expert capacity. Expert IDs travel in the
  activation payload, so the method does not use a metadata collective. The receiver and sender
  capacity factors are 1.15. `ragged_all_to_all` is the intended default and takes the hero back
  once it stops hanging an 11-rack run after a watch step
  ([#8870](https://github.com/marin-community/marin/issues/8870)): one update carries each (peer,
  local expert) pair, so rows arrive grouped by expert, and it reaches XLA's device-initiated
  (NCCL LSA) kernel, which needs Marin's patched PJRT build, installed on GB200 through the `gpu`
  extra (`lib/marin/pyproject.toml`).
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Weights: bf16 on device with a pinned-host fp32 master, which the pooled-wave device peak needs.
  A checkpoint written with a master restores natively here. A master-less checkpoint, which the
  hero writes when it keeps fp32 weights on device under the ragged transport, cannot restore into
  this mode: synthesizing a master is refused. The reverse direction migrates in process, so a
  master-bearing checkpoint reaches either mode.
- Runtime: Each GPU has one JAX process. The recipe uses `cuda_async`, no PGLE, and no GPU
  command buffers. The layer carry stays in HBM, which only the ragged transport offloads. Inline
  watch uses collective overlap limit 1. A disabled watch uses limit 4.
- Resources: Each four-GPU worker requests 120 CPU, 890 GB of RAM, and 1 TB of disk.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

Bounded diagnostics write metrics only by default. `--save-checkpoints` writes checkpoints below
`--checkpoint-path` and resumes from the newest complete checkpoint.

## Why this recipe

[#8549](https://github.com/marin-community/marin/pull/8549) selected the ragged transport in a
head-to-head restore of the live hero: 22.87% vs 22.71% MFU against pooled-wave (inside
run-to-run spread), 0.018% vs 2.67% assignments dropped, and 137.9 vs 149.9 GiB device peak.
Dropping the pinned-host fp32 master is worth about 0.4 MFU on this path; pooled-wave needed the
master to fit at all. The earlier pooled-wave gates and their per-run W&B links are in the
[#7279](https://github.com/marin-community/marin/issues/7279) coordination record; the EP ablation
ladder is in [#8062](https://github.com/marin-community/marin/issues/8062).

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

Use `--version dev` for diagnostics, ablations, profiles, and scaling runs in this guide. These
runs write under `users/<username>/grug/...`. Reserve calendar versions for coordinated major
production runs that need a shared checkpoint path under `grug/...`.

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
  --version dev
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
    --run-id "$run_id" --num-steps 200 --version dev --run
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
  --version dev
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
    --run-id "$run_id" --size d1024 --flavor ep --version dev --run
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

Launch or resume the production d6144 hero with `trigger_hero.sh`. The trigger first comments on
[issue #8506](https://github.com/marin-community/marin/issues/8506) with the full `HEAD` commit,
whether the tree has staged, unstaged, or untracked changes, and the coordinator job name. A
missing GitHub CLI login or failed comment aborts the trigger before Iris submission. Iris also
captures its standard launch provenance in `MARIN_PROVENANCE`. The run ID continues to identify
the checkpoint and output lineage across resumptions.

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
