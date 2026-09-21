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
  The production hero trained on `fixed_pooled_wave_all_to_all` through step 81716 (3.77T of its 18T
  tokens); `hero-ragged_a2a-ep-step81k` continues from that checkpoint on the ragged transport.
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
`--checkpoint-path` and resumes from the newest complete checkpoint.

## Hero cutovers and W&B lineage

```python
from experiments.grug.moe_hero_ep.checkpoints import hero_checkpoint_paths

paths = hero_checkpoint_paths()
```

The function returns permanent checkpoint paths for the current run and its ancestors, in step order.
Pass a run ID to select another run. The launcher and function share `CURRENT_HERO_RUN_ID` in [`current_run.py`](current_run.py).

Use the [deployment checklist](../../../.agents/skills/deploy-hero-change/SKILL.md)
for preflight, the 200-step trial, and rollback. `current_run.py` records the
current run ID. `trigger_hero.sh` records the handoff checkpoint and W&B fork point.
Update the run ID, handoff checkpoint, and fork point together and land them on main.
Record the old run's exact launch SHA and command for rollback.
The checkpoint must be complete and protected from cleanup through the trial.

Both commands below require `WANDB_API_KEY`, an authenticated GitHub CLI (`gh`),
and a pristine checkout. Fork creation requires fetched main; subsequent launches
use the same SHA recorded on the child, even if main advances.

For checkpoint `step-N`, the first replayed update logs `global_step=N`. Find the
parent row with `global_step=N-1` and use its actual W&B `_step` in
`WANDB_FORK_FROM='<parent-run-id>?_step=<history-step>'`. Inspect the parent with
`wandb.Api().run("marin-community/marin_moe/<parent>")` and
`run.scan_history(keys=["_step", "global_step"], min_step=N-2, max_step=N+1)`.
These bounds use W&B `_step`; if the indices differ or the row is missing,
adjust the history range and locate `global_step=N-1` explicitly before forking.
Do not inherit parent results for the replayed updates.

Create the child tracker once, outside training retry loops:

```bash
experiments/grug/moe_hero_ep/trigger_hero.sh fork-wandb
```

After preflight and confirmation that the old coordinator is terminal, submit:

```bash
experiments/grug/moe_hero_ep/trigger_hero.sh launch
```

For recovery, verify no child coordinator is live, then use `launch` from the
recorded SHA. Do not fork again. For rollback, use the old run's recorded revision
and command with `IRIS_USER=marin`. One operator owns submissions; verify exactly
one live coordinator after launch.

W&B forks preserve history, while `--initialize-from-checkpoint` restores training
state. Confirm new child progress from its Finelog execution, not inherited W&B
rows. Before submission, the launcher posts source SHA, run and coordinator IDs,
fork point, and checkpoint to [#8506](https://github.com/marin-community/marin/issues/8506);
a failed post aborts submission. Iris also records `MARIN_PROVENANCE`.

## Coordinated garbage collection

Pass `--gc-interval 100` to `python -m experiments.grug.moe_hero_ep.launch_diagnostics`,
or set `gc_interval=100` in `hero_grug_trainer_config`. The default `None` preserves automatic
Python garbage collection. The launcher distributes one configuration to all ranks. After ten
training steps in each process (including after resume), disable automatic cyclic collection
and collect once. Then a training hook collects at completed global steps divisible by the
interval, after evaluation hooks and before checkpoint work. It also collects on the forced
final callback pass. Training collectives bound rank skew; GC adds no barriers. Collect after each
evaluation hook as well, so cycles holding temporary eval buffers do not wait for the next periodic boundary.
Reference-count deallocation continues. The previous GC policy is restored on exit.

`throughput/gc_time` records local startup and hook collection time when GC runs; the `garbage_collection`
profiler annotation also covers evaluation cleanup. Evaluation cleanup is included in callback
and iteration time. `throughput/checkpoint_time` includes the save-decision broadcast and any
synchronous checkpoint work. `throughput/iteration_time` includes batch loading, training,
callbacks, checkpoint work, and GC. The existing `throughput/duration` excludes loading,
callbacks, checkpoints, and GC. Tracker steps are zero-based: collections after completed updates
100/200/300 appear at x=99/199/299. Use elapsed time per update for comparisons.

The [single-rack validation](https://iris.oa.dev/#/job/%2Fmwittmann%2Fgc-sync-9205-hook-20260917-coord)
used the full model with one sequence per GPU on 64 GPUs. Across 300 measured updates after ten
warmup steps, elapsed time including initial and final collection fell from 795.576 to 776.003
seconds (2.46%; 2.652 to 2.587 seconds/update). Before the final callback pass, elapsed time fell
from 795.575 to 774.700 seconds (2.62%). Across all ranks, automatic GC produced 72 full
collections across 48 steps; coordinated GC produced one collection per rank after warmup, at
global updates 100/200/300, and on normal completion at global update 310. Scheduled collections ran through
the training hook before checkpoint decisions. The slowest rank's periodic collections took
1.15–1.22 seconds; final collection took up to 1.30 seconds.

Both arms had identical sampled live HBM (35.094 GiB per rank) and allocator peaks (111.840 GiB),
with no increase above their post-warmup baselines. Live memory was sampled every ten measured
steps and at completion; the allocator peak includes warmup. The largest per-rank RSS increase in treatment was
54.4 MiB above its post-warmup baseline. This single pair exercised checkpoint decisions with writes
and evaluation disabled. The result supports the mechanism and short-term memory behavior at this
batch size. Cycles can still retain device buffers between collections; keep the feature opt-in
pending a production-batch trial that includes evaluation transitions and longer memory tracking.
See [#9205](https://github.com/marin-community/marin/issues/9205).

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
| `--gc-interval` | opts into cyclic GC at shared completed-step boundaries after warmup |
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
the `d6144` hero (which is the hero itself). Every rung shares the data schedule below on the
Marin tokenizer (simulated against 18.75T for small runs; raw sampling above 1e23 training FLOPs),
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

The [new mixture](../../../docs/reports/hero-mixture-log.md) starts at ~27.7%
of training, with cooldown weights at ~80% (hero steps 108,000 and 312,192).
For production launch and recovery, follow
[Hero cutovers and W&B lineage](#hero-cutovers-and-wb-lineage).

```bash
python -m experiments.grug.moe_hero_ep.launch_scaling_ladder \
  --run-id ladder-d768 --size d768 --version dev
```

Submit a rung through the Marin Iris controller as with the launchers above, swapping the module and
passing `--size`. Runs report to W&B group `moe-hero-ep-scaling-ladder`.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
