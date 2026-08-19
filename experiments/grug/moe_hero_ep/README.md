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
  payload, so the method does not use a metadata collective. The receiver capacity factor is 1.15,
  and the sender capacity factor is 1.10.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Runtime: one JAX process per four-GPU worker, BF16 parameters and compute, GPU command buffers
  off, `cuda_async`, PGLE off, and collective overlap limit 4.
- Output: Metrics only by default. `--save-checkpoints` writes checkpoints, but restore with the
  pinned-host optimizer state has a known memory-kind mismatch. Do not use these checkpoints to
  restart a run. Drop metrics include the sender and receiver shares of all assignments. The
  receiver also reports its drop share of assignments that reached it.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

## Results

The current 1.10 sender and 1.15 receiver configuration completed a 20-step, one-rack gate. Median
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

The selected E384 model runs at expert width 3072 and receiver capacity factor 1.15.

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

### Dropless MoK comparison

The comparison branch exposes three EP launchers under the same Torch 2.11+cu130, cuBLAS 13.2,
NVCC 13.0.88 environment:

- `launch` is the historical one-process-per-node fixed-all-to-all baseline.
- `launch_multiprocess` is the same fixed-all-to-all backend with one JAX process per GPU, matching
  MoK's process topology.
- `launch_mok` is the dropless MoK backend. It keeps the shared experts' six canonical parameter
  leaves -- and LatentMoE's two projections and norm gain -- separate for MuonH, but all of them
  are *computed* inside the fused call:

  - At the hero's latent width (the default), the call takes the full-width token, down-projects
    and RMSNorms it internally, routes and combines at `latent_dim` -- which is what makes its
    wire traffic match the all-to-all arms' -- runs both shared experts at the full width, and
    up-projects the combined routed result back before returning. Nothing is left for the block
    or for JAX.
  - With `--latent-dim 0` the two widths coincide, the projections and the norm are skipped, and
    the three latent operands are passed with a zero-length latent axis. Both shared experts are
    still fused.

  `config.model.fuses_shared_experts` is therefore true for MoK in both arms and false for every
  all-to-all backend, and the run records the outcome as `comparison/fused_shared_experts`. Note
  that the latent arm used to pay for a zeroed 256-wide dummy shared expert plus JAX-side
  projections; that overhead is gone, so the MoK arm's MFU improves for reasons `_compute_flops`
  does not model. Attribute that part of any delta to the removed overhead, not to the kernel.

MoK is supplied as an immutable, prebuilt CPython 3.12 Linux wheel for the Iris worker
architecture. Build that wheel from the matching `mark/mok_hacking` checkout against Torch
2.11+cu130 and CUDA 13.0 with `MOK_ARCH=SM100` for GB200; MoK's default SM103 wheel is not the
right target. Source/VCS installs are intentionally unsupported by this launcher because task
dependency installation happens before Iris exposes `nvcc` on `PATH`.

```bash
python -m experiments.grug.moe_hero_ep.launch_multiprocess \
  --run-id mhep-fixed-multiprocess --num-steps 25 --version dev --run

python -m experiments.grug.moe_hero_ep.launch_mok \
  --run-id mhep-mok-dropless --num-steps 25 --version dev \
  --mok-package 'https://storage.googleapis.com/BUCKET/mixture_of_kittens-0.1.0-cp312-cp312-linux_ARCH.whl' \
  --run

# MoK's fully fused mode, for reading the cost of moving the shared experts out.
python -m experiments.grug.moe_hero_ep.launch_mok \
  --run-id mhep-mok-dropless-nolatent --num-steps 25 --version dev --latent-dim 0 \
  --mok-package 'https://storage.googleapis.com/BUCKET/mixture_of_kittens-0.1.0-cp312-cp312-linux_ARCH.whl' \
  --run
```

`launch_mok` and `launch_multiprocess` resolve to the same `GrugModelConfig` in every field except
`moe_implementation`, which is what makes their throughput comparable; `tests/test_moe_hero_ep.py`
asserts that directly. They do not share a process topology with `launch`: MoK drives Torch
symmetric memory and needs one rank per GPU, while the pooled-wave hero's recorded rack-scale
results were all measured at one JAX process per node. `launch_multiprocess` is therefore the
topology-matched control, and `launch` is the pooled-wave arm's own native configuration.

**Run both pooled-wave topologies and score MoK against the better of the two.** One process per
GPU is not exotic for this backend -- `moe_hero_fsdp/launch.py` and the pooled-wave
`small_scale_abl_launch.py` ladder both run one process per GPU -- but there is no *rack-scale*
pooled-wave number at four processes per node, so the sign and size of the topology effect are
unmeasured. Taking `launch_multiprocess` alone as the baseline silently accepts an unmeasured
handicap; taking `launch` alone reintroduces a topology confound. Running both costs one extra
gang and removes the question. See "Fairness controls" below for the ordering.

The metric contract is loss, tokens/s, analytic MFU, and MoE drop fraction. MoK reports zero
drops. Do not put credentials or signed query parameters in `--mok-package`, because the package
spec is part of the recorded run configuration.

#### Where the kernel source lives

`mixture-of-kittens/` is a submodule tracking `mark/mok_hacking` on
[muchanem/mixture-of-kittens](https://github.com/muchanem/mixture-of-kittens), a fork of
[cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens). The bundled wheel is
built from the pinned commit; upstream carries none of the changes the hero shape needs. Clone with
`git submodule update --init experiments/grug/moe_hero_ep/mixture-of-kittens`.

The fork's own history is the readable account of what changed. In dependency order: the routed and
shared paths are given separate token widths, the epilogue gains a routed-only mode, the LatentMoE
projections become fused kernels in `csrc/latent.cuh`, and the XLA FFI adapter in
`csrc/ffi_adapter.cuh` -- the only surface this repo calls -- is widened to carry both. Nothing is
pushed upstream.

#### Reading the metrics across a dropless and a fixed-capacity arm

`_compute_flops` prices the routed term at exactly the selected top-k assignments for every
backend, and the resulting `flops_per_token` is identical in both arms. Analytic MFU is therefore a
fixed multiple of tokens/s, and **ranking the arms by MFU is the same as ranking them by tokens/s**:
the analytic model cannot bias the head-to-head. What it does bias is the reading of either arm's
absolute MFU, and the two backends are biased in opposite directions:

- `fixed_pooled_wave_all_to_all` sizes its receiver buffers as
  `ceil(capacity_factor * assignments_per_shard / (local_experts * num_expert_waves))` and runs
  every slot, filled or not
  (`lib/levanter/src/levanter/grug/_moe/ep_fixed_pooled_wave_all_to_all.py`). Its executed routed
  FLOPs are `capacity_factor` times the analytic count and its all-to-all payload is
  `pooled_transport_capacity_factor` times the analytic rows, **independent of how many
  assignments it drops**. A capacity drop removes useful output; it does not save work or traffic.
  So dropping does *not* flatter the arm's throughput, and its analytic MFU *understates* its
  hardware utilization while *overstating* its useful work.
- MoK is dropless and executes the analytic count up to minibatch padding, so all three readings
  coincide for it.

The run logs the terms needed to reconcile them, as W&B summaries:
`throughput/routed_flops_fraction_analytic` (0.4356 at the hero shape),
`throughput/routed_capacity_multiplier_nominal`, and
`throughput/routed_transport_multiplier_nominal`. With `f` the routed share and `d` the measured
`moe/drop_fraction`:

    utilization MFU  ~= analytic MFU * (1 + f * (capacity_multiplier - 1))
    useful-work MFU  ~= analytic MFU * (1 - f * d)

At the hero shape and the README's measured 19.33% drop rate those are about +6.5% and -8.4%
against the analytic number -- both larger than the differences this comparison is trying to
resolve. Discounting only by the drop fraction, or only by capacity, is wrong in opposite
directions. Report tokens/s as the head-to-head result and carry `moe/drop_fraction` beside it.

The 25-step losses are **not** a head-to-head quality result. MoK computes 100% of the selected
top-8 assignments; the pooled-wave arm resolves roughly 80% of them at this gate. The loss column
is biased against the pooled-wave arm by an amount this run does not measure. Use the drop-free
re-eval route (see the EP ablation ladder above) for a quality claim.

#### Fairness controls and known biases

The user-level rule for this comparison is that the pooled-wave baseline is never handicapped to
flatter MoK. The audited state:

1. **Process topology (mitigated by a third arm).** Run three arms, not two:
   `launch_mok` (four processes per node, forced by Torch symmetric memory),
   `launch_multiprocess` (four processes per node, topology-matched control), and
   `launch` (one process per node, the pooled-wave arm's native and only rack-validated
   topology). Report MoK against `max(tokens/s)` over the two pooled-wave arms. The difference
   between the two pooled-wave arms is itself the measurement of the topology cost, which no
   recorded run currently provides.
2. **Capacity drops are not a throughput advantage.** See the metric section above. The pooled-wave
   arm executes ~`capacity_factor` times the analytic routed FLOPs whether or not it drops. Its
   25-step loss *is* biased against it; its throughput is not biased in its favor.
3. **The overflow counters are one-sided, and stay on.** `model.report_capacity_overflow=True` in
   every arm, but only the fixed-capacity path computes anything: the MoK branch in `model.py`
   short-circuits to `_zero_dropped_assignments()`. The pooled-wave arms therefore pay a per-layer
   `sum` over the assignment mask plus a two-element `psum` across the batch axes, 48 layers per
   step, that MoK does not. This is a real one-sided cost against the baseline. It stays on
   anyway: turning it off would delete `moe/drop_fraction`, which the metric contract needs, and
   would add a seventh key to the arm diff. Record it; do not silently remove it.
4. **Two concurrent arms land on two different racks.** A 16-node gang is bound to one NVLink
   domain and an NVL72 rack holds 18 nodes, so two concurrent gangs cannot share a rack. Rack-run
   variability is ~0.6-1% (`MOK_TREATMENT_BENCHMARKS.md`), the same order as some of the effects
   being measured. The confound is symmetric. If the measured delta lands under ~2%, re-run with
   the arms swapped between racks, or run the arms sequentially on one rack.
5. **Two concurrent 16-node gangs exceed the interactive budget.** With
   `resource_value = 1000 * accelerators + RAM_GB + 5 * CPU_cores`
   (`lib/iris/src/iris/cluster/controller/budget.py`), one training task at
   `GB200 x4 / 850g / 120 cpu` is worth 5,450 and a 16-node arm is worth ~87,200. The interactive
   budget for this user on `cw-us-east-08a` is 128,000 (`lib/iris/config/cw-us-east-08a.yaml`), so
   two arms at once total ~174,400 and any task scheduled after that point is downgraded to the
   BATCH band. The training tasks are `preemptible=True`, so a BATCH-banded retry after a
   preemption is exposed. Running the arms sequentially keeps spend under the limit and removes
   confound 4 at the same time; running them concurrently buys wall clock and accepts both.

A four-GPU node is enough for the reduced MoK distributed correctness tests and a sliced kernel
benchmark. It cannot run this exact 359.6B HERO shape: EP4 would hold 32 routed experts per device
instead of two, and the rack batch would make each symmetric workspace roughly 16 times larger.
Use the full 64-GPU rack for the end-to-end HERO loss/throughput/MFU result.

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

### Small-scale hero-shape ablations

`small_scale_abl_launch.py` runs the hero shape — pooled-wave transport, 384 experts / top-8,
hidden/2-wide experts in a hidden/2 latent, receiver/sender capacity 1.15 with 3 waves — at a
downsized width (`--size` in `d768`…`d2048`) on one GB200 rack. It fixes the batch at ~4M tokens per
step per rack to hold the pooled-wave drop dynamics, and sizes the step count from the model's
active-parameter count: `num_steps` trains `--tokens-per-active-param` (default 750) tokens per
active parameter. `--flavor ep` keeps the 64-way expert axis; `--flavor fsdp-nodrop` runs the same
shape dropless, and `--flavor fsdp-chunk4` runs it with four-chunk capacity. Both pooled-wave gates
are tunable: `--capacity-factor` (receiver) and `--transport-capacity-factor` (sender). Print the
plan without a GPU run:

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
The MoK hot/cold-placement and shared-PGLE results are in
[`MOK_TREATMENT_BENCHMARKS.md`](MOK_TREATMENT_BENCHMARKS.md).
