# BF16 EP64 MFU handoff

## TL;DR

The requested result has not been reached. The best matched-architecture BF16
screen is 23.286% steady median MFU over steps 5-19. The longest qualified BF16
run predating the final overlap flag reached 22.299% tail-30 median MFU over 120
steps with 1.71% mean aggregate assignment drop in that tail and finite loss
falling to 5.995.

The clean handoff branch is `research/rav/7201-ep64-drop3-handoff`. It starts
from `24ee8609095022342afd9f898a3e3e55179fca39`, the last pushed receiver-ECHO
kernel snapshot, and adds only the exact submission recipe, one-node profiling
support, and this handoff. Do not recover the later dirty worktree wholesale:
it mixes rejected FP8, host-offload, attention-pipeline, fused-QKV, and Iris
diagnostic changes.

No EP64 experiments are active. The final custom clone-weight-gradient A/B was
terminated at the user's request before it produced usable throughput data.

## Goal and acceptance gate

The next owner should continue toward:

- at least 25% steady BF16 MFU on one GB200 rack;
- 64 GPUs as 16 four-GPU GB200 nodes on `cw-us-east-08a`;
- exact aggregate post-ECHO assignment drop at or below 3%;
- the original shared-expert intermediate width of 5,120;
- finite, falling loss over at least 120 matched steps;
- drop comparisons over a tail window at the same fraction of the learning-rate
  schedule;
- a final profile from one node only, with HLO enabled and CUDA command buffers
  disabled.

FP8 is excluded by explicit user direction. The 21,504-wide shared-expert
series is an architecture option, not evidence for this target.

Do not use a 20-step final drop value to claim compliance. The learning-rate
schedule is defined over `num_train_steps`; step 19 in a 20-step run is not
comparable to step 19 in a 120- or 350-step run. Prefer the last 30-50 steps of
a 120+ step run.

## Locked model and parallelism

| Setting | Value |
|---|---:|
| Layers | 48 |
| Hidden dimension | 5,120 |
| Routed experts | 256 |
| Experts selected per token | 4 |
| Routed expert intermediate | 2,048 |
| Shared expert intermediate | 5,120 |
| Global batch | 1,024 sequences |
| Sequence length | 4,096 |
| Sliding window | 512 |
| Optimizer | MuonH with SYRK |
| Precision | BF16 |
| Expert parallel axis | 64 |
| Hardware | 64 GB200 GPUs, 16 nodes, 4 GPUs/node |
| JAX processes | 64, one process per GPU |

The global batch is sharded. Each GPU does not receive 1,024 complete
sequences. With 64 batch shards, each GPU owns 16 sequences, or 65,536 tokens,
before top-4 routing.

## Best implementation

The promoted implementation uses:

- `gpu_fa4_cute` flash attention;
- receiver-ECHO routing that preserves the router-selected expert and moves
  execution to a clone shard when the home receiver is full;
- a static BF16 token `all_to_all` envelope;
- two sender-balanced pipeline chunks that preserve the one-chunk acceptance
  set;
- sparse cloned-expert weight movement;
- Sonic dispatch, combine, and slot-gather kernels;
- QuACK SM100 expert GEMMs and grouped weight gradients with 256-by-256 tiles;
- quantile balancing;
- XLA's latency-hiding scheduler;
- `xla_gpu_experimental_parallel_collective_overlap_limit=4`;
- token padding of two experts per sender-destination bucket.

The fixed token transport remains the largest bottleneck. Receiver-ECHO avoids
the sender-local fixed-capacity semantics that motivated the original kernel:
capacity is enforced in receiver pools, and a logical assignment stays on its
selected expert even when execution moves to a clone shard.

Relevant code:

- `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`:
  receiver-ECHO planning, fixed and compact transports, sparse clone-weight
  exchange, sender-balanced chunking, and custom VJPs.
- `lib/levanter/src/levanter/grug/_moe/sonic.py`: dispatch/combine/slot-gather
  and clone-weight-gradient kernels.
- `lib/levanter/src/levanter/grug/_moe/sonic_cute.py`: QuACK expert MLP wiring.
- `lib/levanter/src/levanter/grug/_moe/quack_moe_cute.py`: SM100 grouped GEMM
  launchers and production tile settings.
- `experiments/grug/dispatch.py`: nested-task environment forwarding and
  one-process-per-device dispatch.
- `experiments/grug/moe/launch_cw_scale.py`: environment-to-model/trainer
  construction.
- `experiments/grug/moe/train.py`: analytic MFU and exact aggregate drop
  logging.

## Submit the current best BF16 screen

Use
[`run_best_bf16_ep64.sh`](run_best_bf16_ep64.sh). It requires a unique run ID,
an unused coordinator port, and an existing `WANDB_API_KEY` in the shell. Do
not put the key in a file, command history, echo-log entry, or commit.

```bash
export WANDB_API_KEY=...  # obtain through the team's secret channel

.agents/projects/7201-ep64-drop3/run_best_bf16_ep64.sh \
  ep64-handoff-control-$(date -u +%Y%m%d-%H%M) \
  32401 \
  20
```

Use a fresh coordinator port for every concurrent job. The script submits with
zero retries. A first-rank failure should be investigated directly; gang
retries can obscure the root cause and previously triggered #7650.

For a 120-step qualification, change only the run ID, coordinator port, and
step count:

```bash
.agents/projects/7201-ep64-drop3/run_best_bf16_ep64.sh \
  ep64-handoff-qual120-$(date -u +%Y%m%d-%H%M) \
  32402 \
  120
```

The script uses W&B project `marin-community/rav_moe`, the
`cw-us-east-08a` target cluster, 16 replicas, four GB200 GPUs per replica, and
four supervised processes per task.

### Job operations

```bash
# Find the outer and nested jobs.
.venv/bin/iris --cluster=marin job list --limit 100 | rg '<run-id>|STATUS'

# Follow recent logs.
.venv/bin/iris --cluster=marin job logs /rav/<run-id> \
  --tail --max-lines 5000

# Save a large local log before grepping it.
.venv/bin/iris --cluster=marin job logs /rav/<run-id> \
  --no-tail --max-lines 1000000 > /tmp/<run-id>.log

# Stop the outer job; Iris propagates termination to the nested train job.
.venv/bin/iris --cluster=marin job stop --exact /rav/<run-id>
```

When a rank reports the first `RESOURCE_EXHAUSTED`, `CUDA_ERROR`, traceback, or
coordinator abort, treat later connection-refused and coscheduled-task failures
as teardown until the first error is understood.

## W&B analysis

The run summary includes startup and warmup samples. For short screens, report
the median of `throughput/mfu` from step 5 onward in addition to
`throughput/p50_mfu`.

```python
import statistics

import wandb

run = wandb.Api().run("marin-community/rav_moe/<run-id>")
rows = run.history(samples=10_000, pandas=False)
steady = [row for row in rows if row.get("global_step", -1) >= 5]
print("steady MFU", statistics.median(row["throughput/mfu"] for row in steady))
print(
    "aggregate drop",
    statistics.median(row["train/router/capacity_overflow_rate_mean"] for row in steady),
)
print("final loss", steady[-1]["train/loss"])
```

`train/router/capacity_overflow_rate_mean` is the goal metric: exact aggregate
post-ECHO dropped assignments divided by routed assignments. The
`capacity_overflow_rate_max` series is the worst layer, a useful diagnostic but
not the stated aggregate 3% gate.

## Profile one node

The handoff profiler can capture processes 0-3, which are the four GPUs on one
node. `PROFILE_STEPS` activates profiling, HLO proto is enabled by the launcher,
and the script appends `--xla_gpu_enable_command_buffer=`. Do not compare this
profile run's throughput directly with normal command-buffer-enabled runs.

```bash
PROFILE_START=8 PROFILE_STEPS=4 \
  .agents/projects/7201-ep64-drop3/run_best_bf16_ep64.sh \
  ep64-handoff-profile-$(date -u +%Y%m%d-%H%M) \
  32403 \
  14
```

The job prints `XProf profile:` after upload. Profiles use a 30-day TTL store.
The primary existing profile expires around 2026-08-26:

- [Hosted v156 XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fep64-d5120-sh5120-pad2-qb-pipe2-pgle-overlap4-profile-bf16-v156-20260727-0320)
- `s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep64-d5120-sh5120-pad2-qb-pipe2-pgle-overlap4-profile-bf16-v156-20260727-0320`
- current-machine download:
  `/tmp/marin-profiles/ep64-v156/steps-8-to-12/s29txs64.xplane.pb`
- current-machine summary:
  `/tmp/marin-profiles/ep64-v156/summary.json`

To download and summarize a future profile:

```bash
uv run --with xprof --with protobuf python \
  lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/rav_moe/<run-id> \
  --download-root /tmp/marin-profiles \
  --breakdown-mode exclusive_global \
  --xplane-output-dir /tmp/marin-profiles/<run-id>/xprof-tables \
  --output /tmp/marin-profiles/<run-id>/summary.json

uv run python lib/marin/tools/profile_summary.py report \
  --summary /tmp/marin-profiles/<run-id>/summary.json \
  --output /tmp/marin-profiles/<run-id>/report.md
```

## Current measured frontier

| Change | Run | Result | Decision |
|---|---|---:|---|
| Corrected receiver-ECHO control | [v132](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-weightfix-native-v132-20260726-2143) | 20.932% p50 | Exact post-offset-fix baseline |
| QuACK 256-by-256 tiles | [v134](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-quack256-v134-20260726-2208) | 21.794% p50 | Keep; +0.861pp |
| XLA latency hiding | [v136](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-quack256-lhs-v136-20260726-2221) | 22.393% p50 | Keep; +0.599pp |
| Exact two-chunk token pipeline | [v140](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-quack256-lhs-pipe2-v140-20260726-2340) | 22.640% p50 | Keep; +0.247pp |
| 120-step BF16 control | [v143](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-bf16-120-v143-20260727-0040) | 22.299% tail-30 median | Stable reference |
| AutoPGLE screen | [v152](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-pgle-bf16-v152-20260727-0240) | 22.801% p50 | CUPTI profiles were empty; keep only for matched v153 comparison |
| Overlap limit 4 | [v153](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-pgle-overlap4-bf16-v153-20260727-0249) | 23.286% steps 5-19 median | Best short BF16 screen; +0.427pp over v152 |

The v153 W&B summary p50 is 23.229%. Its steps-5-to-19 median is 23.286%.
Loss fell to 8.283. Its last-five aggregate drop averaged 0.626%, but that
20-step tail is not a qualification measurement.

The 120-step v143 tail from steps 90-119 reached 22.299% median MFU, 1.711%
mean aggregate drop, and finite loss falling to 5.995. It does not include the
overlap-limit-4 gain, so a clean 120-step v153 recipe remains unrun.

## v156 profile evidence

The device-complete start-to-start timeline measured 9.545 seconds per step:

| State | Share |
|---|---:|
| Compute only | 61.21% |
| Compute/communication overlap | 27.64% |
| Communication only | 10.29% |
| Idle | 0.85% |

Aggregate kernel time was 69.75% compute and 30.25% communication. SendRecv was
22.7% of aggregate kernel time. Perfectly hiding the remaining exposed
communication would move the 23.286% control to about 25.7%, so reaching 25%
requires roughly 0.65 seconds per step of exposed-time reduction.

The largest exposed regions were fixed token dispatch/combine SendRecv,
attention all-gathers, and sparse clone-weight work. Generic sparse
clone-weight pack adjoints consumed about 126 ms/step. Host `np.asarray` spans
were asynchronous device synchronization, not a one-second logging bottleneck.

## Most useful unfinished lead

`_pack_sparse_clone_weights` already has a custom Sonic reduction adjoint behind:

```bash
SCALE_A2A_CLONE_SONIC_WEIGHT_GRAD=1
```

The v153 control did not enable it. On the exact local shape
(`local_experts=1`, `packed_count=17`), a four-GB200 microbenchmark measured:

| Gradient | Generic adjoint | Sonic block 512 | Speedup | Max error |
|---|---:|---:|---:|---:|
| W2 | 0.5456 ms | 0.1684 ms | 3.24x | 0 |
| W13 | 0.9535 ms | 0.2060 ms | 4.63x | 0 |

Block 512 was faster than block 1,024 for the real shape. Full-rack v163 was
terminated after step 1 during the handoff and has no usable performance
result. This is the first A/B to rerun, unchanged except for the flag.

If it wins and reduces the temporary arena, the next bounded retry is fused QKV
plus this adjoint. Fused QKV was 11.5-12.8% faster for local QKV forward and
backward with zero output/weight-gradient error and a 4.77e-7 input-gradient
maximum difference, but the rack executable requested an 81.14-GiB temporary
arena and failed after step 2.

## Rejected or parked paths

### Semantics and capacity

- The original sender-bucket fixed `all_to_all` is fast but enforces capacity
  per sender/expert bucket. It is more drop-prone than receiver pooling under
  imbalance. Receiver-ECHO supersedes it for the fidelity-constrained work.
- Compact receiver-pooled ragged transport preserved accepted assignments but
  reached only [19.862%](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-weightfix-ragged-v133-20260726-2143),
  1.07pp below native fixed A2A.
- Padding 1 reached [23.335% steps-5-to-19 median](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad1-qb-pipe2-pgle-overlap4-bf16-v157-20260727-0339),
  only +0.049pp over padding 2, while aggregate drop averaged 3.49% over
  steps 15-19. Reject under the 3% gate.
- Increasing capacity factor is expensive. A prior 350-step sweep moved
  aggregate drop below 3% around `cf=1.15` but cost roughly 3.4pp MFU.
- Same-step spill can re-offer an overflowed assignment to another expert
  already in the token's top-k set. It is cheap because static expert GEMMs
  already process capacity-sized buffers, but it changes assignment placement.
  Top-4 provides at most three alternatives. Use it only if it enables a
  smaller, faster transport envelope and qualify loss.

### Scheduling and rematerialization

- `xla_gpu_experimental_parallel_collective_overlap_limit=2` regressed v138 to
  21.910%. Limit 4 is the validated setting.
- Manual PGLE matched only 217 of 535 profiled instructions and reached
  [23.051%](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-manualpgle-overlap4-bf16-v158-20260727-0358),
  0.235pp below v153. Do not reuse `/tmp/marin-profiles/ep64-v156/profile-steps.pb`.
- Scan unroll 2 reached
  [22.401%](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-scan2-bf16-v151-20260727-0240);
  reject.
- Host-offloading dispatch activations required a 135-GiB pinned-host arena per
  process and, after raising the host limit, reached
  [19.694%](https://wandb.ai/marin-community/rav_moe/runs/ep64-d5120-sh5120-pad2-qb-pipe2-offload-bf16-v155-20260727-0318);
  reject for d5120.
- `save_moe` requested roughly 374 GiB/GPU. It cannot fit.

### Transport kernels

- Destination-grouped MNNVL direct writes reached 19.907%. The prototype waits
  for all peers and copies the persistent fabric buffer into an XLA-owned
  output. A useful replacement must fuse peer arrival with consumption.
- A serial 64-round `ppermute` pipeline exhausted NCCL/XLA memory. A future
  round-robin peer pipeline must bound live state and start expert work as
  chunks arrive.
- Concatenating W13 and W2 clone-weight exchanges regressed 20.51% to 20.21%;
  the larger materialization outweighed one fewer collective.
- Raising NCCL CTAs to 40, 48, or 64 exhausted HBM on the first token A2A.
- Rowwise FP8 transport reduced wire bytes but conversion cost erased the win.
  FP8 is excluded regardless.

### Attention and memory

- Pipeline depth 4, corrected sequential QKV pipelining, and fused QKV all
  reached step 2 and then failed the next executable allocation:

  | Run | Allocation |
  |---|---:|
  | v160 pipeline depth 4 | 82.03 GiB |
  | v161 corrected QKV pipeline | 82.18 GiB |
  | v162 fused QKV | 81.14 GiB |

- The corrected QKV pipeline bug was real: it gathered only `data` while the
  contraction dimension is sharded over `("data", "expert")`, leaving 80 of
  5,120 rows on EP64. The correction matched the reference exactly on four
  GB200 GPUs. It is not in the clean handoff because no rack configuration fit.

### Architecture changes outside the goal

- Shared intermediate 21,504 reached 25.50% tail MFU with about 2% aggregate
  drop over 200 steps. It changes the model's always-on capacity by 4.2x and is
  excluded from the locked target.
- A d6144 4-of-128 run reported 24.594% but dropped 9-13% of assignments. Its
  ~23.9% compliant value is projected, not measured.
- d6144 and wider routed-expert candidates repeatedly exceeded the executable
  or NCCL memory budget. Host offload may be useful only as a fit mechanism,
  not as a d5120 speed mechanism.

### FP8 record

Full FP8 reached 26.775% p50 for 120 steps with finite loss, but it changes
arithmetic, showed backward underflow, and was explicitly removed from scope.
Do not resume it.

## Infrastructure and logistics

- Iris config: `.venv/bin/iris --cluster=marin`.
- Target cluster: `cw-us-east-08a`.
- Kubernetes context used for node-level development:
  `marin-us-east-08a_US-EAST-08A`, namespace `iris`.
- Existing development pod at handoff:
  `iris-rav-dev-gpu-rav-gpu-b200-0-7303c2c9-0`, four GB200 GPUs, node
  `s1nrxs64`, working directory `/app`.
- Local kubeconfig path on the current machine:
  `/tmp/coreweave-iris-08a`. It contains credentials and must not be committed
  or pasted into echo.

Read-only pod check:

```bash
KUBECONFIG=/tmp/coreweave-iris-08a \
  kubectl --context marin-us-east-08a_US-EAST-08A -n iris \
  get pod iris-rav-dev-gpu-rav-gpu-b200-0-7303c2c9-0 -o wide
```

Interactive shell:

```bash
KUBECONFIG=/tmp/coreweave-iris-08a \
  kubectl --context marin-us-east-08a_US-EAST-08A -n iris \
  exec -it iris-rav-dev-gpu-rav-gpu-b200-0-7303c2c9-0 -- bash
```

Code copied into `/app` is not durable. Commit and push changes from the Marin
worktree after copying results back.

JAX's "different incarnation" abort is tracked in
[#7650](https://github.com/marin-community/marin/issues/7650); the Iris retry
fix is [#7651](https://github.com/marin-community/marin/pull/7651) and was still
open at handoff. The error means a global process ID registered with an
existing coordinator session under a new process incarnation, normally because
an old pod and its retry overlap. Use fresh run IDs and coordinator ports, keep
experimental retries at zero, and inspect the first failed attempt.

W&B initialization failures on rank 0 cascade into coordinator failures on
every other rank. Confirm `WANDB_API_KEY` is forwarded before diagnosing later
connection errors.

Checkpoint and watch callbacks caused unrelated long-run failures in early
experiments. Keep `SCALE_CHECKPOINTS=local`,
`SCALE_DISABLE_CHECKPOINT=1`, and `SCALE_WATCH_INTERVAL=0` for disposable
throughput runs.

## References

- Coordinating issue:
  [#7201](https://github.com/marin-community/marin/issues/7201)
- Capacity/drop and scheduling study:
  [#7279](https://github.com/marin-community/marin/issues/7279)
- Recent EP64 comparison:
  [#7201 comment 5084895357](https://github.com/marin-community/marin/issues/7201#issuecomment-5084895357)
- Drop methodology, spill, and collective-volume result:
  [#7279 comment 5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846)
- Initial 17% reproduction:
  [#7201 comment 5036748495](https://github.com/marin-community/marin/issues/7201#issuecomment-5036748495)
- Initial reproducibility branch:
  [`rav/ep-2` at `fe21ea495`](https://github.com/marin-community/marin/tree/fe21ea495)
- Receiver-ECHO snapshot:
  [`24ee86090`](https://github.com/marin-community/marin/tree/24ee8609095022342afd9f898a3e3e55179fca39)
- Detailed experiment ledger:
  [`research.md`](research.md)
- Append-only task log:
  [`7201-ep64-mfu.md`](../../logbooks/7201-ep64-mfu.md)
- NVIDIA Blackwell MoE report:
  [arXiv:2603.07685](https://arxiv.org/abs/2603.07685)
- NVIDIA ECHO implementation:
  [Megatron-LM PR 2368](https://github.com/NVIDIA/Megatron-LM/pull/2368)
- DeepEP:
  [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- Loong-Megatron ECHO integration:
  [baidu-baige/Loong-Megatron PR 7](https://github.com/baidu-baige/Loong-Megatron/pull/7)

The external sources converge on a persistent, device-initiated transport that
uses device-side counts and starts expert computation as data arrives. Marin's
current fixed NCCL A2A has static shapes and exact receiver semantics but does
not fuse arrival, permutation, and expert consumption.

## Communication rules

The user is handling any PR. Do not open a PR or post to #7201/#7279 unless the
user explicitly asks. Agent-authored issue comments must begin with `🤖`.

The shared echo-log project is `grug-moe-mfu`. Read it before resuming:

```bash
~/.claude/skills/echo-log/echo_log.py recent \
  --days 30 --project grug-moe-mfu
```

Echo entries #458, #531, #543, #552, #559, #578, #588, #595, #618, #621,
and the final handoff entry contain the decision milestones. The branch and
this document are authoritative if a transient echo note conflicts with the
committed record.
