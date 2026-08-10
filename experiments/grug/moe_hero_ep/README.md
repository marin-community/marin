# Grug MoE EP Hero

This self-contained variant is the selected one-rack EP64 configuration for GB200 NVL72.

## Configuration

- Model: d6144, 48 layers, 192 routed experts of width 6272, top-4 routing, latent width 3072, and
  two shared experts of width 3072. This is 546.292 B total parameters and 24.680 B active per token.
- Attention: 48 heads, 12 local and 6 global KV heads, head dimension 128, sequence length 4096,
  sliding window 512, and every sixth layer full-causal. SConv and fused RoPE are on.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each. Three whole experts
  land on each device.
- Batch: 1024 sequences.
- Router: top-4 quantile balancing with next-step, stop-gradient expert biases and no auxiliary
  balancing loss.
- MoE backend: `fixed_all_to_all` with gather dispatch, structured custom VJPs, and capacity
  factor 1.33.
- Optimizer: MuonH, with its state offloaded to pinned host memory.
- Runtime: GPU command buffers off and `cuda_async`. The fixed and ring backends use latency hiding
  and collective overlap limit 4. Ragged all-to-all disables latency hiding and uses overlap limit 1;
  the default scheduler corrupts its first backward at larger expert banks, and four-way overlap was
  24% slower in a same-host clean timing comparison. Auto-PGLE is off for the process-per-GPU
  topology because a node cannot run four concurrent profiling sessions. Ragged runs can split each
  peer transfer into multiple contiguous updates with `--ragged-all-to-all-splits-per-peer`; this
  changes kernel parallelism without changing routing capacity or tensor layout.
- Output: Metrics only. This throughput run does not write a checkpoint.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and
`expert` axes. The expert axis stays sharded during Newton-Schulz.

## Result

A five-step, full-watch gate completed at capacity factor 1.33. All 76 norm fields were finite on
each step. Capacity factor 1.34 failed from a CUDA OOM. The highest confirmed routing cell capacity
is 1,829. Capacity factor 1.33 keeps a small margin below that limit.

The matched 200-step result used capacity factor 1.30 and automatic PGLE. Its last-50 mean was
262,683 tokens/s with 3.9642% drops. Its drop-adjusted rate was 252,271 tokens/s, and its mean loss
was 3.2417.

The FSDP chunk-4 reference is a smaller model. It uses 128 full-width experts of width 3072 at
top-4, with 359.6 B total parameters and 20.9 B active parameters. Thus, the EP and FSDP rates do
not measure a same-model parallelism comparison.

### Ragged all-to-all proxy

The clean four-node d6144/L48/E48 process-per-GPU proxy measured 11.2834% MFU with one ragged update
per peer. XProf attributed 41.3% of its step to XLA's custom symmetric-memory peer-write kernel;
each call launched only 16 blocks, one per peer. The surrounding NCCL barriers were much smaller,
so NCCL protocol, channel, NVLS, and SHARP settings do not tune the dominant transfer.

Splitting every peer slice into 32 updates raises that kernel to 512 cooperating blocks. On the same
topology class, clean steps 5-24 reached 18.0770% MFU and 49,674.7 tokens/s, a 60.2% throughput gain,
with finite loss and zero dropped assignments or router overflow.

The remaining gap to fixed all-to-all was local expert compute, not communication. A matching
profile measured 8.965 seconds per step in the ragged `moe_up_down` scope versus 4.502 seconds for
fixed all-to-all; ragged communication was 0.886 seconds faster. The `ep-ragged-cute` flavor keeps
the split-capable ragged transport but uses the existing QuACK/CuTe SM100 grouped GEMMs for the
activation path. Its exact clean run reached 19.6291% mean MFU, 19.7000% median MFU, and 53,939.7
tokens/s with finite loss and zero routing drops.

CuTe XProf measured 18.807 seconds of device-kernel time per step, 1.883 seconds below the prior
ragged trace. The two retained Pallas weight-gradient contractions account for 2.084 seconds per
step and use 128x128 tiles at 18.75% theoretical occupancy. Exact one-GB200 tile sweeps found a
7.54% improvement for the larger contraction and no improvement for the smaller one; combined,
that predicts only 19.74% MFU. The trace already uses NCCL's STMC/NVLS multicast kernels for its
largest all-gathers, and restoring `NCCL_BUFFSIZE` from 1 MiB to 4 MiB was neutral.

Enabling `FUSION,CUSTOM_CALL` command buffers completed cleanly but regressed the exact proxy to
19.3369% mean MFU and 53,136.9 tokens/s. Command buffers remain disabled. Even stacking the best
weight-gradient tile result perfectly with that arm predicts only 19.45% MFU, so the bounded
runtime and kernel-tile changes do not clear the 20% target.

The four-node proxy and one-rack EP64 shape both route 65,536 tokens and hold three experts per GPU,
so the larger run does not improve the local grouped-GEMM shapes or imply a larger-batch uplift.
The full one-rack EP64 shape still needs its own validation. Issue
[#8077](https://github.com/marin-community/marin/issues/8077) contains the profiles and complete
experiment record.

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

Ragged runs also accept `--ragged-all-to-all-splits-per-peer`. It is a transport-parallelism knob,
not a model-shape sweep: increasing it preserves all routed tokens and divides each peer's contiguous
slice into more XLA ragged updates. The default is one; the measured four-node treatment used 32.

`ragged_weight_grad_benchmark.py` screens the two exact local weight-gradient shapes on one GB200.
Submit the whole job tree directly to the GB200 cluster; a coordinator on the federated Marin
cluster cannot send only its child to a peer:

```bash
uv run iris --cluster=cw-us-east-08a job run --no-wait \
  --cpu 1 --memory 2G --disk 5G --priority interactive --extra cpu \
  --job-name ragged-weight-grad-benchmark-coord \
  -- python -m experiments.grug.moe_hero_ep.ragged_weight_grad_benchmark \
    --version dev --run --max-concurrent 1
```

The benchmark writes compile time, five-run steady-state time, throughput, tile parameters, device
provenance, and output deviation to `results.json` in its artifact.

Three quantities move independently, which sets what a sweep can afford on one rack:

- Active routed neurons are top-k multiplied by width.
- Parameters are expert count multiplied by width.
- The all-to-all buffers are tokens multiplied by top-k.

Width appears in the first two and not the third, thus width is the cheap way to buy active compute
and top-k is the expensive way. Six buffers scale with top-k, and one of them is float32: top-6
costs 30.75 GiB against 20.50 GiB for top-4 at this shape.

The selected E192 model fits at expert width 6272 and capacity factor 1.33. Width 6400 fails at
capacity factor 1.30. The full experiment record contains the size and capacity searches.

## Launch

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

python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-ragged-ep16-split32 \
  --dp-racks 1 --ep-nodes 4 \
  --num-steps 25 --schedule-steps 17652512 \
  --batch-size 256 --num-experts 48 \
  --flavor ep-ragged-cute --ragged-all-to-all-splits-per-peer 32 \
  --watch-interval 0 --eval-every 0 --profile-steps 0 --no-save-checkpoints \
  --version 2026.08.10
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
    --run-id "$run_id" --dp-racks 1 --num-steps 200 --version 2026.08.05 --run
```

W&B uses the `WANDB_PROJECT` environment variable, or project `marin_moe` when it is unset, with
group `moe-hero-ep` and the supplied run ID. The run output includes the durable W&B metrics
artifact. Give each concurrent gang its own `IRIS_PORT_JAX`: rank 0 binds and registers that port
for the JAX coordinator, and the default 8476 is shared by every run on the cluster.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
