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
with finite loss and zero dropped assignments or router overflow. This result is for the four-node
proxy; the one-rack EP64 shape still needs its own validation. Issue
[#8077](https://github.com/marin-community/marin/issues/8077) contains the profile and experiment
record.

The post-treatment trace measured 0.777 seconds per step in the ragged kernel and 0.295 seconds in
its barriers. Removing both entirely would reach only about 19.04% MFU. Restoring `NCCL_BUFFSIZE`
from the fallback's 1 MiB setting to 4 MiB was neutral, so further conventional NCCL tuning is not
expected to close the 20% gap on this proxy.

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
  --flavor ep-ragged --ragged-all-to-all-splits-per-peer 32 \
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
