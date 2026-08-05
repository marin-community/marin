# Grug MoE EP Hero

This self-contained variant is the one-rack EP64 baseline for GB200 NVL72.
It starts from the FSDP hero at PR 7876 and changes only the parts that EP64 requires.

## Result

`MHEP-004` is the selected 25-step stack. Receiver-ECHO and three-choice spill reduced drops but did not
increase MFU, so they are not in the final code. `MHEP-008` passed the 200-step gate with 23.6969% median
MFU, 382,902 last-sample tokens/s, 7.4113% final MoE drop rate, and 3.3119 final loss.

## Model shapes

`--shape` selects which model shape runs on the EP64 mesh:

- `ep` (default): the native EP hero shape below.
- `fsdp`: the `experiments/grug/moe_hero_fsdp` shape, for a transport comparison at one model shape.

The `fsdp` shape keeps every field of the FSDP hero model except the two that expert parallelism
cannot honor. `sonic_cute` is a local grouped-GEMM backend with no EP collectives, thus the run uses
`fixed_all_to_all`. `moe_mlp` rejects `expert_chunks` greater than one when the expert axis is
larger than one, because EP shards the expert bank instead of a gather of it. As a result, both
sharding strategies keep the same analytic FLOP count, and their MFU values are directly comparable.
The `fsdp` shape also keeps the FSDP hero's host offload of the optimizer state.
`tests/test_moe_hero_ep.py` fails if the two model specs drift apart in more fields.

Capacity is the one behavior that the transport changes. The two backends enforce it differently,
and EP discards more: the measured pair below drops 9.9683% against 1.8779%. MFU credits every
assignment in the analytic FLOP count, thus a run that drops more is credited for work it did not
do. Read the drop fraction with the MFU value.

## Measured result at the FSDP shape

One rack, 25 steps, batch 1024, one hour apart on `cw-us-east-08a`.

| | EP64 (`--shape fsdp`) | FSDP64 (`moe_hero_fsdp --dp-racks 1`) |
| --- | --- | --- |
| median MFU | 27.7544% | 19.3951% |
| tokens/s (last sample) | 316,473 | 235,125 |
| step time (last sample) | 13.2533 s | 17.8386 s |
| MoE drop fraction | 9.9683% | 1.8779% |
| final loss | 6.0498 | 6.0754 |

EP measures 8.3593 percentage points more median MFU, or 43.1% relative. A first-order correction
for the assignments each arm skipped leaves EP about 6 points ahead. Both runs took 26 MFU samples
that include compile and warmup, so their deviations are 6.5623 and 5.5456. Use the medians, and
run 200 steps before you trust a smaller difference.

## Configuration

- Model: d5120, 48 layers, 256 routed experts, top-8 routing, and one shared expert.
- Attention: 40 heads, 10 KV heads, sequence length 4096, and sliding window 2048.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each.
- Batch: 1024 sequences.
- Router: top-8 quantile balancing with next-step, stop-gradient expert biases and no auxiliary balancing loss.
- MoE backend: `fixed_all_to_all` with gather dispatch, structured custom VJPs, and capacity factor 1.0.
- Optimizer: MuonH with on-device optimizer state.
- Runtime: GPU command buffers off, `cuda_async`, PGLE off, and collective overlap limit 4.
- Output: Metrics only. This throughput run does not write a checkpoint.

The attention, shared-expert, language-model-head, and optimizer states use the combined `data` and `expert` axes.
The expert axis stays sharded during Newton-Schulz.

## Launch

Print the plan without a GPU run:

```bash
python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-008-final-200 \
  --num-steps 200 \
  --version 2026.08.02

python -m experiments.grug.moe_hero_ep.launch \
  --run-id mhep-009-fsdp-shape-25 \
  --num-steps 25 \
  --shape fsdp \
  --version 2026.08.04
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="mhep-008-final-200"
uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 21600 \
  --job-name "${run_id}-coord" \
  -e WANDB_MODE offline \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id "$run_id" --num-steps 200 --version 2026.08.02 --run
```

W&B uses project `marin_moe`, group `moe-hero-ep`, the supplied run ID, and offline mode.
The run output includes the durable W&B metrics artifact.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
