# Grug MoE EP Hero

This self-contained variant is the one-rack EP64 baseline for GB200 NVL72.
It starts from the FSDP hero at PR 7876 and changes only the parts that EP64 requires.

## Current Gate

`MHEP-001` completed 25 steps with the existing JAX ragged all-to-all backend.
`MHEP-002` completed 25 steps with fixed-capacity all-to-all dispatch.
`MHEP-003` completed 25 steps with gather dispatch.
`MHEP-004` completed 25 steps with structured gather transposes that avoid scatter in the backward pass.
`MHEP-005` completed 25 steps with a fixed dispatch capacity factor of 1.0625. It was slower than `MHEP-004`.
`MHEP-006` completed 25 steps with receiver-ECHO. It reduced drops but lost 5.9132 median MFU points.
`MHEP-007` completed 25 steps with three spill attempts. It reduced drops but lost 0.0401 median MFU points
and added 106 net backend and dispatch lines. `MHEP-004` is the selected minimal stack.
`MHEP-008` is the final 200-step gate for that stack.

## Configuration

- Model: d5120, 48 layers, 256 routed experts, top-8 routing, and one shared expert.
- Attention: 40 heads, 10 KV heads, sequence length 4096, and sliding window 2048.
- Mesh: 64-way expert parallelism across 16 workers with four GB200 GPUs each.
- Batch: 1024 sequences.
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
  --run-id MHEP-008-final-200 \
  --num-steps 200 \
  --version dev
```

Submit the one-rack gate through the Marin Iris controller:

```bash
run_id="MHEP-008-final-200"
iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 21600 \
  --job-name "${run_id}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe_hero_ep.launch \
    --run-id "$run_id" --num-steps 200 --version dev --run
```

W&B uses `marin-community/marin_moe`, group `moe-hero-ep`, and the supplied run ID.
Use `WANDB_PROJECT` to select a different project.

## Result Record

The experiment record is in [`.agents/logbooks/7279-moe-hero-ep.md`](../../../.agents/logbooks/7279-moe-hero-ep.md).
Issue [#7279](https://github.com/marin-community/marin/issues/7279) is the coordination record.
