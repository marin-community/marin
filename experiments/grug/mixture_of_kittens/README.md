# Mixture of Kittens

This variant measures XLA's device-initiated ragged all-to-all kernel in the Grug expert-parallel path. It also contains a JAX implementation of the reference destination schedule.

## Configuration

- Model: d6144, 48 layers, top-4 routing, and two shared experts of width 3072.
- Mesh: One to 16 workers with four GB200 GPUs each. The default expert count is two per GPU.
- Batch: 16 sequences per GPU. The global batch scales from 64 on one worker to 1,024 on 16 workers.
- MoE backend: `ragged_all_to_all` with a receiver capacity factor of 1.0.
- Precision: BF16 compute with float32 parameters.
- Output: W&B and Finelog metrics plus a five-step XProf capture. The run writes no checkpoint.

Both arms install the hash-pinned JAX `0.11.1.dev20260809` CUDA 13 wheels. This nightly pins XLA `7c3dd1936addd297d7c6fa46f6183986fc4160c3`, which includes the device-initiated kernel. Stable Marin jobs keep the repository JAX pin.

Each run requires one XLA implementation:

| Implementation | XLA flags |
| --- | --- |
| `one-shot` | One-shot copies and the NCCL device barrier on; device kernel and NCCL fallback off |
| `device` | One-shot path, NCCL barrier, and NCCL fallback off; symmetric buffers for `raggedalltoall` and device kernel on |

The launcher replaces existing values for all controlled flags. It keeps unrelated `XLA_FLAGS` values. Both arms disable the latency-hiding scheduler and use one collective-overlap stream. These common settings keep the first backward pass finite at larger expert banks.

The broad NCCL switch stays off because it can change other collectives. The device arm requests symmetric allocation only for ragged all-to-all. It does not use `xla_gpu_ragged_all_to_all_mode=symmetric`, which selects a different Put and Signal path. The device arm disables the one-shot path and NCCL fallback, so missing device-kernel requirements cause a clear error.

## Local checks

```bash
uv run pytest -n 0 tests/test_mixture_of_kittens.py
uv run pytest -n 0 tests/test_grug_variant_contracts.py
```

The first test checks route ordering, rank offsets, 256-row expert padding, JIT tracing, overflow reporting, and runtime flags.

## Launch

Print each plan before submission:

```bash
uv run python -m experiments.grug.mixture_of_kittens.launch \
  --run-id mok-jax-002-one-shot-25 \
  --implementation one-shot \
  --num-nodes 1 \
  --num-steps 25 \
  --version 2026.08.10

uv run python -m experiments.grug.mixture_of_kittens.launch \
  --run-id mok-jax-002-device-25 \
  --implementation device \
  --num-nodes 1 \
  --num-steps 25 \
  --version 2026.08.10
```

Submit each arm through Iris with a different `IRIS_PORT_JAX`. Use the same code snapshot, topology, step count, and model overrides for both arms.

Start with one four-GPU worker. Expand the same pair to two workers only after both local arms complete.

## Records

The [task logbook](../../../.agents/logbooks/8108-mixture-of-kittens.md) contains commands and results. Issue [#8108](https://github.com/marin-community/marin/issues/8108) is the coordination record.
