# EP25 Direction 2 Relay Commands

Run these from `/home/marin/projects/marin/.worktrees/ep25-d2-bakeoff`, one at a time. Wait for the outer Iris job and its nested `grug-train-*` job to finish before running the next command. These are EP4 transport/toolchain smokes because a one-replica allocation has four GPUs; the rack runs restore EP64 for fixed and ragged.

The commands use the compatible Iris environment from the direction-4 worktree while forcing all repository imports and the submitted bundle to come from this worktree.

## Fixed A2A + gather smoke

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  /home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris \
  --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-fixed-gather-smoke3-20260725 \
  -e RUN_ID ep25d2-fixed-gather-smoke3-20260725 \
  -e SCALE_GPU_IMAGE nvcr.io/nvidia/jax:26.06-py3 \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_URI s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 \
  -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-fixed-gather-smoke3-20260725.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-ngc2606-dev --run
```

## Ragged A2A, one-shot disabled smoke

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u SCALE_A2A_FIXED -u SCALE_A2A_GATHER_DISPATCH -u SCALE_A2A_CHUNKS -u SCALE_A2A_NO_BARRIER \
  IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  /home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris \
  --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-ragged-smoke3-20260725 \
  -e RUN_ID ep25d2-ragged-smoke3-20260725 \
  -e SCALE_GPU_IMAGE nvcr.io/nvidia/jax:26.06-py3 \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_URI s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194 \
  -e XLA_FLAGS "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false --xla_gpu_nccl_termination_timeout_seconds=600" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-ragged-smoke3-20260725.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-ngc2606-dev --run
```

## ring_cute EP4 smoke

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u SCALE_A2A_FIXED -u SCALE_A2A_GATHER_DISPATCH -u SCALE_A2A_CHUNKS -u SCALE_A2A_NO_BARRIER \
  IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  /home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris \
  --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-ring-cute-ep4-smoke3-20260725 \
  -e RUN_ID ep25d2-ring-cute-ep4-smoke3-20260725 \
  -e SCALE_GPU_IMAGE nvcr.io/nvidia/jax:26.06-py3 \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_URI s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ring_cute \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-ring-cute-ep4-smoke3-20260725.metrics \
  -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-ngc2606-dev --run
```
