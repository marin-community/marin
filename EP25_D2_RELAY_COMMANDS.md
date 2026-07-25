# EP25 Direction 2 Rack Relay Commands

Run these from `/home/marin/projects/marin/.worktrees/ep25-d2-bakeoff`, one at a time. Wait for the outer Iris job and its nested `grug-train-*` job to finish before running the next command. Both arms use the d5120, 48-layer, 8-of-256, batch-1024, sequence-4096 operating point on one 64-GPU GB200 rack and log the dropped-assignment fraction.

The fixed arm is not repeated. Compare both arms with d1's matched fixed-path results: 20.61% p50 MFU with autodiff and 24.04% with the custom adjoint.

## Ragged A2A, one-shot disabled

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u SCALE_A2A_FIXED -u SCALE_A2A_GATHER_DISPATCH -u SCALE_A2A_CHUNKS -u SCALE_A2A_NO_BARRIER \
  IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  /home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris \
  --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-rack-ragged-120-20260725 \
  -e RUN_ID ep25d2-rack-ragged-120-20260725 \
  -e SCALE_GPU_IMAGE nvcr.io/nvidia/jax:26.06-py3 \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_URI s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194 \
  -e XLA_FLAGS "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false --xla_gpu_nccl_termination_timeout_seconds=600" \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-rack-ragged-120-20260725.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-ngc2606-dev --run
```

## ring_cute EP64

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env -u XLA_FLAGS -u SCALE_A2A_FIXED -u SCALE_A2A_GATHER_DISPATCH -u SCALE_A2A_CHUNKS -u SCALE_A2A_NO_BARRIER \
  IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  /home/marin/projects/marin/.worktrees/ep25-d4-pipelined/.venv/bin/iris \
  --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-rack-ring-ep64-120-20260725 \
  -e RUN_ID ep25d2-rack-ring-ep64-120-20260725 \
  -e SCALE_GPU_IMAGE nvcr.io/nvidia/jax:26.06-py3 \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_URI s3://marin-us-east-02a/tmp/ttl=30d/cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so \
  -e MARIN_NGC_XLA_CUDA_PLUGIN_SHA256 e420223a7a3ce7e5a816be50286e3610dacb10971984935ce986b316f47d8194 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
  -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 \
  -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ring_cute \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-rack-ring-ep64-120-20260725.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-ngc2606-dev --run
```
