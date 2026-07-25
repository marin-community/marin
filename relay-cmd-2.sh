ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-ep4-treatment-20260725-v2 \
  -e RUN_ID ep25d2-mxfp8-ep4-treatment-20260725-v2 \
  -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
  -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 1 \
  -e SCALE_EXPERT_AXIS 4 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
  -e SCALE_HIDDEN_DIM 2048 -e SCALE_NUM_LAYERS 4 \
  -e SCALE_INTERMEDIATE 512 -e SCALE_SHARED_INTERMEDIATE 2048 \
  -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 64 -e SCALE_SLIDING_WINDOW 2048 \
  -e SCALE_STEPS 4 -e SCALE_MOE_IMPL ragged_all_to_all \
  -e SCALE_A2A_FIXED 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
  -e SCALE_A2A_CUSTOM_ADJOINT 1 -e SCALE_MOE_QB 1 \
  -e SCALE_MOE_MXFP8 1 \
  -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
  -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
  -e SCALE_TRACKER json_logger \
  -e SCALE_JSON_LOGGER ep25d2-mxfp8-ep4-treatment-20260725-v2.metrics \
  -e SCALE_REPORT_DROPS 1 -e SCALE_DISABLE_CHECKPOINT 1 \
  -- python -m experiments.grug.moe.launch_cw_scale \
  --version ep25d2-mxfp8-r6-v2-dev --run
