#!/usr/bin/env bash
# Auto-generated missing evals
# 1 checkpoint(s) with missing results

set -euo pipefail

# exp_temp_sft_qwen3_4b_selfinstill_ot4_math_n1_uq1_round1 step-500 (missing: MATH500, AIME24, AIME25, AMC23, HMMT)
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_temp_sft_qwen3_4b_selfinstill_ot4_math_n1_uq1_round1" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_temp_sft_qwen3_4b_selfinstill_ot4_math_n1_uq1_round1-8397ac/hf/step-500" \
    --force_run_failed true
