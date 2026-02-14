#!/usr/bin/env bash
# Auto-generated missing evals
# 2 checkpoint(s) with missing results

set -euo pipefail

# exp_instilloracle_sft_qwen3_4b_selfinstill_mot_math_45k_n1_uq1_round1 step-1000 (missing: MATH500, AIME24, AIME25, AMC23, HMMT)
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_selfinstill_mot_math_45k_n1_uq1_round1" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_selfinstill_mot_math_45k_n1_uq1_r-bb75b3/hf/step-1000" \
    --force_run_failed true

# exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math step-2929 (missing: MATH500, AIME24, AIME25, AMC23, HMMT)
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math-263ee5/hf/step-2929" \
    --force_run_failed true
