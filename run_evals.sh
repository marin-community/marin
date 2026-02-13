#!/usr/bin/env bash
# Re-run failed evalchemy evals (9 of 18 checkpoints).
# The --force_run_failed flag will retry FAILED seeds while skipping SUCCESS ones.

set -euo pipefail

# =============================================================================
# exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math (4 failed checkpoints)
# =============================================================================
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math-263ee5/hf/step-500/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math-263ee5/hf/step-1000/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math-263ee5/hf/step-2000/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_mixtureofthoughts_math-263ee5/hf/step-2929/" \
    --force_run_failed true

# =============================================================================
# exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math (3 failed checkpoints)
# =============================================================================
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math-3d6ce3/hf/step-1000/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math-3d6ce3/hf/step-1500/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_selfinstill_openthoughts4_math-3d6ce3/hf/step-1738/" \
    --force_run_failed true

# =============================================================================
# exp_instilloracle_sft_qwen3_4b_openthoughts4_math (2 failed checkpoints)
# =============================================================================
python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_openthoughts4_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_openthoughts4_math-d5c87e/hf/step-500/" \
    --force_run_failed true

python lib/marin/src/marin/run/ray_run.py --no_wait --extra eval,vllm \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY stanford-mercury \
    --env_vars WANDB_PROJECT instill-eval \
    -- python experiments/evals/exp_evalchemy_eval.py \
    --experiment "exp_instilloracle_sft_qwen3_4b_openthoughts4_math" \
    --checkpoint "gs://marin-us-central1/checkpoints/exp_instilloracle_sft_qwen3_4b_openthoughts4_math-d5c87e/hf/step-1000/" \
    --force_run_failed true
