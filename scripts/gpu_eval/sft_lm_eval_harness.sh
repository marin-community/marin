#!/bin/bash

MODEL=$1

HF_ALLOW_CODE_EVAL=1 lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096,max_gen_toks=4096" \
  --tasks leaderboard_ifeval,gsm8k_cot,humaneval,leaderboard_math_hard,mmlu,leaderboard_mmlu_pro,leaderboard_gpqa,leaderboard_musr,leaderboard_bbh \
  --batch_size auto \
  --output_path ./local-eval-results \
  --confirm_run_unsafe_code \
  --apply_chat_template
