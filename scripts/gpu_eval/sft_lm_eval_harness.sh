#!/bin/bash
# This is not great Marin practice - we want everything to run inside the Executor framework!
# However, for the 05-19 Launch, there were a wide variety of issues for Evals on TPU
# - Disk Space Fills up when running too many evaluations at once
# - VLLM isn't releasing TPUs causing crashed jobs due to resource contention
# - OLMo was unexplainably OOMing when Marin model was not.
# In order to get the evaluations done on time and to keep the evaluation setup consistent
# across models I ended up using the following script for Marin 8B SFT evals!
# Documenting here for posterity, but this should not become a standard practice - Will

MODEL=$1

HF_ALLOW_CODE_EVAL=1 lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096,max_gen_toks=4096" \
  --tasks leaderboard_ifeval,gsm8k_cot,humaneval,leaderboard_math_hard,mmlu,leaderboard_mmlu_pro,leaderboard_gpqa,leaderboard_musr,leaderboard_bbh \
  --batch_size auto \
  --output_path ./local-eval-results \
  --confirm_run_unsafe_code \
  --apply_chat_template
