#!/bin/bash
# We evaluate the models not supported on TPU (e.g. Gemma 3, Nemotron) with this script

set -euo pipefail

DEFAULT_MODELS=(
  "meta-llama/Meta-Llama-3-70B"
  "marin-community/marin-32b-base"
  "allenai/OLMo-2-0325-32B"
  "Qwen/Qwen2.5-32B"
  "allenai/OLMo-3-1125-32B"
  "google/gemma-3-27b-pt"
  "meta-llama/Llama-3.1-8B"
  "marin-community/marin-8b-base"
  "allenai/OLMo-2-1124-7B"
  "Qwen/Qwen3-8B-Base"
  "allenai/OLMo-3-1025-7B"
)

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

for MODEL in "${MODELS[@]}"; do
  echo "Running lm_eval for model: ${MODEL}"
  HF_ALLOW_CODE_EVAL=1 lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096,max_gen_toks=4096" \
    --tasks paloma_c4_en,leaderboard_musr,anli,triviaqa,drop,truthfulqa_mc2,squadv2,race,toxigen,blimp,nq_open,xsum,uncheatable_eval,agieval_lsat_ar,arc_easy,arc_challenge,leaderboard_bbh,boolq,commonsense_qa,copa,leaderboard_gpqa,gsm8k_cot,hellaswag,humaneval,lambada_openai,minerva_math,mmlu,leaderboard_mmlu_pro,openbookqa,piqa,winogrande,wsc273 \
    --batch_size auto \
    --output_path ./local-eval-results \
    --confirm_run_unsafe_code \
    --apply_chat_template
done
