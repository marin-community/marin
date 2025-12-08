#!/bin/bash
# We evaluate the models not supported on TPU (e.g. Gemma 3, Nemotron) with this script

MODEL=$1



HF_ALLOW_CODE_EVAL=1 lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096,max_gen_toks=4096" \
  --tasks paloma_c4_en,leaderboard_musr,anli,triviaqa,drop,truthfulqa_mc2,squadv2,race,aime,toxigen,eq_bench,blimp,nq_open,xsum,uncheatable_eval,agieval_lsat_ar,arc_easy,arc_challenge,leaderboard_bbh,boolq,commonsense_qa,copa,leaderboard_gpqa,gsm8k_cot,hellaswag,humaneval,lambada_openai,minerva_math,mmlu,leaderboard_mmlu_pro,openbookqa,piqa,winogrande,wsc273 \
  --batch_size auto \
  --output_path ./local-eval-results \
  --confirm_run_unsafe_code \
  --apply_chat_template
