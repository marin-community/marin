#!/usr/bin/env bash
# Decode-setting ablation, generation side. On one whole H100x8 node:
#   1. stage the checkpoint, serve it with the MarinSkyRL frozen vLLM runtime;
#   2. run the full Evalchemy IFBench once per decode cell, concurrently, with every
#      sampling field passed explicitly through --gen_kwargs;
#   3. extract each cell's responses to the 100 trace prompts into a trace file;
#   4. self-score those traces against the same server (generational_perplexity --traces);
#   5. upload stores, traces, self-scores and the vLLM request log to DEST.
#
# usage: decode_ablation_generate.sh MODEL DEST LIMIT CELL...
#   MODEL  grugsft471 | snowballbase | glm47flash | kimilinear | qwen36
#   LIMIT  "full" or a prompt count (smoke)
#   CELL   greedy | shipped | t06
set -euo pipefail

model="$1"
dest="$2"
limit="$3"
shift 3
cells=("$@")
status=0

workspace="$PWD"
runtime_dir=experiments/evaluation/mini_policy_runtime
templates="$workspace/$runtime_dir/chat-templates"
keys="$workspace/$runtime_dir/trace-sets/ifbench-trace-prompt-keys.txt"
model_dir=/tmp/pp/model
environment=/tmp/skyrl-env
runtime_file=/tmp/skyrl-runtime.env
work=/tmp/da
mkdir -p "$work"
quick=s3://marin-rno2a/tmp/ttl=30d/quick-serve-models

score_args=(--template-family template --enable-thinking false)
case "$model" in
  grugsft471)
    author_model=open-athena/Grug-67B-A2B-Datakit-SFT-Repro-262K-Step471-2026.10.25@51669ee19b8972fc2e74c10b39eaef2eb244c44a
    mirror=$quick/open-athena_Grug-67B-A2B-Datakit-SFT-Repro-262K-Step471-2026.10.25_51669ee19b8972fc2e74c10b39eaef2eb244c44a-cc6512b344b08123
    template="$templates/grug-67b-a2b-datakit-sft.jinja"
    serve_args=(--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
      --max-num-seqs 32 --max-num-batched-tokens 7168 --tool-call-parser hermes --chat-template "$template")
    score_args+=(--chat-template-file "$template")
    shipped="temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,seed=42"
    ;;
  snowballbase)
    author_model=open-athena/snowball-67b-a2b-base-262k-qk175-skew8@3bbf729f0555965c8667600ee91c3c109ac21b19
    mirror=$quick/open-athena_snowball-67b-a2b-base-262k-qk175-skew8_3bbf729f0555965c8667600ee91c3c109ac21b19-722f147d8996ed84
    template="$templates/grug-67b-a2b-datakit-sft.jinja"
    serve_args=(--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
      --max-num-seqs 32 --max-num-batched-tokens 7168 --tool-call-parser hermes --chat-template "$template")
    score_args+=(--chat-template-file "$template")
    shipped="temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,seed=42"
    ;;
  glm47flash)
    author_model=zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67
    hf_repo=zai-org/GLM-4.7-Flash
    hf_revision=7dd20894a642a0aa287e9827cb1a1f7f91386b67
    serve_args=(--tensor-parallel-size 2 --data-parallel-size 4 --enable-prefix-caching
      --reasoning-parser glm45 --tool-call-parser glm47
      --default-chat-template-kwargs '{"enable_thinking":true}')
    shipped="temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,seed=42"
    ;;
  kimilinear)
    author_model=moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a447157d4658b573f9a695d57658590e9
    hf_repo=moonshotai/Kimi-Linear-48B-A3B-Instruct
    hf_revision=e1df551a447157d4658b573f9a695d57658590e9
    template="$templates/kimi-linear-48b-a3b-instruct.jinja"
    serve_args=(--tensor-parallel-size 4 --data-parallel-size 2 --enable-prefix-caching
      --tool-call-parser kimi_k2 --chat-template "$template" --trust-remote-code)
    score_args+=(--chat-template-file "$template" --trust-remote-code)
    shipped="temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,seed=42"
    ;;
  qwen36)
    author_model=Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0
    mirror=$quick/Qwen_Qwen3.6-35B-A3B_995ad96eacd98c81ed38be0c5b274b04031597b0-567d286455fd6825
    serve_args=(--tensor-parallel-size 2 --data-parallel-size 4 --enable-prefix-caching
      --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --tool-call-parser qwen3_coder)
    shipped="temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,seed=42"
    ;;
  *)
    echo "unknown model: $model" >&2
    exit 2
    ;;
esac

gen_kwargs_for() {
  case "$1" in
    greedy) echo "temperature=0,top_p=1.0,top_k=-1,min_p=0.0,seed=42" ;;
    shipped) echo "$shipped" ;;
    t06) echo "temperature=0.6,top_p=0.95,top_k=-1,min_p=0.0,seed=42" ;;
    *) echo "unknown cell: $1" >&2; return 2 ;;
  esac
}

echo "MARIN_PREFIX=${MARIN_PREFIX:-unset} MARIN_TEMP_PREFIX=${MARIN_TEMP_PREFIX:-unset} DEST=$dest"
if [[ -n "${mirror:-}" ]]; then
  uv run --frozen fsutil rsync "$mirror" "$model_dir"
else
  uv run --frozen --with huggingface_hub hf download "$hf_repo" --revision "$hf_revision" --local-dir "$model_dir"
fi
if [[ "$model" == snowballbase ]]; then
  # The base export ships no generation_config.json, so vLLM would stop only on
  # <|end_of_text|>; add <|eot_id|> (128009) as the end-of-turn stop.
  echo '{"eos_token_id": [128001, 128009], "bos_token_id": 128000}' > "$model_dir/generation_config.json"
fi
echo "served generation_config.json:"
cat "$model_dir/generation_config.json" || echo "(none)"

(cd "$workspace/vendor/skyrl" && bash cloud/iris/bootstrap_runtime.sh "$PWD" "$environment" "$runtime_file" megatron production)
source "$runtime_file"
"$environment/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
port=$("$environment/bin/python" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")

vllm serve "$model_dir" \
  --served-model-name "$model" \
  --max-model-len 73728 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --enable-log-requests \
  --max-log-len 64 \
  "${serve_args[@]}" \
  --port "$port" > "$work/vllm.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

(cd "$workspace/vendor/evalchemy" && uv sync --frozen --python 3.12 --extra ifbench --extra serve-eval --quiet)

until "$environment/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health')" 2>/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming healthy" >&2
    tail -n 200 "$work/vllm.log" >&2
    exit 1
  fi
  sleep 10
done
echo "vLLM server healthy"
grep -m 5 -i -E "generation config|sampling param|default.*sampling" "$work/vllm.log" || true

if [[ "$limit" == selfscore ]]; then
  # Self-score-only rerun: reuse the trace files an earlier run uploaded.
  for cell in "${cells[@]}"; do
    uv run --frozen fsutil cp "$dest/$model/traces-$model-$cell.jsonl" "$work/traces-$model-$cell.jsonl"
  done
else
limit_args=()
[[ "$limit" == full ]] || limit_args=(--limit "$limit")
model_args="model=$model,base_url=http://127.0.0.1:$port/v1/chat/completions,api_key=EMPTY,tokenizer_backend=none,tokenized_requests=False,num_concurrent=64,max_retries=8,timeout=3600,max_length=73664,chat_template_kwargs={\"enable_thinking\":false}"
pids=()
for cell in "${cells[@]}"; do
  gen_kwargs=$(gen_kwargs_for "$cell")
  echo "cell $cell gen_kwargs $gen_kwargs"
  (cd "$workspace/vendor/evalchemy" && uv run --frozen --no-sync python -m eval.eval \
    --model local-chat-completions \
    --model_args "$model_args" \
    --tasks IFBench \
    --num_fewshot 0 \
    --batch_size 1 \
    --apply_chat_template \
    --seed 42 \
    --gen_kwargs "$gen_kwargs" \
    "${limit_args[@]}" \
    --output_path "$work/store-$cell") > "$work/eval-$cell.log" 2>&1 &
  pids+=($!)
done
status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Evalchemy failed for cell ${cells[$index]}" >&2
    tail -n 80 "$work/eval-${cells[$index]}.log" >&2
    status=1
  fi
done
grep -h -E "strict_prompt_accuracy|Pool finished" "$work"/eval-*.log | tail -n 20 || true
[[ $status == 0 ]] || exit 1

for cell in "${cells[@]}"; do
  uv run --frozen fsutil rsync "$work/store-$cell" "$dest/$model/store-$cell" > /dev/null
done
gzip -kf "$work/vllm.log"
uv run --frozen fsutil cp "$work/vllm.log.gz" "$dest/$model/vllm-generation.log.gz"


for cell in "${cells[@]}"; do
  (cd "$workspace/vendor/evalchemy" && uv run --frozen --no-sync python "$workspace/$runtime_dir/decode_ablation_extract.py" \
    --store "$work/store-$cell" --author "$model" --author-model "$author_model" --cell "$cell" \
    --gen-kwargs "$(gen_kwargs_for "$cell")" --keys "$keys" \
    --out-traces "$work/traces-$model-$cell.jsonl" --out-summary "$work/summary-$model-$cell.json")
done

# Evidence that the sampling fields reached vLLM: the engine-side SamplingParams it logged.
"$environment/bin/python" - "$work/vllm.log" > "$work/sampling-params-$model.txt" <<'EOF'
import re, sys
from collections import Counter
fields = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "repetition_penalty", "stop_token_ids")
counts = Counter()
for line in open(sys.argv[1], errors="replace"):
    if "Received request" not in line or "params: SamplingParams(" not in line:
        continue
    params = line.split("params: SamplingParams(", 1)[1]
    counts[tuple(f"{f}={m.group(1)}" for f in fields if (m := re.search(rf"\b{f}=(\[[^\]]*\]|[^,)]*)", params)))] += 1
for key, n in counts.most_common():
    print(n, ", ".join(key))
EOF
cat "$work/sampling-params-$model.txt"
fi

spids=()
for cell in "${cells[@]}"; do
  "$environment/bin/python" -m experiments.evaluation.mini_policy_runtime.generational_perplexity \
    --model "$model_dir" --label "self-$model" --engine vllm-server \
    --server-url "http://127.0.0.1:$port" --served-model "$model" "${score_args[@]}" \
    --traces "$work/traces-$model-$cell.jsonl" --trace-output "$work/selfscore-$model-$cell.jsonl.gz" \
    > "$work/selfscore-$cell.log" 2>&1 &
  spids+=($!)
done
for index in "${!spids[@]}"; do
  if ! wait "${spids[$index]}"; then
    echo "self-score failed for cell ${cells[$index]}" >&2
    tail -n 60 "$work/selfscore-${cells[$index]}.log" >&2
    status=1
  fi
done
tail -n 2 "$work"/selfscore-*.log

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
gzip -kf "$work/vllm.log"
uploads=("$work"/selfscore-*.jsonl.gz "$work"/selfscore-*.log)
[[ "$limit" == selfscore ]] || uploads+=("$work"/traces-*.jsonl "$work"/summary-*.json "$work"/vllm.log.gz "$work"/sampling-params-*.txt "$work"/eval-*.log)
for file in "${uploads[@]}"; do
  [[ -e "$file" ]] && uv run --frozen fsutil cp "$file" "$dest/$model/$(basename "$file")"
done
exit $status
