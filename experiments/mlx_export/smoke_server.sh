#!/bin/bash
# Server checks for an MLX build: thinking split, thinking off, tool calls, tool
# round trip, streaming, and a ~20k-token prompt. Mac only.
# Usage: ./smoke_server.sh <mlx build dir> <work dir for logs> [port]
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?usage: smoke_server.sh <mlx build dir> <work dir> [port]}
WORK=${2:?usage: smoke_server.sh <mlx build dir> <work dir> [port]}
PORT=${3:-8080}
BASE="http://127.0.0.1:${PORT}"
mkdir -p "$WORK"
LOG=$WORK/server-smoke.log
PIDFILE=$WORK/server-smoke.pid

fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "PASS: $*"; }

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  fail "port $PORT busy: reuse that server or pick another port"
fi

echo "starting mlx_lm.server --model $MODEL on :$PORT (log: $LOG)"
nohup uv run mlx_lm.server --model "$MODEL" --trust-remote-code \
  --max-tokens 32768 --port "$PORT" --log-level DEBUG > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
trap 'kill "$(cat $PIDFILE)" 2>/dev/null || true; rm -f "$PIDFILE"' EXIT

# Wait for a real 1-token completion; /v1/models alone can answer early.
for i in $(seq 1 120); do
  if curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"default_model","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' >/dev/null 2>&1; then
    break
  fi
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -30 "$LOG"; fail "server exited during startup"; }
  sleep 5
done
pass "server up"

chat() { # chat <json-body>
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"
}

# 1. basic chat
R=$(chat '{"model":"default_model","max_tokens":256,"messages":[{"role":"user","content":"Name the capital of France in one word."}]}')
CONTENT=$(echo "$R" | jq -r '.choices[0].message.content // empty')
FINISH=$(echo "$R" | jq -r '.choices[0].finish_reason // empty')
[ -n "$CONTENT" ] || fail "basic chat empty content: $R"
[ "$FINISH" = "stop" ] || fail "basic chat finish_reason=$FINISH"
pass "1 basic chat (finish=stop)"

# 2. thinking on by default, split into reasoning
R=$(chat '{"model":"default_model","max_tokens":2048,"messages":[{"role":"user","content":"Is 391 prime? Think carefully."}]}')
REASONING=$(echo "$R" | jq -r '.choices[0].message.reasoning // empty')
CONTENT=$(echo "$R" | jq -r '.choices[0].message.content // empty')
[ -n "$REASONING" ] || fail "no reasoning field by default; mlx-lm pin or model file not loaded: $(echo "$R" | head -c 400)"
case "$CONTENT$REASONING" in *"<|start_think|>"*|*"<|end_think|>"*) fail "thinking markers leaked into text" ;; esac
pass "2 thinking default (reasoning split, no leaked markers)"

# 3. enable_thinking=false suppresses reasoning in the response.
R=$(chat '{"model":"default_model","max_tokens":512,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Is 391 prime?"}]}')
REASONING=$(echo "$R" | jq -r '.choices[0].message.reasoning // empty')
[ -z "$REASONING" ] || fail "reasoning present with enable_thinking=false"
pass "3 thinking disabled via chat_template_kwargs (no reasoning)"

# 4. tool call
TOOLS='{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}'
R=$(chat "{\"model\":\"default_model\",\"max_tokens\":512,\"chat_template_kwargs\":{\"enable_thinking\":false},\"tools\":[$TOOLS],\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}]}")
FINISH=$(echo "$R" | jq -r '.choices[0].finish_reason // empty')
NAME=$(echo "$R" | jq -r '.choices[0].message.tool_calls[0].function.name // empty')
ARGS=$(echo "$R" | jq -r '.choices[0].message.tool_calls[0].function.arguments // empty')
[ "$FINISH" = "tool_calls" ] || fail "tool finish_reason=$FINISH: $(echo "$R" | head -c 400)"
[ "$NAME" = "get_weather" ] || fail "tool name=$NAME"
echo "$ARGS" | jq -e . >/dev/null || fail "tool arguments not valid JSON: $ARGS"
pass "4 tool call (finish=tool_calls, name=get_weather, args=$ARGS)"

# 5. tool round trip
TOOL_MSG=$(python3 - "$ARGS" <<'EOF'
import json, sys
args = sys.argv[1]
msgs = [
  {"role": "user", "content": "What is the weather in Paris?"},
  {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": json.loads(args)}}]},
  {"role": "tool", "name": "get_weather", "content": "18C, light rain, wind 12 km/h"},
]
print(json.dumps({"model": "default_model", "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}))
EOF
)
R=$(chat "$TOOL_MSG")
CONTENT=$(echo "$R" | jq -r '.choices[0].message.content // empty')
case "$CONTENT" in *18*|*rain*) pass "5 tool round trip answered from result" ;; *) fail "round trip content unexpected: $CONTENT" ;; esac

# 6. streaming: reasoning deltas and tool-call deltas
STREAM=$(chat '{"model":"default_model","max_tokens":2048,"stream":true,"messages":[{"role":"user","content":"Is 391 prime? Think carefully."}]}')
grep -q '"reasoning"' <<<"$STREAM" && pass "6a streaming reasoning deltas" || fail "no reasoning delta in stream"
# An answer given without thinking streams as content, as with vLLM's grug reasoning parser.
STREAM=$(chat '{"model":"default_model","max_tokens":256,"stream":true,"messages":[{"role":"user","content":"Name the capital of France in one word."}]}')
CONTENT=$(echo "$STREAM" | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -r '.choices[0].delta.content // empty' | tr -d '\n')
[ -n "$CONTENT" ] && pass "6c streamed answer is content ($CONTENT)" || fail "streamed answer has no content: $(echo "$STREAM" | head -c 400)"
STREAM=$(chat "{\"model\":\"default_model\",\"max_tokens\":512,\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":false},\"tools\":[$TOOLS],\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Paris?\"}]}")
grep -q '"tool_calls"' <<<"$STREAM" && pass "6b streaming tool-call deltas" || fail "no tool_calls delta in stream"

# 7. ~20k token prompt completes; report prefill/decode tok/s and RSS
PROMPT_FILE=$WORK/long_prompt.txt
python3 - "$PROMPT_FILE" <<'EOF'
import sys
body = ("The surveyor kept a journal of the cuttings and the embankments, the navvies and "
        "their barrows, the engineers with their theodolites, the inns full of argument about "
        "speed and distance and time. The steam whistle had replaced the cuckoo as the marker "
        "of the morning. ") * 330
body += " Summarize the passage in one sentence."
open(sys.argv[1], "w").write(body)
EOF
START=$(date +%s)
BODY=$(python3 -c "
import json
body = open('$PROMPT_FILE').read()
print(json.dumps({'model': 'default_model', 'max_tokens': 256,
  'chat_template_kwargs': {'enable_thinking': False},
  'messages': [{'role': 'user', 'content': body}]}))")
R=$(chat "$BODY")
echo "$R" | START=$START python3 -c "
import json, sys, time, os
r = json.loads(sys.stdin.read())
usage = r.get('usage') or {}
prompt_toks = usage.get('prompt_tokens', 0)
gen_toks = usage.get('completion_tokens', 0)
elapsed = max(time.time() - float(os.environ['START']), 1e-6)
assert prompt_toks > 15000, f'prompt too short: {prompt_toks}'
print(f'PASS: 7 long prompt ({prompt_toks} prompt tokens, {gen_toks} generated, '
      f'{prompt_toks/elapsed:.0f} prefill tok/s incl. queue, {gen_toks/elapsed:.1f} decode tok/s wall)')
" || fail "long prompt request failed: $(echo "$R" | head -c 300)"
RSS_KB=$(ps -o rss= -p "$(cat "$PIDFILE")" | tr -d ' ')
echo "server RSS: $((RSS_KB / 1024 / 1024)) GiB"

echo "ALL SMOKE CHECKS PASSED"
