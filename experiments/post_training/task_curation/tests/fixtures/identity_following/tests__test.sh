#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt

if [[ ! -s /app/response.txt && ! -s /app/answer.txt ]]; then
  printf '{"reward": 0.0}\n' > /logs/verifier/reward.json
  exit 0
fi

if [[ ! -s /app/response.txt ]]; then
  cp /app/answer.txt /app/response.txt
fi

if [[ -f /tests/deterministic_gate ]]; then
  set +e
  REWARDKIT_DETERMINISTIC_ONLY=1 python3 /tests/deterministic_gate \
    >> /logs/verifier/test-stdout.txt 2>&1
  gate_status=$?
  set -e
  if [[ ${gate_status} -eq 1 ]]; then
    printf '{"reward": 0.0}\n' > /logs/verifier/reward.json
    exit 0
  fi
  if [[ ${gate_status} -ne 0 ]]; then
    exit "${gate_status}"
  fi
fi

if [[ -f /tests/judge.toml ]]; then
  export LITELLM_DROP_PARAMS=1
  rewardkit /tests --max-concurrent-llm 1 \
    >> /logs/verifier/test-stdout.txt 2>&1
else
  printf '{"reward": 1.0}\n' > /logs/verifier/reward.json
fi
