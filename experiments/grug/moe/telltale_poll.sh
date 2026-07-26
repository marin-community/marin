#!/usr/bin/env bash
# Poll a running grug task's telltale page from inside the pod, appending
# "epoch step drop_fraction loss tok_s" rows to a local TSV. Re-resolves the
# telltale address every cycle (tasks can be rescheduled to a new node/port).
# Usage:
#   bash experiments/grug/moe/telltale_poll.sh <child-task-id> <job-prefix> <out.tsv> <n-polls> <interval-s>
set -u
TASK="$1"; JOBPAT="$2"; OUT="$3"; N="${4:-60}"; DT="${5:-180}"
for i in $(seq 1 "$N"); do
  ADDR=$(IRIS_USER=mwittmann .venv/bin/iris --cluster=marin endpoints list "$JOBPAT" 2>/dev/null \
    | grep "telltale/$TASK##" | awk -F'http://' '{print $2}' | awk '{print $1}')
  # Fall back: match the full task path suffix /0 (task 0).
  if [ -z "${ADDR:-}" ]; then
    ADDR=$(IRIS_USER=mwittmann .venv/bin/iris --cluster=marin endpoints list "$JOBPAT" 2>/dev/null \
      | grep "telltale" | grep '/0 ' | head -1 | awk -F'http://' '{print $2}' | awk '{print $1}')
  fi
  if [ -n "${ADDR:-}" ]; then
    line=$(IRIS_USER=mwittmann .venv/bin/iris --cluster=marin task exec "$TASK" -- bash -c \
      "curl -sf --max-time 10 http://$ADDR/metrics | grep -E '^levanter_(step|moe_drop_fraction|train_loss|throughput_tokens_per_second) '" \
      2>/dev/null | grep -v '^I2026' | awk '{print $1"="$2}' | paste -sd' ')
    echo "$(date +%s) $line" >> "$OUT"
  else
    echo "$(date +%s) NO_ADDR" >> "$OUT"
  fi
  sleep "$DT"
done
