#!/usr/bin/env bash
# Poll a running grug task's telltale page from inside the pod, appending
# "epoch step drop_fraction loss tok_s" rows to a local TSV. Usage:
#   bash experiments/grug/moe/telltale_poll.sh <child-job-task-id> <pod-ip:port> <out.tsv> <n-polls> <interval-s>
set -u
TASK="$1"; ADDR="$2"; OUT="$3"; N="${4:-60}"; DT="${5:-180}"
for i in $(seq 1 "$N"); do
  line=$(IRIS_USER=mwittmann .venv/bin/iris --cluster=marin task exec "$TASK" -- bash -c \
    "curl -sf --max-time 10 http://$ADDR/metrics | grep -E '^levanter_(step|moe_drop_fraction|train_loss|throughput_tokens_per_second) '" \
    2>/dev/null | grep -v '^I2026' | awk '{print $1"="$2}' | paste -sd' ')
  echo "$(date +%s) $line" >> "$OUT"
  sleep "$DT"
done
