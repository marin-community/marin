#!/usr/bin/env bash
# One side of the policy-transfer screen. Both jobs run this with the same
# ROUND; only ROLE (hence --rank-base) differs.
#
#   ROLE=source|dest  ROUND=1|2|3  [WIDTH=N]  bash bench/run_matrix.sh
#
# ROUND 1 is the bulk floor plus the fan-out canary plus the width calibration,
# ROUND 2 the full-payload run at the calibrated width, ROUND 3 a repeat of the
# headline 1->1 stream on a fresh placement.
#
# The task installs a CUDA build of torch (the Iris task image has none and
# GB200 nodes are aarch64), then blocks until an operator injects the
# rendezvous address into /tmp/master_addr. That gate is what lets the
# NVLink-domain labels of both jobs be verified before any bytes move.
set -euo pipefail

ROLE=${ROLE:?set ROLE=source|dest}
ROUND=${ROUND:-1}
WIDTH=${WIDTH:-8}
TORCH_VERSION=${TORCH_VERSION:-2.11.0+cu128}
PARAMS=${PARAMS:-359.6e9}
LOCAL_RANKS=${LOCAL_RANKS:-4}
NUM_SOURCE=${NUM_SOURCE:-8}
WORLD_SIZE=${WORLD_SIZE:-16}
# Tasks with index >= MAX_TASKS hold their node but stay out of the world. A
# gang is sometimes sized larger than the measurement needs purely to force it
# onto a different NVLink domain than the peer job.
MAX_TASKS=${MAX_TASKS:-0}
VENV=/tmp/ptb

case "$ROLE" in
  source) RANK_BASE=0 ;;
  dest)   RANK_BASE=$NUM_SOURCE ;;
  *) echo "bad ROLE=$ROLE" >&2; exit 2 ;;
esac

echo "[run_matrix] role=$ROLE round=$ROUND host=$(hostname) task=$IRIS_TASK_ID"
TASK_INDEX=${IRIS_TASK_ID##*/}; TASK_INDEX=${TASK_INDEX%%:*}
if [ "$MAX_TASKS" -gt 0 ] && [ "$TASK_INDEX" -ge "$MAX_TASKS" ]; then
  echo "[run_matrix] task $TASK_INDEX is placement padding; holding the node"
  sleep 86400
  exit 0
fi

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match "torch==$TORCH_VERSION"

echo "[run_matrix] waiting for /tmp/master_addr"
while [ ! -s /tmp/master_addr ]; do sleep 5; done
MASTER_ADDR=$(cat /tmp/master_addr)
echo "[run_matrix] master_addr=$MASTER_ADDR"

run() {  # run <tag> <port> <extra args...>
  local tag=$1 port=$2; shift 2
  echo "[run_matrix] === $tag ==="
  "$VENV/bin/python" bench/policy_transfer_bench.py \
    --world-size "$WORLD_SIZE" --num-source "$NUM_SOURCE" \
    --rank-base "$RANK_BASE" --local-ranks "$LOCAL_RANKS" \
    --params "$PARAMS" --master-addr "$MASTER_ADDR" --master-port "$port" \
    --tag "$tag" "$@"
}

if [ "$ROUND" = "1" ]; then
  # Primary bulk floor, then the small fan-out canary, then the width
  # calibration at a reduced payload.
  run p2p-1to1-fullS        29501 --mode p2p                                    --reps 5
  run bcast-1to8-fullS      29502 --mode broadcast --active-source 1 --active-dest 8 --reps 3
  run striped-N1-cal        29503 --mode striped --active-source 1 --active-dest 1 --payload-fraction 0.1 --reps 3
  run striped-N4-cal        29504 --mode striped --active-source 4 --active-dest 4 --payload-fraction 0.1 --reps 3
  run striped-N8-cal        29505 --mode striped --active-source 8 --active-dest 8 --payload-fraction 0.1 --reps 3
elif [ "$ROUND" = "2" ]; then
  # Full-payload run at the width the calibration justified.
  run "striped-N${WIDTH}-fullS" 29601 --mode striped --active-source "$WIDTH" --active-dest "$WIDTH" --reps 5
else
  # Repeat the headline 1->1 stream on a fresh placement.
  run p2p-1to1-fullS-rep2 29701 --mode p2p --reps 5
fi

echo "[run_matrix] done role=$ROLE round=$ROUND"
