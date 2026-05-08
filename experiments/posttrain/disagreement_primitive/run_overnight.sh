#!/usr/bin/env bash
# Auto-chain Phases 1 → 2 → 3 → 4. Idempotent: each phase skips already-done work.
# Phase 1 may already be running in the background — this script polls the
# responses file until 920 rows are present, then proceeds.
#
# Usage: bash experiments/posttrain/disagreement_primitive/run_overnight.sh

set -e
cd /lfs/skampere3/0/ahmedah/code/marin/.claude/worktrees/align

source .env
source .env2

LOG_DIR=experiments/posttrain/disagreement_primitive/overnight_logs
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RESPONSES=experiments/posttrain/disagreement_primitive/e9_opposite_mode_responses.jsonl

# ----- Phase 1 (or wait for existing Phase 1) -----
N=$(wc -l < "$RESPONSES" 2>/dev/null || echo 0)
if [ "$N" -ge 920 ]; then
  echo "[$(date -u +%H:%M:%S)] Phase 1 already complete ($N rows in $RESPONSES)"
else
  echo "[$(date -u +%H:%M:%S)] Phase 1 — Grok-opposite generation"
  if pgrep -f e9_run_opposite_mode_generation.py > /dev/null; then
    echo "  Phase 1 already running in background; polling for completion..."
    while true; do
      N=$(wc -l < "$RESPONSES" 2>/dev/null || echo 0)
      if [ "$N" -ge 920 ]; then break; fi
      if ! pgrep -f e9_run_opposite_mode_generation.py > /dev/null; then
        echo "  Phase 1 process exited at $N/920 rows — re-running to catch any failures"
        break
      fi
      sleep 30
    done
  fi
  # If still incomplete, run (idempotent — skips already-done cells)
  N=$(wc -l < "$RESPONSES" 2>/dev/null || echo 0)
  if [ "$N" -lt 920 ]; then
    .venv/bin/python experiments/posttrain/disagreement_primitive/e9_run_opposite_mode_generation.py \
      > "$LOG_DIR/phase1_${TS}.log" 2>&1
  fi
  N=$(wc -l < "$RESPONSES" 2>/dev/null || echo 0)
  echo "[$(date -u +%H:%M:%S)] Phase 1 done ($N rows)"
fi

# ----- Phase 2: GPT + Gemini sync judging on new Grok cells -----
echo "[$(date -u +%H:%M:%S)] Phase 2 — GPT + Gemini sync judging on new Grok cells"
.venv/bin/python experiments/posttrain/disagreement_primitive/e9_phase2_judge_grok_sync.py \
    > "$LOG_DIR/phase2_${TS}.log" 2>&1
echo "[$(date -u +%H:%M:%S)] Phase 2 done"

# ----- Phase 3: Submit Claude batches -----
echo "[$(date -u +%H:%M:%S)] Phase 3 — Submitting Claude batches"
.venv/bin/python experiments/posttrain/disagreement_primitive/e9_phase3_submit_claude_batches.py \
    > "$LOG_DIR/phase3_${TS}.log" 2>&1
echo "[$(date -u +%H:%M:%S)] Phase 3 done (batches submitted)"

# ----- Phase 4: Poll + fetch + integrate -----
echo "[$(date -u +%H:%M:%S)] Phase 4 — Polling + fetching Claude batches (may block hours)"
.venv/bin/python experiments/posttrain/disagreement_primitive/e9_phase4_fetch_claude_batches.py \
    > "$LOG_DIR/phase4_${TS}.log" 2>&1
echo "[$(date -u +%H:%M:%S)] Phase 4 done (Claude judgments integrated)"

echo "[$(date -u +%H:%M:%S)] === ALL PHASES COMPLETE ==="
echo "  per-phase logs: $LOG_DIR"
echo "  Run analysis next: .venv/bin/python experiments/posttrain/disagreement_primitive/e9_kappa_with_opposite.py"
