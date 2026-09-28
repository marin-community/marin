#!/bin/bash
# Grades MT-MBPP generations as TPU tasks complete (every 15 minutes) across the us-east1 (Olmix), europe-west4 and
# us-east5 plans, until all 68 checkpoint-language pairs are graded in some region, or every chain has ended.
# Needs Docker (OrbStack). Progress: grading/grade.log; per-region outputs under grading/<region>/.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
P=.agents/projects/mt_mbpp_exec_20260926
mkdir -p $P/grading
while :; do
  plans=()
  for f in $P/plan_east1.json $P/plan_euw4.json $P/plan_east5b.json $P/plan_east5.json; do [[ -f $f ]] && plans+=(--plan "$f"); done
  uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.grade_generations "${plans[@]}" \
    --signatures $P/signatures.jsonl.gz --translations $P/tests/translations.jsonl --validation $P/tests/validation.jsonl \
    --output $P/grading --log $P/grading/grade.log >> $P/grading/loop.out 2>&1 || echo "$(date -u +%FT%TZ) grading pass failed; see loop.out" >> $P/grading/grade.log
  n=$(uv run --offline --no-sync python -c "import json; print(len({(s['checkpoint'], s['task']) for s in json.load(open('$P/grading/summary.json'))}))" 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) pass done: $n/68 checkpoint-language pairs graded" >> $P/grading/grade.log
  [[ $n -ge 68 ]] && { echo "$(date -u +%FT%TZ) DONE" >> $P/grading/grade.log; break; }
  if [[ -f $P/outcome_east1 ]] && grep -qE "DONE|STOP" $P/euw4/auto_release.log 2>/dev/null && grep -qE "DONE|STOP" $P/east5b/auto_release.log 2>/dev/null; then
    [[ -n ${final:-} ]] && { echo "$(date -u +%FT%TZ) STOP: chains ended with $n/68 graded" >> $P/grading/grade.log; break; }
    final=1; continue
  fi
  sleep 900
done
