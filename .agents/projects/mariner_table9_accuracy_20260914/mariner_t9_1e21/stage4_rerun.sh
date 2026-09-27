#!/bin/bash
# Rerun of auto_release.sh stage 4 (grade, then report) after Docker was started on 2026-09-22.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
D=.agents/projects/mariner_table9_accuracy_20260914/mariner_t9_1e21
LOG=$D/auto_release.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "stage4 rerun: grading (docker up)"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.grade_table9_accuracy --plan "$D/backfill_plan.json" > "$D/grade.log" 2>&1 || { log "STOP: grading failed again; see grade.log"; exit 1; }
log "stage4 rerun: report"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.report_table9_accuracy --plan "$D/backfill_plan.json" --overlap-plan "$D/overlap_plan.json" --output "$D/coverage" > "$D/report.log" 2>&1 || { log "STOP: report failed; see report.log"; exit 1; }
log "DONE: coverage/ written"
