#!/bin/bash
# Release the full Table-9 backfill for the MARINER OlmoBaseEval Easy 1e21 checkpoint once both canaries succeed,
# wait for the full runs and the overlap suite, then grade and report. Detached from any session; progress goes to
# auto_release.log next to this file. Stops and logs instead of retrying if a parent fails.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
D=.agents/projects/mariner_table9_accuracy_20260914/mariner_t9_1e21
LOG=$D/auto_release.log
IRIS="uv run --no-sync iris --config lib/iris/config/marin.yaml"
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
PLAN=$D/backfill_plan.json
OVERLAP=$D/overlap_plan.json
CAN_C=/calvinxu/table9-accuracy-choices-v6e4-canary-mariner-t9-1e21-20260921
CAN_G=/calvinxu/table9-accuracy-generation-v6e4-canary-mariner-t9-1e21-20260921
FULL_C=/calvinxu/table9-accuracy-choices-v6e4-full-mariner-t9-1e21-20260921
FULL_G=/calvinxu/table9-accuracy-generation-v6e4-full-mariner-t9-1e21-20260921
OVERLAP_JOB=/calvinxu/mariner-ladder-accuracy-mariner-t9-1e21-v5p-20260921
SAFETY_ARGS="--expected-child-zone us-east5-b"
REDACT='s/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g'

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
state() { $IRIS job describe "$1" 2>/dev/null | awk '/^State:/{print $2; exit}'; }
children() {  # succeeded/running/pending/failed counts of a parent's direct children
  $IRIS job list --prefix "$1/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// {c[$2]++} END {printf "succeeded=%d running=%d pending=%d failed=%d killed=%d", c["succeeded"], c["running"], c["pending"], c["failed"], c["killed"]}'
}

log "chain started (pid $$)"
# Stage 1: canaries
while :; do
  sc=$(state $CAN_C); sg=$(state $CAN_G)
  log "stage1 canaries: choices=$sc generation=$sg; overlap=$(state $OVERLAP_JOB)"
  [[ $sc == succeeded && $sg == succeeded ]] && break
  if [[ $sc == failed || $sc == killed || $sg == failed || $sg == killed ]]; then
    log "STOP: canary ended choices=$sc generation=$sg; nothing released"; exit 1
  fi
  sleep 300
done

# Stage 2: release the full runs (idempotent via marker)
if [[ ! -f $D/full.released ]]; then
  while IFS= read -r cmd; do
    [[ -z $cmd ]] && continue
    job=$(sed -E 's/.*--job-name ([^ ]+).*/\1/' <<<"$cmd")
    if ! uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety $SAFETY_ARGS --command "$cmd" >> "$LOG" 2>&1; then
      log "STOP: east5 safety validation failed for $job"; exit 1
    fi
    ( set -a; source ~/.zshrc.secrets; set +a; eval "$cmd" ) 2>&1 | sed -E "$REDACT" > "$D/submit_$job.log"
    if ! grep -q "Job submitted: /calvinxu/$job" "$D/submit_$job.log"; then
      log "STOP: submission of $job did not confirm; see submit_$job.log"; exit 1
    fi
    log "released $job"
    uv run fieldbook job add --experiment $EXP --name "$job" --status running --external-system iris --external-id "/calvinxu/$job" --launcher iris --started-at "$(date -u +%FT%TZ)" --attr "protocol.plan=mariner_t9_1e21/backfill_plan.json" --json >/dev/null 2>&1 || log "warning: fieldbook registration failed for $job"
  done < "$D/launch_full_commands.txt"
  touch "$D/full.released"
fi

# Stage 3: wait for both full parents and the overlap suite
t0=$(date +%s)
while :; do
  fc=$(state $FULL_C); fg=$(state $FULL_G); fo=$(state $OVERLAP_JOB)
  log "stage3 full: choices=$fc [$(children $FULL_C)] generation=$fg [$(children $FULL_G)] overlap=$fo elapsed=$(( ($(date +%s) - t0) / 60 ))min"
  [[ $fc == succeeded && $fg == succeeded && $fo == succeeded ]] && break
  if [[ $fc == failed || $fc == killed || $fg == failed || $fg == killed || $fo == failed || $fo == killed ]]; then
    log "STOP: a parent ended choices=$fc generation=$fg overlap=$fo; grading not started (resubmitting the same command resumes it)"; exit 1
  fi
  sleep 600
done
for j in $FULL_C $FULL_G $OVERLAP_JOB; do
  uv run fieldbook job add --experiment $EXP --external-system iris --external-id "$j" --status succeeded --finished-at "$(date -u +%FT%TZ)" --update-existing --json >/dev/null 2>&1 || log "warning: fieldbook update failed for $j"
done

# Stage 4: grade and report
log "stage4 grading"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.grade_table9_accuracy --plan "$PLAN" > "$D/grade.log" 2>&1 || { log "STOP: grading failed; see grade.log"; exit 1; }
log "stage4 report"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.report_table9_accuracy --plan "$PLAN" --overlap-plan "$OVERLAP" --output "$D/coverage" > "$D/report.log" 2>&1 || { log "STOP: report failed; see report.log"; exit 1; }
log "DONE: coverage/ written"
