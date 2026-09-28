#!/bin/bash
# Release the full r11 Table-9 accuracy runs once both r11 canaries succeed, then grade, report and analyze.
# Detached from the interactive session; progress goes to auto_release_r11.log next to this file.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
D=.agents/projects/mariner_table9_accuracy_20260914
LOG=$D/auto_release_r11.log
IRIS="uv run --no-sync iris --config lib/iris/config/marin.yaml"
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
PLAN=$D/plan_v6e4_r12.json
OVERLAP=.agents/projects/mariner_ladder_accuracy_20260914/plan_v5p_cpu_staging.json
NATIVE=.agents/projects/mariner_ladder_accuracy_20260914/comparison/native_wandb_summaries.json
CAN_C=/calvinxu/table9-accuracy-choices-v6e4-canary-20260915-r12
CAN_G=/calvinxu/table9-accuracy-generation-v6e4-canary-20260915-r12
FULL_C=/calvinxu/table9-accuracy-choices-v6e4-full-20260915-r12
FULL_G=/calvinxu/table9-accuracy-generation-v6e4-full-20260915-r12
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
  log "stage1 canaries: choices=$sc generation=$sg"
  [[ $sc == succeeded && $sg == succeeded ]] && break
  if [[ $sc == failed || $sc == killed || $sg == failed || $sg == killed ]]; then
    log "STOP: canary ended choices=$sc generation=$sg; nothing released"; exit 1
  fi
  sleep 300
done

# Stage 2: release the full runs (idempotent via marker)
if [[ ! -f $D/auto_release_r12.released ]]; then
  while IFS= read -r cmd; do
    [[ -z $cmd ]] && continue
    job=$(sed -E 's/.*--job-name ([^ ]+).*/\1/' <<<"$cmd")
    if ! uv run python -m experiments.domain_phase_mix.east5_launch_safety $SAFETY_ARGS --command "$cmd" >> "$LOG" 2>&1; then
      log "STOP: east5 safety validation failed for $job"; exit 1
    fi
    ( set -a; source ~/.zshrc.secrets; set +a; eval "$cmd" ) 2>&1 | sed -E "$REDACT" > "$D/submit_$job.log"
    if ! grep -q "Job submitted: /calvinxu/$job" "$D/submit_$job.log"; then
      log "STOP: submission of $job did not confirm; see submit_$job.log"; exit 1
    fi
    log "released $job"
    uv run fieldbook job add --experiment $EXP --name "$job" --status running --external-system iris --external-id "/calvinxu/$job" --launcher iris --started-at "$(date -u +%FT%TZ)" --attr "protocol.plan=plan_v6e4_r12.json" --json >/dev/null 2>&1 || log "warning: fieldbook registration failed for $job"
  done < "$D/launch_v6e4_r12_full_commands.txt"
  touch "$D/auto_release_r12.released"
fi

# Stage 3: wait for both full parents
t0=$(date +%s)
while :; do
  fc=$(state $FULL_C); fg=$(state $FULL_G)
  log "stage3 full: choices=$fc [$(children $FULL_C)] generation=$fg [$(children $FULL_G)] elapsed=$(( ($(date +%s) - t0) / 60 ))min"
  [[ $fc == succeeded && $fg == succeeded ]] && break
  if [[ $fc == failed || $fc == killed || $fg == failed || $fg == killed ]]; then
    log "STOP: full run ended choices=$fc generation=$fg; grading not started"; exit 1
  fi
  sleep 600
done
for j in $FULL_C $FULL_G; do
  uv run fieldbook job add --experiment $EXP --external-system iris --external-id "$j" --status succeeded --finished-at "$(date -u +%FT%TZ)" --update-existing --json >/dev/null 2>&1 || log "warning: fieldbook update failed for $j"
done

# Stage 4: grade, report, analyze
log "stage4 grading"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.grade_table9_accuracy --plan "$PLAN" > "$D/grade_r11.log" 2>&1 || { log "STOP: grading failed; see grade_r11.log"; exit 1; }
log "stage4 report"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.report_table9_accuracy --plan "$PLAN" --overlap-plan "$OVERLAP" --output "$D/coverage_r11" > "$D/report_r11.log" 2>&1 || { log "STOP: report failed; see report_r11.log"; exit 1; }
log "stage4 analyze"
uv run --offline --no-sync python experiments/domain_phase_mix/analyze_table9_accuracy_vs_bpb.py --coverage "$D/coverage_r11/coverage.json" --native-summary "$NATIVE" --baseline proportional --candidate unimax8 --output "$D/accuracy_vs_bpb_r11" > "$D/analyze_r11.log" 2>&1 || { log "STOP: analysis failed; see analyze_r11.log"; exit 1; }
log "DONE: coverage_r11/ and accuracy_vs_bpb_r11/ written"
