#!/bin/bash
# Table-9 accuracy for the Olmix OlmoBaseEval Easy 1e21 checkpoint, choices and generation on v6e-4 in us-east1-d.
# Calvin (2026-09-26) moved this off us-east5-b so it cannot compete with OLM-U's v6e-64 slices; only the checkpoint
# was copied to us-east1 (copy_checkpoint.log), and the request files were uploaded from their local frozen copies.
# The overlap suite stays on v5p-8 in us-east5-a beside the original checkpoint. Submits the overlap suite and both
# canaries, releases the full runs when both canaries succeed, waits for everything, then grades and reports.
# Detached from any session; progress goes to auto_release.log. Stops and logs instead of retrying if a parent fails.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
D=.agents/projects/mariner_table9_accuracy_20260914/olmix_t9_1e21_east1
LOG=$D/auto_release.log
IRIS="uv run --no-sync iris --config lib/iris/config/marin.yaml"
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
PLAN=$D/backfill_plan.json
OVERLAP=.agents/projects/mariner_table9_accuracy_20260914/olmix_t9_1e21/overlap_plan.json
CAN_C=/calvinxu/table9-accuracy-choices-v6e4-canary-olmix-t9-1e21-east1-20260926
CAN_G=/calvinxu/table9-accuracy-generation-v6e4-canary-olmix-t9-1e21-east1-20260926
FULL_C=/calvinxu/table9-accuracy-choices-v6e4-full-olmix-t9-1e21-east1-20260926
FULL_G=/calvinxu/table9-accuracy-generation-v6e4-full-olmix-t9-1e21-east1-20260926
OVERLAP_JOB=/calvinxu/mariner-ladder-accuracy-olmix-t9-1e21-v5p-20260926
EAST1_GUARD="--expected-region us-east1 --expected-zone us-east1-d --expected-bucket-prefix gs://marin-us-east1"
EAST5_GUARD="--expected-child-zone us-east5-a"
REDACT='s/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g'

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
state() { $IRIS job describe "$1" 2>/dev/null | awk '/^State:/{print $2; exit}'; }
children() {  # succeeded/running/pending/failed counts of a parent's direct children
  $IRIS job list --prefix "$1/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// {c[$2]++} END {printf "succeeded=%d running=%d pending=%d failed=%d killed=%d", c["succeeded"], c["running"], c["pending"], c["failed"], c["killed"]}'
}
submit_all() {  # $1 = guard flags, $2 = command file, $3 = fieldbook plan tag
  while IFS= read -r cmd; do
    [[ -z $cmd ]] && continue
    job=$(sed -E 's/.*--job-name ([^ ]+).*/\1/' <<<"$cmd")
    # shellcheck disable=SC2086
    if ! uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety $1 --command "$cmd" >> "$LOG" 2>&1; then
      log "STOP: region safety validation failed for $job"; exit 1
    fi
    ( set -a; source ~/.zshrc.secrets; set +a; eval "$cmd" ) 2>&1 | sed -E "$REDACT" > "$D/submit_$job.log"
    if ! grep -q "Job submitted: /calvinxu/$job" "$D/submit_$job.log"; then
      log "STOP: submission of $job did not confirm; see submit_$job.log"; exit 1
    fi
    log "submitted $job"
    uv run fieldbook job add --experiment $EXP --name "$job" --status running --external-system iris --external-id "/calvinxu/$job" --launcher iris --started-at "$(date -u +%FT%TZ)" --attr "protocol.plan=$3" --json >/dev/null 2>&1 || log "warning: fieldbook add failed for $job"
  done < "$2"
}

log "chain started (pid $$)"
# Stage 0: overlap suite (us-east5) and both canaries (us-east1), idempotent via marker
if [[ ! -f $D/canaries.released ]]; then
  submit_all "$EAST5_GUARD" "$D/launch_overlap_command.txt" "olmix_t9_1e21/overlap_plan.json"
  submit_all "$EAST1_GUARD" "$D/launch_canary_commands.txt" "olmix_t9_1e21_east1/backfill_plan.json"
  touch "$D/canaries.released"
fi
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
  submit_all "$EAST1_GUARD" "$D/launch_full_commands.txt" "olmix_t9_1e21_east1/backfill_plan.json"
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
# Stage 4: grade and report (grading needs Docker)
log "stage4 grading"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.grade_table9_accuracy --plan "$PLAN" > "$D/grade.log" 2>&1 || { log "STOP: grading failed; see grade.log"; exit 1; }
log "stage4 report"
uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1 python -m experiments.domain_phase_mix.report_table9_accuracy --plan "$PLAN" --overlap-plan "$OVERLAP" --output "$D/coverage" > "$D/report.log" 2>&1 || { log "STOP: report failed; see report.log"; exit 1; }
log "DONE: coverage/ written"
