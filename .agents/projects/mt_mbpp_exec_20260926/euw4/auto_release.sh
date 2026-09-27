#!/bin/bash
# MT-MBPP on v6e-4 in europe-west4-a for Proportional, UniMax-8 and MARINER, moved there because us-east5-b had no
# v6e-4 capacity (Calvin approved the one-time copy of the three checkpoints on 2026-09-26). Waits for the copy,
# relocates plan_east5.json (same request manifest, recorded parity), submits the canary, releases the full run on
# canary success, then waits. Detached; progress in euw4/auto_release.log; stops and logs instead of retrying.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
P=.agents/projects/mt_mbpp_exec_20260926
E=$P/euw4
LOG=$E/auto_release.log
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
GUARD="--expected-region europe-west4 --expected-zone europe-west4-a --expected-bucket-prefix gs://marin-eu-west4"
CAN=/calvinxu/mt-mbpp-accuracy-v6e4-canary-euw4-20260926
FULL=/calvinxu/mt-mbpp-accuracy-v6e4-full-euw4-20260926
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
state() { uv run --no-sync iris --config lib/iris/config/marin.yaml job describe "$1" 2>/dev/null | awk '/^State:/{print $2; exit}'; }
children() {
  uv run --no-sync iris --config lib/iris/config/marin.yaml job list --prefix "$1/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// {c[$2]++} END {printf "succeeded=%d running=%d pending=%d failed=%d killed=%d", c["succeeded"], c["running"], c["pending"], c["failed"], c["killed"]}'
}
submit() {  # $1 = command file
  local cmd job
  cmd=$(cat "$1"); job=$(sed -E 's/.*--job-name ([^ ]+).*/\1/' <<<"$cmd")
  # shellcheck disable=SC2086
  uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety $GUARD --command "$cmd" >> "$LOG" 2>&1 || { log "STOP: guard failed for $job"; return 1; }
  eval "$cmd" > "$E/submit_$job.log" 2>&1
  grep -q "Job submitted: /calvinxu/$job" "$E/submit_$job.log" || { log "STOP: submission of $job did not confirm"; return 1; }
  log "submitted $job"
  uv run fieldbook job add --experiment $EXP --name "$job" --status running --external-system iris --external-id "/calvinxu/$job" --launcher iris --started-at "$(date -u +%FT%TZ)" --attr "protocol.plan=mt_mbpp_exec_20260926/plan_euw4.json" --json >/dev/null 2>&1 || log "warning: fieldbook registration failed for $job"
}
wait_for() {  # $1 = job, $2 = label; returns 0 on success
  local s
  while :; do
    s=$(state "$1"); log "$2=$s [$(children "$1")]"
    [[ $s == succeeded ]] && return 0
    [[ $s == failed || $s == killed ]] && { log "STOP: $2 $s"; return 1; }
    sleep 300
  done
}
log "chain started (pid $$)"
until grep -q "^.* DONE" "$E/copy_checkpoints.log" 2>/dev/null; do
  grep -q FAILED "$E/copy_checkpoints.log" 2>/dev/null && { log "STOP: checkpoint copy failed"; exit 1; }
  sleep 60
done
log "copy verified"
if [[ ! -f $P/plan_euw4.json ]]; then
  uv run --offline --no-sync python -m experiments.domain_phase_mix.evaluate_table9_accuracy --plan $P/plan_euw4.json \
    --relocate-from $P/plan_east5.json --region europe-west4 --requests $P/requests >> "$LOG" 2>&1 || { log "STOP: relocation failed"; exit 1; }
  log "relocated plan written"
fi
[[ -f $E/canary.submitted ]] || { submit "$E/launch_canary_command.txt" && touch "$E/canary.submitted"; } || exit 1
wait_for $CAN canary || exit 1
[[ -f $E/full.submitted ]] || { submit "$E/launch_full_command.txt" && touch "$E/full.submitted"; } || exit 1
wait_for $FULL full || exit 1
log "DONE"
