#!/bin/bash
# Executable MT-MBPP accuracy on the four 1e21 checkpoints: us-east5 (Proportional, UniMax-8, MARINER; plan_east5.json)
# and us-east1 (Olmix, whose checkpoint copy is there; plan_east1.json), v6e-4 children. Both canaries were submitted
# by hand at 13:01 PDT on 2026-09-26. Each region's full run is released as soon as its own canary parent succeeds;
# then the chain waits for both full parents. Grading is separate (it needs the translated tests).
# Detached; progress goes to auto_release.log. Stops a region and logs instead of retrying if its parent fails.
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin || exit 1
P=.agents/projects/mt_mbpp_exec_20260926
LOG=$P/auto_release.log
IRIS="uv run --no-sync iris --config lib/iris/config/marin.yaml"
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
GUARD_east5="--expected-child-zone us-east5-b"
GUARD_east1="--expected-region us-east1 --expected-zone us-east1-d --expected-bucket-prefix gs://marin-us-east1"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
state() { $IRIS job describe "$1" 2>/dev/null | awk '/^State:/{print $2; exit}'; }
children() {
  $IRIS job list --prefix "$1/" 2>/dev/null | awk 'NR>1 && $1 ~ /^\// {c[$2]++} END {printf "succeeded=%d running=%d pending=%d failed=%d killed=%d", c["succeeded"], c["running"], c["pending"], c["failed"], c["killed"]}'
}
release() {  # $1 = region short name
  local r=$1 cmd job guard
  cmd=$(grep -- "-full-$r-" $P/launch_full_commands.txt)
  job=$(sed -E 's/.*--job-name ([^ ]+).*/\1/' <<<"$cmd")
  guard=GUARD_$r
  # shellcheck disable=SC2086
  if ! uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety ${!guard} --command "$cmd" >> "$LOG" 2>&1; then
    log "STOP $r: region safety validation failed for $job"; return 1
  fi
  eval "$cmd" > "$P/submit_$job.log" 2>&1
  if ! grep -q "Job submitted: /calvinxu/$job" "$P/submit_$job.log"; then
    log "STOP $r: submission of $job did not confirm; see submit_$job.log"; return 1
  fi
  touch "$P/full_$r.released"
  log "released $job"
  uv run fieldbook job add --experiment $EXP --name "$job" --status running --external-system iris --external-id "/calvinxu/$job" --launcher iris --started-at "$(date -u +%FT%TZ)" --attr "protocol.plan=mt_mbpp_exec_20260926/plan_$r.json" --json >/dev/null 2>&1 || log "warning: fieldbook registration failed for $job"
}

log "chain started (pid $$)"
# Region outcomes are marker files (macOS /bin/bash 3.2 has no associative arrays): $P/outcome_<region>.
t0=$(date +%s)
while :; do
  line=""
  for r in east5 east1; do
    [[ -f $P/outcome_$r ]] && { line+=" $r=$(cat $P/outcome_$r)"; continue; }
    can=/calvinxu/mt-mbpp-accuracy-v6e4-canary-$r-20260926
    full=/calvinxu/mt-mbpp-accuracy-v6e4-full-$r-20260926
    if [[ ! -f $P/full_$r.released ]]; then
      s=$(state $can)
      line+=" $r canary=$s [$(children $can)]"
      if [[ $s == succeeded ]]; then release $r || echo stopped > $P/outcome_$r; fi
      if [[ $s == failed || $s == killed ]]; then log "STOP $r: canary $s; full run not released"; echo stopped > $P/outcome_$r; fi
    else
      s=$(state $full)
      line+=" $r full=$s [$(children $full)]"
      if [[ $s == succeeded ]]; then echo succeeded > $P/outcome_$r; fi
      if [[ $s == failed || $s == killed ]]; then log "STOP $r: full parent $s (resubmitting the same command resumes completed tasks)"; echo stopped > $P/outcome_$r; fi
    fi
  done
  log "status$line elapsed=$(( ($(date +%s) - t0) / 60 ))min"
  [[ -f $P/outcome_east5 && -f $P/outcome_east1 ]] && break
  sleep 300
done
log "DONE east5=$(cat $P/outcome_east5) east1=$(cat $P/outcome_east1)"
