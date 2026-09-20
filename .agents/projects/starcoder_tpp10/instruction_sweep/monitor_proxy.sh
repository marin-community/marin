#!/bin/zsh
# Detached watch for the instruction (Dolmino FLAN) proxy sweep: logs parent/child states every 15 minutes and, once the
# parent is terminal, collects the archived plan and writes measurements plus the analysis.
# Start with: nohup zsh monitor_proxy.sh > monitor_proxy.nohup 2>&1 &
set -uo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin
D=.agents/projects/starcoder_tpp10/instruction_sweep
JOB=${SWEEP_JOB:-/calvinxu/tpp10-instruction-proxy-sweep}
PLAN=${SWEEP_PLAN:-plan_proxy.json}
RESULTS=${SWEEP_RESULTS:-results}
LOG=$D/monitor_${RESULTS}.log
IRIS="uv run --offline --no-sync iris --config lib/iris/config/marin.yaml"
export MARIN_PREFIX=gs://marin-us-central1

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
redact() { sed -E 's/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g'; }

log "monitor started for $JOB (poll every 15 min)"
tick=0
while true; do
  tick=$((tick + 1))
  state=$(eval $IRIS job describe "$JOB" 2>/dev/null | redact | awk '/^State:/ {print $2; exit}')
  children=$(eval $IRIS query "\"SELECT state, COUNT(*) AS n FROM jobs WHERE root_job_id='$JOB' AND depth=2 GROUP BY state\"" 2>/dev/null | grep -v 'Loading config\|scopes' | tail -n +2 | tr -s ' ' | tr '\n' ';')
  log "tick $tick parent=${state:-unknown} children(state;n)=[${children}]"
  case "$state" in
    succeeded|failed|killed)
      log "parent terminal: $state; collecting"
      if uv run --offline --no-sync python -m experiments.domain_phase_mix.launch_tpp10_instruction_sweep \
           --plan $D/$PLAN --collect $D/$RESULTS/measurements.csv >> "$LOG" 2>&1; then
        log "collected: $D/$RESULTS/measurements.csv and analysis.json"
        uv run --offline --no-sync python - >> "$LOG" 2>&1 <<'PY'
import json
import os; a = json.load(open(os.path.join(".agents/projects/starcoder_tpp10/instruction_sweep", os.environ.get("SWEEP_RESULTS", "results"), "analysis.json")))
for m in ["sciq_bpb", "arc_easy_bpb", "arc_challenge_bpb", "openbookqa_bpb", "qasc_bpb", "dolmino_flan_heldout_bpb", "macro_bpb", "paloma_bpb"]:
    r = a["metrics"][m]
    print(f"{m:<24} min {r['minimum_epochs']:.2f} ep (p={r['minimum_percent']}%) boundary={r['boundary_minimum']} "
          f"neighbors={ {k: round(v, 2) for k, v in r['neighbor_excess_percent'].items()} } "
          f"excess%={[round(v, 1) for v in r['excess_percent']]}")
PY
      else
        log "collect failed (see above); leaving results for manual collection"
      fi
      log "monitor exiting"
      exit 0
      ;;
  esac
  sleep 900
done
