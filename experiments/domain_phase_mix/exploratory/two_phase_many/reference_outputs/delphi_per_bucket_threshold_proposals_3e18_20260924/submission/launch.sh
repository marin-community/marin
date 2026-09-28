#!/bin/bash
# One-threshold-per-bucket MARINER ablation validated at 3e18: six runs (three trainer seeds x two objectives) on the
# protocol of the 2026-09-14 convex/additive batch. TPU=v6e-8 (default, us-east5-b, as every Table 3 row) or
# TPU=v5p-8 (us-east5-a). Direct form: the executor is the top-level job (a nested `iris job run` inside a task is
# refused since 2026-09-24). Submitted from a subshell that sources the secrets file (output redacted), validated by
# east5_launch_safety on the equivalent `iris job run` command, registered in Fieldbook. SUBMIT=1 actually submits;
# without it the script stops after the safety check.
set -euo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin
O=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_per_bucket_threshold_proposals_3e18_20260924
TPU=${TPU:-v6e-8}
case "$TPU" in v6e-8) ZONE=us-east5-b; TAG=v6e8;; v5p-8) ZONE=us-east5-a; TAG=v5p8;; *) echo "unknown TPU $TPU"; exit 1;; esac
JOB=dm-delphi-3e18-per-bucket-threshold-$TAG-20260924
MODULE=experiments.domain_phase_mix.launch_delphi_per_bucket_threshold_3e18
ARGS="--tpu-type $TPU --tpu-region us-east5 --tpu-zone $ZONE --max-concurrent 6 --prefix gs://marin-us-east5"
INCLUDES="--working-dir-include experiments/domain_phase_mix/exploratory/general_scaling_models.py --working-dir-include experiments/domain_phase_mix/exploratory/dsre_ceq_tools.py --working-dir-include $O/candidate_weights.csv"
EXCLUDES="--working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip --working-dir-exclude cache/ --working-dir-exclude tmp/"
TAIL="--no-wait --no-preemptible --job-name $JOB --region us-east5 --zone us-east5-a --priority interactive --enable-extra-resources --cpu 1 --memory 4GB --disk 20GB --timeout 604800 --extra cpu -e MARIN_PREFIX gs://marin-us-east5 -e MARIN_EXECUTOR_STRICT 1 -e HF_HUB_DISABLE_XET 1 -- python -m $MODULE $ARGS"
EQUIV="iris --config lib/iris/config/marin.yaml job run $TAIL"
WRAP="UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml $EXCLUDES $INCLUDES -- $TAIL"
echo "$EQUIV" > "$O/submission/equivalent_iris_command_$TAG.txt"; echo "$WRAP" > "$O/submission/launch_command_$TAG.sh"
uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety --expected-child-zone $ZONE --command "$EQUIV" 2>&1 | grep -v "no effect" | tee "$O/submission/east5_launch_safety_$TAG.log" | tail -2
grep -q 'safety check passed' "$O/submission/east5_launch_safety_$TAG.log"
if [ "${SUBMIT:-0}" != "1" ]; then echo "safety check passed for $JOB; set SUBMIT=1 to submit"; exit 0; fi
( set -a; source ~/.zshrc.secrets; set +a; bash "$O/submission/launch_command_$TAG.sh" ) 2>&1 | sed -E 's/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g' | tee "$O/submission/submit_$TAG.log" | grep -i 'bundle size\|submitted\|error\|Traceback\|exceed' | cut -c1-200
grep -q "Job submitted: /calvinxu/$JOB" "$O/submission/submit_$TAG.log"
if [ ! -f "$O/submission/fieldbook_experiment_id.txt" ]; then
  uv run --offline --no-sync fieldbook experiment create --json --name "MARINER one-threshold-per-bucket ablation validated at 3e18" --description "Table 1 ablation freeing the harm threshold per bucket, fitted on the frozen Qwen3 3e18 swarm by the heldout-stage protocol and optimized without cap or penalty (cmp_u_pbt_cap06, cmp_t9_pbt_cap08); three trainer seeds per objective at the seed-matched data seeds, paired with the frozen procedure's validation and repeats (Table 3 row)." --tag data-mixing --tag validation --tag delphi-3e18 2>/dev/null | uv run --offline --no-sync python -c "import json,sys; print(json.load(sys.stdin)['id'])" > "$O/submission/fieldbook_experiment_id.txt"
fi
EXP=$(cat "$O/submission/fieldbook_experiment_id.txt")
uv run --offline --no-sync fieldbook job add --experiment "$EXP" --name "$JOB" --status running --external-system iris --external-id "/calvinxu/$JOB" --launcher experiments/domain_phase_mix/launch_delphi_per_bucket_threshold_3e18.py --command "$WRAP" --started-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --attr "launch.runs=cmp_u_pbt_cap06 x3 seeds, cmp_t9_pbt_cap08 x3 seeds" --attr "launch.run_ids=7490000+,7490100+" --attr "launch.tpu=$TPU $ZONE" | tail -1
echo "submitted $JOB at $(date '+%H:%M %Z'); Fieldbook experiment $EXP"
