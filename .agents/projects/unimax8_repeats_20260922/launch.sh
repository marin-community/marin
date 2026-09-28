#!/bin/bash
# UniMax-8 trainer-seed repeats at Delphi 3e18 (seeds 1 and 2 at the ladder run's data seed 660700) with chained
# Table 9 BPB evaluations, via the issue #6607 baseline launcher (--trainer-seeds/--data-seed added 2026-09-22).
# Same command shape as the cap sweep's launch.sh: east5 guard, secrets sourced in a subshell with redacted output,
# Fieldbook registration.
set -euo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin
D=.agents/projects/unimax8_repeats_20260922
JOB=dm-delphi-unimax8-repeats-3e18-20260922
EXP=exp_01kvvvv6zxrf0j7tkp4f7k6y66
MODULE=experiments.domain_phase_mix.launch_delphi_baseline_mixtures
ARGS="--mixtures unimax8 --target-budgets 3e18 --tpu-region us-east5 --tpu-zone us-east5-a --table9-tpu-zone us-east5-b --run-id-base 660740 --data-seed 660700 --trainer-seeds 1 2 --with-table9-eval --max-concurrent 4"
TAIL="--no-wait --no-preemptible --job-name $JOB --region us-east5 --zone us-east5-a --priority interactive --enable-extra-resources --cpu 1 --memory 2GB --disk 16GB --timeout 604800 --extra cpu -e MARIN_PREFIX gs://marin-us-east5 -e MARIN_EXECUTOR_STRICT 1 -e HF_HUB_DISABLE_XET 1 -- python -m $MODULE $ARGS"
INNER="iris --config lib/iris/config/marin.yaml job run $TAIL"
WRAP="UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip --working-dir-exclude cache/ --working-dir-exclude tmp/ --working-dir-include experiments/domain_phase_mix/exploratory/general_scaling_models.py --working-dir-include experiments/domain_phase_mix/exploratory/dsre_ceq_tools.py -- $TAIL"
echo "$INNER" > "$D/inner_command.txt"; echo "$WRAP" > "$D/launch_command.sh"
uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety --expected-child-zone us-east5-a --expected-table9-child-zone us-east5-b --command "$INNER" 2>&1 | grep -v "no effect" | tee "$D/east5_launch_safety.log" | tail -2
grep -q 'safety check passed' "$D/east5_launch_safety.log"
( set -a; source ~/.zshrc.secrets; set +a; bash "$D/launch_command.sh" ) 2>&1 | sed -E 's/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g' | tee "$D/submit.log" | grep -i 'bundle size\|submitted\|error\|Traceback\|exceed' | cut -c1-200
grep -q "Job submitted: /calvinxu/$JOB" "$D/submit.log"
uv run fieldbook job add --experiment $EXP --name "$JOB" --status running --external-system iris --external-id "/calvinxu/$JOB" --launcher experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py --command "$WRAP" --started-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --attr "launch.repeats=unimax8 trainer seeds 1 and 2 at data seed 660700, 3e18, with table9 evals" --attr "launch.run_ids=660740-660741" --attr placement.parent=us-east5-a --attr placement.child=us-east5-a --attr placement.table9=us-east5-b --json | python3 -c "import json,sys; d=json.load(sys.stdin); print('fieldbook job', d.get('id'))" | tee "$D/fieldbook_job.txt"
