#!/bin/bash
# Four Uncheatable optima (MARINER, matched Olmix, tuned RegMix, released RegMix) trained at Llama 160M/1.2B and
# 200M/6B (Figure 28 deployed markers), eight v5p-8 runs in us-east5-a (the 39-bucket pools exist only in us-east5;
# the Olmix 1e21 runs use v6e-64 in us-east5-b, a different accelerator pool).
# Direct form: the top-level job runs the executor itself, which launches the training children through the
# in-task Iris client. The earlier double-wrap form (a nested `iris job run` inside the task) was refused with
# HTTP 403 Forbidden at 19:16 PDT on 2026-09-24 although the same form worked on 2026-09-22.
# Validated by east5_launch_safety on the equivalent `iris job run` command, submitted from a subshell that sources
# the secrets file (output redacted), registered in Fieldbook. The runtime mixture CSVs live under the excluded
# exploratory/ tree, so they are added explicitly, as is exploratory/two_phase_many/two_phase_many.csv, which the
# qsplit240 launcher chain reads at import time (retry1 failed on it at 19:38 PDT; found with a staged dry-run).
# retry2 (19:45 PDT) raised under MARIN_EXECUTOR_STRICT=1 because the launcher built its steps outside
# executor_context(); retry3 builds them inside it (strict dry run passes).
# retry3 (19:55 PDT) failed in both cache_eval_datasets steps: the step returned a str path and the executor's
# ArtifactRecord.result requires a mapping; retry4 carries the fix in lib/marin/.../eval_dataset_cache.py.
# retry4 (20:18 PDT) trained 7 of 8 runs; the tuned-RegMix 200M/6B run stalled near step 18k through 17 preemptions (attempts of
# 2-20 min, 4 min of compilation each, 10-min temporary checkpoints). retry5 (25 Sep) resumes it with 3-minute checkpoints; output
# paths are unchanged, so the executor serves the seven finished runs from cache.
set -euo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin
D=.agents/projects/mariner_optimum_scale_transfer_20260924
JOB=dm-mopt-scale-transfer-20260924-retry5
MODULE=experiments.domain_phase_mix.launch_mariner_optimum_scale_transfer
ARGS="--tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a --max-concurrent 8 --checkpoint-minutes 3"
R=experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs
INCLUDES="--working-dir-include experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv --working-dir-include experiments/domain_phase_mix/exploratory/general_scaling_models.py --working-dir-include experiments/domain_phase_mix/exploratory/dsre_ceq_tools.py --working-dir-include $R/delphi_corrected_screen_20260908/materialized_flat15_nocap/runtime_materialization/candidate_weights.csv --working-dir-include $R/delphi_matched_olmix_3e18_20260908/candidate_weights.csv --working-dir-include $R/delphi_comparator_proposals_3e18_20260909/candidate_weights.csv --working-dir-include $R/regmix_official_rerun_20260913/candidate_weights.csv"
EXCLUDES="--working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip --working-dir-exclude cache/ --working-dir-exclude tmp/"
TAIL="--no-wait --no-preemptible --job-name $JOB --region us-east5 --zone us-east5-a --priority interactive --enable-extra-resources --cpu 1 --memory 2GB --disk 16GB --timeout 604800 --extra cpu -e MARIN_PREFIX gs://marin-us-east5 -e MARIN_EXECUTOR_STRICT 1 -e HF_HUB_DISABLE_XET 1 -- python -m $MODULE $ARGS"
EQUIV="iris --config lib/iris/config/marin.yaml job run $TAIL"
WRAP="UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml $EXCLUDES $INCLUDES -- $TAIL"
echo "$EQUIV" > "$D/equivalent_iris_command_retry5.txt"; echo "$WRAP" > "$D/launch_command_retry5.sh"
uv run --offline --no-sync python -m experiments.domain_phase_mix.east5_launch_safety --expected-child-zone us-east5-a --command "$EQUIV" 2>&1 | grep -v "no effect" | tee "$D/east5_launch_safety_retry5.log" | tail -2
grep -q 'safety check passed' "$D/east5_launch_safety_retry5.log"
( set -a; source ~/.zshrc.secrets; set +a; bash "$D/launch_command_retry5.sh" ) 2>&1 | sed -E 's/(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+/\1=<redacted>/g; s/[0-9a-f]{40}/<redacted-40hex>/g' | tee "$D/submit_retry5.log" | grep -i 'bundle size\|submitted\|error\|Traceback\|exceed' | cut -c1-200
grep -q "Job submitted: /calvinxu/$JOB" "$D/submit_retry5.log"
EXP=$(cat "$D/fieldbook_experiment_id.txt")
uv run --offline --no-sync fieldbook job add --experiment "$EXP" --name "$JOB" --status running --external-system iris --external-id "/calvinxu/$JOB" --launcher experiments/domain_phase_mix/launch_mariner_optimum_scale_transfer.py --command "$WRAP" --started-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --attr "launch.runs=mariner_u,olmix_u,regmix_tun_u,regmix_rel_u at 60m_1p2b and 300m_6b" --attr "launch.run_ids=795000-795003" --attr "launch.tpu=v5p-8 us-east5-a" --attr "launch.form=direct (executor is the top-level job)" | tail -2
echo "submitted $JOB at $(date '+%H:%M %Z'); Fieldbook experiment $EXP"
