#!/bin/bash
# Regrade the four 1e21 checkpoints with the name-agnostic MBPP grader, then rebuild coverage, merges, analyses and
# the paper's accuracy tables (2026-09-26). Needs Docker (OrbStack). Log: results/adopt.log.
set -u
cd /Users/calvinxu/Projects/Work/Marin/marin
D=.agents/projects/mariner_table9_accuracy_20260914
R=$D/mbpp_regrade/results
A=$D/mariner_t9_1e21/accuracy_vs_bpb
UV="uv run --no-sync --with math-verify==0.8.0 --with antlr4-python3-runtime==4.11.1"
log() { echo "$(date -u +%FT%TZ) $*" >> $R/adopt.log; }
grade() { $UV python -m experiments.domain_phase_mix.grade_table9_accuracy --plan "$1" > "$2" 2>&1; }
report() { $UV python -m experiments.domain_phase_mix.report_table9_accuracy --plan "$1" --overlap-plan "$2" --output "$3" > "$4" 2>&1; }
log "start grading (r12 in parallel with MARINER then Olmix)"
grade $D/plan_v6e4_r12.json $R/grade_r12.log & r12=$!
{ grade $D/mariner_t9_1e21/backfill_plan.json $R/grade_mariner.log && grade $D/olmix_t9_1e21_east1/backfill_plan.json $R/grade_olmix.log; } & rest=$!
wait $r12 || { log "STOP: r12 grading failed; see grade_r12.log"; exit 1; }
wait $rest || { log "STOP: MARINER or Olmix grading failed; see grade_mariner.log / grade_olmix.log"; exit 1; }
log "graded"
report $D/plan_v6e4_r12.json .agents/projects/mariner_ladder_accuracy_20260914/plan_v5p_cpu_staging.json $D/coverage_r11 $R/report_r12.log || { log "STOP: r12 report"; exit 1; }
report $D/mariner_t9_1e21/backfill_plan.json $D/mariner_t9_1e21/overlap_plan.json $D/mariner_t9_1e21/coverage $R/report_mariner.log || { log "STOP: MARINER report"; exit 1; }
report $D/olmix_t9_1e21_east1/backfill_plan.json $D/olmix_t9_1e21/overlap_plan.json $D/olmix_t9_1e21_east1/coverage $R/report_olmix.log || { log "STOP: Olmix report"; exit 1; }
log "reported"
uv run --offline --no-sync python -c "
import json
D = '$D'
parts = json.load(open(f'{D}/coverage_r11/coverage.json')) + json.load(open(f'{D}/mariner_t9_1e21/coverage/coverage.json'))
open(f'{D}/mariner_t9_1e21/accuracy_vs_bpb/coverage_merged.json', 'w').write(json.dumps(parts, indent=1) + '\n')
" || { log "STOP: MARINER merge"; exit 1; }
for base in proportional unimax8; do
  uv run --offline --no-sync python experiments/domain_phase_mix/analyze_table9_accuracy_vs_bpb.py --coverage $A/coverage_merged.json \
    --native-summary $A/native_summaries_with_mariner.json --baseline $base --candidate mariner --output $A/vs_$base \
    > $A/analyze_vs_$base.log 2>&1 || { log "STOP: analysis vs $base"; exit 1; }
done
uv run --offline --no-sync python $D/olmix_t9_1e21_east1/accuracy_vs_bpb/merge_inputs.py > $R/merge_olmix.log 2>&1 || { log "STOP: Olmix merge"; exit 1; }
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/build_accuracy_component_table_20260922.py > $R/tables.log 2>&1 || { log "STOP: tables"; exit 1; }
log "DONE"
