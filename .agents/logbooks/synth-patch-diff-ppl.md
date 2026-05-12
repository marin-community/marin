# Synthetic patch-diff PPL probe: Research Logbook

## Scope
- Goal: Archive the synthetic target-only PPL provider and pinned HF dataset for future reference without adding it to the regular eval suite.
- Primary metric: Marin 32B minus Qwen3 32B bpb on the all-available dashboard run.
- Constraints: Keep this branch isolated to the provider, focused tests where present, and this logbook.

## Baseline
- Issue: https://github.com/marin-community/marin/issues/5662
- Parent issue: https://github.com/marin-community/marin/issues/5618
- Base branch: origin/codex/supervised-lm-format
- HF dataset: marin-community/synth-patch-diff-ppl
- HF revision: 7b7e44357aef62325a69d9b3e56241d90a277e5c
- Dashboard run: main_gap_all_available_diag_50eb41089_v18_surface_form_probes
- Result: Marin minus Qwen = -0.0685 bpb over 1,536 docs and 217,344 scored bytes.

## Experiment Log
### 2026-05-11 - Archive snapshot
- Hypothesis: Surface-form continuation probes can explain targeted Marin/Qwen gaps without becoming recurring benchmark obligations.
- Command: Providers were scored in Iris job /dlwh/main-gap-32b-all-available-diag-50eb41089-v18-surface-form-probes.
- Config: supervised target-only rows with input/target fields, 1000 validation rows per HF config.
- Result: Marin minus Qwen = -0.0685 bpb over 1,536 docs and 217,344 scored bytes.
- Interpretation: Keep as archived diagnostic evidence. Re-run only when investigating related surface-form failures.
- Next action: Do not wire this branch into the regular eval suite unless the probe graduates into a maintained benchmark.
