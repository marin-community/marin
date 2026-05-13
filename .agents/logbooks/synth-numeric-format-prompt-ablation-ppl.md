# Synthetic numeric format prompt-ablation PPL probe: Research Logbook

## Scope
- Goal: Archive the synthetic numeric/structured-format target-only PPL providers and pinned HF datasets for future reference without automatically adding the old role-framed suite to routine training-run tracking.
- Parent issue: https://github.com/marin-community/marin/issues/5614
- Tracking selection issue: https://github.com/marin-community/marin/issues/5686
- Base branch: origin/main
- HF datasets:
  - `marin-community/synth-numeric-format-ppl` @ `57cd2ab0f7b507cc2598d0cceeb95db47a739653`
  - `marin-community/synth-numeric-format-prompt-ablation-ppl` @ `bca4b9413bc72ae66614da99dafcc87ab7bc074f`
- Dashboard runs:
  - `main_gap_all_available_diag_50eb41089_v16_numeric_format_ppl`
  - `main_gap_all_available_diag_50eb41089_v17b_numeric_prompt_ablation_mmlu_subject_fallback`

## Experiment Log
### 2026-05-13 - Archive snapshot
- Hypothesis: Numeric and structured formatting gaps should be separated from chat-role framing artifacts.
- Result: The old `User:` / `Assistant:` framing produced large first-target-token artifacts, especially for Qwen. Neutral `arrow` and `equals` variants are the better candidates for routine tracking.
- Next action: If #5686 graduates a small `format_readiness`/`surface_form_sentinels` suite, use the neutral prompt-ablation subsets and avoid the old role-framed aggregate as a headline metric.
