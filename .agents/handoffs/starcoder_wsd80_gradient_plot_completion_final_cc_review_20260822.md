# StarCoder WSD80 gradient plot-completion v8: final CC review

Date: 2026-08-22

Reviewer: Claude Code, `claude-opus-5`, maximum reasoning, read-only

Session: `16995bea-1942-4e0a-8fa6-e74088d7ab1c`

Account and billing provenance: `plambdafour@proton.me`, Claude subscription; API-key fallback disabled.

## Verdict

**PASS. No blockers.** The reviewer independently reconciled the delivered tables and all seven HTML panels. It confirmed the expected row counts, evidence-role partition, registered-state coverage for every cohort, structural terminal-state missingness, phase-boundary markers, frozen-table identity, and consistency between rendered prose and measured values.

Independent row-count reconciliation:

- Source geometry: `45,056` rows.
- Target-source utilities: `218,944` rows.
- Target-source choice alignment: `190,784` rows.
- Historical-overlap comparison: `9,856` rows and `49,280` scalar comparisons.

The maximum historical-overlap difference was `4.440892098500626e-16`, one floating-point ulp and far below the preregistered `5e-6` tolerance. This confirms that the historical recovery runtime reproduces the v10 calculations.

## Delivery Clarifications

`final` is absent from the three target-source panels by construction. At `final`, the learning rate is zero, so every corrected optimizer update is the zero vector and target-update cosine is undefined, not missing. Final rows request source gradients only, which is why `final` appears in the source-source panels and not the target-source panels.

The delivered provenance record is pinned by:

- Full strict audit: `288/288` groups, SHA-256 `a087c0c3c4a23430725d8855fd3aa0bad35d69bb28419c5ababa42e8d913ce64`.
- Multiplicity audit input: SHA-256 `851bafe4312e2995dcb7297313b0963728d88719e4ce635f1db266e4a600b8e9`.
- Base all-states geometry input: SHA-256 `89dc67e3fdbc0acc2359efb6a243b33ebee2ca146585501dca95e98b8d746546`.
- Render manifest: SHA-256 `755009eb78eeaf0666f02579ae653f650ebff1cc19eaa6402f0e84f3db17d688`.

## Non-Blocking Polish

The three target-source panels do not repeat the terminal-LR explanation in their local footnotes, although the index, source panels, release report, analysis contract, and materialization audit all state it. The reviewer recommended documenting it in this delivery note rather than changing the frozen plotter and invalidating the release hashes.

The render manifest itself does not embed the two external input hashes. They are pinned above and in the immutable release's `plot_inputs` record.

No changes to the frozen tables, plots, or release are required.

## Delivered Artifacts

- Coverage and recovery report: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_plot_completion_v8_20260822/report.md`.
- Complete tables: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_plot_complete_tables_v8_20260822/`.
- Complete interactive plots: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_mechanism_complete_plots_v8_20260822/`.

All 288 recovered groups were computed from saved checkpoints. No trajectory was retrained, and endpoint metrics were inaccessible to the recovery jobs.
