# StarCoder WSD80 self-contained gradient report v9: CC review

Date: 2026-08-22

Reviewer: Claude Code, `claude-opus-5`, max effort, read-only tools, OAuth subscription account
`plambdafour@proton.me`, with `ANTHROPIC_API_KEY` removed from the child environment.

Reviewed files:

- `experiments/domain_phase_mix/exploratory/two_phase_many/render_starcoder_wsd80_gradient_mechanism_self_contained_20260822.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_mechanism_self_contained_plots_v9_20260822/`
- frozen scientific release `starcoder_wsd80_gradient_plot_completion_v8_20260822/release.json`

## Review sequence

The first review verified that label rewriting preserved Plotly visibility masks, target-source checkpoints were placed
at their true training-update fractions, and the v8 scientific values and provenance remained unchanged. It found five
presentation blockers: a split m100a dropdown group, two-decimal zoom ticks, incomplete H3 descriptions, inconsistent
evidence names, and misleading `StarCoder support` wording.

After those fixes, the second review verified all five were resolved. It found one remaining cross-panel blocker: the
fixed 0.80T LR-decay reference had been added to moved-switch target trajectories but not moved-switch source-alignment
trajectories. The index therefore promised a reference line that one temporal panel lacked.

The renderer was updated so both temporal panels distinguish the dark data-switch boundary from the slate dashed
0.80T LR-decay reference. The index now states that the dark 0.80T line on tied policies is LR-decay onset and does not
represent a mixture change. A final wording check caught and corrected one stale `orange dashed` phrase after the
reference color changed to slate.

## Final verdict

PASS. The final review confirmed the rendered target-trajectory footnote calls the `#6f7f87` LR-decay reference a
slate dashed bar, does not point to the orange StarCoder series, and matches the renderer. No scientific values were
changed.
