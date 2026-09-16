# StarCoder WSD80 gradient-mechanism contextual report review

## Scope

Presentation-only revision at
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_mechanism_contextual_plots_v17_20260822`.
The frozen v8 scientific tables and multiplicity analysis were not changed.

## Review history

The first Opus 5 review found two blockers: moved-switch source panels were incorrectly described as H1 rather than
H5, and tied `0.80T` markers still said `Phase 2 begins` despite representing learning-rate decay. A follow-up found
that tied source curves pool preregistered H1 checkpoints with later H2 or H3 checkpoints, so describing the whole
curve as H1 was also inaccurate.

The renderer now explains that mixed provenance separately for Full, m100a, and m100b tied source selections;
describes moved-switch selections as H5; and labels tied and moved-switch temporal markers as LR-decay and data-switch
events respectively.

## Final verdict

**PASS.** The final read-only Opus 5 review verified:

- Contextual explanations cover every selector button: `5/5`, `5/5`, `5/5`, `11/11`, and `53/53` across the five
  interactive panels.
- Full, m100a, and m100b tied source explanations accurately identify pooled H1 plus H2/H3 checkpoint provenance.
- Both moved-switch source selections remain correctly identified as H5.
- No stale `Phase 2 begins`, `phase boundary`, `policy switch`, or global selector-glossary wording remains.
- All four target-source trajectory axes remain linear true-time axes over `[0.075, 1.025]` with `.3~f` tick format.
- Frozen scientific hashes match the v8 release and `scientific_values_changed` remains false.

Claude Code was invoked through the `plambdafour@proton.me` OAuth subscription with `claude-opus-5`, max effort,
and read-only `Read`, `Grep`, and `Glob` tools. `ANTHROPIC_API_KEY` was removed from the child environment.
