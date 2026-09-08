## Opus 5 review: StarCoder WSD80 LR-onset gradient probe

Review provenance: Claude Code subscription session `9e714b02-1f3c-4e1b-8a60-1f230ecfed0b`, resumed after the first review found two blockers. Model `claude-opus-5`, maximum reasoning, read-only tools. The final review inspected the current runtime, tests, Levanter data-loader and LR-schedule behavior, ArtifactStep path construction, projection machinery, and the frozen training contract.

The repaired probe is safe to freeze and launch.

- The two 1,024-sequence halves now use a shared 2,048-sequence cyclic view with starts 0 and 1,024. The review traced Levanter's loader offsets and confirmed that both StarCoder and Nemotron consume disjoint logical indices 0-1,023 and 1,024-2,047. The behavioral test draws the complete panels and checks disjointness and union size.
- Split-half disattenuation is correctly wired. Both within-source signal dot products must be positive before the corrected cosine is defined; negative signals cannot manufacture a valid denominator.
- The primary is the noise-corrected raw-gradient statistic on projected trainable-trunk geometry, matching the existing gradient-onset artifact. Unprojected raw geometry is a sensitivity analysis and optimizer-update cosine remains secondary.
- Frozen expected state-equivalence classes are correct at all three schedule boundaries. Per-seed audits require identical model and optimizer fingerprints within each class and distinct fingerprints across classes, plus a shared training key.
- Result discovery now includes the ArtifactStep version segment, so completed rows are visible to the stage gate. Stage 1 is gated on complete outputs, identity, reference-batch hashes, state partitions, and no-decay reference reliability. Low reliability in decay arms is recorded as advisory and makes the scientific result inconclusive rather than blocking execution.
- Realized LR checks use a scale-sensitive tolerance that distinguishes all three late-decay schedules. The runtime, manifest, training release, projection code, and both freeze modules are hash-pinned and re-derived at launch. Probe checkpoints are statically required to be retained by training.
- Checkpoint restoration, optimizer counters, locality, create-only durable outputs, result-marker binding, and idempotent retry behavior remain fail-closed.

Non-blocking caveats: the no-decay reliability threshold was inherited rather than recalibrated for projected geometry; the StarCoder reference is a true shared holdout while Nemotron is sampled from retained training support; and the endpoint-blind probe does not identify endpoint performance by itself.

VERDICT: PASS
