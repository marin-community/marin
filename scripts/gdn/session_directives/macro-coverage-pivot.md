# Session Directive: Coverage Pivot To Decoder-Layer Shell And Whole-Layer Boundaries

Goal:
- fully demote same-boundary GDN Pallas hillclimbing from the mainline,
- keep the fixed `3/4` GDN regime,
- spend iteration budget on the largest unresolved question:
  - how to eliminate the hybrid-only decoder-layer shell tax.

Coverage rule for this session:
- Before spending another mainline iteration on same-boundary GDN shell/tape/kernel work,
  complete at least one **validated** attempt for each of:
  - `S2` decoder-layer-shell attribution widening,
  - `L2` specialized whole-layer design/skeleton,
  - `P2` one serious whole-layer prototype.
- `U` CE side-arm work is optional and bounded. Use it only if the widened attribution again points to CE.

Selection order guidance:
1. `S2` decoder-layer-shell attribution.
2. `L2` specialized whole-layer design/skeleton.
3. `P2` one serious whole-layer prototype.
4. `U` bounded CE side-arm only if it has a clear end-to-end path.
5. Same-boundary GDN shell/tape/kernel work only as diagnostic side-arm.

Repeat-avoidance rules:
- Do not spend the next mainline iteration on another same-boundary GDN structural move unless:
  - `decoder_layer_shell_budget_ms` is already explained well enough to show it is on the critical path, or
  - the iteration is explicitly diagnostic and not promotable.
- Do not repeat a kernel-local move just because it lowered a closed-call bucket.
- Classify `train_path_budget_ms down, decoder_layer_shell_budget_ms flat/up, step flat/up` as
  `wrong-boundary progress`, not as a near-win.

Writeup requirement:
- At the top of each iteration writeup, include:
  - `Coverage slot: S2 | L2 | P2 | U | diagnostic`
  - `Change class: decoder shell attribution | whole-layer boundary | CE backend | diagnostic side-arm | inner kernel math`
  - `Why this is mainline-worthy now:`
- In the perf section, always include:
  - `gdn_layer_fraction`
  - `upper_bound_gap_ms`
  - `gap_explained_by_train_path`
  - `gap_explained_by_decoder_layer_shell`
  - `decoder_layer_shell_budget_ms`
  - `decoder_layer_shell_topk`
  - `remainder_budget_ms`
  - `remainder_topk`
  - `ce_backend_selected`
  - `ce_bwd_mode`

Guardrails:
- Hold CE fixed at `pallas_tpu` + `pallas` during `S2/L2/P2` unless the iteration is explicitly CE-focused.
- Treat the attention-only control only as a ceiling/diagnostic, not as a product recommendation.
- Keep `3/4` GDN fixed; do not propose reducing GDN fraction as the optimization result.
