# Session Directive: Coverage Pivot To Remainder Attribution And Model Boundary

Goal:
- stop treating same-boundary GDN Pallas hillclimbing as the mainline,
- spend iteration budget on the biggest unresolved questions:
  1. what is inside the unexplained full-step remainder,
  2. how much throughput penalty comes from GDN layer fraction itself,
  3. whether any bounded CE side-arm is still worth taking.

Coverage rule for this session:
- Before repeating a same-boundary GDN structural move (`M`, `N`, `O`, `R`) as the mainline,
  complete at least one **validated** attempt for each of:
  - `S` remainder attribution against the attention-only control,
  - `T` model-boundary sweep over `gdn_layers_per_block in {0, 1, 2, 3}` with `gdn_block_size=4`.
- `U` CE side-arm work is optional and bounded. Use it only after `S` or `T`, or when attribution becomes unclear again.

Selection order guidance:
1. `S` remainder attribution / gap accounting.
2. `T` model-boundary sweep.
3. `U` bounded CE side-arm only if it has a clear end-to-end path.
4. `O` or `M` only as diagnostic control arms after `S/T`.
5. `R` and other same-boundary GDN train-path variants only as research branches, not the mainline.

Repeat-avoidance rules:
- Do not spend the next mainline iteration on another same-boundary GDN shell/tape change unless:
  - `remainder_budget_ms` is already explained well enough to show it is on the critical path, or
  - the iteration is explicitly diagnostic and not promotable.
- Do not repeat a kernel-local or same-boundary move just because it lowered a closed-call bucket.
- Classify `train_path_budget_ms down, step_duration_ms flat/up` as `off-critical-path`, not as a near-win.

Writeup requirement:
- At the top of each iteration writeup, include:
  - `Coverage slot: S | T | U | diagnostic`
  - `Change class: attribution | model boundary | CE backend | outer control structure | inner kernel math`
  - `Why this is mainline-worthy now:`
- In the perf section, always include:
  - `gdn_layer_fraction`
  - `upper_bound_gap_ms`
  - `gap_explained_by_train_path`
  - `remainder_budget_ms`
  - `remainder_topk`
  - `ce_backend_selected`
  - `ce_bwd_mode`

Guardrails:
- Hold CE fixed at `pallas_tpu` + `pallas` during `S` and `T`.
- Do not mix new GDN kernel changes into the `T` model-boundary sweep.
- Treat the attention-only control as the practical ceiling reference for this benchmark family.
