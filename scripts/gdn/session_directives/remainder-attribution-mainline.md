# Session Directive: Remainder Attribution Is The Mainline

Goal:
- stop hillclimbing blind inside the tracked GDN train-path budget,
- quantify what the loop is still not explaining in the hybrid-vs-attention gap,
- treat unexplained remainder as the primary optimization target.

Current evidence:
- the validated hybrid regime is around `6.09 MFU` and `~0.166 s` step time,
- the attention-only control on the same benchmark family and CE settings is about `21.09 MFU`
  and `~0.0579 s` step time,
- the tracked hybrid train-path budget is only about `42.7 ms`, while the hybrid-vs-attention
  step gap is about `108 ms`,
- therefore more than half of the gap is outside the currently tracked GDN closed-call + CE while budget.

Mainline requirements:
1. Treat `upper_bound_gap_ms = hybrid_step_ms - attn_only_step_ms` as a first-class metric.
2. Treat `gap_explained_by_train_path = train_path_budget_ms / upper_bound_gap_ms` as a first-class metric.
3. Record top remainder categories, not just GDN forward/backward closed-call and CE while.
4. Reject any candidate whose tracked train-path budget improves but whose step time does not.

Required metrics in iteration writeups:
- `step_duration_ms`
- `train_path_budget_ms`
- `remainder_budget_ms`
- `upper_bound_gap_ms`
- `gap_explained_by_train_path`
- `remainder_topk`
- `gdn_layer_fraction`
- `ce_backend_selected`
- `ce_bwd_mode`

Selection rules:
- Prefer pure diagnostic / attribution iterations over same-boundary kernel rewrites.
- If the next iteration does not change the full-step attribution picture, classify it as low information.
- Do not treat lower GDN closed-call time as success unless the full step also improves.

Guardrails:
- Hold CE fixed at `pallas_tpu` + `pallas` unless the iteration is explicitly a CE side-arm.
- Do not spend a fresh mainline iteration on another same-boundary GDN shell/tape move before
  recording a current hybrid-vs-attention remainder comparison.
