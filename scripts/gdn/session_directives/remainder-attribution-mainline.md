# Session Directive: Decoder-Layer Shell Attribution Is The Mainline

Goal:
- stop hillclimbing blind inside the tracked GDN train-path budget,
- widen attribution from train-path-only to the whole GDN-bearing decoder-layer shell,
- treat the hybrid-only decoder-layer shell tax as the primary optimization target.

Current evidence:
- the fixed `3/4` GDN regime is around `6.06-6.09 MFU` and `~166-167 ms` step time,
- the attention-only control on the same benchmark family and CE settings is about `21.0 MFU`
  and `~58 ms` step time,
- the tracked train-path budget explains only about `39%` of that gap,
- the dominant hybrid-only remainder buckets are decoder-layer shell categories under
  `HackableDecoderLayer/*` and `transpose(jvp(HackableTransformer))/HackableDecoderLayer/*`.

Mainline requirements:
1. Treat `decoder_layer_shell_budget_ms` as a first-class metric.
2. Treat `gap_explained_by_decoder_layer_shell` as a first-class metric.
3. Break the shell budget down into at least:
   - `ad_shell_budget_ms`
   - `sharding_shell_budget_ms`
   - `layout_shell_budget_ms`
4. Record `decoder_layer_shell_topk`, not just generic `remainder_topk`.
5. Reject any candidate whose train-path budget improves but whose decoder-layer shell budget or step time does not.

Required metrics in iteration writeups:
- `step_duration_ms`
- `train_path_budget_ms`
- `decoder_layer_shell_budget_ms`
- `ad_shell_budget_ms`
- `sharding_shell_budget_ms`
- `layout_shell_budget_ms`
- `remainder_budget_ms`
- `upper_bound_gap_ms`
- `gap_explained_by_train_path`
- `gap_explained_by_decoder_layer_shell`
- `decoder_layer_shell_topk`
- `remainder_topk`
- `gdn_layer_fraction`
- `ce_backend_selected`
- `ce_bwd_mode`

Selection rules:
- Prefer pure attribution or whole-layer boundary iterations over same-boundary chunk-kernel rewrites.
- If the next iteration does not change the whole-layer attribution picture, classify it as low information.
- Do not treat lower GDN closed-call time as success unless decoder-layer shell budget and full step also improve.

Guardrails:
- Hold CE fixed at `pallas_tpu` + `pallas` unless the iteration is explicitly a CE side-arm.
- Keep `3/4` GDN fixed.
- Do not treat the model-boundary sweep as a live product recommendation; use it as proof that shell tax scales with GDN-bearing layers.
