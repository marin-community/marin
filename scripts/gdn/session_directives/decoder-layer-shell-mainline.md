# Session Directive: Make Decoder-Layer Shell Cost First-Class

Goal:
- turn the current anonymous hybrid-only remainder into an actionable decoder-layer-shell budget,
- drive the next iterations by `decoder_layer_shell_budget_ms`, not only by `train_path_budget_ms`.

Required named budgets:
- `decoder_layer_shell_budget_ms`
- `ad_shell_budget_ms`
- `sharding_shell_budget_ms`
- `layout_shell_budget_ms`
- `gap_explained_by_decoder_layer_shell`
- `decoder_layer_shell_topk`

Grouping rule:
- treat buckets with prefixes:
  - `HackableDecoderLayer/`
  - `jvp(HackableTransformer)/HackableDecoderLayer/`
  - `transpose(jvp(HackableTransformer))/HackableDecoderLayer/`
  as the shell budget family.

Promotion rule:
- a candidate is not promotable unless `step_duration_ms` improves,
- and either `decoder_layer_shell_budget_ms` improves or the iteration is explicitly diagnostic.
- classify `train_path_budget_ms down, decoder_layer_shell_budget_ms flat/up` as `wrong-boundary progress`.
