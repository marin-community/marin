# Session Directive: Optimize Canonical Shell Budgets, Not Old Bucket Names

- Treat `dispatch_shard_shell_delta_ms` as the immediate target.
- Treat `ad_wrapper_shell_delta_ms` as the second target.
- The obvious outward `HackableDecoderLayer` / `HackableDecoderBlock` boundaries are already rejected.
- Move the next ownership boundary inward to the hybrid-specific GDN branch.
- Do not accept vanished old bucket names as progress unless the canonical shell budgets and full step improve too.
- If old train-path buckets shrink but shell deltas, `interaction_remainder_ms`, or `xprof_idle_attributed_ms` stay flat/up, reject and revert.
