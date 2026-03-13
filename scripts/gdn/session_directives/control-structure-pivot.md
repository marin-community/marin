# Session Directive: Boundary And Bucket Names Must Both Improve

Current diagnosis:
- the unresolved hybrid-vs-attention gap is now split across:
  - tracked GDN train path,
  - a hybrid-specific generic shell delta,
  - and a large interaction remainder.
- the broad `HackableDecoderLayer/*` family is only a coarse upper bound.
- Iteration 90 proved that an outer wrapper can erase the old train-path labels while the real cost simply reappears under `HackableDecoderBlock/*`.

Implications for this session:
- do not optimize toward smaller old bucket names alone,
- do not treat vanished `HackableDecoderLayer/*` buckets as wins unless the namespace-invariant shell delta also improves,
- prefer:
  - `S3` hybrid-specific shell-delta attribution,
  - `A3` AD-boundary prototypes that move the manual backward outward,
  - or `P3` whole-block prototypes with bespoke backward/sharding.

Hard guardrail:
- If the result is another iteration where old buckets shrink but:
  - `dispatch_shard_shell_delta_ms` stays flat/up,
  - or `ad_wrapper_shell_delta_ms` stays flat/up,
  - or `xprof_idle_attributed_ms` stays flat/up,
  - or `step_duration_ms` stays flat/up,
  classify it as renamed-bucket / waiting-dominant progress, revert it, and pivot again.
