# Claude Code review: TPP40 bridge Uncheatable sidecar

Review session: `273344f8-d193-4e00-8806-9f545d237356`

Model: `claude-opus-5[1m]`, maximum effort, read-only tools

Authentication: `plambdafour@proton.me`, `stripe_subscription`, with `ANTHROPIC_API_KEY` removed from the child environment

## Verdict

**NO-GO pending two blocking fixes.** The reviewer found the reconstruction faithful, checkpoint selection exact, validation data leak-free, regional placements valid, and East5/Europe evaluations comparable on identical `v6e-8` hardware. It identified two defects that must be fixed before either sidecar is submitted.

1. `_uncheatable_metrics` recomputed the seven-component macro in Python float64 but compared it to Levanter's float32-reduced macro at `1e-9` absolute tolerance. Real rows can differ by several times (10^{-8}), causing every TPU evaluation to finish and then fail without writing `bridge_result.json`. Reconstruct the macro with Levanter's float32 arithmetic and add a realistic regression test.
2. The evaluator rebuilt the model configuration from the live scaling-fit artifact without repeating training's exact parameter-count assertion, while the horizon fields were omitted from the sidecar's identity checks. Reassert the parameter count, pin phase-boundary step `21855` and endpoint step `27335`, and record a deterministic model-config identity in every result.

## Additional findings

- Bind completed results to the current `checkpoint_metadata_sha256`; otherwise a replaced checkpoint can leave a stale result accepted.
- Reject missing or nonnumeric `macro_bpb` explicitly instead of allowing `float(None)` to raise an uninformative `TypeError`.
- The seven-set sidecar metric is valid for the paired East5-versus-Europe bridge, but it is not directly comparable to training-time 23-set W&B values because per-tag BPB depends on co-evaluated batch composition.
- Ready-only scheduling and the exact checkpoint path are sound. Metadata is written after the Orbax commit callback and therefore serves as a valid completion marker.
- Re-run focused tests, both regional dry runs, launch-safety validation, and a follow-up review after the blockers are fixed.

## Follow-up review

The same subscription-authenticated `claude-opus-5[1m]` session reviewed the repaired evaluator and tests. Its final verdict is **GO**: the float32 macro reconstruction, explicit malformed-macro handling, exact parameter-count and horizon assertions, model-config identity, and current-checkpoint binding close both original blockers. It confirmed that the ready-only launch remains fail-closed until at least one exact checkpoint materializes.

The reviewer identified no remaining blocker. Its non-blocking cautions are to mirror Levanter's masked float32 reduction in a test, confirm the first East5 and Europe results produce the same deterministic model-config hash, and verify the first materialized phase-boundary path before interpreting the current `0/8 ready` state. The masked-reduction regression was added after this review; the two live-result checks remain acceptance gates rather than assumptions.
