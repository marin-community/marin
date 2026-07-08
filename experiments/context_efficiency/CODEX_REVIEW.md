# Codex peer review of PLAN.md (2026-07-08)

Verdict: **PROCEED-WITH-FIXES**. Reviewer: `codex exec` (read-only sandbox) over
PLAN.md + RESEARCH_BRIEF.md + EXPERIMENT_LOG.md. Adopted fixes are tracked in
the M1.5 experiment-log entry and folded into PLAN.md v2.

**Prioritized Issues**

1. **Denominator risk: raw tokens and price-weighted “budget” are being mixed.**
   The plan’s `billed(C,t,T)` is price-weighted base-input equivalent, but M0 headline totals are raw usage fields. Since cache reads are 0.1x and output is about 5x, `% of total budget saved` can move materially depending on denominator.
   **Fix:** report separate denominators: raw context tokens, base-price-equivalent input cost, and full dollar-equivalent including output. Make the headline use one explicitly.

2. **The per-chunk cost model is too clean for prefix caching.**
   `1.25C + 0.10C(T-t)` is a useful first-order marginal model only if the chunk is cached once, remains in the active prefix, every later turn hits cache, and no compaction/context editing removes it. Actual usage is whole-prefix, breakpoint-based, with TTL expiry, 1h writes, 20-block lookback misses, and possible uncached suffix tokens.
   **Fix:** validate the model by reconstructing per-session modeled cost and comparing to actual `cache_creation/cache_read/input`. Add variants using actual `ephemeral_5m/1h` writes and timestamp gaps.

3. **The “fixed prelude dominates cache_creation” inference is not yet established.**
   Subtracting estimated visible body content from `cache_creation` does not prove the residual is fixed prelude. It may include missed assistant/thinking/tool content, re-created body after TTL/lookback failure, summaries, attachments, or transcript fields the parser ignores. The experiment log’s output-token/body-token mismatch is a warning sign.
   **Fix:** do per-turn reconciliation before using the residual as a docs/prelude savings pool. Show residual by session, version, model, and turn; sample inspect high-residual sessions.

4. **Turn index `t` is recoverable only with careful request reconstruction.**
   Tool results are user records joined to prior tool uses, but their first paid use is the next assistant request. Assistant output becomes paid input only on later turns. Compaction changes lifetime. A naive record `turn_idx` will produce off-by-one and over-lifetime errors.
   **Fix:** define `first_in_prompt_turn` and `last_in_prompt_turn`, not just block record turn. Treat compaction/summary/context-edit events as censoring points.

5. **Within-session rereads are discounted, not low-value.**
   The plan is right not to price them at base input, but wrong to imply they are mostly irrelevant. An early duplicate read in a 500-turn session still costs `0.1 * C * remaining_turns`, can trigger compaction earlier, and consumes scarce context.
   **Fix:** keep separate pools: cross-session memory savings, within-session context-hygiene savings, and compaction-avoidance savings. Do not bank within-session as “shared memory,” but do include it in intervention ranking.

6. **Cross-session “cold read” logic has a hole around shared/prewarmed prefixes.**
   File-read body content probably does not share cache across sessions because the preceding conversation differs, but fixed prelude/tool schemas might. Concurrent sessions or prewarming can also share exact prefixes.
   **Fix:** distinguish body reads from fixed-prefix costs. Use observed first-turn `cache_read/cache_creation` to estimate how often sessions actually start warm.

7. **“Reads beyond first per path” is an unstable redundancy definition.**
   Same path across branches/weeks is not the same information need. Same-content is a good split, but byte-changed content may still have stable API/symbol facts; byte-identical content may still be legitimately reverified.
   **Fix:** key by repo, branch/commit when available, path, content hash, and edit history. Add tiers: byte-identical, API/signature-identical, semantically similar, changed. Human-label a stratified sample.

8. **The Bash stability classifier is too command-name based.**
   Excluding all `git show` loses stable commit/path facts. Including all `rg` risks mixing unrelated searches if args are stripped. `gh pr view #N` and `ls` can be volatile. `git diff` is not memory-addressable cross-session, but repeated diff inspection is a real context-efficiency signal.
   **Fix:** classify by target immutability and output hash, not just command. Preserve normalized patterns/paths for `rg`, commit refs for `git show`, PR numbers plus timestamps for `gh`.

9. **The uplift model is currently hand-wavy.**
   Hit-rate × pool with conservative/likely/optimistic ranges is acceptable only if the hit rates are grounded. Otherwise the final recommendation will look chosen by priors.
   **Fix:** label 200-500 high-weight redundancy events: replaceable?, by which intervention?, replacement token cost?, verification still needed?, stale risk? Derive hit-rate ranges and bootstrap intervals from that audit.

10. **Double-counting across wiki/RAG/docs is likely.**
   Hot file rereads, repeated grep, and docs gaps are overlapping symptoms of the same missing repo knowledge. Summing intervention pools will overstate combined savings.
   **Fix:** assign every content block/event an intervention mask and compute union savings. For combined estimates, use priority allocation or Shapley-style attribution, and count replacement-entry cost once per session.

11. **Missing threats to validity would make the headline fragile.**
   Add: hidden/sub-agent transcripts, retained-transcript survivorship, chars/4 error by content type, unmeasured output-token re-derivation, cache sharing/concurrency scope, parser-missed record types, version drift in Claude Code behavior, changed repo state over time, and quality/rework costs from stale memory.

12. **Reviewer-trust issue: no falsification test yet.**
   The plan needs one or two checks that could disprove the headline.
   **Fix:** run a top-K ablation: for the heaviest 100 sessions, manually classify the top 80% of modeled savings. If the manual replaceable savings is much lower than modeled, scale the whole estimate down.

**What’s Solid**

Using real `assistant.message.usage` is the right anchor. The heavy-tail correction is important and prevents a bad 34x global multiplier. Splitting same-content from changed-content reads is directionally right. Calling out `git diff` as a volatility trap is also correct; it prevents the most obvious overclaim.

Verdict: **PROCEED-WITH-FIXES**.
