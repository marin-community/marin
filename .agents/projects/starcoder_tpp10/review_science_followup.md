I verified each claimed change against the files rather than the summary. Every arithmetic claim I could recompute independently reproduces exactly.

## Resolved

**Batch/step mismatch (prior B4).** `design.md:21,32` now sets proxy batch 32. Verified in code: `starcoder_tpp10.py:150-152` rounds the horizon in 128-sequence units, so 2,532 × 32 × 2,048 = 165,937,152 exactly, and the calibration variant's `row.total_steps // 4` (`:206`) = 633 × 128 × 2,048 = the *same* horizon. The screen therefore varies batch at fixed tokens, which is the right contrast. `calibration_summary` (`:270-291`) computes the gap from **unmatched** losses only, requires both finite trainer-seed endpoints in every cell (`:279-280`), and exposes `batch32_loss_screen_passed = gap <= 0.01`; matched means are carried but unused. `launch_starcoder_tpp10.py:295-298` raises before the canary if that flag is false, and `:238` rejects nonfinite endpoints at collection. The screen is 4 batch-32 runs already in the pilot grid plus 4 calibration-only artifacts (`:210`, `select_runs:252-258`), i.e. 183 + 4 = 187, matching `spec.md:32`.

**Parent ordering (prior B8).** `parent_order_audit.json` is decisive: unmatched prefixes at p=10/50/100 each touch **49/49 shards** with maximum deviation from equal of 0.50%, 0.15%, 0.04%; all three matched subsets touch 49/49. At p=100 the prefix is 81,024 of 92,928 sequences (0.8719 epochs), confirming no wrap. Composition drift across p is no longer a live confound. Pairwise subset overlaps (252/260/252) sit just under the 282 expected under independent sampling — unremarkable.

**Undefined promotion gate (prior B3).** `design.md:42` now reads "There is no signal-size or interior-optimum promotion gate," and `spec.md:34,46` states the release check does not require matched selection to improve. The leak path is closed.

**Scale-control transparency (prior B5/B6).** `design.md:17,20,26` adds nonembedding parameters, nonembedding TPP, and available-tokens-per-parameter. I recomputed all of these from the stated geometry: 8,395,008 / 268,473,344 params, TPP 19.7662 / 11.2201, and 0.6322 / 0.6318 — exact. `design.md:26` correctly labels the token/parameter match as "a control, not an explanation of the prior result," and states the parent-size rationale.

**Cross-experiment framing (prior B7).** `design.md:3` now lists the changed axes and states outcome changes "cannot be attributed to TPP alone," and links the prior RESULTS.md.

**Other prior items.** `research.md` exists (link resolves). Optimizer routing, clipping 1.0, absent decoupled weight decay and zero LR floor are explicit (`design.md:32`). Web weights are explicitly "held fixed as training-token proportions … they do not preserve identical document proportions" (`design.md:30`). `p`, regret, ties and the shared-metric requirement are defined (`design.md:3,38`). Reporting is per-subset with all six matched and two unmatched selections and no independence claim (`design.md:38`, `spec.md:42`). `audit_allocations:333-341` hard-fails on unmatched wrap, target web wrap, and >1% epoch mismatch.

## Remaining pre-pilot blocker

**One, and it is small.** `design.md:34` asserts that "Stable component names and explicit shuffle keys make the surviving web stream independent of subset identity and zero-weight component removal; this is checked offline." `sequence_allocation:318` does drop zero-weight components (`if w > 0`), so at p=0 StarCoder is genuinely absent from the mixture while it is present at p≥0.05. This is the exact mechanism that previously produced a wrong parent and a corrected re-run (`starcoder_epoch_matching/.../RESULTS.md:48`). I found no recorded artifact for that offline check — `parent_order_audit.json` covers shard coverage only, not stream identity under component removal.

Note the run-count half of my earlier concern is now moot: `build_design:186` (`if percent:`) emits no matched rows at p=0, so aliasing is structural, and 57/183 verify.

**Smallest correction:** emit the zero-weight stream-identity check as a durable artifact alongside `parent_order_audit.json` — the web sequence-identity sequence at p=0 versus p=0.05, with the subset seed varied — and cite it from `design.md:34`. Scientifically the exposure is bounded: p=0 is shared by all three arms, so any offset largely cancels in the arm comparison; the risk is to curve shape and to p=0's eligibility as a selected minimum.

## Calibration prerequisite versus canary

**Prerequisite to prepare/run calibration:** the frozen design and pins (`load_design:236-245`), materialized regional caches with receipts (`spec.md:20,28`), the allocation audit, and the artifact above. Nothing about optimizer behavior is knowable first — that is what the screen is for.

**What the canary must still establish** (not decidable offline, and not addressed by the batch screen): MuonH stability at the 256/8, two-head geometry across a *full* horizon at the grid endpoint — loss/gradient traces, no spikes; that Paloma programming-languages BPB is finite and non-degenerate at 16.6M parameters and TPP 10; the child-runtime receipt and regional cache receipts; and per-run wall-clock, since `design.md:34` correctly concedes "chip-hours require canary timing." The calibration runs at p=0.5 do not cover the p=1 repetition endpoint, where the matched arm's 15.8 epochs are most likely to destabilize.

## Limitations that should remain explicit

1. **The screen is one-sided and two-point.** It guards against batch 32 regressing versus the historical 128; it establishes nothing about batch 16 or 8 being better. Do not report it as recipe optimization.
2. **Steps are still unequal.** 2,532 versus 11,491 (4.54×), and tokens-per-step-per-parameter remains 4.54× higher for the proxy (3.95e-3 vs 8.70e-4). Improved from 18×, not removed; equalizing would need proxy batch ≈ 7.
3. **Nonembedding TPP remains mismatched 1.76×** (19.77 vs 11.22), embeddings are still 49.4% of the proxy, and the token/parameter match at 0.632 holds only under *total*-parameter normalization.
4. **Scope.** One target seed, one finite parent, three overlapping conditional subsets, code-only BPB (`spec.md:42`), and a repetition regime set by the chosen 190.3M parent.
5. **The audit's own caveat** — midpoint shard attribution, "counts do not certify corpus representativeness" (`parent_order_audit.json:4`) — belongs in the write-up, not just the JSON.