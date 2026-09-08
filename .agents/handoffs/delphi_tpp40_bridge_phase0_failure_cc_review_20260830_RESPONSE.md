I have what I need. One caveat up front: this session has no shell, so I could not execute the noise estimator — the power figures below are computed analytically from the SD already frozen in the contract, and I flag exactly where that matters.

# Verdict: Path 3, via an explicit amendment — but the gate failed because it was mis-specified, not because Europe is unsafe

The one-pair gate cannot answer the question it was built to answer. Its margin sits below the metric's own noise floor, so both the original fail-closed reading (path 1) and the sequential rescue (path 2) over-read a coin flip.

## The specification defect

Contract v2 states the Uncheatable tolerances are *"deliberate absolute tolerances, **not** functions of the Table-9 noise estimate"* (`bridge_acceptance_contract_v2.json:75`). They were never noise-calibrated, and v4 inherited them verbatim. Meanwhile the only frozen noise figure — `single_run_sd = 0.0034194` — is for `table9_macro_bpb`, **not** for the metric the binding phase-0 gate actually tests.

The two sides are the same `run_order` on different accelerators, so after 21855 steps they are two chaotically-diverged trajectories from one init — the correct null is the seed-repeat distribution. Under that null the paired delta has SD `σ√2 ≈ 0.004836`:

| Quantity | Value |
|---|---|
| Observed \|δ\| | 0.003671 |
| Expected \|δ\| under a **zero** region effect | 0.003858 (mean), 0.003262 (median) |
| Observed in paired-SD units | **0.76 σ** |
| P(pass 0.002 gate) if region effect is exactly zero | **32%** |
| P(pass 0.002 gate) if there is a real +0.003 harm | **27%** |

The gate's pass probability is nearly independent of the thing it is meant to detect — a likelihood ratio of about 1.2. The observed value is the *typical* outcome under a perfectly benign null. Failing it is close to uninformative.

This is structural, not fixable by more rows. A TOST equivalence claim at ±0.002 with 80% power needs **~36 paired rows** — more compute than the entire Europe production allocation. Even the original four-row mean had a ~41% false-failure rate. **No affordable amount of bridging can ever pass this gate reliably.** Chasing it is not a path to launch.

## Q2 — Yes, path 2 is an unacceptable rescue, and quantifiably so

Predeclaring the *rows* doesn't save it, because what changed after seeing row 2 fail is the **decision rule**, not the row list. Rows 120/240/260 were frozen as supplementary and explicitly "must not block this gate"; promoting them to a first-to-pass ladder converts pass probability from 32% → **69%** under the benign null, while discrimination *falls* (LR 1.2 → 1.13). It buys permission, not evidence.

It is also the wrong estimator: the mean of paired deltas is the sufficient statistic; first-to-pass is a minimum over noise. This codebase has already characterized that exact failure mode — `analyze_prefix_search_evidence_20260819.py:244`, *"THE ORACLE IS NOT A CEILING: best observed, corrected for selection on noise."*

The rows are still worth running — just pooled, as an **estimate with a CI and a large-effect screen**, never as pass/fail. A 3-pair mean has SE ≈ 0.0028: useless for certifying ±0.002, but it would catch a genuinely alarming offset (e.g. 0.010 would land at 3.6 SE). That is the honest version of path 2.

## Q3 — Noisy macro dominated by two subsets, not a broad accelerator shift

Two of seven components carry **82%** of the macro delta. Dropping BBC News and Wikipedia English, the other five average **+0.00093** — comfortably inside the 0.002 margin. Decisively, `arxiv_computer_science` is **negative** (Europe better); a systematic degradation cannot improve a component. The movers are the high-entropy prose subsets while the structurally stable code/arXiv subsets sit near zero — the standard per-component variance ordering, not a lesion.

The 6/7 positive signs do *not* discriminate: components share one model, so any whole-model quality difference of either origin moves them together (sign test p ≈ 0.125 regardless). This cannot be settled without per-component seed SDs — which are computable today (below).

## Q4 — Yes, stratified production is defensible, and it is the better design

The failed pair does not indicate deeper incompatibility; it indicates the bridge never had the resolution to say. **The production panel is a far better region-offset estimator than any bridge can be.** Region is balanced across strata (imbalance ≤ 1, per the multiregion review), so the region contrast over the 253 newly-assigned rows has SE ≈ `σ√(1/126 + 1/127)` ≈ **0.00043 BPB** — about 11× tighter than the one-pair bridge, resolving the 0.002 margin at ~4.7σ, for free.

An additive offset balanced against the design is absorbed into the intercept and does not bias surface shape or coordinate ranking. Two residual risks remain, both handled: selection bias at frontier picks (killed by common-region re-evaluation) and a region × mixture interaction (not absorbed by a fixed effect, but testable from the same balanced panel). Estimate the offset on the 253 balanced rows only — the 27 completed rows are East-only and contiguous in `run_order`, so they carry their own confound.

Launching does not spend the evidence; it generates it. The rows are not wasted even if an offset exists, because it is estimable and correctable after the fact.

Path 3 does, however, **directly contradict v4's frozen `decision` clause**. That requires a dated amendment, not a silent reinterpretation.

---

# Blockers before launch

1. **Compute and freeze the Uncheatable noise scale.** `analyze_prefix_search_evidence_20260819.py:148` already implements the exact estimator, and `endpoint_components.csv` carries all seven `eval/uncheatable_eval/*/bpb` columns over 1864 clean rows / 21 seed-repeat groups. Reconstruct the bridge's exact unweighted 7-component macro, compute its seed SD **and per-component SDs**. Zero accelerator cost. Every power claim above — including mine — is conditional on `σ_uncheatable ≈ σ_table9` until this exists, and the per-component SDs are what settle Q3.
2. **Dated amendment superseding v4's `decision` clause.** It must record that ±0.002 was never noise-calibrated (quoting v2:75), the achieved power, and the replacement rule. The current clause reads as a hard block; leaving it silently contradicted is the same preregistration breach already flagged as B2 in the multiregion review.
3. **Read the row-2 endpoint results before launching.** Already in flight, so waiting costs nothing. Table-9 macro is the *only* metric here with a frozen, validated noise scale — it is the single highest-information pending result.
4. **Per-row region must be recoverable in the production output contract.** If region is not recorded per row, the entire path-3 remedy is unavailable. This is a design requirement, not a nicety.
5. **Carry over the still-open production blockers**: B1 (resolve the 153-row East graph against live GCS), B2 (re-mint or amend the frozen assignment), B3 (issue + logbook). Also `EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "UNFROZEN"`, and the `--allow-incomplete` guard at `analyze_delphi_tpp40_bridge_acceptance.py:978` if the analyzer's exit code is to be trusted. (The component-order blocker is evidently fixed — report v3 has a populated phase-0 pair, which was impossible before.)
6. **Commit to common-region re-evaluation of all selected and near-frontier checkpoints**, as a launch precondition rather than a later intention. This is what makes the offset tolerable, and it is the preregistered remedy already written into v4's `cross_region_evaluator_followup`.

# Optional diagnostics — none of these should gate launch

- Rows 120 and 260 as a **pooled 3-pair estimate** with a CI plus a large-effect screen. Cheap (Europe checkpoints already committed), improves the offset prior, and is legitimate *only* if never reported as pass/fail.
- Per-component z-score table for row 2 against the newly computed per-component SDs.
- Row 240 resume — low value, skip.
- The preregistered cross-region evaluator follow-up (already non-blocking).

**On the one-passing-pair criterion:** no single pair can supply it. A pair that "passes" this gate is 32% likely under zero effect and 27% likely under real harm — passing would not have been meaningful evidence either. The defensible substitute that delivers what you actually want is blockers 1–6: they need no East5 training capacity, and item 3 is already running.
