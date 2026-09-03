# DeepSeek review of the single-phase Observatory benchmark and successor (2026-09-02)

Independent read-only review by deepseek-v4-pro (max reasoning) through the DeepSeek harness wrapper, 57 assistant messages, provenance verified. Brief: `reviews/deepseek_instructions_final.txt` in the session scratchpad (six claims: minimal formula, revision legitimacy, scramble controls, cache validity, statistics and heldout, Michael families).

The review is complete. Here is my report.

---

# Review: single-phase Observatory benchmark and successor

**Verdict up front:** the benchmark plumbing (panels, splits, hashing, reconstruction, heldout stage, promotions) is verified sound, and the harm/inventory/head mechanisms genuinely survive their matched ablations. But the report's central claim — that `weibull_softplus_unscaled` is "the smallest model that the benchmark supports" — is **falsified by the successor's own Certify ablation set**, and the head revision is post-hoc Screen selection dressed up as an independent ablation win.

---

## Blockers

### B1. The "smallest supported formula" claim is falsified by `weibull_softplus_unscaled@log_deficit_link` at Certify

The report's completion claim (sections 0 and 12.2) says "everything it drops … either ties or hurts." The dropped log-deficit link does neither — it **beats the successor on all 8 Certify aggregate cells**:

| cell | parent | @log_deficit_link | diff |
|---|---:|---:|---:|
| 60M Uncheatable | 0.0076 | 0.00689 | −0.0007 |
| 60M Table 9 | 0.0211 | 0.01794 | **−0.0032** |
| 300M Uncheatable | 0.0052 | 0.00391 | −0.0013 |
| 300M Table 9 | 0.0116 | 0.01143 | −0.0002 |
| Delphi Uncheatable | 0.0092 | 0.00811 | −0.0011 |
| Delphi Table 9 | 0.0302 | 0.02597 | **−0.0042** |
| dclm | 0.1171 | 0.10074 | −0.0164 |
| high-quality | 0.0864 | 0.06865 | −0.0178 |

At the Certify-scope pooled test the log-deficit variant is better on **135 of 194 units (p = 4.9e-8)** (`pooled_screen_contrasts_certify_scope.csv`). The report drops it only because its StarCoder dense-horizon family macro explodes (RMSE 120.0 vs the parent's 0.081, matched-onset 0.0276 vs 0.0117 — those numbers are correct in the artifact). That trade-off is disclosed, but it makes the "everything dropped ties or hurts" sentence false on every paper-primary cell, so the minimum-formula claim needs to be either weakened (e.g., "minimum under the StarCoder shape gate") or the link added as a second paper-primary-optimized candidate.

Related report error, same claim family: section 12.2 says the successor "is the only model whose Delphi Table 9 shortlist contains the measured optimum." **False in the CSV**: `grp_pair_power` and `family_onset_power_grp` also have `top5_regret = 0` (Delphi T9). This also contradicts section 5's "their top-5 shortlists still contain the best coordinate," which is itself wrong for `bucket_family_power_grp` and both hierarchical models (`top5_regret = 0.013245`).

### B2. The Weibull-vs-saturation support is confounded and does not transfer to the paper-primary aggregates

The successor's `@exp_benefit` ablation changes three things at once: benefit family (Weibull→exponential saturation), nonlinear dof (3→2), and grid size (168→42 shapes; `SUCCESSOR_EXP_SHAPES` in the registry). This is exactly the confound Codex flagged for the parents (fixed there with `@weibull_benefit_matched`/`@power_benefit_matched`), but no budget-matched variant was built for the successor's own ablation set.

- Screen: 27/11 units (p = 0.014, uncorrected; largest single contribution the StarCoder matched-onset family unit, 0.099 vs 0.012).
- Certify aggregates: fold-paired differences vs the parent are ≤ 0.0013 on every 39-bucket cell, every interval includes zero; it is *worse* on high-quality (+0.0055, interval excludes zero).

So "a shared saturating Weibull benefit … better than exponential saturation" (section 0) is overstated; what survives is "some saturating shared benefit." (Note the Certify-scope 194-unit sign test does favor Weibull over saturation, 126/68 p = 3.8e-5 — so the evidence is mixed, not absent.) The same aggregate-level caution applies to Weibull-vs-power, which section 10 already admits is a tie.

### B3. The shared→unscaled revision is post-hoc selection on Screen; its "matched ablation" is circular

The revision was decided by `weibull_softplus_shared@unscaled_head` winning 35/38 Screen units (registry note and Fieldbook decision `note_01m1hntk01pjwksppmahme9krx` both state this). The revised model's mirror `@scaled_head` ablation (Screen 35/3; Certify scope 163/31) is the *same comparison re-run from the other side* — it carries no independent information, and presenting it among "every retained mechanism survives its matched ablation (… unscaled head 35/3)" overstates.

What does independently support the revision:
- Finalist five-repeat tables: the scaled predecessor explodes on Michael repeats (dclm 38 ± 54, high-quality 4,546 ± 5,709) while the unscaled revision is stable (0.113 ± 0.003 / 0.093 ± 0.006) — verified in `finalist/aggregate_metrics.csv`, fresh partitions, a real replication.
- Heldout selection parity or better (Delphi T9 rank 8 / regret 0.0132 vs the shared model's rank 11 / 0.0155).
- The mechanical explanation is code-verifiable: `_nonnegative_solve` scales columns by the *training* RMS with a 1e-8 floor, so near-zero-in-training harm columns amplify on extrapolated test mixtures.

So: legitimate as a finalist-stage robustness fix, illegitimate as "the one change its matched ablations supported" — that phrasing is selection-on-Screen with a circular mirror control.

---

## Answers to the six claims

**(1) Smallest supported formula.** The harm block, the true-inventory epoch coordinate, the weight-vs-epoch coordinate, and the nonnegative head survive scrutiny: `@no_harm` 31/7, `@row_scrambled_harm` 34/4, `@permuted_inventory` 33/1, `@weight_coordinate` 29/9, `@signed_head` 35/3 — all confirmed in `screen/pooled_screen_contrasts.csv`, and the harm/coordinate/head effects replicate at Certify (148/46, 152/42, 144/46, 137/57, 175/19 of 194 units). The Weibull term specifically is the weak link (B2), the identity link is contradicted (B1), and the unscaled-head "ablation" is circular (B3). Additionally, the unit-level sign tests treat 38 heavily correlated units (18 anchors on 3 panels, 16 Michael tasks on 2 panels, 4 curve-family macros pooling 3–28 curves) as independent, and no multiplicity correction is applied across ~90 ablation contrasts; the promotion amendment is post-hoc by admission. p = 0.014 (exp_benefit) and p = 0.0017 (weight_coordinate) don't survive even modest corrections; the p ≤ 1e-4 claims (harm, inventory, head sign) do. The claim "every kept term has a matched ablation that loses at Screen with p ≤ 0.014" is therefore accurate as arithmetic but not as statistical support.

**(2) The revision.** Legitimate only as a finalist-replicated robustness fix (verified: 5-repeat tables, no explosions, heldout parity, mechanical explanation in code); the Screen "35/38" is the selection event itself, and `@scaled_head` on the revised base is its mirror. No 25-fold interval against canonical DSP excludes zero for the revised model on any 39-bucket cell (verified: 60M U −0.00255 [−0.0135, 0.0084], … all straddle zero); intervals against OLMix do exclude zero on 60M/300M/Delphi Uncheatable and dclm, as reported.

**(3) Scramble controls.** Both verified in `family_design` and by the tests. `scrambled_harm` permutes harm-block columns before the per-bucket softplus — for `softplus_bucket` harm this is a pure column reorder, and the 39-bucket anchor metrics match the parent to 6 decimals (vacuous, as reported). Caveat: on the Michael panels the QR-reduced NNLS path makes it numerically non-identical (e.g., dclm mt_mbpp 0.300069 vs 0.300375; both explode on high-quality with very different magnitudes under the scaled head) — "fit is identical" holds up to solver numerics only. `row_scrambled_harm` permutes exposure rows with a fixed seed after the benefit block is computed; each harm column's value multiset is preserved (capacity-matched: same columns, ridge grid, threshold search) while row alignment with the response is destroyed. The permutation is input-only, applied once to the full-panel design before train/test slicing, so there is **no label leakage across folds**. Two approximations worth stating: (a) the fixed permutation is global, so under mixture-blocked folds ~20% of rows receive a harm value from a mixture in their own block (partial alignment retained); (b) ~1/242 rows are permutation fixed points. Neither changes the interpretation that the block's value is overexposure alignment information, not capacity.

**(4) Cache validity.** Yes — a behaviour change in the models module can escape the description hash. `describe_model` serializes only non-default configuration fields plus a manual `DESIGN_REVISIONS` whitelist that contains exactly one entry (`crs_plus_design`). Invisible to the description hash: module constants used as dataclass defaults (`DSP_MAXITER`, `DSP_LOG_RATE_BOUND`, `DSP_LINEAR_REG`, `HARM_SCRAMBLE_SEED`, … — a changed default equals the instance value, so the field is skipped), and every shared helper (`family_design`, `softplus_harm`, `weibull_response`, `_benefit`, `fit_head`, `_nonnegative_solve`, `_sobol`, …). The gen1/gen2 acceptance gates compare only `(entry, panel)` description equality, so such a change would silently accept pre-change shards. Gen1-era shards additionally carry no harness fit-path hash at all (the gen1 payload has no `fit_path_hash` field). This is load-bearing today, not hypothetical: I verified that sampled parent shards (`dsp_total_exposure`, `bucket_family_power_grp`, `crs_plus_family_overload`, `bucket_family_power_grp@no_harm` on 60M/U/c0/r0/f0) are accepted via gen1 keys written under models-module hash `d0cc0259…`, two module revisions before the final `7c89b11d…`. Their acceptance works and the review-fix record suggests the DSP path never changed, but no stored hash proves behaviour equality — the gate proves configuration equality only. Not a current-data blocker, but a real gap in the cache design and its description in section 3 of the report.

**(5) Reconstruction, Nadeau-Bengio, heldout.** All verified sound. Aggregates are reconstructed from atomic out-of-fold predictions with the frozen evaluator weights (Uncheatable payload-specific, Table 9 unweighted 51-mean, Michael 1/8 mean; completeness enforced). `corrected_contrast` implements the NB correction with `1/n + realized test/train ratio` (realized ratios 0.255–0.267 in the finalist CSVs) — correct, modulo the minor approximation of averaging the per-fold ratio. The claim "no 25-fold interval against canonical DSP excludes zero while the unit-level sign tests do" is **true** in both directions (verified in `finalist/paired_model_contrasts.csv` vs `screen/pooled_screen_contrasts.csv`); the framing "the Screen unit-level test is the decisive one" is a modeling choice that inherits the dependence/multiplicity problems from (1) and deserves the same caveat. The heldout stage is label-safe (refit on the fit panel only with `heldout_inner_folds`, predictions frozen in shards and hashed before joining), regret/rank/top-k are in native BPB with correct strata and an exact random-ranking baseline. Overstatements found: the Delphi T9 shortlist "only model" sentence (B1) and the section-5/section-12.2 contradiction on the power models' shortlists.

**(6) Michael families.** Verified correct. `families_from_buckets` parses `cXX_qY` into 24 cluster families (sizes 4–5 on both panels, dclm 118 buckets, high-quality 120), records quality 0–4 but leaves `quality_ordered=False`, so `Families.pairs` is empty and the quality axis is genuinely inert in `pair_discount`/`softplus_group_sum` (matching the report's "order undeclared" statement; `grp_pair` on Michael also collapses to discount=1.0 and per-bucket signal, as its build function checks `features.families.pairs`). The family-pooling-hurts claim is Screen-level: `bucket_family_power_grp@no_families` wins 26/8 units (p = 0.0029) at Screen and 111/79 (p = 0.024) at Certify scope, while the aggregate five-fold intervals all straddle zero — so "hurts" is unit-level evidence; the aggregate is a tie, which section 10 says.

---

## Verified-as-claimed (spot checks all matched)

- 36 promoted ablations (18 frozen-rule, 18 amended), promotion table identical between `screen/` and top level; successor ablations were never subject to the promotion rule (they are not in `registry.ABLATIONS`) — the successor's Screen evidence is post-hoc by construction.
- 38 family-macro units (18 anchors + 16 Michael tasks + 4 StarCoder family macros) built as claimed; 45 curves macro-averaged by family before pooling.
- Verifier `reduction_equivalence.csv` 46/46 pass (including the corrected crs_plus image and ladder parity at 1.3e-13); it now exits nonzero on failure (line 533–536).
- `failures.csv` empty at top level and in `finalist/`; finalist = 4,975 shards per model (4,350 tabular + 400 Michael + 225 curves), 0 failed, matching the report.
- All quoted finalist RMSE/±SD and heldout rank/regret numbers reproduce exactly from the CSVs.
- Fieldbook note: the current ledger fails to open via `fieldbook` ("unable to open database file" — SQLite error); I read it directly read-only. The experiment record and its decision notes are consistent with the handoffs; the DeepSeek review remains outstanding (quota), so handoff completion criterion 5 is still unmet — the report discloses this.

## Minor notes

- `single_phase_observatory_models_20260902.py:84-87` has a duplicated comment block (lines 84-85 repeat 86-87) — cosmetic only.
- The promotion-rule pooled contrasts use the fixed 1/4 ratio instead of the realized ratio (minor; the per-unit contrasts use realized).
- The gen2 legacy-acceptance path exists for fit shards but not heldout shards (conservative direction — would force refits, not accept stale data).

I did not edit any files, as instructed.

---
verified: 57 assistant messages, all from deepseek-v4-pro at max effort, read-only
