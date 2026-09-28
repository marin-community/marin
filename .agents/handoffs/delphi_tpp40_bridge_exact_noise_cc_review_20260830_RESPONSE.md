I've reviewed the audit script, both frozen contracts, the acceptance report, the noise estimator it depends on, and the cross-region data audits. Verdict below.

---

# Verdict

**Launch gate: NO-GO, already determinate.** Not "pending further evidence" — `bridge_acceptance_report_v3.json` records `numerical_acceptance_passed: false`, `production_launch_authorized: false`, `idempotence.passed: false`, and five blocking errors. Contract v4's decision rule requires *every* threshold to pass. Phase-0 has failed one on a completed, frozen measurement. The endpoint cannot rescue it.

**Science: the delta is real.** It is not explicable by run-to-run noise at any plausible correction to the noise estimate. But the mechanism is undiagnosed, and its fingerprint looks more like an *event* than like smooth numerical drift.

## What I verified independently

- **Internal arithmetic is sound.** Mean of the seven component deltas = `0.0036705221` vs. the reported macro `0.0036705732` (5.1e-8 apart, float32-level). `z = 0.0036706/0.00094420 = 3.8875` ✓. `P(|N(0,σ_p)|<0.002) = 0.9659` ✓. `P(-0.002<N(0.003,σ_p)<0.002) = 0.1448` ✓.
- **The estimator is not bespoke.** `_repeat_groups`/`_median_within_group_sd` (`audit_delphi_tpp40_bridge_noise.py:72-85`) reproduce the pre-existing `run_noise()` at `analyze_prefix_search_evidence_20260819.py:148` exactly — same 8-decimal rounding, `ddof=1`, median across groups. Only the *metric* changed. That change was correct and necessary: the gate metric is the Uncheatable macro, so its noise reference must be too. Interpreting an Uncheatable gate with Table-9's σ was a genuine category error.
- **The report is pinned to v4, not the superseded v3.** Report v3 carries a populated `east5_reference_mirror` with `mirror_manifest_sha256: 08c6160b…`, matching `bridge_acceptance_contract_v4.json:44`. Contract v3 has no such block. The chain is coherent.
- **I excluded the obvious training-data confound.** `stack_hydration_cross_region_metrics_diff.json` reports `status: different_metrics`, 80 differing tasks — but `stack_hydration_delta_transfer_size.json` bounds the actual corpus difference at **68 blobs / 257 KB uncompressed**, all stack_edu *code* files. Too small by orders of magnitude to move BBC News 0.012 BPB, and the wrong family.

**Could not verify (no shell in this session):** none of the SHA-256 digests, and I could not re-run the audit. The script digest in the handoff matches what the artifact records for *itself* — that is self-consistency, not independent confirmation. One command settles it; see "evidence that would authorize."

## Answers

**Q1 — Real drift, and the "underpowered gate" defense is now dead.** This is the audit's most decision-relevant consequence and the handoff undersells it. Under the old Table-9 σ, paired SD = 0.004836, and the 0.002 gate would have passed only **32%** of the time under *zero* true shift — a 68% false-failure rate, which would have been a legitimate underpowered-gate defense. Under the corrected σ the gate passes **96.6%** under zero shift. A 3.4% false-failure rate is well-calibrated. The gate is fine; the result is bad.

**Q2 — There is no four-row contract to restore.** This is the load-bearing finding. `run_orders: [2,120,240,260]` exists only in **v2**, which was superseded *twice before any result existed*. Both successors are one-pair gates that explicitly demote the other three:

> v3: "Europe run_orders 120, 240, and 260 remain supplementary evidence and do not block production."
> v4: "The capacity-blocked East5 evaluation parents remain supplementary and do not block production training."

Promoting them now, after an adverse primary, is not restoration — it is reversion to a superseded contract, chosen because the binding one failed. Row-level cherry-picking isn't the hazard here; **contract-level** cherry-picking is.

**Q2 also mischaracterizes v2's rule, in the weakening direction.** Your Q3 says "pooled **signed** mean." v2 says *"the four-run **mean absolute** paired delta must be at most 0.002 and **every** individual run-order macro absolute paired delta must be at most 0.005."* The report key is `mean_absolute_paired_delta_max`. A mean of absolute values cannot be cancelled by opposite signs; a signed mean can. Adopting "signed mean" would be exactly the silent weakening you said must not happen.

Worth noting: under v2's *actual* rule, four rows share a budget of 4 × 0.002 = 0.008 in summed |δ|. Row 2 consumes 0.00367, leaving the other three an average of **≤ 0.00144**. The four-row rule was never a rescue mechanism.

**Q3 — If you run them, freeze v5 *before* reading any of the three phase-0 boundaries.** Non-negotiable terms: row 2 is a mandatory member (cannot be dropped, re-run, or replaced); **mean-absolute** stays at 0.002 and any-row at 0.005, verbatim from v2; the artifact itself must record that this is an outcome-contingent amendment and why; and a pass produces a *new* four-row claim — it does not retroactively convert the v4 one-pair no-go, which stays in the record.

Add one thing v2/v3/v4 all lack: **a pre-committed per-component bound.** All three say component cells are "completeness inputs, not rows for this threshold." That is precisely why a 12σ BBC blowout sits inside a macro only 1.8× over its limit. Averaging seven components is what hides this failure mode, and pooling four rows would hide it further.

**Q4 — No, not for the shard as specified.** Contract v4 forbids it outright, and your own stated bar was one passing matched trajectory. Row 2 *is* that trajectory, and it failed.

But there is a clean way to use the idle slices today that needs no amendment: **stop trying to certify Europe as poolable with East5, and run self-contained sub-experiments there whose comparisons are Europe-vs-Europe only.** Their absolute BPB never joins an East5 row, so the bridge is irrelevant to them. That is a different thing from the gated production shard — but it converts idle capacity into real science now, which is what you actually want.

**Q5 — Flaws, by materiality:**

1. **Scale transfer (largest, unrecorded).** The pool is `delphi_3e18_append_only_heldouts_20260714`, filtered to `coordinate_disjoint` — the **3e18-FLOP** panel (`swarm39_harness` covers "60m, 300m, and 3e18"). The bridge is **TPP40**: 27,335 steps, 4.3 GB checkpoints. σ is imported across a ~100× compute gap with no check that seed noise is scale-invariant, and neither the script nor the artifact records the limitation. Inherited from v2/v3/v4, so it is standing methodological debt rather than a new sin — but the handoff presents σ as if it characterizes the bridge's own noise, and it does not.
2. **The median is downward-biased.** Group sizes {2:7, 3:9, 4:2, 8:1, 26:2} are mostly 1–2 df. For n=2, median(s)/σ = 0.674; n=3 → 0.833; n=4 → 0.888. The across-group median lands near the n=3 region, so σ is likely understated ~15–25%. **This cuts against the finding** — but survives it: at σ×1.2, z = 3.24 (p ≈ 0.0012); at the extreme σ×1.48, z = 2.63 (p ≈ 0.009). A pooled estimator on 88 df (7·1 + 9·2 + 2·3 + 7 + 2·25) is strictly better and free.
3. **The artifact throws away the per-group SDs.** `_median_within_group_sd` computes and returns `group_sds`, and `build_audit` discards it (`audit_delphi_tpp40_bridge_noise.py:98`). No reviewer can recompute a pooled σ, check heterogeneity, or see whether the two 26-member groups dominate, without re-running. For a frozen auditable record that is a real gap, and a two-line fix.
4. **The independence assumption is wrong, but conservatively so** — this is the one you were most worried about, and it runs in your favor. v3's scope note states the pair preserves "data, seeds, model, optimizer, schedule, and mixture coordinates," differing only in data-parallel width (4 v5p → 8 v6e). The historical groups differ *by seed* — a strictly larger perturbation. Chaotic amplification means both saturate at the same attractor spread, so **√2·σ_seed is the ceiling of the null**, reachable only under full decorrelation. The observed delta exceeds even that ceiling. Fixing this assumption strengthens the finding.
5. **n=1.** No within-bridge variance estimate. The p-value is borrowed; the drift *magnitude* has no confidence interval from bridge data.

**The 3.89 is not the strongest evidence — the fingerprint is.** BBC News +0.01222 (12.3σ) and Wikipedia English +0.00881 (6.8σ), against arxiv_cs −0.33σ, github_python 0.36σ, github_cpp 0.62σ, arxiv_physics 0.64σ. Chaotic divergence moves components roughly coherently; it does not blow out two and leave five at ~0. Inflate every σ by 2× to absorb every bias above and BBC is still 6.1σ. And it is not an eval artifact: report v3 shows both sides on byte-identical payloads (`79546f8d…`, 13,432,369 bytes, 42 objects) and v4 evaluates both on the *same* Europe v6e-8 evaluator. The difference is in the weights.

That pattern reads like a **bug, not drift** — and step 21855 is a resume/restore boundary.

## Blockers

1. Contract v4 numerical acceptance failed. No production submission is permissible under the frozen rule.
2. Europe endpoint (27335) uncheatable + table9 missing; both executor statuses `None`.
3. Idempotence unaudited — a v4 `required_completion` item, still unsatisfied.
4. The BBC/Wikipedia mechanism is undiagnosed.
5. No frozen contract exists that would make supplementary rows admissible.

## Recommended next action

**Diagnose before spending three slices.** In priority order:

1. **Measure the evaluator's own reproducibility floor** — re-evaluate both step-21855 checkpoints twice with the same evaluator. This is the cheapest, highest-value missing number: it discharges blocker #3, and it lets you interpret the delta *without importing 3e18-scale σ at all*.
2. **Inspect phase-0 boundary training state**, not just eval output: LR at 21855, optimizer state norms, tokens consumed, data-iterator position, both sides. A resume/offset bug shows up here and nowhere else.
3. **Let the endpoint finish.** Already running, costs nothing, and tells you whether the gap grows, persists, or closes — gate-irrelevant, diagnostically decisive.
4. **Freeze v5 now, before submitting anything.** It costs nothing and preserves the option; freezing it *after* the boundaries are readable forecloses it permanently.

On the supplementary rows themselves: **run them as characterization, not as an appeal.** Three more rows sharing the BBC/Wikipedia fingerprint would confirm a real, characterized effect with a within-bridge variance estimate on n=4. That is worth three idle slices. It is not worth them as a re-gate. The resume command is clean for this — same experiment name, wandb groups, and TPU zone as the row-2 command, differing only in `--run-orders` and `--max-concurrent 3`.

## Evidence that would authorize or reject

**Rejects Europe production outright** (any one):
- Endpoint 27335 delta also exceeds 0.002, or BBC/Wikipedia stay ≫ 5σ → systematic, confirmed.
- The diagnostic shows a training-side defect at the phase-0 boundary → fix and re-bridge from scratch; the current pair is void either way.
- Supplementary rows under a pre-frozen v5 give mean-absolute > 0.002 or any row > 0.005.

**Authorizes** (all required):
- A v5 contract frozen and independently reviewed *before* any of the three boundaries is read, retaining mean-absolute 0.002 / any-row 0.005, including row 2 as mandatory, adding a per-component bound, and recording itself as an outcome-contingent amendment.
- Four-row mean-absolute ≤ 0.002 **with row 2's 0.00367 included** — the other three must average ≤ 0.00144.
- No row's BBC or Wikipedia component exceeding the pre-stated per-component bound.
- Endpoint pairs complete for all four; idempotence audited and passing; mirror inventories unchanged.
- A recorded mechanism for row 2 consistent with the pooled result. Without this, a pooled pass means "row 2 was an unexplained 12σ outlier we averaged away" — which is not a finding you want in the record behind a production launch.

**Housekeeping (one command each):** confirm the handoff's two digests, and confirm `source_sha256.acceptance_report = 55b73a50…` still matches `bridge_acceptance_report_v3.json` — if the report changed after the audit ran, the audit is stale. I could not check any of these here.
