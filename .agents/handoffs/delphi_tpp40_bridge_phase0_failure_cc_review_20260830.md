# Delphi TPP40 bridge: phase-0 failure and next-step review

## Decision needed

Review the scientifically defensible next step after the preregistered one-pair bridge failed its phase-0 Uncheatable tolerance. The user has explicitly said that one passing pair is sufficient to unblock production and that the goal is to launch TPP40 on as much compute as possible. Do not edit files.

## Frozen design and observed result

- Contract: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v4.json`
- Report: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_report_v3.json`
- Frozen primary pair: `run_order=2`, East5 v5p-8 training versus Europe v6e-8 training, both evaluated on Europe v6e-8 with identical payloads.
- Phase-0 macro Uncheatable BPB: East5 `1.0538958311`, Europe `1.0575664043`, signed delta `+0.0036705732` (Europe worse).
- Frozen one-pair mean limit: `0.002` BPB; any-row limit: `0.005` BPB. The primary pair therefore fails the binding mean limit.
- Largest component deltas are BBC News `+0.0122187` and Wikipedia English `+0.0088098`; the other five are within about `0.00174` absolute.
- The endpoint continuation is already resuming from exact step 21855 and should finish; endpoint Uncheatable and Table-9 remain pending.

## Supplementary rows already available

The earlier four-row bridge preregistered rows 120, 240, and 260 as supplementary evidence. Before the four-row parent was operationally cancelled to release capacity:

- Europe rows 120 and 260 committed complete step-21855 checkpoints.
- Europe row 240 did not commit step 21855 but has resumable work.
- Canonical East5 trajectories for all three exist.
- No supplementary acceptance result has been inspected yet.

## Candidate paths

1. Keep the original fail-closed decision: do not launch Europe production from any later supplementary pass. Finish row 2 only for diagnosis.
2. Predeclare now, before inspecting supplementary outcomes, a deterministic sequential rule: test rows 120 then 260, and use the first row that passes the unchanged phase-0, endpoint, and Table-9 thresholds. One passing pair would satisfy the user's stated operational criterion, but this changes the original primary population after seeing row 2 fail.
3. Launch Europe production without claiming raw metric pooling, requiring all selected/near-frontier checkpoints to be re-evaluated in one common region. This treats the bridge as a pooling/calibration gate rather than a training-safety gate. Assess whether the phase-0 delta supports that distinction.
4. Another option that is more defensible and still avoids waiting for East5 training capacity.

## Questions

1. Which path is scientifically defensible under the existing record, and why?
2. Is sequentially testing supplementary rows after the primary failure an unacceptable post-hoc rescue, even though they were predeclared before outcomes?
3. Does the observed component pattern suggest a broad accelerator-induced training shift or a noisy/evaluation-sensitive macro dominated by two subsets?
4. Can production training proceed if Europe and East5 results are kept region/accelerator-stratified and frontier candidates are re-evaluated in one common region, or does the failed phase-0 pair indicate a deeper incompatibility?
5. What exact additional evidence is the minimum needed before launch, given the user's explicit one-passing-pair criterion?

Return a concise verdict with blockers separated from optional diagnostics.
