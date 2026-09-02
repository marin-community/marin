# Review request: coupled-onset refinement and confirmation launch

Please perform a read-only, blocker-focused review of the exact frozen successor experiment below. The scientific question is whether the selected two-phase gain at coupled onset 0.60T is genuinely smaller than at both 0.80T and 0.90T.

## Files

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_starcoder_wsd80_coupled_onset_refinement_confirmation_20260901.py`
- `experiments/domain_phase_mix/starcoder_wsd80_coupled_onset_refinement_confirmation_design_20260901.json.gz`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_refinement_confirmation_20260901/report.md`
- `experiments/domain_phase_mix/launch_starcoder_wsd80_coupled_onset_refinement_confirmation.py`
- `tests/test_starcoder_wsd80_coupled_onset_refinement_confirmation.py`
- Prior review contract: `.agents/handoffs/three_modeling_workstreams_cc_review_20260901_RESPONSE.md`

The design hash is `79943e36932e942e9c42a5070663fae00f2c5b4e3cdb5b942fdeb1af7abac8a5`. The inventory is 24 discovery-only local Bayesian-refinement rows plus 72 fixed confirmation rows over eight previously reserved fresh seeds. Adaptive outcomes are prohibited from changing any confirmation cell or statistic. E1 is the primary intersection-union test: `gain_0.80 > gain_0.60` and `gain_0.90 > gain_0.60`, each one-sided at alpha 0.05. The manifest now explicitly defines per-seed gains and paired cross-arm one-sample t tests. E2 and E3 are secondary transport/sensitivity estimands. C4 is descriptive only.

The initial global GP was rejected before launch because posterior variance drove acquisitions to remote corners. The current acquisition fits normalized local GPs only on eligible untied observations within 0.008 BPB of each arm incumbent. Its posterior SDs are 0.0009-0.0023 BPB. BO remains a discovery-only under-sampling falsifier.

The first review returned `BLOCK`. The revised package now (1) refuses to freeze unless this response ends in `APPROVE`, (2) uses and audits terminal-only permanent checkpoint retention for every row, and (3) freezes the exact endpoint metric, paired statistic, all-row completeness requirement, exact-identity retry/no-drop rule, analysis order, and selection-bias caveat. Focused tests, pre-commit, materialization audit, and graph lowering must be rerun against this revision before freeze.

## Exact proposed command

```bash
uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude checkpoints/ --working-dir-exclude .experiments.zip --working-dir-exclude .experiments/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-include experiments/domain_phase_mix/exploratory/two_phase_many/design_starcoder_wsd80_coupled_onset_refinement_confirmation_20260901.py --working-dir-include experiments/domain_phase_mix/manifests/starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_v1_20260901/release.json --working-dir-include .agents/handoffs/starcoder_coupled_onset_refinement_confirmation_cc_review_20260901_FINAL_RESPONSE.md -- --no-wait --job-name dm-starcoder-wsd80-coupled-onset-refinement-confirmation-central2-v4-v1-20260901 --region us-central2 --zone us-central2-b --priority interactive --no-preemptible --cpu 2 --memory 12GB --disk 20GB --enable-extra-resources --extra marin-core:tpu -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -e MARIN_PREFIX gs://marin-us-central2 -- python -m experiments.domain_phase_mix.launch_starcoder_wsd80_coupled_onset_refinement_confirmation --max-concurrent 96 --confirmation I_AUTHORIZE_THE_STARCODER_WSD80_COUPLED_ONSET_REFINEMENT_CONFIRMATION
```

Please answer:

1. Is the adaptive/fixed separation actually fail-closed, with no leakage into E1-E3?
2. Is the local acquisition scientifically defensible for detecting 0.60T under-sampling, without pretending the BO rows are confirmatory?
3. Do the inventory, seeds, policy cells, checkpoint/evaluation configuration, region-local placement, release hashing, and command preserve the completed experiment's training contract?
4. Identify every blocker that should prevent submission. End with exactly `APPROVE` or `BLOCK`.
