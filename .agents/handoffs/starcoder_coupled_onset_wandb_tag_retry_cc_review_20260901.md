# Review: StarCoder coupled-onset W&B tag repair and idempotent retry

Review this production retry read-only. Inspect the referenced launcher, test, release, original failure, and exact command. Do not edit the repository.

## Incident

The frozen 96-row parent /calvinxu/dm-starcoder-wsd80-coupled-onset-refinement-confirmation-central2-v4-v1-20260901 emitted all 96 children. Every child reached TPU/JAX initialization, then failed before the first model step with:

    Value error, Tag 'starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4' is 65 characters. Tags must be between 1 and 64 characters

No row trained. The common output paths remain failed Marin artifacts.

## Repair

- Launcher: experiments/domain_phase_mix/launch_starcoder_wsd80_coupled_onset_refinement_confirmation.py
- Test: tests/test_starcoder_wsd80_coupled_onset_refinement_confirmation.py
- Release: experiments/domain_phase_mix/manifests/starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_v1_20260901/release.json

The only runtime change is panel_tag from the 65-character string to wsd80_coupled_onset_successor_c2v4. A regression test enforces the W&B 64-character limit. The scientific design, 96 run names, output root, deployment version, seeds, mixtures, schedules, cache paths, TPU type, and evaluation/checkpoint contracts are unchanged. run(..., force_run_failed=True) retries the failed artifact identities.

Post-repair evidence:

- Focused tests: 5 passed.
- Targeted pre-commit: passed.
- Audit: 96 rows and 6 representative runtime configs; Central2 StarCoder cache validated.
- Dry run: 96 frozen training graphs lowered.
- Re-frozen release SHA-256: 561f2abbddfb17635fdab07785ea9eebc459bb47cfd2a10dc818ceffac1d310d.

## Exact retry command

~~~sh
uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude checkpoints/ --working-dir-exclude .experiments.zip --working-dir-exclude .experiments/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-include experiments/domain_phase_mix/exploratory/two_phase_many/design_starcoder_wsd80_coupled_onset_refinement_confirmation_20260901.py --working-dir-include experiments/domain_phase_mix/manifests/starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_v1_20260901/release.json --working-dir-include .agents/handoffs/starcoder_coupled_onset_refinement_confirmation_cc_review_20260901_FINAL_RESPONSE.md --working-dir-include .agents/handoffs/starcoder_coupled_onset_wandb_tag_retry_cc_review_20260901_RESPONSE.md -- --no-wait --job-name dm-starcoder-wsd80-coupled-onset-refinement-confirmation-central2-v4-v1-retry1-20260901 --region us-central2 --zone us-central2-b --priority interactive --no-preemptible --cpu 2 --memory 12GB --disk 20GB --enable-extra-resources --extra marin-core:tpu -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -e MARIN_PREFIX gs://marin-us-central2 -- python -m experiments.domain_phase_mix.launch_starcoder_wsd80_coupled_onset_refinement_confirmation --max-concurrent 96 --confirmation I_AUTHORIZE_THE_STARCODER_WSD80_COUPLED_ONSET_REFINEMENT_CONFIRMATION
~~~

Check:

1. The fix is sufficient for the observed failure.
2. No scientific/runtime identity besides observability metadata changed.
3. The retry is idempotent and will reuse the same 96 artifact/output identities.
4. The command is region-local and includes every untracked file required at runtime.
5. No additional canary is scientifically or operationally necessary because all rows failed at the same pre-training validation point.

End with exactly APPROVE if there is no blocker, otherwise BLOCK and the minimum correction.
