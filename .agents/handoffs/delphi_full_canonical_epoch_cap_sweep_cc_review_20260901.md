# Delphi full-canonical epoch-cap sweep: pre-submit review

## Decision requested

Return `GO` only if the exact 16-row sweep below is mechanically safe, scientifically labeled correctly, region-local, hash-pinned, idempotent, and cannot collide with the completed shared-shape sweep. Otherwise return `NO-GO` with concrete blockers.

## Scientific intent

Train tied one-phase mixtures obtained by optimizing the full per-bucket canonical DSP response separately for Uncheatable and Table-9 under whole-run per-bucket epoch caps (2,4,6,8,10,12,14,16). This is an empirical stress test of the materially different full-canonical optima, not a claim that their predicted BPB is calibrated. The model has strong held-out rank correlation but weakly identified per-bucket shape parameters and extrapolative high-cap optima.

All 16 runtime mixtures are distinct. They use the same data seed and trainer seed as the completed shared-shape epoch-cap sweep so endpoint differences have common random numbers.

## Files to inspect

- `experiments/domain_phase_mix/exploratory/two_phase_many/materialize_delphi_one_phase_dsp_epoch_cap_sweep_20260828.py`
- `experiments/domain_phase_mix/launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18.py`
- `experiments/domain_phase_mix/launch_delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18.py`
- `tests/test_delphi_one_phase_dsp_epoch_cap_sweep_3e18.py`
- `tests/test_delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/manifest.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/candidate_summary.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/candidate_weights.csv`

Frozen candidate SHA-256: `909f7240eee5ee2b1ecc2fa88a987fe471b867609eff4993b31a7b765f35d84d`.

Local verification already completed:

- Fresh materialization exactly matched the prior independent diagnostic after restricting it to caps at most 16.
- `29 passed` across the two launcher suites and East5 safety tests.
- Focused pre-commit checks passed.
- Dry run emitted 16 runtime-distinct full-horizon tied trainings.
- The exact command passed `east5_launch_safety` with parent `us-east5-a`, child `us-east5-b`, and all GCS state under `gs://marin-us-east5`.

## Exact command

```sh
uv run iris --cluster=marin job run --no-wait --priority interactive --job-name dm-delphi-3e18-onephase-fullcanonical-dsp-epochcaps-v6e8-20260901 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 8GB --disk 20GB --extra marin:tpu --bundle-include 'experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/candidate_weights.csv' --bundle-include 'experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/manifest.json' --exclude '^checkpoints/' --exclude '^\.experiments\.zip$' --exclude '^\.agents/' --exclude '^docs/' --exclude '^tests/' --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/(?!delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901/(?:candidate_weights\.csv|manifest\.json)$)' --exclude '^experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/' --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/dsre_ceq_debug/' --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many\.csv$' --exclude '\.pdf$' -e MARIN_PREFIX gs://marin-us-east5 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18 --tpu-type v6e-8 --tpu-region us-east5 --tpu-zone us-east5-b --max-concurrent 16
```

Pay particular attention to candidate identity, full-run epoch accounting, old-sweep reproducibility after the generic launcher refactor, output/run-ID collisions, bundle inclusion of gitignored candidate files, regional placement, and whether every final checkpoint receives both inline Uncheatable and native Table-9 evaluation.
