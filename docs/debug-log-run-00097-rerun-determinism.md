# Debugging log for run_00097 rerun determinism

Investigate why `run_00097` and its rerun `baseline_dsre_observed_consensus` were not bitwise identical despite using the same schedule.

## Initial status

- `run_00097` and `baseline_dsre_observed_consensus` used the same frozen phase weights.
- Their metrics differed substantially on MMLU and slightly on perplexity-style metrics.

## Hypothesis 1

The rerun changed some configuration field that affects the training trajectory, most likely a seed.

## Changes to make

- Inspect the domain-phase mix training config path.
- Compare W&B configs and code provenance for both runs.

## Results

- The run weights are the same by construction:
  - `baseline_dsre_observed_consensus` loads its phase weights directly from the `run_00097` row in `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv`.
- The code version is the same:
  - both W&B runs report `git_commit = c43345016345281ca9554acb1b3f49ae48d89f85`.
- The training data seed is different:
  - `run_00097`: `data_seed = 97`
  - `baseline_dsre_observed_consensus`: `data_seed = 242`
- This comes from `experiments/domain_phase_mix/experiment.py`, where `MixtureExperiment.create_train_config` sets `data_seed = run_id`.
- `lib/levanter/src/levanter/main/train_lm.py` then overrides the data RNG with `config.data_seed`.
- `trainer.seed` remained fixed at `0` for both runs.

## Conclusion

These two runs were not configured for bitwise-identical replay. They used the same schedule and same code commit, but different `data_seed` values because the rerun had a different `run_id`. That changes the data order and breaks deterministic replay.

## Future Work

- [ ] Add an explicit replay path that lets a rerun pin `data_seed` independently of `run_id`.
- [ ] Decide whether rerun baselines should default to inheriting the source run's `data_seed`.
