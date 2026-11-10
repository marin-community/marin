# MARINER release: run and value aliases (internal)

The public release (the Hugging Face dataset `Calvin-Xu/mariner-data`, the swarm checkpoint repositories, and
`lib/mariner`) uses public names only. This note records the other names the same runs and values carry in the
research tree, on W&B, and on GCS.

## Records

- `experiments/domain_phase_mix/release/public_name_aliases.csv`: every value the export renames, one row per value,
  with its table, column, observation id (run labels only), legacy name, and public name. The export writes it
  (`public_names.publish`); `test_committed_aliases_match_the_export` keeps the committed copy current.
- `.agents/projects/mariner_release/swarm_checkpoint_manifest.csv`: one row per swarm run (871), with its public and
  legacy run label, W&B run id, hash-less training run name, design kind, fit inclusion, its name in the single-phase
  benchmark, and its final HF export on GCS. The benchmark name is empty for the runs the benchmark did not score:
  the proportional repeats, the proportional calibration run, and the Llama 160M/1.2B deletions, which are held out.

## Name layers of a run

1. W&B run id (`observation_id`): the training run name, `-`, and six hex digits. It is also the run's folder on GCS,
   and the public dataset keeps it as the key.
2. Training run name: the W&B id without its hash, such as `singleavg_run_00125`. Launch manifests and the benchmark
   harness use it.
3. Benchmark name: the single-phase benchmark (`single_phase_observatory_*`) prefixes `singleavg_` to the one-phase
   runs reused from a two-phase swarm, so W&B id `fit_241_pctrl_del_dolma3_cc_art_and_design_high-d491f2` is
   `singleavg_fit_241_pctrl_del_dolma3_cc_art_and_design_high` there. The frozen held-out components key Llama
   160M/1.2B runs as `60m::<W&B id>`.
4. Legacy dataset label, used until 2026-11-09: deletions `pctrl_del_<internal bucket>`, repeats
   `proportional_repeat_<letter>` at Qwen3 (a to j) and `proportional_repeat_<trainer seed>` at the Llama settings
   (10000 to 10009).
5. Public label (`run_label`): `run_NNNNN`, `baseline_<name>`, `deletion_<bucket id with / written _>`, and
   `proportional_repeat_00` to `proportional_repeat_09`. It also names the run's folder in the checkpoint repositories.

## Internal names of the settings and objectives

- Qwen3 360M/1.6B: Delphi, 3e18, benchmark panel `delphi_3e18_39bucket`.
- Llama 160M/1.2B: 60M, `60m_1p2b`, benchmark panel `60m_39bucket`.
- Llama 200M/6B: 300M, `300m_6b`, `qsplit240_300m_6b`, benchmark panel `300m_39bucket`.
- OlmoBaseEval Easy: Table 9, `table9`.

## Where the final exports live (gs://marin-us-east5)

- Qwen3 360M/1.6B: sampled runs in `pinlin_calvin_xu/data_mixture/delphi_one_phase_augmented_swarm_3e18_20260715/`;
  the baselines and deletions in the two-phase swarm's `pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/`;
  the repeats in `pinlin_calvin_xu/data_mixture/delphi_3e18_baseline_noise_panel_20260703/`.
- Llama 160M/1.2B: sampled runs and baselines in
  `checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_single_phase_exposure_average_60m_1p2b/`; deletions in
  `checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_pctrl60/`; repeats in
  `checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_variable_subset_noise_60m_1p2b/`.
- Llama 200M/6B: sampled runs and two baselines in `checkpoints/calvin/dm/spavg_q240_300m/`; the stratified baseline
  in `checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b/`; repeats in
  `checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_variable_subset_noise_300m_6b/`. The executor
  shortened the deletions' long output paths, one folder per run, at varying depths (`checkpoints/pi~25c13de4/`,
  `checkpoints/pinlin_calvin_xu/data_mixture/ng~950eecd9/`), so take their paths from the manifest.

## Renamed values

`public_name_aliases.csv` also maps the 25 archived development campaigns, `archive::<folder>` in
`heldout.proposal_method` and `fits_learning_curves.stratum`, to descriptive names; the launches named after old
paper tables (`table5_repeats`, `table9_qa_subset`, `table9_snr_subset`); and the reliability sources named after
Delphi (`snr_fit_*_delphi`). The other policy codes in `proposal_method`, `variant`, and `predictor` keep their
internal names, which the dataset card says.
