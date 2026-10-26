# Delphi 3e18 checkpoint inventory for the SFT probe

Inventory taken 2026-10-14 from GCS object listings plus small metadata reads: each run's `.executor_status` and
`.executor_info`, two Levanter `metadata.json` files, and one HF `config.json`. No weights were read or copied.
Row-level data: `checkpoint_inventory.csv` beside this file. Arms A, B, B' and C refer to `PLAN.md` beside this file.

Every run in panels 1-6 below still has its final checkpoint. The CSV has 1,058 rows over 1,016 distinct training
directories (panel 2's 42 aliases reuse panel-1 directories). Each directory holds:

- a Hugging Face export at `hf/step-3006` with `config.json`, a 1,433,239,408-byte `model.safetensors` and tokenizer files;
- a Levanter checkpoint at `checkpoints/step-3006` (`metadata.json`, `manifest.ocdbt`, `d/`);
- executor status `SUCCESS`.

The sampled HF config is `Qwen3ForCausalLM` (hidden size 896, 10 layers, Llama 3 vocabulary). The two sampled Levanter
checkpoints are permanent (`is_temporary: false`). Nothing has been cleaned up. All checkpoints are in
`gs://marin-us-east5` (region us-east5).

## Terms

- Arms (from `PLAN.md`): A is MARINER's OlmoBaseEval Easy optimum `lwspu_t9_snc_cap08`; B is the chosen two-phase
  schedule; C is B's phase-weighted average trained one-phase; B' is a Dolmino-late schedule whose phase average is A's
  mixture.
- Anchor: the one-phase mixture whose aggregate a fixed-aggregate panel holds constant. Panels 3 and 4 use two anchors.
  `table9_frontier` is the mixture of the decoupled-panel tied run `dphase_t9b075_tied` (run-name codes `a1` in panel 3
  and the random-phase panel, `fiber_1` in panel 4, `t9b075` in the decoupled panel). `uncheatable_frontier` is the
  mixture of `dphase_unch05_tied` (codes `a0`, `fiber_0`).
- Tied control (`contrast_family` `center_control`): a run of the anchor mixture itself, identical in both phases.
- Seed block: an index that sets the data seed. A treatment and the tied control of its anchor in the same block share a
  data seed.
- Table-9: OlmoBaseEval Easy BPB, the unweighted mean over its 51 components. Lower BPB is better.

## Counts

GCS prefixes are relative to `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/`.

| CSV `panel` | Fieldbook experiment | GCS prefix | Runs | HF found | Missing | Two-phase / one-phase |
|---|---|---|---|---|---|---|
| p1_two_phase_swarm | exp_01kxhhy6k73fwz111ab42q5ssq | `delphi_augmented_swarm_3e18_20260714/` | 280 | 280 | 0 | 238 / 42 |
| p2_one_phase_swarm | exp_01kxm37amarnnd1wey37nj8ed5 | `delphi_one_phase_augmented_swarm_3e18_20260715/` (238 runs); the 42 aliases point into the p1 prefix | 280 | 280 | 0 | 0 / 280 |
| p3_aggressive_phase_asymmetry | exp_01ky48sv686wh45tnyc1pnrgym | `delphi_3e18_aggressive_phase_asymmetry_20260722/` | 290 | 290 | 0 | 258 / 32 |
| p4_phase_fiber_doe | exp_01kxwpm5236b4ej68ph3xvra2t | `delphi_3e18_frontier_phase_fiber_20260719/` | 200 | 200 | 0 | 192 / 8 |
| p5_mariner_3e18 | exp_01m1xv9n8j3v1e9175xf5te225 (trainer seed 0, archived), exp_01m1yjft203bdg75hg0rdv4b5g (trainer seeds 1, 2) | `delphi_frozen_procedure_validation_3e18_20260908/`, `delphi_fairness_repeats_3e18_20260908/` | 6 | 6 | 0 | 0 / 6 |
| p6_table9_two_phase_frontier | exp_01kwc2ha76te8g6277h9hq520h (archived) | `delphi_decoupled_phase_information_validation_3e18_20260712/` | 2 | 2 | 0 | 1 / 1 |

All runs use batch size 128 and 3,007 steps. The final step is 3006 and the phase boundary is step 2400. Panels 1-4 and 6
trained on v5p-8 in us-east5-a. The MARINER runs trained on v6e-8 in us-east5-b. `hf/` and `checkpoints/` hold only
`step-3006`, so no checkpoint exists at the phase boundary.

## CSV conventions

- The 13 requested columns come first: `panel` through `bucket_region`. `checkpoint_uri` is the HF export and `exists`
  is true when its `config.json` and `model.safetensors` are listed. Extra columns: `seed_block` (panels 3-4),
  `phase_tv`, `levanter_checkpoint_uri`, `levanter_exists`, `fieldbook_experiment_id`, `fieldbook_run_id`, `notes`.
- `anchor_or_design_id` holds the source design name for panels 1-2 (for example `run_00002`), the anchor for panels 3, 4
  and 6, and the policy name for panel 5.
- `phase_tv` is the total variation distance between the phase-0 and phase-1 weights in the checkpoint's runtime config
  (`.executor_info`).
- `policy_class` follows the same runtime weights: `one_phase` when the phases are identical (max absolute difference at
  most 1e-12). Fieldbook labels all 280 panel-1 rows `two_phase`. 42 of them are phase-constant: the three baselines
  (`baseline_proportional`, `baseline_unimax`, `baseline_stratified`) and the 39 `pctrl_del_*` domain deletions. Panel 2
  reuses these 42 checkpoints. In panels 3, 4 and 6 the one-phase rows are the `center_control` tied controls.
- `run_name` is the Fieldbook run name, for example `fiber_1_domain_13_minus` or `dphase_t9b075_can_e0p005`. The GCS
  directory name is the launcher run name plus the six-character executor hash, for example
  `fiber_131_fiber_1_domain_13_minus-05f1fe` or `dphase_t9b075_can_e0p005_3e18-6c15f1`. The W&B training run id equals
  the directory name.
- `contrast_family` for panels 1-2 holds the source-panel family (`baseline`, `qsplit_signal`, `domain_deletion`).
- `objective_bpb_table9` is `table9_macro_bpb` from Marin's native OlmoBaseEval Easy evaluator.
- `objective_bpb_uncheatable` is the seven-corpus Uncheatable aggregate. All these runs predate the schema-2 evaluator of
  2026-09-12, so the logged value equals the paper's fixed-weight value.

## How paths were derived and checked

Each launcher names the training step `<experiment>/<run name>` and writes to `this_output_path()`. The final export is
`hf/step-{train_steps - 1}`, which is also the checkpoint the native Table-9 step evaluates. Sources per panel:

| Panel | Launcher | Run list and design | Directory and BPB source |
|---|---|---|---|
| p1 | `launch_delphi_augmented_swarm_3e18.py` | `delphi_augmented_swarm_3e18_20260714/training_manifest.csv`, `run_specs.json` | `delphi_3e18_observed_components_20260724/observed_component_panel.csv` (`training_run_id`); equals `delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv` exactly |
| p2 | `launch_delphi_one_phase_augmented_swarm_3e18.py` | `delphi_one_phase_augmented_swarm_3e18_20260715/training_manifest.csv` (`disposition`, `reused_two_phase_run_name`) | same observed-component panel |
| p3 | `launch_delphi_aggressive_phase_asymmetry_panel_3e18.py` | `delphi_3e18_aggressive_phase_asymmetry_20260722/candidate_manifest.csv` | `delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results.csv` (`training_wandb_id`) |
| p4 | `launch_delphi_frontier_phase_fiber_panel_3e18.py` | `delphi_3e18_frontier_phase_fiber_20260719/candidate_manifest.csv` | `delphi_3e18_frontier_phase_fiber_results_20260719/observed_results.csv` (`training_wandb_name`) |
| p5 | `launch_delphi_frozen_procedure_validation_3e18.py`, `launch_delphi_fairness_repeats_3e18.py` | Fieldbook runs | `eval_metrics_uri` in `delphi_frozen_procedure_validation_3e18_20260908/measured_results.csv` and `delphi_fairness_repeats_3e18_20260908/measured_results.csv` |
| p6 | `launch_delphi_decoupled_phase_information_panel.py` | `decoupled_phase_information_validation_panel_20260712/` | `decoupled_phase_information_validation_results_20260712/observed_results.csv`, `delphi_3e18_append_only_heldouts_20260714/heldout_current.csv` |

Launchers are in `experiments/domain_phase_mix/`. Manifests and results are under
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/`. BPB values were compared across sources
where two exist. Panel 1 matches the panel-1 wide CSV exactly. Panels 2-4 and 6 match the 3e18 heldout registry
`delphi_3e18_append_only_heldouts_20260714/heldout_current.csv` to within 5e-7, the rounding of the CSV. Panel 5 matches
the four-decimal values and means in `delphi_fairness_repeats_3e18_20260908/fairness_summary.csv`.

Verification listed each experiment prefix. All expected run directories are present. In the panel 1-4 prefixes the only
other children are `manifest-*` and `source/`. For every directory, the scan then listed `hf/`, `hf/step-3006/`,
`checkpoints/` and `checkpoints/step-3006/`, and read `.executor_status` and `.executor_info`. For panels 1-5 the CSV
takes seeds, steps and phase weights from `.executor_info`. These match Fieldbook attrs and the local manifests on every
row, with zero mismatches. For panel 6, `.executor_info` holds the data seed and the mixture path. Its phase weights come
from the local mixture CSVs, whose MD5 hashes match the launch copies on GCS. Its trainer seed comes from the W&B tag
`trainer_seed=0`, which matches the launcher's fixed trainer seed of 0.

None of the Fieldbook experiments in this inventory has `checkpoint` artifacts. The scaling ladder
exp_01m1zy8yths4dqp5bgc0ffzztp has them, but it has no 3e18 runs (2e19, 3e20 and 1e21 only).

## Storage

`gs://marin-us-east5` has Autoclass enabled, with ARCHIVE as the terminal class. Its lifecycle rules delete only
`tmp/ttl=*` prefixes, and soft delete keeps deleted objects for 7 days. Autoclass has moved the weight files to colder
classes. A sample of `model.safetensors` objects shows:

- NEARLINE for one run each from panels 1, 3, 4 and 5, about 30 days after creation;
- COLDLINE for both panel-6 runs since 2026-10-11, 91 days after creation.

Panels 1-4 reach 90 days in mid to late October, so their weights will likely move to COLDLINE soon. Each transition
updates the object's `updated` time: August 2026 for panels 1-4 and October for panels 5-6. `timeCreated` and the object
generation are unchanged, so no file was rewritten. Autoclass charges no retrieval fee for colder objects, and a read moves
the object back to Standard.

## Pairing between panels 1 and 2

Pair rows by `anchor_or_design_id`, the source design name (for example `run_00002`). Each design keeps the same launcher
`run_id`, data seed and trainer seed in both panels. The data seed equals the `run_id`, 7141000 plus the run order
(7141000-7141279), and the trainer seed is 0 for all 280 pairs. Four records agree on every pair: Fieldbook
`research.data_seed` (panel 1), Fieldbook `seed.data` (panel 2), both training manifests, and the runtime config of
every checkpoint.

The panel-2 weights equal the realized phase average α0·w0 + (1 − α0)·w1, with α0 = 2400/3007 = 0.798138, to 1e-16.
The phase boundary falls at step 2400 of 3007. The nominal average 0.8·w0 + 0.2·w1 differs from the trained weights by at
most 0.0012 in total variation.

238 designs form true two-phase/one-phase pairs. Their two-phase schedules are far from tied (phase TV 0.38-0.66, median
0.51). The 42 panel-2 aliases point to the panel-1 checkpoint of the same design. Over the 238 pairs, the two-phase run
has higher (worse) BPB on average: Table-9 by 0.0033 (SD 0.027) and Uncheatable by 0.0018 (SD 0.011).

## Control pairing in panels 3 and 4

In panel 3, each treatment pairs with the tied control that shares its anchor (`anchor_or_design_id`:
`uncheatable_frontier` or `table9_frontier`) and its `seed_block` (0-15). The panel's results report
(`reference_outputs/delphi_3e18_aggressive_phase_asymmetry_results_20260723/report.md`, 2026-07-23) calls this the
same-seed control. The data seed is 7224000 + seed_block for both anchors, and the trainer seed is 0. Blocks are
assigned this way:

- balanced partition k uses block k (k = 0-15) for both signs, all three target TVs and both anchors;
- handcrafted recipe r uses block r (r = 0-7) at all three TVs;
- Dolmino-late replicate 0 uses blocks 8, 9 and 10 (late Dolmino share 0.75, 0.90, 1.00);
- Dolmino-late replicates 1 and 2 use blocks 14 and 15 for all three shares, so blocks 14 and 15 each hold the tied control
  and all three late shares at one data seed;
- each anchor has one tied control per block, 16 per anchor.

Panel 4 uses the same rule with four blocks per anchor. The data seed is 7192000 + block for `uncheatable_frontier` and
7192010 + block for `table9_frontier`. Each block holds one tied control and 24 treatments, and plus/minus pairs share a
block.

The `table9_frontier` tied control is one mixture across panels 3 and 4, panel 6 (`dphase_t9b075_tied`) and the July
random-phase population panel (`delphi_3e18_frontier_random_phase_population_20260720`, not inventoried here). Its 25 runs
at different data seeds score 1.0641 Table-9 BPB on average (SD 0.0040, range 1.0553-1.0717). The
`uncheatable_frontier` control is also one mixture across panels 3 and 4. Its 16 panel-3 runs have an Uncheatable SD of
0.0010.

## MARINER 3e18 proposals (arm A and its Uncheatable counterpart)

Both policies are one-phase. Each was trained at three trainer seeds and one data seed.

| Policy | Trainer seed | Data seed | Table-9 | Uncheatable | HF checkpoint |
|---|---|---|---|---|---|
| lwspu_t9_snc_cap08 | 0 | 662009 | 1.068202 | 1.000443 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_validation_3e18_20260908/table9_seed662009/lwspufp_table9_seed662009_lwspu_t9_snc_cap08-a835d3/hf/step-3006` |
| lwspu_t9_snc_cap08 | 1 | 662009 | 1.062406 | 0.998080 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_fairness_repeats_3e18_20260908/table9_seed662009_t1/lwspufr_table9_seed662009_t1_lwspu_t9_snc_cap08-067b9a/hf/step-3006` |
| lwspu_t9_snc_cap08 | 2 | 662009 | 1.072729 | 1.000560 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_fairness_repeats_3e18_20260908/table9_seed662009_t2/lwspufr_table9_seed662009_t2_lwspu_t9_snc_cap08-6ed4ac/hf/step-3006` |
| lwspu_u_snc_cap06 | 0 | 666200 | 1.087603 | 0.981424 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_validation_3e18_20260908/uncheatable_seed666200/lwspufp_uncheatable_seed666200_lwspu_u_snc_cap06-8cf646/hf/step-3006` |
| lwspu_u_snc_cap06 | 1 | 666200 | 1.096239 | 0.983437 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_fairness_repeats_3e18_20260908/uncheatable_seed666200_t1/lwspufr_uncheatable_seed666200_t1_lwspu_u_snc_cap06-dd4b1e/hf/step-3006` |
| lwspu_u_snc_cap06 | 2 | 666200 | 1.088767 | 0.982546 | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_fairness_repeats_3e18_20260908/uncheatable_seed666200_t2/lwspufr_uncheatable_seed666200_t2_lwspu_u_snc_cap06-fd115f/hf/step-3006` |

The three-seed means are 1.0678 Table-9 (SD 0.0052) for `lwspu_t9_snc_cap08` and 0.9825 Uncheatable (SD 0.0010) for
`lwspu_u_snc_cap06`, which match the paper. Replace `hf/step-3006` with `checkpoints/step-3006` for the Levanter
checkpoint. No other inventoried run uses data seed 662009 or 666200.

## Table-9 two-phase frontier and the best two-phase runs

The panel-3 results report names `dphase_t9b075_can_e0p005_3e18` (Table-9 1.056690) as the existing Table-9 frontier.
It is one run, launched as `dphase_t9b075_can_e0p005_3e18` and recorded in Fieldbook as `dphase_t9b075_can_e0p005`
(run_01kxb62emy0y9vzt23ytze0pdv in exp_01kwc2ha76te8g6277h9hq520h, archived on 2026-08-01). It comes from the 79-run
decoupled phase-information validation panel. The 3e18 heldout registry (1,969 runs up to 2026-07-28) has no other run
of this mixture.

- Two-phase, with phase TV 0.115: anchor `t9b075` (`table9_frontier`), canonical family, phase-information budget 0.005.
- Phase 1 has more `dolma3_stack_edu` (+0.034), `dolmino_synth_qa` (+0.030), `dolmino_stack_edu_fim` (+0.016),
  `dolmino_synth_math` (+0.015) and `dolma3_finemath_3plus` (+0.006) than phase 0. It has less `dolmino_common_crawl_hq`
  (−0.034) and `dolma3_cc/literature_high` (−0.011).
- Full phase weights: `reference_outputs/decoupled_phase_information_validation_panel_20260712/mixtures/dphase_t9b075_can_e0p005.csv`.
  Its MD5 matches the launch copy at
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_decoupled_phase_information_validation_20260712/mixtures/dphase_t9b075_can_e0p005.csv`.
- Data seed 690300, trainer seed 0, v5p-8.
- Table-9 1.056690; Uncheatable 0.997804.
- HF checkpoint: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_decoupled_phase_information_validation_3e18_20260712/dphase_t9b075_can_e0p005_3e18-6c15f1/hf/step-3006`.
  Levanter: same directory, `checkpoints/step-3006`.

Its 0.8/0.2 phase average equals the `table9_frontier` tied mixture exactly. The realized α0 shifts the average by TV
0.0002. The tied run used the same data and trainer seeds, so this pair already exists at one seed:
`gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_decoupled_phase_information_validation_3e18_20260712/dphase_t9b075_tied_3e18-c0ea32/hf/step-3006`
(Table-9 1.057530, 0.00084 above the two-phase run).

The lowest Table-9 among two-phase runs in panels 1, 3 and 4 is `fiber_1_domain_13_minus` in panel 4. Its 1.056213 is
below the 1.056690 of the frontier run named in the panel-3 report. Its HF checkpoint is
`gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_frontier_phase_fiber_20260719/fiber_131_fiber_1_domain_13_minus-05f1fe/hf/step-3006`
(data seed 7192011, trainer seed 0). Its schedule moves `dolma3_cc/food_and_dining_low` toward phase 0 and spreads the
difference thinly over the other buckets in phase 1 (phase TV 0.011). Per-panel minima are `fit_125_run_00125`
(1.115476) in panel 1 and `agphase_a1_recipe_synthetic_all_plus_tv010` (1.057097) in panel 3.

Candidates for arm B. All four have the `table9_frontier` aggregate (to within TV 0.0002), and Δ is the run minus its
same-seed tied control.

| Run | Source panel (runs) | Phase TV | Table-9 | Same-seed tied control | Δ |
|---|---|---|---|---|---|
| `rphase_216_random_phase_a1_d21_r50` | random-phase population (296; not in the CSV) | 0.032 | 1.053311 | `rphase_149_random_phase_a1_center_s1`, 1.058290 | −0.0050 |
| `fiber_1_domain_13_minus` | p4 (200) | 0.011 | 1.056213 | `fiber_1_center_s1`, 1.060305 | −0.0041 |
| `dphase_t9b075_can_e0p005` | decoupled phase-information (79); p6 | 0.115 | 1.056690 | `dphase_t9b075_tied`, 1.057530 | −0.0008 |
| `agphase_a1_recipe_synthetic_all_plus_tv010` | p3 (290) | 0.100 | 1.057097 | `agphase_a1_center_center_tv000_seed01`, 1.067283 | −0.0102 |

The panel-3, panel-4 and panel-6 runs and their controls are rows of the CSV. The two random-phase runs are outside the
CSV. Listings confirm that both have `hf/step-3006` and `checkpoints/step-3006`, with trainer seed 0 and data seed
7212001. Their HF exports are:

- `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_frontier_random_phase_population_20260720/rphase_216_random_phase_a1_d21_r50-09005b/hf/step-3006`
- `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_frontier_random_phase_population_20260720/rphase_149_random_phase_a1_center_s1-8cfa07/hf/step-3006`

## Caveats for the plan

- Each candidate is the lowest two-phase Table-9 run of its panel and has one seed. Three of the four lie inside the
  1.0553-1.0717 range of the 25 tied replicates of their shared aggregate. `rphase_216` lies 0.0020 below the lowest
  replicate. The lowest Table-9 value in panels 1-4 belongs to a tied control (`fiber_1_center_s2`, 1.055328).
- `fiber_1_domain_13_minus` and `rphase_216` are small perturbations of the tied mixture (phase TV 0.011 and 0.032).
  `dphase_t9b075_can_e0p005` and the panel-3 synthetic recipe move code, math and synthetic data late (phase TV about 0.1).
- The tied `table9_frontier` mixture averages 1.0641 Table-9 over 25 data seeds. That is 0.0037 below
  `lwspu_t9_snc_cap08`'s 1.0678 over three trainer seeds at one data seed. The two means vary different seeds, so they do
  not rank the mixtures, but arm C need not score worse than arm A on base-model Table-9.
- No inventoried two-phase run has MARINER's mixture as its phase average. No B or C candidate uses MARINER's data seed
  662009. B', and B or C runs paired with A by (data seed, trainer seed), need new training. A ran on v6e-8, and every
  panel run ran on v5p-8.
- No checkpoint exists at the phase boundary (step 2400). SFT from intermediate points would need new training.
