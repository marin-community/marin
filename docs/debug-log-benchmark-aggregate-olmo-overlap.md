# Debugging log for 60M benchmark aggregate OLMoBase easy-overlap ingestion

## Initial status

The 60M benchmark aggregate fitter reported 0/242 OLMoBase easy-overlap rows even though we previously ran the easy-overlap suite for the qsplit240 60M swarm.

## Hypothesis 1: local metric registry was stale

The local `metric_registry/metrics_wide.csv` was dated Apr 17 and did not contain `lm_eval/olmo_base_easy_overlap/*` columns. Code search showed `build_metric_registry.py` has explicit GCS collectors for `gcs_qsplit240_olmo_base_easy_overlap`, so absence from the local wide CSV was not evidence that the evals were missing.

## Results

`gsutil ls gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv` found the expected 8 shard result CSVs. A direct read of one shard showed the expected OLMoBase easy-overlap task metrics and `lm_eval/olmo_base_easy_overlap/macro_bpb`.

The qsplit240 rerun has 240 rows. The 60M fit swarm has 242 rows because it also includes selected baseline rows. The selected-baseline OLMoBase rerun exists separately at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_selected_baselines_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv` and contains 3 rows; the fitter now reads both sources and validates that every fit-swarm run name is covered.

One fit-swarm row, `baseline_olmix_loglinear`, is not covered by those reruns. The selected-baseline rerun contains `baseline_olmix_loglinear_uncheatable_bpb`, but its phase weights differ substantially from `baseline_olmix_loglinear`, so it is not a safe metric substitute. The fitter therefore treats OLMoBase objectives as objective-specific panels with 241 rows instead of incorrectly imputing the missing baseline.

Launched a one-off OLMoBase easy-overlap rerun for `baseline_olmix_loglinear` under `ngd3dm2_olmix_bpb_olmo_base_easy_overlap_rerun_*`. The result landed successfully, and the OLMoBase panels are now 242/242 rows.

## Accuracy aggregate fitting result

The OLMoBase easy-overlap mean-accuracy aggregate is fit on all 242 rows. I compared raw probability, logit/sigmoid, arcsin-sqrt, probit, rank-normal, per-task z-scored mean, per-task mean-logit, per-task mean-probit, and per-task rank-normal aggregates.

The best predictive variants are the simple probability-like targets:

- raw mean accuracy: OOF RMSE 0.003002, Spearman 0.640984
- arcsin-sqrt mean accuracy: OOF RMSE 0.003002, Spearman 0.640859
- probit mean accuracy: OOF RMSE 0.003001, Spearman 0.640320
- logit/sigmoid mean accuracy: OOF RMSE 0.003009, Spearman 0.637347

The per-task transformed aggregates did not improve rank quality enough to justify replacing the raw mean-accuracy objective. The model correctly identifies `baseline_olmix_loglinear` as the best observed row, but the unrestricted raw optimum remains over-optimistic and structurally suspect: it predicts roughly 0.397-0.399 mean accuracy versus the best observed 0.380723, and the optimized phase-1 mixture is nearly all broad-text.

Conclusion: aggregation makes benchmark accuracy substantially more modelable than MMLU alone, but GRP no-L2 should not yet be used as an unconstrained benchmark-accuracy optimizer. Treat the observed-row ranking as useful and use constrained/trust-region optimization for deployment candidates.

## Family partition sweep

I added a `--family-scheme` option to the aggregate fitter and tested alternative three-way GRP family partitions. This keeps the GRP model structure fixed and changes only how domains are assigned to `broad_text`, `tech_code`, and `reasoning`.

Triage sweep candidates:

- `default`: original GRP partition, with `synth_qa` treated as `broad_text`.
- `qa_reasoning`: move `synth_qa` from `broad_text` to `reasoning`.
- `qa_tech`: move `synth_qa` from `broad_text` to `tech_code`.
- `academic_reasoning`: move academic/math/QA synthetic domains into `reasoning`.
- `academic_reasoning_stack_tech`: same, but keep StackEdu in `tech_code`.
- `synthetic_reasoning`: move all `dolmino_synth_*` domains into `reasoning`.
- `synthetic_tech`: move all `dolmino_synth_*` and StackEdu domains into `tech_code`.
- `cc_high_vs_rest`: use CC-high as broad, CC-low/web as reasoning, everything else as tech.

The stronger rerun kept the four most relevant candidates:

| family_scheme | OOF RMSE | Spearman | raw predicted optimum | nearest observed regret | nearest observed TV | phase-1 broad | phase-1 tech | phase-1 reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.003013 | 0.6360 | 0.3951 | 0.0000 | 0.6015 | 0.9997 | 0.0000 | 0.0003 |
| qa_reasoning | 0.002975 | 0.6535 | 0.4021 | 0.0096 | 0.8127 | 0.9985 | 0.0000 | 0.0015 |
| qa_tech | 0.002988 | 0.6396 | 0.3990 | 0.0000 | 0.6828 | 0.6001 | 0.3998 | 0.0001 |
| synthetic_tech | 0.002965 | 0.6523 | 0.4522 | 0.0167 | 0.8633 | 0.9768 | 0.0232 | 0.0000 |

Interpretation: `qa_reasoning` and `synthetic_tech` improve predictive ranking, but both make the raw optimum less trustworthy. `qa_tech` is the best tradeoff because it preserves best-observed selection and reduces the family-level phase-1 collapse. It still does not make raw simplex optimization safe.

## Hypothesis 2: full metric-registry rebuild is too broad for this sprint

A full registry rebuild with GCS collectors touched broader import paths and local environment issues, then ran slowly. For this sprint, the right fix is targeted ingestion: read the 8 easy-overlap shard CSVs directly inside the aggregate fitter, cache the combined 242-row table, and merge on `run_name`.

## Changes

Updated `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_no_l2_benchmark_aggregates.py` to:

- read the 8 OLMoBase easy-overlap shard `results.csv` files directly from GCS,
- read the 3 selected-baseline OLMoBase easy-overlap shard `results.csv` files directly from GCS,
- read optional one-off Olmix BPB OLMoBase easy-overlap result CSVs directly from GCS,
- validate run names are unique and keep uncovered fit-swarm rows as missing for objective-specific filtering,
- cache the merged table as `benchmark_aggregate_60m/grp_no_l2_fits/olmo_base_easy_overlap_results.csv`,
- merge same-kind OLMoBase metrics into the 60M fit-swarm frame before target construction.

## Future Work

- [ ] Make `build_metric_registry.py` cheaper and less import-heavy for GCS collector refreshes.
- [ ] After GSM8K/HumanEval finishes, rerun the aggregate fitter with the downstream results CSV and compare OLMo-only vs all-accuracy aggregate fits.
