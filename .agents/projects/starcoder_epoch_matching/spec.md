# Experiment and artifact contracts

## Frozen design

`experiments/domain_phase_mix/starcoder_epoch_matching.py` defines:

```python
def build_design() -> ExperimentDesign: ...
def load_design(path: Path = DESIGN_PATH) -> ExperimentDesign: ...
def select_runs(design: ExperimentDesign, stage: str) -> tuple[RunSpec, ...]: ...
```

`build_design` reconciles the 26 C40 observations with the historical run manifest, requires tied policies, the reference seed and target horizon, and checks that full-pool aliases never exhaust the intended parent. It rejects changed source checksums or missing coordinates. `load_design` validates the saved canonical SHA-256 without contacting remote storage. `select_runs` returns cumulative stages `pilot`, `refinement`, `primary`, or `replicated`; unknown stages raise `ValueError`. The adaptive `refinement` release selects the completed pilot plus both reference-seed proxy arms at p=0.2, 0.4, 0.5, 0.6, and 0.9. It uses existing frozen run identities; the original stage sets and design checksum are unchanged.

The JSON file is `experiments/domain_phase_mix/starcoder_epoch_matching_design_20260908.json`. Its fields are `design_version`, `primary_metric`, `design_sha256`, `runs`, `component_order`, `component_shuffle_keys`, `training_environment`, `source_sha256`, and `target_observations`. The checksum covers every field except itself. Generation is idempotent for identical input and refuses to overwrite an existing different manifest. A revised reviewed protocol must carry a new version and execution identity.

Each `RunSpec` contains `run_name`, `arm`, `starcoder_weight`, `trainer_seed`, `data_seed`, `total_steps`, `boundary_step`, `materialized_tokens`, `starcoder_support_batches`, `first_stage`, and `coordinate_id`. Arm is `unmatched`, `matched`, or `target`. Only the corrected p=1 target endpoint is newly trained. The matched p=0 policy aliases the unmatched policy at the same trainer seed and must not be counted as a second independent run.

Each `TargetObservation` contains `starcoder_weight`, `observed_bpb`, `source_run_name`, `source_observation_id`, `reusable`, and `reason`. The historical p=1 value has `reusable=false` and is never a substitute for the corrected target endpoint. The 25 eligible measurements come from the frozen source file hashes, including two no-wrap full-pool aliases.

## Fixed source identity

`LmDataConfig` (also exported as `LMMixtureDatasetConfig`) adds:

```python
train_component_shuffle_keys: dict[str, tuple[int, int]] | None = None
```

The tuple contains the raw uint32 JAX shuffle key for that named training component. Overrides apply to the existing run-order shuffle, including the legacy shuffle before the cap. They do not change mixture sampling, validation data, or explicitly seeded subset selection. Unspecified components retain their historical sequential keys. Unknown components, disabled shuffling, and malformed keys are rejected.

All seven keys are pinned in the manifest. The StarCoder key is `(898005854, 446240491)`, the historical interior key. The new experiment leaves `experiment_budget`, `target_budget`, `simulated_epoch_subset_seed`, and `max_train_batches_subset_seed` unset; `max_train_batches` implements the finite parent or its nested matched subset. This avoids applying a second independent permutation or global slicing of the web sources.

## Execution

`experiments/domain_phase_mix/launch_starcoder_epoch_matching.py` builds the requested stage as normal Marin training artifacts. Default execution produces a local plan. Remote cache reads and materialized-runtime validation require audit mode or submission; actual training requires explicit submission mode. Runtime configuration and source fingerprints are part of artifact identity. A successful artifact can be reused only when its durable fingerprint matches the expected configuration; an existing path alone is insufficient.

The cache path, component order, model geometry, optimizer, data keys, caps, token accounting, phase alignment, and runtime PRNG environment are checked before submission. Both `refinement` and `primary` require the pilot to have completed; `replicated` requires the primary stage. The adaptive refinement does not change the original promotion dependencies. Jobs and all source/output paths remain in us-central1/us-central1-a under `gs://marin-us-central1`. The parent command carries explicit region and zone as well as child placement. Submission concurrency is the number of remaining selected jobs, not a guess about available TPU capacity.

Submission also requires `--reuse-audit PATH`, a reviewed JSON receipt establishing equivalence to the historical target configuration and packed-sequence mapping. It must contain the following fields; the example deliberately does not pass validation:

```json
{
  "design_sha256": "3191f3d005ebc3e1c653f1de664bfb0c3a81c291665a7a06b872903a0b4f9e0b",
  "status": "unverified",
  "historical_config_uri": null,
  "historical_config_sha256": null,
  "packed_starcoder_sequence_count": null,
  "parent_indices_sha256": null,
  "matched_indices_sha256": null,
  "legacy_parent_match": false,
  "nested_subset_match": false
}
```

The passed receipt needs a central1 historical-config URI, three lowercase 64-character SHA-256 digests, the actual packed source length (at least 136,704 sequences), `status="passed"`, and both checks true. Hash parent and matched source-index arrays as contiguous little-endian signed 64-bit integers in draw order; retain those arrays and the config used in the audit. The matched array contains 5,120 indices and must equal the prefix of the 136,704-element parent array. The receipt is a reviewed evidence record, not an automated substitute for those checks. Its contents appear in the submitted plan. The current-cache audit does not produce a passed historical-reuse receipt. At submission time, the launcher recomputes every source index under the observed runtime and requires both array hashes to match this receipt. The intended new training environment is JAX/JAXlib 0.11.1 and NumPy 2.3.5, checked against the bundled lock file and recorded separately from the historical 0.10.1 runtime. Array identity is not a claim of numerical training equivalence.

Before scheduling, submission persists its complete plan and reuse receipt at `gs://marin-us-central1/experiments/starcoder_epoch_matching_20260908/<design_sha256>/<stage>/launch_plan.json`, also exposed as `submission_plan_uri`. Identical plans can be resumed; different contents at that path are rejected without overwriting. Local dry runs and runtime audits do not publish a remote plan.

## Measurement and analysis

The primary metric is the native evaluation key `eval/paloma/dolma_100_programing_languages-llama3/bpb`. Measurements CSV columns are `run_name`, `step`, `metric`, `value`, `design_sha256`, `config_fingerprint`, and `status`. Each row must be a verified successful run's endpoint at `total_steps - 1`. The analyzer also requires `--plan PATH`, the JSON emitted by the launcher for the same stage. Its design hash, metric, exact run set, and per-run settings must match the manifest; every measurement's fingerprint must equal the corresponding plan fingerprint. Results outside that stage are rejected. The report records the checked plan's SHA-256. Conflicting, stale, nonfinite, incomplete, or wrong-metric records fail validation. Collection reads the durable `checkpoints/eval_metrics.jsonl` records and binds each metric to its recorded step, rather than using an unbound tracker summary.

`experiments/domain_phase_mix/exploratory/two_phase_many/analyze_starcoder_epoch_matching.py` consumes that CSV and the manifest, computes per-arm minima on the selected stage's common grid, and evaluates target loss at the selected fractions. It reports the full exact tie set and chooses the smallest tied p. Replicated-stage selection averages all three proxy trainer seeds. Target regret uses the target minimum on the same grid. Pilot-grid regret is explicitly distinguished from adaptive-refinement and full-grid regret. The refinement uses p=0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, and 1; its analysis is a separate artifact and does not overwrite the five-point pilot report. No surrogate fit or target-informed tuning of proxy choices is used.

The analysis writes a machine-readable report and a curves CSV with provenance and p=0 alias indicators. It also reports the absolute distance from each selected fraction to the lowest-weight exact target minimum, and matched-minus-unmatched target regret. Optional excess-loss plots subtract each arm’s observed mean-curve minimum while retaining the raw plot. Optional figures show only observed measurements. Historical p=1 is always excluded; missing corrected-target data prevents a complete three-curve result. Intervals or significance claims for expected target performance are outside this experiment because target trainer and parent-subset uncertainty are not replicated.

## Scope

The package prepares the experiment, stores its review record in Fieldbook, and provides dry-run, audit, execution, collection and analysis commands. It does not submit jobs during preparation, modify current manuscript figures, test fixed epoch caps, fit a new surrogate, or claim an independent confirmation from the previously observed target curve.
