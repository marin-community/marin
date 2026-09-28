# CC review: finite-pool epoch-matching experiment

Pilot complete: all ten runs succeeded. On the prespecified five-point grid, epoch matching did not improve target selection: target regret was 0.033350 BPB for the matched choice (p=0.3) versus 0.031826 for the unmatched choice (p=1), with the target minimum at p=0.7. This is descriptive, without a significance claim. See [pilot results and next-stage assessment](pilot_results/RESULTS.md). The full-grid and replication stages remain unsubmitted. Ten additional proxy runs have now been submitted as an adaptive refinement (one running and nine queued at 21:40:01 PDT); see [the release and handoff](refinement/README.md).

The ten-run pilot was submitted after CC review and the user’s authorization. Iris acknowledged `/calvinxu/starcoder-epoch-matching-pilot` on 8 September at 17:29:55 PDT. Its CPU parent passed the corpus audit in us-central1/us-central1-a and submitted all ten training children. At 17:32:18 PDT one child had reached Qwen3/W&B initialization and nine were queued for capacity, with no reported failures. Full-grid and replication stages remain unsubmitted. Historical metadata, source length and reconstructed index mappings passed verification.

Read [design.md](design.md) for the scientific protocol, [spec.md](spec.md) for interfaces, [target_reuse_audit.md](target_reuse_audit.md) for source evidence, and [research.md](research.md) for prior work and limitations.

## Decision and budget

Reuse 25 observations from C40: the fixed Qwen3 model trained for 7.408B tokens with a 279.970M-token StarCoder parent. Replace the old p=1 observation: removing zero-weight web components changed its shuffle key and therefore its corpus. The new named-component key override preserves the historical interior parent at every mixture fraction.

Both proxy arms train for 277.873M tokens. The unmatched arm sees the full parent; the matched arm sees its first 10.486M tokens. Maximum nominal epochs are 0.9925 and 26.5000, respectively, compared with 26.4607 for the target. The local allocator replay puts the largest relative epoch discrepancy at 0.174% after mixture-block rounding.

| Cumulative stage | New jobs | Estimated training FLOPs |
|---|---:|---:|
| Five-point pilot | 10: nine proxies and corrected target endpoint | 8.705e18 |
| Full 26-point grid | 52 | 1.896e19 |
| Three proxy trainer seeds, full grid | 154 | 4.386e19 |

The pilot grid is p=0, 0.10, 0.30, 0.70, 1.00. The shared p=0 proxy is trained once per seed. Later stages reuse completed artifacts and require the previous stage to finish. Promotion requires technical completion and a separate release decision; all valid pilot outcomes are retained, including ties or an unfavorable ordering. No outcome threshold is imposed on this irregular five-point grid. A one-seed pilot cannot estimate a repeat-noise threshold.

The primary comparison is target loss at each proxy-selected mixture and regret on the common grid. Also report the absolute displacement from the target grid minimum and matched-minus-unmatched target regret. An optional figure subtracts each curve’s own observed minimum while retaining the raw BPB values. Absolute proxy and target losses need not align. The target curve was chosen using existing outcomes, and target regret is conditional on one historical seed and parent. Trainer repeats do not test subset-selection uncertainty. Independently redrawing a nested subset within the same parent is a useful follow-up, requiring its own identity and protocol; this pilot keeps its reviewed subset fixed. In particular, the proxy has only 6.06 tokens per nonembedding parameter (1.32 per total parameter), and its approximately 10M-token subset may be unrepresentative; the pilot tests that concern.

## Code changed

- `experiments/domain_phase_mix/starcoder_epoch_matching.py`: frozen design builder, target eligibility, stages and accounting.
- `experiments/domain_phase_mix/starcoder_epoch_matching_design_20260908.json`: 154 new run identities and 26 archived observations, with historical p=1 explicitly excluded.
- `experiments/domain_phase_mix/launch_starcoder_epoch_matching.py`: dry-run planning, runtime and reuse gates, resumable submission, and final-step metric collection.
- `experiments/domain_phase_mix/audit_starcoder_epoch_matching_indices.py`: reproducible source-index audit using the actual packed corpus length without reading token payloads.
- `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_starcoder_epoch_matching.py`: plan-verified measurements, grid selection, target regret, outputs and plotting.
- `lib/levanter/src/levanter/data/text/datasets.py`: optional named training-component shuffle keys, retaining historical behavior when unset.
- `tests/test_starcoder_epoch_matching_launcher.py`, `tests/test_starcoder_epoch_matching_analysis.py`, and additions in `lib/levanter/tests/test_text.py`: behavioral coverage.

The experiment manifest SHA-256 is `3191f3d005ebc3e1c653f1de664bfb0c3a81c291665a7a06b872903a0b4f9e0b`. The [review file inventory](review_files.json) records code hashes, and [code_changes.patch](code_changes.patch) collects this experiment's code changes. No existing paper figure or result was replaced. The paper's outline and this round's change note record the proposal for coordination.

## Verification completed

- 100 targeted tests passed across the launcher, analyzer and full Levanter text-data test file (26, 27 and 47 respectively), in 18.83 seconds under JAX 0.11.1.
- The earlier 93-test version also passed under historical JAX/JAXlib 0.10.1 and NumPy 2.3.5; those earlier results are archived separately.
- The repository pre-commit entry point passed on all eight changed Python files; Pyrefly also passed.
- Frozen-manifest regeneration is identical. Default launcher planning performs no remote reads or submissions; the three stage plans have been materialized locally.
- Local allocation replay confirms the unmatched arm uses at most 135,680 StarCoder sequences from its 136,704-sequence parent.
- A synthetic plot was rendered and visually checked for overlap and cropping. It is a layout test, not an experiment result; actual curves still require visual review.
- Independent reviews caught and resolved the missing plan-fingerprint check, manifest-metric drift, and missing historical-reuse gate.

The broad affected-test runner's dry-run selected seven packages from 247 worktree changes, most unrelated to this experiment. The actual test run was scoped to the changed behavior above. Live cache, runtime configuration and full index-hash checks passed under JAX 0.11.1 before submission. Accelerator training has not yet been verified.

## Historical verification and runtime

The downloaded representative interior run is:

```text
gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_surfaces_20260808/dss_r3d28260_m100_c109_s0711/2026.07.11
```

Its `.artifact.json`, `.executor_info`, final evaluation records, tracker configuration and W&B environment records were retrieved. [download_manifest.json](historical_metadata/download_manifest.json) records their hashes and GCS generations. The materialized configuration and worker-recorded configuration agree after normal runtime resolution. Canonical record sorting occurs after execution; the historical execution path passes the ordered configuration through cloudpickle.

[config_comparison.json](historical_metadata/config_comparison.json) confirms the complete model and optimizer, tied weights, seven training cache paths, steps 28,260, batch 128, sequence length 2,048, seed 20260711, legacy block shuffle `(256, 512, feistel)`, packing/masking, and disabled global epoch slicing. The durable native metric `eval/paloma/dolma_100_programing_languages-llama3/bpb` is exactly 0.7880429029464722 at step 28,259, matching the archived p=0.7 observation.

[cache_audit.json](historical_metadata/cache_audit.json) verifies 216,567,300,822 tokens from two scalar offset metadata reads: 105,745,752 full sequences and 726 trailing tokens. No token payload was transferred. The index audit checked all 136,704 parent positions and all 5,120 matched-subset positions under JAX 0.10.1. The explicit parent matches the legacy interior mapping, and the matched subset is its exact prefix. The old endpoint differs at every position. [index_audit.json](historical_metadata/indices_training_runtime/index_audit.json) records the array hashes and runtime. The same hashes were obtained under JAX 0.11.1, the version in the repository lock used for new TPU jobs.

The historical child requirements confirm JAX/JAXlib 0.10.1 and NumPy 2.3.5. A cached uv overlay reproduced this historical runtime. New runs use JAX/JAXlib 0.11.1 and NumPy 2.3.5 from the bundled `uv.lock`. Submission recomputes every parent and matched-subset source index and compares their hashes against the reviewed receipt. The plan records the historical runtime, observed parent runtime, intended child dependencies and lock hash separately. Matching source indices establishes corpus identity across these versions; it does not establish identical numerical training behavior. Threefry keys and x64 disabled remain required. The historical records omit an exact launch Git SHA, so the audit establishes configuration and loader equivalence rather than a bitwise reproduction of the original training binary.

The archived August 8 shuffler was also executed under the historical runtime: all 136,704 positions match the current implementation and stored parent array. [verified_historical_config.md](verified_historical_config.md) documents the source comparison, transport ordering and remaining provenance limitation. A [passed reuse receipt](reuse_audit.json) now records the reviewed configuration and mapping evidence; the pilot was then authorized and submitted after CC's review.

## Commands

Run from `/Users/calvinxu/Projects/Work/Marin/marin`.

Generate a local plan, with no submission:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_epoch_matching \
  --stage pilot --plan-path .agents/projects/starcoder_epoch_matching/pilot_plan.json
```

Repeat the read-only runtime and source-index audit in the repository environment:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_epoch_matching \
  --stage pilot --audit-runtime \
  --reuse-audit .agents/projects/starcoder_epoch_matching/reuse_audit.json \
  --plan-path .agents/projects/starcoder_epoch_matching/pilot_plan.json
```

Reproduce the index audit with the verified cache length:

```bash
uv run python -m experiments.domain_phase_mix.audit_starcoder_epoch_matching_indices \
  --packed-sequences 105745752 \
  --output-dir .agents/projects/starcoder_epoch_matching/historical_metadata/indices_current_runtime_v2
```

[planned_pilot_submission.txt](planned_pilot_submission.txt) contains the submitted central1 parent command, with credential references rather than values. The exact command passed the region-safety validator. Its 24.2 MiB workspace bundle preserves all Python source, the dependency lock, frozen design and reuse receipt; local outputs and unrelated non-Python artifacts are excluded. The CPU extra selects CPU dependencies for the parent; TPU children resolve their own TPU extra. Before scheduling, the launcher saves the actual plan and receipt at the `submission_plan_uri` in its plan. Identical resumes reuse it; changed contents are rejected. Use this durable plan for analysis; preparation plans do not substitute for a changed configuration.

Collect only successful, fingerprint-matching final observations:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_epoch_matching \
  --stage pilot \
  --plan-path .agents/projects/starcoder_epoch_matching/pilot_collection_plan.json \
  --collect-results .agents/projects/starcoder_epoch_matching/pilot_measurements.csv
```

Analyze with the saved submission plan:

```bash
gcloud storage cp \
  gs://marin-us-central1/experiments/starcoder_epoch_matching_20260908/3191f3d005ebc3e1c653f1de664bfb0c3a81c291665a7a06b872903a0b4f9e0b/pilot/launch_plan.json \
  .agents/projects/starcoder_epoch_matching/pilot_submission_plan.json

uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.analyze_starcoder_epoch_matching \
  --stage pilot \
  --plan .agents/projects/starcoder_epoch_matching/pilot_submission_plan.json \
  --measurements .agents/projects/starcoder_epoch_matching/pilot_measurements.csv \
  --output-dir .agents/projects/starcoder_epoch_matching/pilot_results \
  --plot-pdf .agents/projects/starcoder_epoch_matching/pilot_results/curves.pdf \
  --plot-png .agents/projects/starcoder_epoch_matching/pilot_results/curves.png \
  --plot-excess-pdf .agents/projects/starcoder_epoch_matching/pilot_results/excess_curves.pdf \
  --plot-excess-png .agents/projects/starcoder_epoch_matching/pilot_results/excess_curves.png
```

Use `primary` or `replicated` and distinct stage filenames for subsequent releases. Any change to the reviewed protocol requires a new design version and execution identity.

## Suggested review focus

Please check the historical-input receipt, the named-key loader change, the unusually short proxy horizon, the fixed-subset conditioning, and the common-grid regret calculation. The user has authorized the pilot. Full-grid and replication releases remain separate decisions. The proposed figure is evidence of horizon/repetition mismatch in a controlled two-bucket setting; it is not independent evidence that the broader compute-allocation rule is optimal.

Fieldbook experiment: `exp_01m21p4aw15bpvhnswz8gtwn0d` (StarCoder finite-pool epoch-matching demonstration). Parent: [Iris dashboard](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fstarcoder-epoch-matching-pilot). Fieldbook parent job: `job_01m21s421dw5vb3p9c2d67q9ts`. Ten datapoint records are linked to the parent. See [submission acknowledgment](pilot_submission_ack.log) and [initial status](pilot_initial_status.txt).

Startup verified at 17:32:18 PDT: all ten training children were acknowledged; one was running and had reached Qwen3/W&B initialization, while nine were queued for central1 resource capacity. No failures or preemptions were reported. The persisted GCS plan exactly matches the local audited plan. Child IDs and timestamps are in pilot_initial_child_snapshot.json. The nine queued logical-run mappings remain unassigned rather than inferred from asynchronous submission order.
