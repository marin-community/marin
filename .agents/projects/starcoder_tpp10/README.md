# StarCoder TPP10 review packet

This package implements three curves with matched total-parameter TPP, a nonrepeating unmatched proxy, and three independently sampled matched subsets. It is a new experiment; the earlier pilot and refinement remain unchanged. Preparation, calibration, the three canaries and all 57 seven-coordinate pilot artifacts are complete. The pilot parent `/calvinxu/starcoder-tpp10-pilot` finished on 10 September 2026 at 21:34 PDT. The target's grid minimum is p=0.7; unmatched proxies select p=1.0 and all matched subsets select p=0.5. Matching reduces observed target selection regret from 0.03636 to 0.00985 BPB. See the [pilot results](live/pilot_results/report.md), [curves](live/pilot_results/curves.png) and [launch state](live/launch_state.json). A focused 45-run refinement at 40%, 55%, 60%, 65%, and 80% was submitted on 11 September as `/calvinxu/starcoder-tpp10-refinement`; the remaining dense coordinates are unreleased. The paper now shows the completed pilot curves without fitted overlays.

- [Design and decision rule](design.md)
- [Implementation contract](spec.md)
- [Source investigation](research.md)
- [CC review dispositions and change list](CC_CHANGES.md)
- [Scientific review](review_science.md)
- [Implementation review](review_implementation.md)
- [Implementation closure](review_implementation_followup.md) and [scientific closure](review_science_closure.md)
- [Final validation](validation.json)

The proxy has 16.6M total parameters and trains on 165.9M tokens. The target has 301.2M parameters and trains on 3.012B tokens. Target and unmatched use the same 190.3M-token StarCoder parent; matched uses a 10.49M-token subset. Maximum nominal repetition is 0.872 epochs for unmatched, 15.825 for matched, and 15.828 for target.

The staged program is: bounded regional CPU preparation; eight fixed-mixture batch-calibration runs; three reusable p=1 canary runs; a seven-coordinate pilot; then a 21-coordinate dense grid. Pilot and dense stages contain 57 and 183 cumulative primary artifacts. Four extra calibration artifacts bring the complete program to 187. The target accounts for most of the estimated 1.44e20 dense-grid training FLOPs. These are algorithmic FLOPs, not measured chip-hours.

## Local plans and checks

Run commands from the repository root. These commands do not submit jobs or read training data from GCS.

```bash
uv run python -m experiments.domain_phase_mix.starcoder_tpp10 \
  --audit-output .agents/projects/starcoder_tpp10/allocation_audit.json
uv run python -m experiments.domain_phase_mix.prepare_starcoder_tpp10 \
  --output .agents/projects/starcoder_tpp10/data_plan.json
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10 \
  --stage calibration --plan-path .agents/projects/starcoder_tpp10/calibration_plan.json
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10 \
  --stage canary --plan-path .agents/projects/starcoder_tpp10/canary_plan.json
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10 \
  --stage pilot --plan-path .agents/projects/starcoder_tpp10/pilot_plan.json
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10 \
  --stage dense --plan-path .agents/projects/starcoder_tpp10/dense_plan.json
uv run pytest tests/test_starcoder_tpp10.py -q
```

The frozen asset manifest rejects code, tokenizer or source drift. `--write-design` accepts only a new file or an identical existing manifest; it refuses to overwrite a different design. A source change can also change training fingerprints even when it does not alter the scientific design. Preserve the reviewed checkout until this experiment finishes. `selected_source_snapshot.tar.gz` preserves the pinned files for restoration into an isolated checkout; it is not a complete repository archive. Iris also archives the actual submitted workspace. Retain that full job bundle after release.

## Regional preparation

The reviewed command is in [planned_preparation.txt](planned_preparation.txt); the [actual submission](live/submitted_preparation.txt) schedules all 12 preparation artifacts, with dependencies controlling readiness. The parent was accepted on 9 September 2026. It bounds tokenization to the prescribed source prefixes, writes resumable per-shard receipts, materializes the shuffled parent and three subsets, and audits the finished caches.

Validate the exact command with:

```bash
uv run python -m experiments.domain_phase_mix.east5_launch_safety \
  --expected-region us-central1 --expected-zone us-central1-a \
  --expected-bucket-prefix gs://marin-us-central1 \
  --command "$(cat .agents/projects/starcoder_tpp10/planned_preparation.txt)"
```

After preparation, the parent prints an immutable `cache_audit_uri` and its `cache_audit_sha256`. The audit checks successful artifact fingerprints, recipe receipts, cache metadata and exact finite lengths. Keep that URI with the Fieldbook experiment. Preparation failures preserve completed matching shards; rerunning the same command resumes them. If a bounded source cannot supply its token quota, stop and revise its reviewed recipe rather than silently changing the corpus.

## Training releases

Each stage needs its own release file at `.agents/projects/starcoder_tpp10/releases/STAGE.json`:

```json
{
  "approved": false,
  "reviewer": "",
  "stage": "calibration",
  "plan_sha256": "COPY_FROM_THE_REVIEWED_STAGE_PLAN",
  "cache_audit_sha256": "COPY_FROM_THE_COMPLETED_CACHE_AUDIT"
}
```

This template deliberately does not authorize training. Record authorization for the concrete stage and hashes before changing it. [The calibration command](planned_calibration.txt) is ready for the same regional safety validation as preparation. Replace the stage, job name, release filename and output plan together for subsequent releases. Include the release file in the job bundle; the supplied exclusion expression preserves that directory and all experiment assets.

Set `--no-preemptible` explicitly on each CPU parent. Its 2-CPU/8-GB request exceeds Iris's automatic coordinator heuristic (at most 1 CPU and 4 GiB), so omission permits placement on preemptible TPU workers. This exposes the parent to worker loss and consumes accelerator-worker memory while children queue. Preemptibility is a per-job policy and is not inherited by TPU children. Keep `--max-retries 0 --max-preemption-retries 0`; after a parent failure, confirm the old tree is terminal and reuse only verified successful artifacts before resubmitting unfinished work.

The submitted calibration command uses `await_starcoder_tpp10_calibration` with the user's recorded conditional authorization. It waits for the preparation parent to succeed, verifies every cache and the immutable audit, then records the exact audit hash in a calibration release and invokes the reviewed launcher. A failed preparation stops this coordinator; it cannot release canary, pilot, or dense stages. It waits at most 12 hours within a 48-hour parent timeout. Automatic failure and preemption retries are disabled; inspect surviving descendants before any manual recovery. Completed calibration rows and the batch screen are persisted as `calibration_results.json` beside the immutable plan in GCS.

The launcher rechecks live caches and the real allocator, verifies the release, then skips only successful artifacts with matching fingerprints. The TPU child verifies source, asset and dependency hashes, local tokenizer resolution and actual vocabulary size before training. Parent and children explicitly use the central1 prefix. A canary release requires all calibration endpoints and a passing unmatched batch-loss screen. Pilot requires all canary endpoints; dense requires all pilot endpoints. Human review of optimizer traces remains part of release. Neither a favorable matched result nor an interior minimum is required. The live preparation parent schedules all 12 cache artifacts; the calibration parent schedules all eight training artifacts. `--max-concurrent` changes scheduling without changing training identities.

Record every actual parent submission, child reconciliation, retry and measurement in Fieldbook experiment `exp_01m23ddmn78breygyvkkpzyrq8`. At submission, also record the full Iris workspace-bundle URI; confirm it is retained before the next stage. The preparation parent and eight planned calibration datapoints are recorded in Fieldbook. Do not resubmit under a new name to work around a fingerprint mismatch.

## Collection and analysis

After a stage completes, collect small endpoint records locally or on its regional parent. Collection reads the saved plan directly and does not overwrite it or compare it to today's source tree, so verified historical results remain readable after code maintenance:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10 \
  --stage pilot --plan-path .agents/projects/starcoder_tpp10/pilot_plan.json \
  --collect-results .agents/projects/starcoder_tpp10/pilot_metrics.csv
uv run python -m experiments.domain_phase_mix.analyze_starcoder_tpp10 \
  --plan .agents/projects/starcoder_tpp10/pilot_plan.json \
  --measurements .agents/projects/starcoder_tpp10/pilot_metrics.csv \
  --output .agents/projects/starcoder_tpp10/pilot_results
```

Use `calibration` for the batch screen and `dense` for the final grid. Canary collection verifies endpoints; selection analysis requires a complete pilot or dense grid. Preserve raw losses, every measured point, all three per-subset selections, adverse outcomes and the single-target-seed qualification. Plot outputs must be visually checked before use in the manuscript.

## Focused refinement

The user released five fractions after the pilot: 40%, 55%, 60%, 65%, and 80%. The [selection wrapper](../../../experiments/domain_phase_mix/launch_starcoder_tpp10_refinement.py) preserves every frozen child identity. Its [plan](live/refinement_plan.json), [release](releases/refinement.json), [submitted command](live/submitted_refinement.txt) and [preflight](live/refinement_preflight.json) specify 45 runs. The full dense stage remains unreleased.

Collect the refinement with:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10_refinement \
  --plan-path .agents/projects/starcoder_tpp10/live/refinement_plan.json \
  --collect-results .agents/projects/starcoder_tpp10/live/refinement_metrics.csv
```

The collector requires all 45 exact final endpoints. Combine them with the 57 verified pilot artifacts only after both input plans pass their own metric verification. The original analyzer requires an explicit extension to the combined twelve-coordinate grid; do not pass the focused release off as the complete dense grid. Update measured selections, target regret and the paper plot together, keeping fitted curves out of the main figure.
