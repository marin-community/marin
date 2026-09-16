# Adaptive refinement of the StarCoder pilot

The user authorized ten additional proxy runs after reviewing the completed five-point pilot. Both arms add StarCoder fractions 0.2, 0.4, 0.5, 0.6, and 0.9. The matched curve's sparse minimum and the unmatched curve's upper end motivate these coordinates. No target run is needed: the archived target already covers all five points.

The release adds 2.4411e18 estimated training FLOPs. Each new proxy uses 277,872,640 tokens and the reference trainer/data seed 20260711. Model, optimizer, schedule, parent membership, nested subset and all training identities are unchanged. The cumulative selection contains twenty artifacts: ten completed pilot runs and ten new proxies. Durable success and configuration fingerprints control reuse.

[release.json](release.json) records the exact ten identities, the original design checksum, authorization scope and hashes of the completed pilot's plan, measurements and report. The frozen design remains `3191f3d005ebc3e1c653f1de664bfb0c3a81c291665a7a06b872903a0b4f9e0b`.

The common expanded grid is p=0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, and 1. Select each proxy's observed minimum and report target loss, target regret, selected-weight displacement, matched-minus-unmatched regret, and raw and excess-over-own-minimum curves. Exact ties use the existing lower-fraction rule. Shared p=0 is counted once, and the corrected target endpoint replaces the historical p=1 observation.

The [original pilot result](../pilot_results/RESULTS.md) stays intact: unmatched selected p=1 and matched selected p=0.3, with target regrets 0.031826 and 0.033350 BPB. Matching induced turnover but did not improve selection on that grid. The expanded grid was chosen after seeing those results, so it is an adaptive follow-up. It does not independently confirm a selection benefit. All valid outcomes are retained, and the short horizon, fixed subset and historical/new-runtime limitations remain.

Only these ten new proxies are released. The full-grid and replicated stages remain unsubmitted. Submission evidence is recorded below; do not launch another parent while this release is active.

## Collection after completion

Use the actual persisted submission plan, rather than a preparation plan. Save the expanded-grid outputs in this directory so the pilot artifacts remain unchanged.

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_epoch_matching \
  --stage refinement --collect-results .agents/projects/starcoder_epoch_matching/refinement/measurements.csv \
  --plan-path .agents/projects/starcoder_epoch_matching/refinement/collection_plan.json

uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.analyze_starcoder_epoch_matching \
  --stage refinement --plan .agents/projects/starcoder_epoch_matching/refinement/submission_plan.json \
  --measurements .agents/projects/starcoder_epoch_matching/refinement/measurements.csv \
  --output-dir .agents/projects/starcoder_epoch_matching/refinement/results \
  --plot-png .agents/projects/starcoder_epoch_matching/refinement/results/curves.png \
  --plot-pdf .agents/projects/starcoder_epoch_matching/refinement/results/curves.pdf \
  --plot-excess-png .agents/projects/starcoder_epoch_matching/refinement/results/excess_curves.png \
  --plot-excess-pdf .agents/projects/starcoder_epoch_matching/refinement/results/excess_curves.pdf
```

Visually inspect both rendered plots before sharing them. Compare the expanded-grid result with the original pilot separately.

## Submission

Iris acknowledged [starcoder-epoch-matching-refinement](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fstarcoder-epoch-matching-refinement) on 8 September at 21:38:15 PDT. The [exact command](planned_submission.txt) passed central1 region validation. Fieldbook parent: `job_01m227arf2tf0yakj4pj4157cw`; experiment: `exp_01m21p4aw15bpvhnswz8gtwn0d`. The [submission acknowledgment](submission_ack.log) and [live reuse check](pending_audit.json) record acceptance and exactly ten pending training runs.

The 57 launcher and analysis tests passed. Scoped repository lint, live runtime/index checks and exact-command regional validation passed. The 24.2 MiB submitted workspace preserves the frozen design, dependency lock and reuse receipt. At 21:40:01 PDT, Iris showed all ten training children acknowledged: one running and nine queued, with zero failures or preemptions. The parent uses 2 CPUs, 8 GiB memory and 32 GiB disk; children request v5p-8 TPUs in central1. The actual persisted launch plan is byte-identical to the audited local plan (SHA-256 `0b605cb0e0ff772d18798a7980d977dcd116d2385168b339bc79e39d407f2008`). The stage contains twenty logical artifacts but trains only the ten incomplete proxies. Snapshot: `refinement/initial_snapshot.json`. No active job should be resubmitted.
