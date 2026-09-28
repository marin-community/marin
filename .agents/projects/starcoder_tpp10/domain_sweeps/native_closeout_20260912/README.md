# Native survey closeout

The four resumed FineMath target runs completed the original 28-point survey. The old repair plan still names their canceled source jobs, so its unmodified collector would omit their successful continuations. `closeout_tpp10_native_results.py` validates the immutable repair and completion plans and resolves exactly those four job identities to `/calvinxu/tpp10-finemath-target-completion`.

The wrapper requires both source trees to be wholly terminal, the completion parent to have succeeded, and the continuation receipts to preserve the original fingerprints and full-state restore requirement. It reuses the three verified checkpoint audits and exact evaluation population counts. The existing repair implementation checks final permanent checkpoints, runtime/domain receipts, all saved metrics, exact W&B agreement, expired leases and native artifact fingerprints before finalizing completion records. W&B access is read-only.

The operation launches no training or checkpoint evaluation. It runs once on one on-demand CPU and 4 GiB in us-central1-a. `submit.sh` is the exact command; `bundle_preflight.json` records the 23.3 MiB bundle and 18 verified source/input pins. Region safety, Ruff, Black and focused Pyrefly checks pass. Fieldbook job: `job_01m2bqgr7c7k35gh44b4r1eeda`; Iris job: `/calvinxu/tpp10-native-record-closeout`, submitted 12 September at 21:14 UTC.

The original repair plan, numerical audit receipts, evaluation populations, metric definition and scientific run identities remain unchanged. Operational source-resolution and publication receipts are written under `record_closeouts/` in the original repair namespace. The latest corrected table is published only after all 28 records verify, alongside an immutable content-addressed snapshot. Existing immutable snapshots are retained.

The final native plots use the existing measured-point plotter in a new output directory. No manuscript edits or changes to the separately submitted MATH-500/GSM8K evaluator are part of this closeout.

## Verified result

The coordinator succeeded in 2 minutes 6 seconds. All 28 native completion records verify. The corrected publication has 28/28 rows, `complete=true`, no omitted points and no pending artifact recoveries. Its immutable result hash is `2c3b8286420202d3a30a34c920780473ba8eece3e48f33b5464de4e493de330f`; exact GCS paths are in `validation.json` and `publication.json`.

The 24 previously published requests, corrected metrics and legacy metrics are exactly unchanged. Population counts and both shared controls are exactly unchanged. The four additions therefore extend the same native evaluation population and ratio-of-totals BPB definition. All three saved-checkpoint numerical audits were reused and revalidated.

| Domain | Scale | Observed Uncheatable minimum | Materialized epochs | BPB |
| --- | --- | ---: | ---: | ---: |
| Wikipedia | Matched proxy | 30% | 4.76836 | 1.530924 |
| Wikipedia | Target | 30% | 4.76846 | 1.113183 |
| FineMath-3+ | Matched proxy | 30% | 4.76836 | 1.493869 |
| FineMath-3+ | Target | 30% | 4.76846 | 1.084207 |

FineMath's completed target curve has its lowest observed Uncheatable loss at 30%; 50% is higher by 0.001422 BPB. All four curves have the same observed optimum on this common objective. This remains distinct from the MATH-500/GSM8K comparison, whose four final target evaluations are owned by the parent task. One trainer seed and a coarse grid do not resolve continuous optima or statistical uncertainty.

The overview and eight-component comparison are in `plots/completed_sweeps.pdf` and `plots/component_tradeoffs.pdf`, with PNG previews, exact `completed_points.csv`, per-component `curves.json` and numerical `summary.json`. Both rendered figures were visually checked. The plotter's obsolete sentence that FineMath targets stop at the last completed point was removed; no measured values or axes were altered.

Reproduce the immutable local snapshot:

```bash
uv run python -m experiments.domain_phase_mix.plot_tpp10_domain_sweeps \
  --repair-plan .agents/projects/starcoder_tpp10/domain_sweeps/repairs_20260911/plan.json \
  --output .agents/projects/starcoder_tpp10/domain_sweeps/native_closeout_20260912/plots
```
