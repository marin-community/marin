# Recipe: Agent-Driven Profiling (xprof/TensorBoard/Perfetto)

## Overview
Use this recipe to turn a `jax_profile` artifact into a deterministic, agent-consumable summary and a concrete optimization workflow:

1. capture a representative profile,
2. ingest to `profile_summary.v1`,
3. query hotspots and bottlenecks,
4. patch/configure,
5. re-profile and compare.

## Scope
Primary ingestion source of truth:
- xprof-exported trace JSON inside Levanter `jax_profile` artifacts:
  - `plugins/profile/<timestamp>/perfetto_trace.json.gz` (preferred)
  - `plugins/profile/<timestamp>/*.trace.json.gz` (fallback)

Supplementary analysis source:
- direct `*.xplane.pb` parsing via xprof when you need program/op-family attribution that Perfetto summaries do not expose cleanly

Important constraint:
- xprof raw conversion is currently reliable only on a Linux host with xprof installed.
- When using `xprof.convert.raw_to_tool_data`, force `use_saved_result=False`. Without that flag, matched before/after traces from the same TPU host can collapse to identical output and destroy the comparison.

## Capture Profiles
Use Levanter profiler flags so profiles are uploaded consistently as `jax_profile` artifacts:

```bash
uv run ... \
  --trainer.profiler true \
  --trainer.profiler_start_step 5 \
  --trainer.profiler_num_steps 50 \
  --trainer.profiler_perfetto_link false
```

Reference:
- `lib/levanter/docs/Performance-Guide.md`
- `docs/recipes/add_pallas_kernel.md`

For better profile readability, use `haliax.jax_utils.named_call` and `jax.named_scope` liberally in model code.
These names flow into trace annotations and make region-level summaries far more actionable.

## Ingest to Structured Summary
Pick a download location for pulled profile artifacts:
- ephemeral/local machine: `/tmp`
- in-repo working area: `scratch/`

Examples:

```bash
# /tmp (ephemeral)
uv run python lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/marin/<run_id> \
  --download-root /tmp/marin-profiles \
  --breakdown-mode exclusive_global \
  --output /tmp/profile_summary.json

# in-repo scratch (kept with your workspace)
mkdir -p scratch/profiles
uv run python lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/marin/<run_id> \
  --download-root scratch/profiles \
  --breakdown-mode exclusive_global \
  --output scratch/profile_summary.json
```

### Option A: From a W&B artifact reference

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --artifact marin-community/marin/run-grug-125m-profile-apples-pallas_tpu-20260217-225239-055ab2-profiler:v0 \
  --download-root /tmp/marin-profiles \
  --output /tmp/profile_summary.json
```

### Option B: From a W&B run target (auto-pick latest profile artifact)

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/marin/grug-125m-profile-apples-pallas_tpu-20260217-225239-055ab2 \
  --alias latest \
  --download-root /tmp/marin-profiles \
  --output /tmp/profile_summary.json
```

`--run-target` accepts:
- bare run id (requires `--entity` and `--project`),
- `entity/project/run_id`,
- full W&B run URL.

### Option C: From a local artifact directory

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --profile-dir /path/to/jax_profile_artifact_dir \
  --output /tmp/profile_summary.json
```

### Option D: From a specific trace file

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --trace-file /path/to/perfetto_trace.json.gz \
  --output /tmp/profile_summary.json
```

Summary version tag:
- `profile_summary.v1`

Generate a deterministic markdown root-cause report:

```bash
uv run python lib/marin/tools/profile_summary.py report \
  --summary /tmp/profile_summary.json \
  --output /tmp/profile_report.md
```

Trace quality checks are surfaced in `trace_overview`:
- `suspected_truncation`: `true` when event counts match a known export cap pattern.
- `quality_warnings`: warnings to treat hotspot/gap attribution with extra caution.

## XPlane / xprof Comparison
Use xprof when Perfetto leaves a large unattributed remainder and you need:
- framework-op self-time deltas,
- op-profile category deltas,
- shell-family attribution (dispatch/sharding, AD/wrapper, layout, residual/add).

The base comparison CLI lives in `marin.profiling`:

```bash
uv run --with xprof python -m marin.profiling.cli xprof-compare \
  --before-xplane /path/to/before.xplane.pb \
  --after-xplane /path/to/after.xplane.pb \
  --normalize-positive-deltas-ms 47.972424 \
  --output /tmp/xprof_compare.json
```

For local macOS workflows, use the `gdnctl` wrapper to stage the XPlane files to a held dev TPU and run the Linux-side analysis there:

```bash
uv run python scripts/gdn/gdnctl.py xprof-compare \
  --cluster us-east5-a \
  --tpu-name calvinxu-gdn \
  --before-xplane scratch/xprof_iter92/attn/plugins/profile/2026_03_12_22_48_45/t1v-n-a0df86f9-w-0.xplane.pb \
  --after-xplane scratch/xprof_iter92/hybrid/plugins/profile/2026_03_12_22_47_02/t1v-n-a0df86f9-w-0.xplane.pb \
  --normalize-positive-deltas-ms 47.972424 \
  --output scratch/xprof_iter92/iter92_xprof_compare.json
```

Use `--sync-repo` if the held dev TPU does not already have the current repo state.

If you already know the matched W&B runs and want `gdnctl` to download the latest profile artifacts before comparing:

```bash
uv run python scripts/gdn/gdnctl.py xprof-compare-runs \
  --cluster us-east5-a \
  --tpu-name "$USER-gdn" \
  --before-run-target <attn_run_url_or_id> \
  --after-run-target <hybrid_run_url_or_id> \
  --normalize-positive-deltas-ms 47.972424 \
  --output scratch/xprof_iter92/iter92_xprof_compare.json
```

To merge xprof shell accounting into the same matched hybrid-vs-attention attribution JSON used by the hillclimb loop:

```bash
uv run python scripts/gdn/gdnctl.py summary-attribution \
  --summary <hybrid_summary.json> \
  --baseline-summary <attn_summary.json> \
  --step-duration-ms <hybrid_step_ms> \
  --baseline-step-duration-ms <attn_step_ms> \
  --upper-bound-step-ms <attn_step_ms> \
  --xprof-compare-json scratch/xprof_iter92/iter92_xprof_compare.json \
  --output scratch/xprof_iter92/iter92_summary_attribution.json
```

The current comparison report includes:
- `framework_op_stats.positive_deltas`
- `framework_op_stats.family_positive_deltas`
- `op_profile_category.positive_deltas`

The shell-family classifier currently maps hybrid-only wrapper paths into:
- `dispatch_shard_shell`
- `ad_wrapper_shell`
- `layout_shell`
- `residual_add_shell`

Treat xprof as a complement to Perfetto, not a replacement:
- Perfetto remains the source of truth for promotion decisions and step-level budgets.
- xprof is the right tool for decomposing `interaction_remainder_ms` into concrete op/program families before deciding what boundary to widen next.

## Agent Queries
Top ops:

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "What are the top 10 ops by exclusive time?"
```

Compute vs comm and collective bottlenecks:

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "Is comm or compute dominating? Which collective is worst?"
```

Specific pre-op gap lookup:

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "gap before _linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined.1"
```

Pre-op gap attribution is marker-aware:
- `gap_before_ops[].payload_op`: op where useful work starts after the idle period.
- `gap_before_ops[].marker_op`: first op observed after the gap (often lightweight setup like `iota.*`).

This avoids over-attributing stalls to tiny "first op" markers when the actual payload starts immediately after.

Hierarchical semantic regions (derived from `tf_op` paths when available):

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "show hierarchical regions"
```

Contextualize a noisy op (for example, `copy.*`) in model/module terms:

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "show context for op copy.564"
```

Suggested optimizations from evidence:

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "What should we try next?"
```

## Optimization Workflow
Use a strict workflow:

1. **Measure**: generate `before.json`.
2. **Change**: apply one bounded patch/config tweak.
3. **Re-measure**: generate `after.json`.
4. **Compare**:

```bash
uv run python lib/marin/tools/profile_summary.py compare \
  --before /tmp/profile_before.json \
  --after /tmp/profile_after.json \
  --strict-provenance
```

5. **Track** (thresholded pass/warn/fail + history):

```bash
uv run python lib/marin/tools/profile_summary.py track \
  --before /tmp/profile_before.json \
  --after /tmp/profile_after.json \
  --label "pallas-kernel-attempt-3" \
  --history /tmp/profile_regression_history.jsonl
```

6. **History summary** (regression trend tracking):

```bash
uv run python lib/marin/tools/profile_summary.py history \
  --history /tmp/profile_regression_history.jsonl
```

7. **One-shot compare bundle** (if you want all artifacts in one command):

```bash
uv run python lib/marin/tools/profile_summary.py bundle \
  --before-run-target marin-community/marin/<baseline_run_id> \
  --after-run-target marin-community/marin/<candidate_run_id> \
  --output-dir /tmp/profile_bundle \
  --history /tmp/profile_regression_history.jsonl
```

8. **Publish summary/report back to W&B** (store alongside experiment artifacts):

```bash
uv run python lib/marin/tools/profile_summary.py publish \
  --summary /tmp/profile_summary.json \
  --report /tmp/profile_report.md \
  --alias latest
```

The comparison reports:
- steady-state step-time delta,
- step class deltas (light/heavy when detected),
- compute/comm/host/stall share deltas,
- semantic family deltas with workload-normalized metrics (from trace-derived shape signatures),
- provenance checks (trace hash/run identity),
- regressed/improved ops by exclusive duration.

## First Workloads
Start with one representative training workload:
- Grug 125M TPU training profile artifact (example above)

Then extend to:
- eval workloads,
- inference workloads,
- RL workloads.

## Success Metrics
MVP is successful when:
- one representative profile is summarized reproducibly into `profile_summary.v1`,
- queries produce deterministic structured answers for top ops and comm/compute breakdown,
- one end-to-end before/after comparison bundle is completed and either:
  - throughput improves measurably, or
  - a clear root-cause report is produced with profile evidence.
