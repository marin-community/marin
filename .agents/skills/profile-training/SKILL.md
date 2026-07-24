---
name: profile-training
description: Profile JAX training and analyze hotspots. Use when profiling or optimizing training throughput.
---

# Skill: Agent-Driven Profiling (XPlane/xprof/TensorBoard/Perfetto)

## Overview
Turn a Levanter profile directory into a deterministic, agent-consumable
summary and a concrete optimization workflow:
1. capture a representative profile,
2. ingest to `profile_summary.v1`,
3. query hotspots and bottlenecks,
4. patch/configure,
5. re-profile and compare.

## Scope
Ingestion sources:
- XPlane protobufs inside Levanter profile directories (source of truth):
  - `plugins/profile/<timestamp>/*.xplane.pb`
  - explicit local `*.xplane.pb` files via `--xplane-file`
- xprof aggregate tables exported from the same XPlane protobuf when the
  optional `xprof` package is available: step overview timing, kernel stats,
  collective breakdowns, xprof bottleneck statements.
- Perfetto trace JSON as an explicit/fallback source for older profiles:
  - `plugins/profile/<timestamp>/perfetto_trace.json.gz`
  - `plugins/profile/<timestamp>/*.trace.json.gz`

Prefer XPlane protobuf for new work. Perfetto trace JSON commonly hits the trace
event cap; XPlane contains the uncapped timeline events needed for named-scope
regions, pre-op gaps, gap context, process/thread metadata, and xprof aggregate
tables. Use `--trace-file` only for a specific Perfetto JSON trace or an older
profile with no XPlane protobuf.

## Capture Profiles
Use Levanter profiler flags so profiles land under
`<trainer.log_dir>/<run_id>/profiler`. Remote Marin runs also upload to
`MARIN_PREFIX` TTL storage and print an XProf link:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 10 \
  --trainer.profiler.upload.ttl_days 30
```

For profiles where xprof/HLO protobuf tables matter, enable JAX profile options
through the Levanter profiler config:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 5 \
  --trainer.profiler.profile_options.host_tracer_level 1 \
  --trainer.profiler.profile_options.python_tracer_level 0 \
  --trainer.profiler.profile_options.device_tracer_level 0 \
  --trainer.profiler.profile_options.enable_hlo_proto true
```

HLO metadata increases artifact size, so keep these profile windows short. The
`XProf profile:` link appears after upload. Set
`--trainer.profiler.upload.enabled false` for local-only capture. Do not copy
profiles to another GCS region for inspection.

Known-good TensorBoard scope recipe from CoreWeave Grug MoE profiling:
`trainer.profiler.enabled=true`, `trainer.profiler.start_step=3`,
`trainer.profiler.num_steps=2`, `trainer.profiler.perfetto_link=false`,
`trainer.profiler.profile_options.host_tracer_level=1`,
`trainer.profiler.profile_options.python_tracer_level=0`, and
`trainer.profiler.profile_options.enable_hlo_proto=true` preserved useful
`jax.named_scope` / `named_call` regions in TensorBoard for
`GM2560-MAY-120S4096-W2048-B8-R1-E8M1-FA4PROFILE-S3B-N1-cw-20260617-2353`.
Leave `device_tracer_level` unset unless device timelines are specifically
needed; this profile still had useful hierarchical host/XLA metadata.

On GPU, command buffers can collapse or suppress the visible name stack in
TensorBoard/Perfetto. For profile-readability runs, disable command buffers:

```bash
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=''"
```

This hurts performance, so use it only when the goal is semantic trace
attribution; leave it out of throughput comparisons unless command-buffer
behavior is the axis being tested.

For GPU throughput runs, keep profile-readability flags separate from XLA code
generation and scheduling flags. Start from JAX's GPU performance guide,
especially the code generation flags section:
<https://docs.jax.dev/en/latest/gpu_performance_tips.html#code-generation-flags>.
The exact set of useful XLA flags is `jaxlib`-version dependent, so record the
full `XLA_FLAGS` value with each profile or W&B run.

For better profile readability, use `haliax.jax_utils.named_call` and
`jax.named_scope` liberally in model code; these names flow into trace
annotations and make region-level summaries far more actionable.

Reference:
- `lib/levanter/docs/Performance-Guide.md`
- `.agents/skills/add-pallas-kernel/`
- JAX GPU performance tips:
  <https://docs.jax.dev/en/latest/gpu_performance_tips.html>

## Ingest to Structured Summary
Pick a download location for pulled profile artifacts: `/tmp` for
ephemeral/local, `scratch/` for an in-repo working area.

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

### Option B: From a W&B run target

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/marin/grug-125m-profile-apples-pallas_tpu-20260217-225239-055ab2 \
  --download-root /tmp/marin-profiles \
  --output /tmp/profile_summary.json
```

`--run-target` accepts: a bare run id (requires `--entity` and `--project`),
`entity/project/run_id`, or a full W&B run URL. The profiler directory is
resolved from `trainer.log_dir` in the run config.

### Option C: From a local artifact directory

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --profile-dir /path/to/profiler_dir \
  --output /tmp/profile_summary.json
```

If the directory contains `*.xplane.pb`, `--profile-dir` uses the XPlane path
automatically. When both `*.xplane.pb` and Perfetto trace JSON are present,
`--profile-dir` reads the XPlane protobuf by default (Perfetto exports are often
capped). Use `--trace-file` to force a specific Perfetto JSON file.

### Option D: From a specific trace file

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --trace-file /path/to/perfetto_trace.json.gz \
  --output /tmp/profile_summary.json
```

### Option E: From a specific XPlane protobuf

Direct XPlane timeline parsing uses `protobuf` and does not require
TensorFlow-generated `xplane_pb2` modules. If `xprof` is installed, ingestion
also exports compact xprof table JSON and augments the timeline summary with
aggregate step, kernel, collective, and bottleneck evidence.

```bash
uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize \
  --xplane-file /path/to/profile.xplane.pb \
  --xplane-output-dir /tmp/profile_xprof_tables \
  --xplane-count-trace-events \
  --output /tmp/profile_summary.json
```

Without `--xplane-output-dir` the command still parses XPlane timeline events
directly. Add `--with xprof` for xprof aggregate table augmentation; add
`--xplane-output-dir` to preserve the exported table JSON (this flag requires
the optional `xprof` package).

XPlane summaries expose hierarchical named-scope regions, pre-op gaps, gap
region context, process/thread/timeline event metadata, step timing (when step
markers or xprof overview rows exist), xprof bottleneck statements, kernel
stats, collective breakdowns, and optimization candidates.

Summary version tag: `profile_summary.v1`

Generate a deterministic markdown root-cause report:

```bash
uv run python lib/marin/tools/profile_summary.py report \
  --summary /tmp/profile_summary.json \
  --output /tmp/profile_report.md
```

Trace quality checks are surfaced in `trace_overview`:
- `suspected_truncation`: `true` when event counts match a known export cap.
- `quality_warnings`: warnings to treat hotspot/gap attribution with caution.

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
- `gap_before_ops[].marker_op`: first op observed after the gap (often
  lightweight setup like `iota.*`).

Hierarchical semantic regions (derived from `tf_op` paths when available):

```bash
uv run python lib/marin/tools/profile_summary.py query \
  --summary /tmp/profile_summary.json \
  --question "show hierarchical regions"
```

Contextualize a noisy op:

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

7. **One-shot compare bundle**:

```bash
uv run python lib/marin/tools/profile_summary.py bundle \
  --before-run-target marin-community/marin/<baseline_run_id> \
  --after-run-target marin-community/marin/<candidate_run_id> \
  --output-dir /tmp/profile_bundle \
  --history /tmp/profile_regression_history.jsonl
```

8. **Publish summary/report back to W&B**:

```bash
uv run python lib/marin/tools/profile_summary.py publish \
  --summary /tmp/profile_summary.json \
  --report /tmp/profile_report.md \
  --alias latest
```

The comparison reports: steady-state step-time delta, step class deltas
(light/heavy when detected), compute/comm/host/stall share deltas, semantic
family deltas with workload-normalized metrics, provenance checks (trace
hash/run identity), and regressed/improved ops by exclusive duration.

## Success Metrics
MVP is successful when:
- one representative profile is summarized reproducibly into `profile_summary.v1`,
- queries produce deterministic structured answers for top ops and comm/compute
  breakdown,
- one end-to-end before/after comparison bundle is completed and either
  throughput improves measurably or a clear root-cause report is produced with
  profile evidence.
</content>
