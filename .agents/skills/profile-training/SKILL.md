---
name: profile-training
description: Profile JAX training and analyze hotspots. Use when profiling or optimizing training throughput.
---

# Profile Training

Turn a Levanter profile into a reproducible `profile_summary.v1`, investigate
hotspots, make one bounded change, and compare before/after evidence. XPlane
protobufs are the source of truth for new profiles:
`plugins/profile/<timestamp>/*.xplane.pb` or explicit `--xplane-file`. Use
Perfetto JSON only for a specified trace or an older profile with no XPlane;
trace exports may be capped. Do not copy profiles between GCS regions.

## Capture

Profiles land under `<trainer.log_dir>/<run_id>/profiler`; remote runs may upload
to `MARIN_PREFIX` TTL storage and print an XProf link. Keep windows short,
especially with HLO metadata:

```bash
uv run ... \
  --trainer.profiler.enabled true --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 10 --trainer.profiler.upload.ttl_days 30
```

For xprof/HLO tables:

```bash
uv run ... \
  --trainer.profiler.enabled true --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 5 \
  --trainer.profiler.profile_options.host_tracer_level 1 \
  --trainer.profiler.profile_options.python_tracer_level 0 \
  --trainer.profiler.profile_options.device_tracer_level 0 \
  --trainer.profiler.profile_options.enable_hlo_proto true
```

Use `--trainer.profiler.upload.enabled false` for local-only capture. On GPU,
`XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=''"` improves
semantic trace names but hurts performance; use only for readability, not
throughput comparisons. Record the full version-dependent `XLA_FLAGS` for each
profile/W&B run. Prefer `haliax.jax_utils.named_call` and `jax.named_scope` in
model code when region attribution is needed.

## Ingest

Use `/tmp` for ephemeral downloads or `scratch/` for retained workspace files.
The tool prefers XPlane automatically when both formats exist:

```bash
uv run python lib/marin/tools/profile_summary.py summarize \
  --run-target marin-community/marin/<run_id> \
  --download-root /tmp/marin-profiles --breakdown-mode exclusive_global \
  --output /tmp/profile_summary.json

uv run python lib/marin/tools/profile_summary.py summarize \
  --artifact marin-community/marin/<run>-profile:v0 \
  --download-root /tmp/marin-profiles --output /tmp/profile_summary.json

uv run python lib/marin/tools/profile_summary.py summarize \
  --profile-dir /path/to/profiler_dir --output /tmp/profile_summary.json

uv run python lib/marin/tools/profile_summary.py summarize \
  --trace-file /path/to/perfetto_trace.json.gz \
  --output /tmp/profile_summary.json
```

`--run-target` accepts a bare ID (with `--entity`/`--project`),
`entity/project/run_id`, or a full W&B URL. For direct XPlane parsing and
optional xprof aggregate tables:

```bash
uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize \
  --xplane-file /path/to/profile.xplane.pb \
  --xplane-output-dir /tmp/profile_xprof_tables \
  --xplane-count-trace-events --output /tmp/profile_summary.json
```

The summary includes named regions, pre-op gaps, process/thread metadata, step
timing, kernels, collectives, bottleneck statements, and optimization
candidates. Check `trace_overview.suspected_truncation` and
`quality_warnings`. Generate a deterministic report with:

```bash
uv run python lib/marin/tools/profile_summary.py report \
  --summary /tmp/profile_summary.json --output /tmp/profile_report.md
```

## Query and compare

Use `query` for top exclusive-time ops, compute-vs-communication, worst
collectives, specific gaps, hierarchical regions, op context, and next-step
recommendations. Gap results distinguish `payload_op` from `marker_op`.

Measure one representative profile, make one bounded patch/config change, then
re-measure and compare with provenance checks:

```bash
uv run python lib/marin/tools/profile_summary.py compare \
  --before /tmp/profile_before.json --after /tmp/profile_after.json \
  --strict-provenance

uv run python lib/marin/tools/profile_summary.py track \
  --before /tmp/profile_before.json --after /tmp/profile_after.json \
  --label "pallas-kernel-attempt-3" \
  --history /tmp/profile_regression_history.jsonl

uv run python lib/marin/tools/profile_summary.py history \
  --history /tmp/profile_regression_history.jsonl

uv run python lib/marin/tools/profile_summary.py bundle \
  --before-run-target marin-community/marin/<baseline_run_id> \
  --after-run-target marin-community/marin/<candidate_run_id> \
  --output-dir /tmp/profile_bundle \
  --history /tmp/profile_regression_history.jsonl
```

The comparison must cover steady-state step time, light/heavy step classes,
compute/comm/host/stall shares, workload-normalized semantic families,
trace/run provenance, and regressed/improved exclusive-duration ops. Publish
only after checking the report against the raw profile:

```bash
uv run python lib/marin/tools/profile_summary.py publish \
  --summary /tmp/profile_summary.json --report /tmp/profile_report.md \
  --alias latest
```

Success is a reproducible summary, deterministic hotspot queries, and a
before/after bundle showing measurable throughput improvement or a clear
evidence-backed root-cause report. See `lib/levanter/docs/Performance-Guide.md`
and `add-pallas-kernel` for kernel-specific work.
