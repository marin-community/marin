# Kernel Profiling Reference

Use this guide for kernel-focused profiling details that are more specific than
`lib/levanter/docs/Performance-Guide.md`.

## Scope

This reference is for profiling kernel changes and validating that a code change
actually improved steady-state performance.

For generic profiling setup and UI usage, start with:
- `lib/levanter/docs/Performance-Guide.md`

For research-loop cadence and execution environment policy, use:
- `.agents/skills/agent-research/SKILL.md`

## What to report

For every headline result, report at least:

- hardware type and device count,
- exact shape/dtype,
- compile-including timing (time-to-first-step),
- steady-state timing (after warmup),
- selected implementation and block-size config.

## Profiling capture patterns

### 1) Trainer-integrated profiling (preferred when using trainer loops)

Enable Levanter profiler flags:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 50 \
  --trainer.profiler.perfetto_link false
```

Notes:
- this captures profiles consistently and uploads `jax_profile` artifacts to trackers,
- tune `start_step` to skip compile/warmup noise,
- keep `num_steps` large enough for stable steady-state signal.

### 2) Standalone microbench profiling

If the benchmark is not a trainer loop, wrap the steady-state region with
`levanter.callbacks.profile_ctx`.

Reference example:
- `lib/levanter/src/levanter/main/sample_lm.py`

Guidelines:
- perform at least one compile/warmup run before entering profile region,
- profile only steady-state iterations,
- keep benchmark shape/config fixed while comparing variants.

## Artifacts and inspection

Primary artifact:
- `jax_profile` (includes Perfetto and TensorBoard-compatible traces)

Typical path inside artifact:
- `plugins/profile/<timestamp>/perfetto_trace.json.gz`

Primary tools:
- Perfetto: detailed timeline and host/device gaps,
- TensorBoard profile plugin: op summaries and high-level breakdowns.

For deterministic extraction and before/after comparisons at scale, use:
- `.agents/skills/agent-profiling/SKILL.md`

## Comparison workflow

Use a strict loop:

`profile -> hypothesis -> change -> tests -> microbench -> profile`

1. capture baseline profile and timing table,
2. apply one bounded change,
3. capture candidate profile with identical setup,
4. compare steady-state timings and trace evidence,
5. keep or revert based on evidence.

Avoid multi-change profiling passes; they are hard to attribute.

## Common pitfalls

- reporting only compile-including time and not steady-state,
- profiling different effective shapes between baseline/candidate,
- changing environment flags between runs,
- comparing while accelerator contention differs,
- using too few iterations for stable steady-state measurements.
