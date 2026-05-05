---
name: accelerator-stack-validation
description: Plan a layered real GPU/TPU cluster validation ladder for Marin training or inference/evals accelerator-stack changes, get one explicit approval for the ladder, then run approved work to a final report. Use for JAX/jaxlib/libtpu, CUDA/cuDNN/cuBLAS/NCCL, vLLM/tpu-inference, Levanter/Grug/Haliax kernels, or Iris resource config that selects those runtimes when correctness, loss, throughput, MFU, profile, or eval behavior must be checked beyond local tests. Not for general Iris debugging, dataset work, or unit-test-only changes.
---

# Accelerator Stack Validation

Use cheap tests to catch obvious runtime failures, then progress to
higher-signal checks when the risk or claim warrants it. The end product is a
clear ship, rollback, or escalate recommendation with evidence.

## Scope

In scope:
- **Training**: Levanter, Grug MoE, Haliax/Pallas kernels, JAX/jaxlib, libtpu,
  CUDA, cuBLAS, NCCL, TPU/GPU dependency extras.
- **Inference/evals**: vLLM, tpu-inference, OpenAI-compatible eval flow,
  LM-eval/Levanter eval entrypoints, eval dependency groups.
- **Iris accelerator plumbing**: only when Iris resource requests, images,
  dependency extras, or cluster configs select the intended TPU/GPU runtime.

Out of scope:
- General Iris controller debugging, cluster lifecycle work, or log-server bugs
  unless they block validation. Use `.agents/skills/debug-infra/` or
  `.agents/skills/babysit-job/`.
- General research management. For multi-day benchmark studies, layer
  `.agents/skills/agent-research/` for branch/issue/logbook cadence.

Hard gate: propose the full ladder once, including cost/time, stop criteria,
and evidence, then get explicit requester approval. After approval, keep working
until the agreed stop condition or final report. Ask again only if the plan
materially changes: higher cost, new hardware/cluster, cluster mutation, stopping
shared jobs, or destructive recovery.

## Working Notes

- Keep a local markdown log in `/tmp` with newest entries first and
  minute-resolution timestamps. Record main actions, outcomes, and learnings.
  Do not commit it unless explicitly requested.
- For commands likely to emit large output, redirect stdout/stderr to a local
  `/tmp` file, then inspect it with `rg`, `tail`, or small excerpts. Record the
  tmp path in the log and final report.

## Authoritative Pointers

Do not read every source. Pick by trigger, and patch the source doc when you
find a sharp edge.

- Real Iris cluster work, GPU/TPU resource flags, CoreWeave, smoke commands:
  `lib/iris/OPS.md`
- Canary/daily ferry is a selected layer, or monitoring must run to terminal:
  `.agents/skills/ferries/SKILL.md`, then `.agents/skills/babysit-job/SKILL.md`
- Fast one-off TPU probe before a full Iris job: `.agents/skills/dev-tpu/SKILL.md`
- MFU/throughput/profile claim or profile artifact analysis:
  `.agents/skills/agent-profiling/SKILL.md`
- Inference/evals stack is touched: `docs/tutorials/run-lm-evals.md` and
  `experiments/evals/evals.py`
- Canary thresholds or recipe interpretation is needed:
  `scripts/canary/validate_canary_metrics.py` and
  `experiments/ferries/canary_ferry.py`

## First Pass

1. Identify the touched surface: `training`, `inference/evals`, or `mixed`.
2. Identify impacted hardware: TPU, GPU, or both. JAX changes often need both;
   CUDA/NCCL usually means GPU; libtpu/tpu-inference usually means TPU.
3. Identify the runtime selector under test: package extra, lockfile section,
   container/image, Iris config, env vars, or model/kernel code path.
4. Propose the whole ladder at once: layers, expected cost/time, continue/stop
   criteria, evidence, and the exact approval requested.
5. After approval, run the ladder without repeatedly interrupting the user.

## Validation Ladder

| Layer | Use When | Cost/Time | Evidence |
|---|---|---:|---|
| Local dependency checks | Lockfile/extras/runtime selector changed | Minutes, no cluster | `uv lock --check`, export/tree checks for every impacted extra; absence of unwanted CUDA/TPU packages on unrelated extras |
| Tiny real-accelerator smoke | Driver/runtime/JAX/vLLM might fail only on hardware | Low; warm GPU/TPU minutes, CoreWeave cold start often 20-30 min | `nvidia-smi` or TPU device probe, `jax.devices()`, backend, tiny matmul or one generate call, warnings |
| Short task smoke | Need launch/compile/data/W&B/profiler coverage | Low to medium | terminal state, first steps/tokens, W&B link, child job ID, traceback-free logs |
| 40-100 step canary/benchmark | Need directional training correctness/perf | Medium | final loss, p50/p90 step time, tokens/sec, MFU, profiler artifact if relevant |
| Full canary ferry | Need production-like training stack confidence | Medium/high; workflow timeouts are hours | canary workflow, Iris parent+child jobs, W&B run, `validate_canary_metrics.py`, profile summary |
| Selected inference/eval smoke | Inference/evals stack touched and a known-working selected task exists | Variable | server startup/generation logs, selected task metrics, W&B/artifacts, retry/preemption behavior |
| Full/key eval run | Current eval suite is known working and its metrics are decision-relevant | High/variable | task metrics plus evidence that failures are from the changed stack, not eval infra |
| Daily/research benchmark | Broad performance or quality claim | High; often 4-5h+ monitoring | issue/logbook, repeated runs, sealed commit/tag, W&B report |

Do not skip straight to expensive layers unless cheaper layers cannot exercise
the risk. Do not stop at a cheap layer when the change can regress loss, MFU,
collectives, profiler stop/upload, or eval serving after warmup.

Post-merge/nightly checks can be acceptable as an added backstop, but choose that
with caution. Do not rely on post-merge coverage as the only signal for a
high-risk change when useful pre-merge checks are available.

## Training

- For GPU runtime changes, use `lib/iris/OPS.md` bounded GPU smoke guidance
  across affected rows. H100-only evidence does not prove GH200/B200 behavior.
- For TPU runtime changes, use `.agents/skills/dev-tpu/` for fast one-off checks;
  use the canary path when production-like behavior matters.
- For Grug/Levanter correctness, run the smallest test slice that exercises the
  changed path, then a short real training smoke if hardware behavior matters.
- For perf claims, compare baseline/candidate on the same hardware shape when
  possible; separate compile/first-step from steady-state.
- For profile-backed claims, use `.agents/skills/agent-profiling/`.
- For ferry-scale validation, use `.agents/skills/ferries/`, then
  `.agents/skills/babysit-job/` until terminal state.

## Inference/Evals

- Validate impacted dependency groups first: usually `eval`, `vllm`, `tpu`, and
  unrelated extras that must not inherit the changed runtime.
- Pass child-worker env through `env_vars=` on the eval helper. Launcher shell
  env stays on the coordinator; Iris `remote()` child tasks only receive env
  that is serialized into the step config.
- Do not assume full/key evals are currently green or high-signal. First check
  recent repo context and the eval docs, then prefer a tiny selected eval or
  server/generate smoke unless a full suite is known working and relevant.

## Hardware Notes

- GPU: request hardware with `--gpu` and dependencies with `--extra`; both are
  required for GPU JAX jobs. Use the cluster/config mapping in `lib/iris/OPS.md`.
- TPU: request devices with `--tpu`; `--reserve` alone does not attach TPU
  devices to the task container.
- Multi-host GPU canary success is correctness/cost evidence unless the run was
  explicitly designed as a perf benchmark.
- Executor parent jobs can be CPU-only while child jobs do the real accelerator
  work. Record both parent and child job IDs.

## Evidence Required

Every validation report must include:
- branch, commit SHA, PR/issue, and exact dependency/runtime selector under test,
- chosen layer and why it is sufficient or intentionally partial,
- cluster/config, hardware type/count, dependency extra, env vars, and command or workflow,
- Iris job IDs, child job IDs when present, W&B run links, log/artifact paths,
- terminal state plus correctness signals: loss, eval metrics, first-loss parity,
  or generated-response smoke result as appropriate,
- performance signals when claimed: warmup policy, measured steps, p50/p90 step
  time, tokens/sec, MFU, compile/first-step time separately, profile summary if used,
- explicit `Not run` section and residual risk.

## What Not To Do

- Do not launch cluster jobs outside the approved ladder. Do not scale
  nodepools, restart clusters, stop shared jobs, or increase hardware scope
  without fresh approval.
- Do not claim performance from tiny smokes, compile-only runs, cold-start time,
  or 2-step training jobs.
- Do not treat `validate_canary_metrics.py` as a quality gate; its thresholds are
  intentionally loose and catch gross failures.
- Do not treat a healthy loss curve as proof of success if the job crashes later.
  Profiler stop/upload, log collection, and teardown can be the failing path.
- Do not hide missing logs. Say which evidence is unavailable and whether W&B,
  Iris summary, workflow logs, or profile artifacts compensate.
- Do not duplicate long Iris/eval/profiling instructions here; patch the source
  docs or skills when you find a sharp edge.

## Recent Regression Lessons

Use current GitHub context before relying on these examples; they are 2026-04/05
patterns, not permanent truth.

- TPU inference/JAX update: targeted real-TPU tests caught a JAX strictness break
  before merge.
- CUDA 13 GPU update: H100/GH200/B200 JAX smokes proved package/driver viability,
  but training smokes/full canary were still needed.
- Grug MoE perf work: paired same-hardware baseline/candidate runs supported the
  speedup claim; full canary supplied loss confidence.
- TPU canary failure: healthy W&B loss/MFU did not imply success; the job failed
  around profiler stop/upload. Terminal state and child task history matter.
