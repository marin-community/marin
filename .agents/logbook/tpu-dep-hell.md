# TPU Dependency Hell: vLLM + tpu-inference Fork Integration

## Document Purpose

This document records the full history, technical constraints, and forward
options for integrating Marin's `tpu-inference` fork with the `vllm-tpu`
packaging ecosystem. It exists because multiple agent sessions across
March 2026 converged on the same packaging wall from different angles,
and the accumulated context is spread across 6+ logbooks and project docs.

Anyone picking this work up should read this document first.

---

## 2026-04-01 Codex Session: `marin-community/vllm` Fork Activation

### Scope

- Turn the newly created `marin-community/vllm` fork into the active Marin TPU
  packaging path.
- Add TPU wheel build/release automation in the fork.
- Replace Marin's `vllm-tpu` pin with a direct `vllm` wheel pin.
- Keep the `marin-community/tpu-inference` override separate.
- Validate with:
  - a CPU-only job that imports `vllm` and prints its version
  - one cheap TPU smoke test, preferring a 1B model and falling back to 8B

### Baseline At Session Start

- Current worktree branch: `worktree-tpu-dep-hell`
- `marin-community/vllm` fork exists as a public fork of `vllm-project/vllm`
  with default branch `main`
- `lib/marin/pyproject.toml` still pins `vllm-tpu==0.13.2.post6`
- root `pyproject.toml` does not yet carry the working `tpu-inference` fork
  override from `vllm_load_fast`
- Existing logbook evidence still points to the same packaging conclusion:
  owning the TPU vLLM wheel is cleaner than staying pinned to Google's release
  cadence

### Plan For This Session

1. Inspect `marin-community/vllm` and `marin-community/tpu-inference` to copy
   the fork/release pattern.
2. Add wheel build/release automation to `marin-community/vllm`.
3. Update Marin packaging to consume the forked TPU wheel and separate
   `tpu-inference` override.
4. Run a CPU-only import/version verification job.
5. Run one TPU smoke test on the smallest practical model path available.

### Open Assumptions

- It is acceptable to use `main` as the release branch for `marin-community/vllm`
  unless a dedicated `marin` branch becomes necessary.
- GitHub Releases are the distribution channel; no PyPI publication is needed.
- For the TPU smoke test, "whatever compute is available" means preferring the
  standard Iris dev TPU workflow over building a larger one-off job wrapper.

### Progress Log

#### 2026-04-01 22:05 PDT - Fork release path wired

- Cloned `marin-community/vllm` and `marin-community/tpu-inference` locally to
  compare release setup.
- Confirmed `tpu-inference` uses a dedicated `marin` branch plus a simple
  GitHub Actions wheel release workflow.
- Confirmed upstream vLLM already contains `tools/vllm-tpu/build.sh` and, more
  importantly, that `setup.py` selects `requirements/tpu.txt` when
  `VLLM_TARGET_DEVICE=tpu`.
- Confirmed the smallest standard dev TPU path available in
  `lib/iris/examples/marin.yaml` is `v6e-4`, which is the leading candidate for
  the smoke test.

Actions taken:

1. Created branch `marin` in `marin-community/vllm`
2. Added `.github/workflows/release.yml` on that branch
3. Pushed commit `7754c1ebd8777355609c8b14f816f8e2f238f5d6`

Workflow state:

- GitHub Actions workflow id: `255184423`
- First run id: `23884849235`
- Status at log update: `in_progress`
- Current live step: `Build TPU wheel`

Current blocker:

- Need the first wheel build to finish so Marin can pin a real release URL
  rather than a placeholder.

#### 2026-04-01 22:12 PDT - First wheel published, Marin patched locally

Fork result:

- `marin-community/vllm` Actions run `23884849235` completed successfully
- Release tag: `marin-7754c1ebd`
- Wheel asset:
  `https://github.com/marin-community/vllm/releases/download/marin-7754c1ebd/vllm-0.0.0.dev20260402+7754c1ebd-py3-none-any.whl`

Local Marin changes applied:

- `lib/marin/pyproject.toml`
  - replaced `vllm-tpu==0.13.2.post6` with the direct `marin-community/vllm`
    wheel URL
- `pyproject.toml`
  - ported the working `tpu-inference` override from `vllm_load_fast`
    (`marin-4abb68f4`)
- native eval/smoke job wiring
  - removed the hardcoded runtime reinstall path that pointed at `vllm-tpu`
  - changed native evaluator/smoke jobs to include the repo `vllm` extra
    directly when `MARIN_VLLM_MODE=native`

Local validation result:

- Command: `uv sync --package marin --extra vllm`
- Result: resolution succeeded, install failed on macOS because
  `nixl==0.3.0` only publishes Linux wheels
- Interpretation: local macOS sync is not a valid validation surface for this
  TPU stack; continue with Linux CPU/TPU remote validation

#### 2026-04-01 22:19 PDT - CPU-only import/version gate passed

Remote validation result:

- Job: `/ahmed/vllm-import-cpu-v6`
- State: `JOB_STATE_SUCCEEDED`
- Command shape:
  `python -c 'import importlib.metadata, platform, vllm; print(platform.platform()); print(vllm.__file__); print(importlib.metadata.version("vllm"))'`
- Environment: Iris CPU job with extras `vllm,tpu`

Observed output:

- Platform: `Linux-6.5.0-1013-gcp-x86_64-with-glibc2.41`
- Module path: `/app/.venv/lib/python3.11/site-packages/vllm/__init__.py`
- Installed version: `0.0.0.dev20260402+7754c1ebd`

Interpretation:

- Marin is now resolving and importing the forked `marin-community/vllm` wheel
  on a remote Linux worker.
- The earlier `torch==2.9.0+cpu` / `GraphCaptureOutput` mismatch is gone after
  aligning the lockfile with `torch==2.10.0`, `torchvision==0.25.0`, and
  `triton==3.6.0`.
- The next experiment is the smallest practical native TPU smoke test on
  `v6e-4`, aiming for a public ~1B Llama-family model first and falling back
  only if the model path itself is unsupported.

#### 2026-04-01 22:21 PDT - Next experiment queued

Planned command:

`uv run --no-sync python -m marin.inference.vllm_smoke_test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --tpu-type v6e-4 --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

Rationale:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is public, small, and Llama-family, so
  it exercises the same general TPU model path without requiring gated HF
  access.
- `--mode native` ensures the test hits the newly wired forked `vllm` wheel
  rather than the Docker sidecar.
- `--tpu-type v6e-4` is the smallest standard TPU slice currently exposed in
  the Marin Iris config for this path.
- `--max-model-len 1024` keeps the smoke bounded; the goal here is package and
  startup validation, not throughput.

#### 2026-04-01 22:23 PDT - First TPU launch surface was wrong

Attempted command:

`uv run --no-sync python -m marin.inference.vllm_smoke_test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --tpu-type v6e-4 --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

Failure:

- The smoke harness called `current_cluster()` from Fray v1.
- Outside a Ray or explicit cluster context, Fray v1 defaults to
  `LocalCluster`, so the job ran on the local macOS host instead of Iris.
- Native startup then failed immediately with
  `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`.

Interpretation:

- This is not a packaging regression on the remote path.
- It is a launch-surface mismatch: the smoke harness can launch local or Ray
  jobs, but not Iris-backed remote jobs directly from the laptop.

Next command:

`uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run --tpu v6e-4 --memory 32GB --disk 100GB --job-name vllm-smoke-tinyllama-v1 --extra eval --extra tpu --extra vllm -- python -m marin.inference.vllm_smoke_test --local --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

Adjustment rationale:

- Iris owns the remote TPU scheduling and environment setup here.
- `--local` makes the smoke script start vLLM directly inside the remote TPU
  job rather than trying to submit a second nested cluster job.

#### 2026-04-01 22:24 PDT - First real TPU smoke found a missing dependency edge

Job:

- `/ahmed/vllm-smoke-tinyllama-v1`
- launch surface: Iris TPU job on `v6e-4`
- command:
  `python -m marin.inference.vllm_smoke_test --local --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

Observed failure:

- The remote job reached the TPU host and started `/app/.venv/bin/vllm`.
- vLLM detected the TPU worker correctly.
- Startup then failed with:
  - `ERROR ... tpu_inference not found, please install tpu_inference to run vllm on TPU`
  - `ImportError: cannot import name 'current_platform' from 'vllm.platforms'`

Interpretation:

- The forked `vllm` wheel is present and the native TPU entry path is executing.
- The remote Iris environment is not ending up with an importable
  `tpu_inference` package, even though the workspace lock contains the forked
  `tpu-inference` wheel via root override.
- Wheel metadata confirms the `marin-community/vllm` wheel still declares
  `Requires-Dist: tpu-inference==0.12.0`, so relying on a root-level override
  alone is too implicit for this remote install surface.

Local fix applied:

- Added the forked `tpu-inference` wheel URL directly to `marin[vllm]` in
  `lib/marin/pyproject.toml`.
- Kept the root `override-dependencies` entry in place.
- Regenerated `uv.lock`.

Next command:

`uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run --cpu 2 --memory 8GB --disk 20GB --job-name tpu-inference-import-cpu-v1 --extra vllm --extra tpu -- python -c 'import importlib.metadata, importlib.util; print(importlib.util.find_spec("tpu_inference")); print(importlib.metadata.version("tpu-inference"))'`

Why this probe first:

- It isolates the packaging question from TPU startup cost.
- If this CPU probe sees the fork wheel, the next TPU retry is worth running.

#### 2026-04-01 22:25 PDT - CPU probe confirmed forked `tpu-inference` install

Job:

- `/ahmed/tpu-inference-import-cpu-v1`
- state: `JOB_STATE_SUCCEEDED`

Observed output:

- `ModuleSpec(... origin='/app/.venv/lib/python3.11/site-packages/tpu_inference/__init__.py', ...)`
- `0.0.0.dev20260322+4abb68f4`

Interpretation:

- The remote Iris environment now installs the forked `tpu-inference` wheel
  when `marin[vllm,tpu]` is requested.
- The explicit `marin[vllm]` dependency edge fixed the gap that the first TPU
  smoke exposed.

Next command:

`uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run --tpu v6e-4 --memory 32GB --disk 100GB --job-name vllm-smoke-tinyllama-v2 --extra eval --extra tpu --extra vllm -- python -m marin.inference.vllm_smoke_test --local --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

#### 2026-04-01 22:29 PDT - TPU retry exposed an API mismatch, not a missing wheel

Jobs:

- `/ahmed/vllm-smoke-tinyllama-v2` on `v6e-4`: failed
- `/ahmed/tpu-platform-import-cpu-v3`: cheap CPU reproduction of the platform import

Key evidence:

- The CPU probe can import top-level `tpu_inference` and reports the correct
  fork version.
- The TPU smoke still logs `tpu_inference not found`, but `vllm.platforms.tpu`
  only emits that message for any `ImportError` raised by
  `from tpu_inference.platforms import TpuPlatform`.
- The targeted CPU reproduction surfaced the real exception:
  `ImportError: cannot import name 'ProcessorInputs' from 'vllm.inputs'`

Root cause:

- The current `marin-community/vllm` fork is on a newer vLLM API surface where
  `ProcessorInputs` moved out of `vllm.inputs`.
- The released `marin-community/tpu-inference` wheel still expects the older
  location and also uses the older `Platform.validate_request(...)` signature.

Fix in flight:

- Patched `marin-community/tpu-inference` on branch `marin` at commit
  `42fb81da8a775ed772d6b1b87d7048829a53c283` to:
  - import `ProcessorInputs` from `vllm.multimodal.processing`
  - import `EngineInput` from `vllm.inputs`
  - update `validate_request(...)` to the current two-argument vLLM interface
- The `tpu-inference` release workflow started automatically after push.

Next step:

- Wait for the new `tpu-inference` wheel release.
- Re-pin Marin to that wheel.
- Re-run the cheap `tpu_inference.platforms` CPU import before spending TPU
  time on another smoke.

#### 2026-04-01 22:33 PDT - User redirected to upstream-aligned fork mains

New directive:

- Stop leaning on the old `marin` branch for `tpu-inference`.
- Make both Marin forks track upstream `main` as closely as possible.
- Undo the local Marin packaging changes that pinned branch-built wheels.

GitHub fork state after redirect:

- `marin-community/tpu-inference`
  - created/updated `main` to upstream `vllm-project/tpu-inference` `main`
    at `cd1c7bfea4ac57324ce7bffe451110fec195447b`
  - changed default branch from `marin` to `main`
- `marin-community/vllm`
  - fast-forwarded `main` to upstream `vllm-project/vllm` `main`
    at `9bd72311068919b8f3430278d47859cf312039fa`
  - deleted the temporary `marin` branch that was added for the TPU wheel
    workflow experiment

Local workspace cleanup:

- Reverted the Marin packaging/runtime edits in:
  - `lib/marin/pyproject.toml`
  - `pyproject.toml`
  - `lib/marin/src/marin/inference/vllm_server.py`
  - `lib/marin/src/marin/inference/vllm_smoke_test.py`
  - native evaluator files that had been rewired to install the repo `vllm`
    extra directly
  - `uv.lock`
- Removed the temporary `.agents/logbook/tpu-dep-hell-codex.md` plan file

Current state:

- The worktree is back to clean code/package state except for this logbook.
- The GitHub forks now have upstream-aligned `main` branches, with
  `tpu-inference` no longer defaulting to `marin`.

Next step from the new direction:

- If Marin still needs custom wheel publication, reintroduce the smallest
  possible fork-only delta on `main` rather than on `marin`.
- Otherwise, treat this session's branch-built wheel experiment as closed.

#### 2026-04-01 22:37 PDT - Minimal main-branch release workflows added

Implementation strategy:

- Keep both forks as close as possible to upstream `main`.
- Add one new workflow file on each fork's `main` branch rather than mutating
  upstream-managed release files.
- Trigger only on pushes to `main`.

Fork commits:

- `marin-community/vllm` `main`: `9e3db15a75e4c5e285c2fae8ffcc3b60f2c4c8d6`
  - adds `.github/workflows/marin-release.yml`
  - builds a TPU wheel with `VLLM_TARGET_DEVICE=tpu`
- `marin-community/tpu-inference` `main`: `4cfc17bcc2ef5bb3c7f5b23883fc925ee92729f7`
  - adds `.github/workflows/marin-release.yml`
  - builds the regular wheel

Workflow runs started automatically on push:

- `marin-community/vllm`
  - workflow file: `marin-release.yml`
  - run id: `23885749797`
- `marin-community/tpu-inference`
  - workflow file: `marin-release.yml`
  - run id: `23885749572`

Current state at log update:

- Both runs are in progress.
- Fork `main` remains the only active path; no `marin` branch was recreated.

#### 2026-04-01 22:39 PDT - Main-branch release path validated

`tpu-inference` result:

- Run `23885749572` completed successfully.
- Release tag: `marin-4cfc17bc`
- Wheel URL:
  `https://github.com/marin-community/tpu-inference/releases/download/marin-4cfc17bc/tpu_inference-0.0.0.dev20260402+4cfc17bc-py3-none-any.whl`

`vllm` result:

- Run `23885749797` started successfully from `main`.
- The workflow is still in the `Build TPU wheel` step at the time of this log
  update.
- Live run URL:
  `https://github.com/marin-community/vllm/actions/runs/23885749797`

Interpretation:

- The fork topology is now what the user requested: upstream-aligned `main`
  branches with a minimal Marin-only release workflow layered on top.
- `tpu-inference` is already publishing from `main`.
- `vllm` is using the same `main`-only model; the only outstanding item is the
  heavier wheel build finishing.

#### 2026-04-01 22:43 PDT - Marin cannot lock against upstream `main` wheels yet

Attempt:

- Re-pointed local Marin packaging to the new `main`-branch wheel URLs:
  - `marin-community/vllm` release `marin-9e3db15a7`
  - `marin-community/tpu-inference` release `marin-4cfc17bc`
- Reapplied the native-mode runtime wiring so jobs would use the repo `vllm`
  extra instead of reinstalling `vllm-tpu`.

Result:

- `uv lock` failed before any import smoke could run.

Exact blocker from resolver:

- `tpu-inference==0.0.0.dev20260402+4cfc17bc` depends on `jax==0.9.2`
- Marin is still pinned to `jax==0.8.0`
- Therefore `marin[vllm]` is unsatisfiable against upstream `tpu-inference`
  `main`

Interpretation:

- The new fork topology is correct.
- But upstream `tpu-inference` `main` has already moved past Marin's current
  JAX baseline.
- So "use fork `main` as-is in Marin" is blocked at dependency resolution,
  before runtime import checks.

Local cleanup after confirming the blocker:

- Reverted the attempted local packaging/runtime edits so the worktree stays
  clean except for this logbook.

Decision boundary:

- To use upstream-aligned `main` wheels in Marin, Marin likely needs a JAX
  uplift to `0.9.2` (or a separate compatibility strategy).
- If Marin must stay on `jax==0.8.0`, then `tpu-inference` cannot be taken from
  upstream `main` verbatim.

#### 2026-04-01 22:48 PDT - JAX uplift path resolved the workspace lock

Applied dependency updates:

- Marin JAX baseline: `0.8.0` -> `0.9.2`
- Marin TPU extra: switched from `jax[tpu]` to explicit
  `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38`
- Levanter TPU extra: same explicit TPU runtime shape
- Fray TPU test extra: same explicit TPU runtime shape
- Levanter storage caps:
  - `gcsfs<2026` -> `gcsfs<2027`
  - `fsspec[http]<2026` -> `fsspec[http]<2027`
- Marin/root datasets cap:
  - `<4.0.0` -> `<5.0.0`

Result:

- `uv lock` now succeeds.
- The lock now contains:
  - `jax==0.9.2`
  - `jaxlib==0.9.2`
  - `libtpu==0.0.38`
  - `tpu-inference==0.0.0.dev20260402+4cfc17bc`
  - `vllm==0.0.0.dev20260402+9e3db15a7`

Next command:

- Run a CPU-only Iris import smoke that verifies:
  - `import vllm`
  - `import tpu_inference`
  - `from tpu_inference.platforms import TpuPlatform`

#### 2026-04-01 22:49 PDT - CPU import smoke passed with `main`-branch wheels

Job:

- `/ahmed/vllm-tpu-main-import-cpu-v1`
- state: `JOB_STATE_SUCCEEDED`

Command surface:

- Iris CPU job with extras `vllm,tpu`
- user command imported:
  - `vllm`
  - `tpu_inference`
  - `from tpu_inference.platforms import TpuPlatform`

Observed output:

- Platform: `Linux-6.5.0-1013-gcp-x86_64-with-glibc2.41`
- `vllm` module path:
  `/app/.venv/lib/python3.11/site-packages/vllm/__init__.py`
- `tpu_inference` module path:
  `/app/.venv/lib/python3.11/site-packages/tpu_inference/__init__.py`
- `vllm` version: `0.0.0.dev20260402+9e3db15a7`
- `tpu-inference` version: `0.0.0.dev20260402+4cfc17bc`
- `find_spec("tpu_inference")`: present
- imported symbol: `TpuPlatform`

Non-blocking note:

- The CPU worker logs a TPU detection warning (`/dev/vfio` missing) and a
  TensorFlow GCS shim warning, but neither blocked imports.

Bottom line:

- Marin now resolves, installs, and imports the upstream-aligned fork `main`
  wheels for both `vllm` and `tpu-inference`.

#### 2026-04-01 22:55 PDT - Smallest idle TPU smoke queued

Selection:

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- TPU: `v6e-4`
- Zone: `us-east1-d`

Why this target:

- `v6e-4` is one of the smallest TPU shapes in the current Marin config.
- The cluster currently shows an idle ready `v6e-4` slice in `us-east1-d`.
- TinyLlama 1.1B is public and small enough for a bounded native bring-up test.

Planned command:

`uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run --tpu v6e-4 --zone us-east1-d --memory 32GB --disk 100GB --job-name vllm-smoke-tinyllama-main-v3 --extra eval --extra tpu --extra vllm -- python -m marin.inference.vllm_smoke_test --local --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode native --max-model-len 1024 --use-completions --prompt "Write a short haiku about TPUs."`

---

## Table of Contents

1. [Background: Why We Need a Fork](#1-background-why-we-need-a-fork)
2. [The Dependency Graph](#2-the-dependency-graph)
3. [What the Fork Contains](#3-what-the-fork-contains)
4. [The Three Failed Integration Attempts](#4-the-three-failed-integration-attempts)
5. [Root Cause: Why vllm-tpu Exists as a Separate Package](#5-root-cause-why-vllm-tpu-exists-as-a-separate-package)
6. [Current State of the Branch](#6-current-state-of-the-branch)
7. [Forward Options](#7-forward-options)
8. [Recommendation](#8-recommendation)
9. [Appendix: Full Timeline](#appendix-full-timeline)
10. [Appendix: Iris Jobs Run](#appendix-iris-jobs-run)
11. [Appendix: Benchmark Results](#appendix-benchmark-results)
12. [Appendix: Key Files Reference](#appendix-key-files-reference)

---

## 1. Background: Why We Need a Fork

### The Cold Start Problem

vLLM on TPU has a painful cold start. For a 70B model (Llama 3.3 70B Instruct,
131 GiB of weights), the breakdown is:

| Phase | Time |
|-------|------|
| Weight download (RunAI streamer) | ~3 min (same-region, 2.1 GiB/s) |
| Model bootstrap (concrete random-init for Llama) | 15-20 min |
| XLA compilation | 20-30 min |
| KV cache allocation (first `sharded_allocate`) | ~19 min |
| **Total cold start** | **~60+ min** |

The weight download itself is fast when same-region. The real bottlenecks are
in `tpu-inference` internals: model bootstrap, sampling RNG initialization,
and KV cache allocation.

### What Marin Built First: 4,300 Lines of Monkeypatches

Starting March 12, 2026, agents built a fast-loading path in Marin that
bypassed the slow `tpu-inference` internals via runtime monkeypatching:

1. **Dummy load + external weight injection**: `load_format="dummy"` to skip
   RunAI, then stream weights from GCS via Levanter's fsspec and inject via
   `sync_weights()`.
2. **Abstract model bootstrap**: Replace `tpu-inference`'s concrete random-init
   with `nnx.eval_shape()` (20 min → 3 seconds).
3. **Bootstrap-safe RNG**: Replace `nnx.Rngs(...).params()` (stalls on
   abstract state) with direct `jax.random.key(seed)`.
4. **KV cache override**: Use upstream `num_gpu_blocks_override` to skip the
   19-minute first allocation.

This worked but accumulated ~4,300 lines of Marin-side monkeypatches that
reached deep into `tpu-inference` internals. Brittle, hard to maintain, and
broke across package updates.

### The Fork Decision

On March 21, 2026, the decision was made to move the fixes into a proper fork
of `tpu-inference` at `marin-community/tpu-inference`. The fork would:

- Absorb the monkeypatch logic into clean, upstreamable feature branches
- Reduce Marin-side code from 4,300 lines to ~10 lines of config passthrough
- Follow the existing forking policy (`docs/dev-guide/forking-policy.md`)
- Enable eventual upstream PRs to `vllm-project/tpu-inference`

---

## 2. The Dependency Graph

Understanding this graph is essential to understanding why integration failed.

### How vllm-tpu Packages Things

```
PyPI: vllm-tpu==0.13.3
├── tpu-inference==0.13.3  (hard-pinned)
│   ├── jax==0.8.1         (hard-pinned)
│   ├── jaxlib==0.8.1      (hard-pinned)
│   ├── torchvision==0.24.0 (hard-pinned, resolves to CUDA wheel on Linux)
│   ├── torchax==0.0.10    (brings in torch transitively)
│   ├── libtpu==0.0.31     (195 MB, the actual TPU runtime)
│   ├── flax==0.11.1
│   ├── qwix==0.1.1
│   ├── numba==0.62.1
│   ├── tpu-info==0.7.1
│   ├── runai-model-streamer[gcs,s3]==0.15.0
│   └── ...
├── ray[default], ray[data]
├── transformers, tokenizers, fastapi, ...
└── (NO torch, torchvision, flashinfer, nvidia-* — stripped from CUDA vllm)
```

### How stock vllm (PyPI) Packages Things

```
PyPI: vllm==0.18.0
├── torch==2.10.0
├── torchvision==0.25.0       (CUDA variant, includes CUDA ops)
├── torchaudio==2.10.0
├── flashinfer-python==0.6.6  (GPU-only)
├── nvidia-cudnn-frontend      (GPU-only)
├── nvidia-cutlass-dsl         (GPU-only)
├── quack-kernels              (GPU-only)
└── ...
```

### Key Insight

**`vllm-tpu` is NOT stock vllm repackaged.** It is a completely separate
build produced by running `VLLM_TARGET_DEVICE=tpu python -m build --wheel`
against the vllm source. This build:

- Uses `requirements/tpu.txt` instead of `requirements/cuda.txt`
- Strips ALL GPU-specific dependencies (torchvision CUDA, flashinfer, nvidia-*)
- Adds TPU-specific dependencies (tpu-inference, ray)
- May patch vllm internals for TPU compatibility

There is no `pip install vllm[tpu]` extra. The TPU variant is a **different
package** on PyPI.

### Marin's Current Dependency Setup

In `lib/marin/pyproject.toml`:
```toml
vllm = [
    "vllm-tpu==0.13.3",
    "triton==3.5.0; platform_system == 'Linux' and platform_machine == 'x86_64'",
]
```

Marin already has CPU-only PyTorch index routing for the `tpu` extra:
```toml
[tool.uv.sources]
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cpu", extra = "tpu" },
    { index = "pytorch-cu128", extra = "gpu" },
]
torch = [
    { index = "pytorch-cu128", extra = "gpu", marker = "sys_platform == 'linux'" },
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cpu", extra = "tpu" },
]
```

This CPU routing works for Marin's own `tpu` extra, but when `vllm-tpu`
brings in `tpu-inference` which hard-pins `torchvision==0.24.0`, the
resolution path may bypass Marin's source overrides because the dependency
originates from a transitive package, not from Marin's own extras.

---

## 3. What the Fork Contains

### Repository

`marin-community/tpu-inference` — forked from `vllm-project/tpu-inference`

### Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Tracks upstream, never commit Marin changes |
| `feat/fast-tpu-bootstrap` | Original feature branch (jax 0.9.2 base) |
| `feat/fast-tpu-bootstrap-v0.13.2` | Rebased on v0.13.2 tag |
| `marin` | Integration branch, merges all feat branches, what Marin pins to |

### Feature: Abstract Model Bootstrap (`model_loader.py`)

**Problem**: For `load_format="dummy"` with non-`LoadableWithIterator` models
(like Llama), `tpu-inference` does concrete random-init: allocates full model
weights with random values, JIT compiles, then discards them when real weights
arrive. For Llama 8B this takes 20+ minutes.

**Fix**: `TpuBootstrapConfig` dataclass with three modes:
- `"default"` — existing behavior (concrete random-init)
- `"abstract_dummy"` — `nnx.eval_shape()` returns abstract model only (for RL,
  where caller injects weights via `_sync_weights()`)
- `"abstract_load"` — `nnx.eval_shape()` then loads real weights via iterator
  (for `vllm serve` / evaluators)

Gated on `model_loader_extra_config`:
```python
{"tpu_bootstrap": {"model_bootstrap": "abstract_load"}}
```

Only activates for:
- `load_format="dummy"` (or compatible)
- Non-`LoadableWithIterator` models (currently Llama)
- No quantization active
- Architecture in `_ABSTRACT_BOOTSTRAP_ARCHITECTURES`

### Feature: Bootstrap-Safe Sampling RNG (`tpu_runner.py`)

**Problem**: `nnx.Rngs(jax.random.key(seed)).params()` stalls when model is in
abstract state.

**Fix**: When abstract bootstrap is active, use `jax.random.key(seed)` directly
instead of going through `nnx.Rngs`.

### Feature: Architecture Alias Registration (`model_loader.py`)

**Problem**: vLLM's `ModelRegistry` remaps `LlamaForCausalLM` →
`MistralForCausalLM` internally. The JAX registry in `tpu-inference` doesn't
know about `MistralForCausalLM`, so it falls back to the slow PyTorch path.

**Fix**: Register `MistralForCausalLM` as alias for `LlamaForCausalLM` in the
JAX model registry. Also added `model_type` fallback lookup.

### Feature: fsspec Streaming Weight Loader (`streaming_weights.py`)

**Problem**: RunAI streamer loads all weight shards into host RAM simultaneously.
For 70B (131 GiB), this requires 131+ GiB host RAM.

**Fix**: New `fsspec_weights_iterator()` that streams one shard at a time using
gcsfs, keeping peak host RAM at ~15 GiB for 70B. Configurable via:
```python
{"tpu_bootstrap": {"model_bootstrap": "abstract_load", "weight_loader": "fsspec_streamer"}}
```

### Feature: Mesh Context Fixes

Five separate mesh/tracing issues discovered and fixed during smoke tests:

| Issue | Fix | Commit |
|-------|-----|--------|
| `nnx.eval_shape` outside mesh context | Moved inside `with mesh:` | `af97bfbd` |
| `with mesh:` doesn't set abstract mesh | Added `jax.sharding.use_abstract_mesh()` | `85bcc9c1` |
| Python `list` layers reject abstract data | Changed to `nnx.List` | `988ed996` |
| TP>1 device_put needs full mesh context | Use `jax.set_mesh(mesh)` | `3afb06f0` |
| vLLM remaps architectures silently | Registry alias + model_type fallback | `a74f6142` |

### CI

Wheel build workflow on `marin` branch: `python -m build --wheel`, uploaded to
GitHub Releases as prerelease. Versioning: `0.0.0.dev{YYYYMMDD}+{short_sha}`.

---

## 4. The Three Failed Integration Attempts

All three attempts occurred on March 21, 2026, during the initial fork
integration session.

### Attempt 1: vllm-tpu==0.13.2.post6 + Fork Wheel (jax 0.9.2)

**Setup**: Keep `vllm-tpu==0.13.2.post6` from PyPI, add fork wheel URL to
`pyproject.toml`.

**Failure**: `uv sync` fails immediately. The fork was based on upstream `main`
which declares `jax==0.9.2`. `vllm-tpu==0.13.2.post6` requires `jax==0.8.0`
(via its `tpu-inference==0.13.2.post6` dependency). Direct version conflict,
resolver cannot satisfy both.

**Lesson**: The fork's jax pin must be compatible with `vllm-tpu`'s transitive
jax pin, or `vllm-tpu` must be replaced entirely.

### Attempt 2: Upgrade jax to 0.9.2 Across the Board

**Setup**: Senior engineers approved a jax upgrade. Updated all jax pins in
`pyproject.toml` (0.8.0 → 0.9.2). Also bumped torch (2.9→2.10), torchvision
(0.24→0.25), triton (3.5→3.6) to match `vllm==0.18.0` requirements.

**Local result**: `uv sync` succeeded.

**Cluster failure**: On Iris worker, `vllm-tpu==0.13.2.post6` crashed with
`PackageNotFoundError: No package metadata was found for vllm`. The old
`vllm-tpu` wheel is binary-incompatible with jax 0.9.2. It was built against
jax 0.8.0 and its internal imports/metadata assume that version.

**Lesson**: You cannot just bump jax and keep the old `vllm-tpu` wheel. The
`vllm-tpu` wheel itself must be rebuilt for the new jax version, and only
Google does that.

### Attempt 3: Drop vllm-tpu, Use Stock vllm==0.18.0 + Fork

**Setup**: Remove `vllm-tpu` entirely. Install stock `vllm==0.18.0` from PyPI
plus the fork's `tpu-inference` wheel separately.

**Rationale**: Stock vllm has TPU platform support via its plugin system.
`tpu-inference` registers itself via `entry_points`. In theory they should work
as separate packages.

**Local result**: `uv sync` succeeded after adding missing runtime deps to the
fork wheel (`tpu-info`, `torchax`, `flax`, `qwix`, `runai-model-streamer`).

**Cluster failure 1**: `tpu_inference not found` — fork wheel's
`install_requires` was initially empty. Fixed by adding deps.

**Cluster failure 2**: `tpu_inference` imports OK, but
`torchvision::nms operator does not exist`. Stock `vllm==0.18.0` hard-depends
on `torchvision==0.25.0` from default PyPI (the CUDA variant). On TPU workers,
CUDA ops don't exist.

**Why Marin's torchvision CPU routing didn't help**: Marin's `tool.uv.sources`
routes torchvision to the CPU index for the `tpu` extra. But `vllm==0.18.0` is
a separate package with its own unconditional `torchvision==0.25.0` dependency.
When both the `vllm` and `tpu` extras are active, the `vllm` package's CUDA
torchvision dep overrides the CPU variant. uv's conflict resolution between
extras couldn't reconcile this — both wanted torchvision but from different
indexes with different versions (0.24.0 CPU vs 0.25.0 CUDA).

**Additional CUDA deps**: Even if torchvision were fixed, stock `vllm==0.18.0`
also hard-depends on `flashinfer-python==0.6.6`, `nvidia-cudnn-frontend`,
`nvidia-cutlass-dsl`, and `quack-kernels`. All GPU-only. Overriding
torchvision alone would not have been sufficient.

**Lesson**: Stock `vllm` from PyPI is fundamentally the CUDA build. You cannot
use it on TPU by just swapping torchvision. The entire dependency tree assumes
GPU. This is exactly why `vllm-tpu` exists as a separate package.

### Summary of Attempts

| Attempt | Approach | Failure Point | Root Cause |
|---------|----------|---------------|------------|
| 1 | Keep vllm-tpu + add fork | `uv sync` | jax version conflict (0.8.0 vs 0.9.2) |
| 2 | Bump jax to 0.9.2 + keep vllm-tpu | Iris worker crash | vllm-tpu wheel incompatible with jax 0.9.2 |
| 3 | Drop vllm-tpu + stock vllm + fork | Iris worker crash | Stock vllm requires CUDA torchvision + other GPU deps |

---

## 5. Root Cause: Why vllm-tpu Exists as a Separate Package

vllm uses a build-time device target to produce different wheels:

```python
# vllm/setup.py
VLLM_TARGET_DEVICE = os.environ.get("VLLM_TARGET_DEVICE", "cuda")
```

- `VLLM_TARGET_DEVICE=cuda` → uses `requirements/cuda.txt` → torch,
  torchvision (CUDA), flashinfer, nvidia-* libs
- `VLLM_TARGET_DEVICE=tpu` → uses `requirements/tpu.txt` → tpu-inference,
  ray, NO torch/torchvision/flashinfer/nvidia-*

The TPU build is published as `vllm-tpu` on PyPI. The CUDA build is published
as `vllm`. They are different packages with different dependency trees, built
from the same source.

**This is the fundamental constraint**: there is no `pip install vllm[tpu]`.
The device-specific dependencies are baked into the wheel metadata at build
time. You must either:

1. Use Google's pre-built `vllm-tpu` wheel (tied to their jax/tpu-inference
   version pins), or
2. Build your own `vllm-tpu` wheel from source with
   `VLLM_TARGET_DEVICE=tpu`.

---

## 6. Current State of the Branch

### Branch: `vllm_load_fast`

The branch has the fork working via `override-dependencies` in the root
`pyproject.toml`:

```toml
override-dependencies = [
    # ... other overrides ...
    "tpu-inference @ https://github.com/marin-community/tpu-inference/releases/download/marin-4abb68f4/tpu_inference-0.0.0.dev20260322+4abb68f4-py3-none-any.whl",
]
```

This overrides the `tpu-inference==0.13.3` that `vllm-tpu==0.13.3` pulls in,
replacing it with the fork's wheel. The override works because:

- `vllm-tpu==0.13.3` still installs (it's the vllm portion)
- `tpu-inference` is replaced by the fork wheel
- jax stays at 0.8.1 (the fork wheel was rebased to be compatible)
- The fork wheel's actual code is from upstream main with Marin patches, but
  its declared dependencies were adjusted to not conflict with jax 0.8.1

### What Works

- Fork wheel installs alongside `vllm-tpu==0.13.3`
- Abstract bootstrap modes (`abstract_dummy`, `abstract_load`) work on cluster
- fsspec streaming weight loader works end-to-end
- Llama 8B smoke tests pass (correct haiku output, 32 GiB host RAM)
- Llama 70B with fsspec passes at 32 GiB (where RunAI OOMs)
- `MODEL_IMPL_TYPE=auto` correctly routes Llama to `flax_nnx` path
- Architecture alias (MistralForCausalLM → LlamaForCausalLM) works

### What Doesn't Work / Is Unfinished

- **Async-native server startup**: The engine subprocess hung at
  `vllm_get_model()`. Fix applied (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) but
  not fully validated on cluster.
- **fsspec throughput**: Sequential implementation, ~109-128 MiB/s vs RunAI's
  2.1 GiB/s. The concurrency from Levanter's original code was stripped during
  porting. Closing this gap requires porting back `asyncio.gather()` with
  `ThreadPoolExecutor`.
- **Compilation cache**: XLA compilation cache behavior is inconsistent. Local
  stable paths showed speedups; GCS cache paths did not reliably hit.
- **Qwen3-MoE routing**: `_VLLM_PREFERRED_ARCHITECTURES` still forces
  `Qwen3MoeForCausalLM` to the PyTorch wrapper path despite a working JAX
  implementation. Fork has `routing_mode="prefer_flax_nnx"` but not validated.
- **Evaluator integration**: `abstract_load` config not yet wired through
  `lm_evaluation_harness_evaluator` or `simple_evaluator`.

### Monkeypatch Status

The 4,300-line monkeypatch files were deleted from the `vllm_load_fast` branch
when the fork was created. Marin now passes config through
`model_loader_extra_config` (~10 lines) instead. However, the startup timing
instrumentation monkeypatches in `vllm_async.py` and `worker.py` remain (these
are diagnostic, not functional).

---

## 7. Forward Options

### Option A: Stay on jax 0.8.1, Use Fork via override-dependencies (Current Approach)

**What it is**: Keep `vllm-tpu==0.13.3` as the dependency. Override
`tpu-inference` with the fork wheel via `override-dependencies`. The fork's
code is from latest upstream but its declared jax pin is relaxed to 0.8.1.

**What to do**:
1. This is already partially working on `vllm_load_fast`
2. Verify fork code works on jax 0.8.1 (the patches use `nnx.eval_shape`,
   `jax.random.key`, `jax.set_mesh` — all available in 0.8.1)
3. Clean up any remaining issues from the smoke test cycle
4. Merge to main

**Dependency config** (already in place):
```toml
# lib/marin/pyproject.toml
vllm = ["vllm-tpu==0.13.3", ...]

# root pyproject.toml
override-dependencies = [
    "tpu-inference @ https://github.com/marin-community/tpu-inference/releases/download/marin-XXXX/tpu_inference-...-py3-none-any.whl",
]
```

**Pros**:
- Least work — largely already done
- No custom vllm builds needed
- Compatible with existing Marin packaging
- `vllm-tpu==0.13.3` is a known-good, Google-tested wheel
- Low risk — only the tpu-inference layer changes

**Cons**:
- Stuck on jax 0.8.1 until Google ships a new `vllm-tpu`
- `override-dependencies` is a blunt instrument — silently replaces
  without version compatibility checking
- If the fork accidentally uses jax 0.9+ APIs, failure is at runtime not
  install time

**Risk**: Low. The fork patches use basic jax/flax APIs. `nnx.eval_shape`,
`jax.random.key`, `jax.set_mesh`, `nnx.List` — all existed in jax 0.8.x.

**Effort**: ~1-2 days to clean up and validate.

---

### Option B: Build Your Own vllm-tpu Wheel from Source

**What it is**: Clone vllm at a commit compatible with the fork's
tpu-inference, build with `VLLM_TARGET_DEVICE=tpu`, publish the wheel.

**What to do**:
1. Clone `vllm-project/vllm` at the commit that corresponds to the
   tpu-inference version you're targeting
2. `VLLM_TARGET_DEVICE=tpu python -m build --wheel`
3. This uses `requirements/tpu.txt` which includes `tpu-inference` (without
   CUDA deps)
4. Publish to GitHub Releases under `marin-community/vllm` or a new repo
5. Pin fork `tpu-inference` separately via `tool.uv.sources`
6. Bump jax to whatever version the fork needs (e.g., 0.9.2)
7. Set up CI to rebuild on push

**Dependency config**:
```toml
# lib/marin/pyproject.toml — replace vllm-tpu with custom wheel
vllm = [
    "vllm @ https://github.com/marin-community/vllm/releases/download/v0.18.0-tpu/vllm-0.18.0-py3-none-any.whl",
]

# root pyproject.toml
[tool.uv.sources]
tpu-inference = { git = "https://github.com/marin-community/tpu-inference.git", branch = "marin" }
```

**Pros**:
- Full control over everything: vllm version, jax version, tpu-inference
  version
- Can use latest jax (0.9.2+) immediately
- No waiting for Google
- Clean dependency tree with no overrides
- Can patch vllm itself if needed (e.g., for TPU-specific fixes)

**Cons**:
- Now maintaining **two fork builds** (vllm + tpu-inference)
- Need to determine which vllm commit is compatible with which
  tpu-inference commit — upstream doesn't document this clearly
- Build may fail — vllm's TPU build target may have undocumented
  dependencies or build requirements
- More CI infrastructure needed
- Version compatibility between vllm HEAD and tpu-inference HEAD is
  fragile — they're developed in lockstep by Google

**Risk**: Medium. The main risk is finding a compatible (vllm commit,
tpu-inference commit, jax version) triple. Google develops these in lockstep
but doesn't publish a compatibility matrix.

**Effort**: ~3-5 days for initial setup + CI, ongoing maintenance.

---

### Option C: Docker Sidecar (Bypass Python Dependency Entirely)

**What it is**: Run vllm in a separate Docker container using Google's official
`vllm/vllm-tpu` Docker image. Marin communicates via HTTP only. No Python
dependency on vllm or tpu-inference at all.

**What to do**:
1. Use `vllm/vllm-tpu:nightly-<tag>` Docker image (or build from
   `tpu-inference/docker/Dockerfile`)
2. Start vllm as a sidecar container via `docker run` on TPU workers
3. Talk to it via OpenAI-compatible HTTP API
4. This approach was already partially implemented in Marin
   (`lib/marin/src/marin/vllm/docker_server.py`)

**Dependency config**: None in pyproject.toml for vllm. Just configure the
Docker image tag via env var `MARIN_VLLM_DOCKER_IMAGE`.

**Pros**:
- Completely eliminates Python dependency conflicts
- Always uses Google's tested combination of vllm + tpu-inference + jax
- Marin's pyproject.toml stays clean
- Docker image versions are independently manageable
- No fork maintenance for vllm itself

**Cons**:
- **Cannot support weight hot-reload** (RL path needs `sync_weights` which
  requires in-process access to vllm engine internals)
- Cannot use Marin's fast-loading path (dummy load + fsspec + sync_weights)
  — the whole point of the fork
- Higher cold start (no abstract bootstrap, no fsspec streaming)
- Requires Docker-alongside-Docker infrastructure on TPU workers
- Adds operational complexity (container lifecycle, port management, log
  aggregation)

**Risk**: Low for eval workloads. **Not viable for RL workloads** that need
weight hot-reload.

**Effort**: ~2-3 days (partially implemented already).

**Verdict**: Good for evaluators. Does not solve the core fast-loading problem
that motivated the fork. Consider as a complementary approach for eval-only
workloads that don't need fast startup.

---

### Option D: Hybrid — Docker for Eval, Fork for RL/Fast-Loading

**What it is**: Use Docker sidecar for evaluation workloads (where cold start
doesn't matter much). Use the fork path (Option A or B) for RL and
fast-loading workloads.

**What to do**:
1. Keep Docker sidecar for `VllmTpuEvaluator` (already partially done)
2. Keep fork for `AsyncLLM` + `WorkerExtension` path used in RL
3. Marin already has backend selection logic in `VllmEnvironment`

**Pros**:
- Each workload type uses the best approach
- Eval path is dependency-conflict-free
- RL path keeps fast weight injection

**Cons**:
- Two serving architectures to maintain
- Complexity in backend selection logic
- Fork maintenance still needed for RL path

---

### Option E: Wait for Google (Not Recommended)

**What it is**: Wait for Google to release a `vllm-tpu` wheel with jax 0.9.x
support.

**Status**: Unknown timeline. `vllm-tpu==0.13.3` (jax 0.8.1) was released in
early 2026. Upstream `tpu-inference` main is on jax 0.9.2 but no matching
`vllm-tpu` release exists.

**Pros**: Zero maintenance on our end.

**Cons**: Unknown timeline. Could be weeks or months. Meanwhile we're stuck on
jax 0.8.1 and can't use upstream improvements.

**Verdict**: Not recommended as the primary strategy. Can be a secondary
strategy — when Google does release, we can simplify our fork.

---

## 8. Recommendation

### Immediate (this week): Option A — Stay on jax 0.8.1 with fork override

This is the pragmatic choice. The fork's patches don't use jax 0.9+ APIs.
The `override-dependencies` mechanism is already working. The smoke tests
pass. The remaining work is:

1. Validate that all fork patches work correctly on jax 0.8.1 (they should —
   `nnx.eval_shape`, `jax.random.key`, `jax.set_mesh` are all 0.8.x APIs)
2. Fix the async-native startup hang (the `VLLM_ENABLE_V1_MULTIPROCESSING=0`
   fix exists but needs cluster validation)
3. Port Levanter's async concurrency back into `streaming_weights.py` to
   close the fsspec throughput gap
4. Wire `abstract_load` config through evaluators
5. Clean up and merge to main

### Medium-term (when needed): Option B — Build your own vllm-tpu

Graduate to this when one of these triggers fires:
- You need a jax 0.9+ feature for a real use case (not just to be current)
- Google's `vllm-tpu` release cadence becomes a blocking constraint
- The fork's patches start hitting jax 0.8.1 incompatibilities

### Complementary: Option C — Docker sidecar for eval-only workloads

Keep the Docker sidecar path available for evaluation workloads that don't
need fast startup. It eliminates dependency conflicts entirely for those
workloads.

---

## Appendix: Full Timeline

| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-03-12 | Codex | Identified RunAI streamer bottleneck. Found existing fast-path machinery in RL codebase. Wrote initial design doc. |
| 2026-03-13 | Codex | Created `vllm_inprocess.py` — in-process vLLM runtime with fsspec loading. |
| 2026-03-17 | Claude | Fixed thread-safety bug in HTTP serving. Built queue-based generation server. 50/50 stress test passed. Fixed Iris log visibility (libtpu stdout redirect). |
| 2026-03-18 | Claude + Codex | Built async-native backend (`vllm_async.py`). Replaced queue-based server with `AsyncLLM` + `WorkerExtension` + standard OpenAI app. |
| 2026-03-18 | Claude | Added startup timing instrumentation. Discovered async-native startup hang. Narrowed to engine subprocess model loading. |
| 2026-03-19 | Claude | Applied `VLLM_ENABLE_V1_MULTIPROCESSING=0` fix for startup hang. Added pre-fork monkeypatching for deeper timing. |
| 2026-03-19 | Claude | 13 Iris debug runs narrowing the hang. Final narrowing: hang inside `vllm_get_model()` after "random weights" log, before function return. |
| 2026-03-20 | Claude | Wrote monkeypatch plan. Implemented `vllm_tpu_bootstrap_patch.py` with abstract model bootstrap, zero-state materialization, runtime monkeypatch infrastructure. |
| 2026-03-21 | Claude + Codex | **Fork created** at `marin-community/tpu-inference`. Three feature branches. CI setup. Codex reviewed (2 rounds, "no remaining correctness issue"). |
| 2026-03-21 | Claude | **Dependency hell begins.** Three integration attempts all fail (jax conflict, vllm-tpu incompatible, stock vllm CUDA deps). |
| 2026-03-21 | Claude | Implemented fsspec streaming weight loader inside the fork. Changed `MODEL_IMPL_TYPE` default from `"vllm"` to `"auto"`. |
| 2026-03-22 | Claude | Five bug-fix iterations on fork (architecture remapping, mesh context, nnx.List, abstract mesh, TP>1). Each required new wheel + Iris smoke test. |
| 2026-03-22 | Claude | **First successful fork smoke test**: `/ahmed/vllm-smoke-fsspec-v8`. Llama 8B, 32 GiB, correct output. |
| 2026-03-22 | Claude | Added `abstract_load` bootstrap mode for non-RL use. Smoke tests pass for both `abstract_load` and `default`. |
| 2026-03-22 | Claude | Head-to-head benchmarks: RunAI 6-18x faster on throughput, fsspec enables 70B at 32 GiB where RunAI OOMs. |
| 2026-03-22 | Claude | Post-mortem on original 53 MiB/s RunAI claim — was a confounded measurement. RunAI actual speed: 2.1 GiB/s same-region. |

---

## Appendix: Iris Jobs Run

### Fork Integration Debug (March 21)

| Job | Config | Result |
|-----|--------|--------|
| `/ahmed/fast-bootstrap-baseline` | default loading, v6e-4 | Failed: `vllm` binary not found |
| `/ahmed/fast-bootstrap-test` | dummy + tpu_bootstrap, v6e-4 | Failed: same |
| `/ahmed/fast-boot-baseline-v3` | default, v6e-4 | Failed: `tpu_inference not found` (empty install_requires) |
| `/ahmed/fast-boot-debug` | `import tpu_inference`, v6e-4 | Succeeded: import works |
| `/ahmed/fast-boot-debug2` | `from vllm.platforms.tpu import TpuPlatform` | Failed: cascading import |
| `/ahmed/fast-boot-debug3` | print source of tpu.py | Succeeded |
| `/ahmed/fast-boot-debug4` | full traceback | Failed: `No module named 'torchax'` |
| `/ahmed/fast-boot-baseline-v5` | default, v6e-4 (after adding deps) | Failed: `torchvision::nms` (CUDA torchvision on TPU) |
| `/ahmed/fast-boot-test-v5` | fast bootstrap, v6e-4 | Failed: same torchvision issue |

### Fork Smoke Tests (March 22)

| Job | Config | Result |
|-----|--------|--------|
| `/ahmed/vllm-smoke-fsspec` | abstract_load + fsspec, v6e-4, 16GB | Failed: MODEL_IMPL_TYPE=vllm forced PyTorch path → OOM |
| `/ahmed/vllm-smoke-fsspec-v2` | same, 32GB | Failed: MistralForCausalLM remap → JAX registry miss → PyTorch fallback → OOM |
| `/ahmed/vllm-smoke-fsspec-v4` | 32GB, with model_type fallback | Failed: model_type=transformer (not llama) |
| `/ahmed/vllm-smoke-fsspec-v5` | 32GB, with alias registration | Failed: nnx.eval_shape outside mesh context |
| `/ahmed/vllm-smoke-fsspec-v6` | 32GB, with mesh fix | Failed: abstract mesh not set |
| `/ahmed/vllm-smoke-fsspec-v7` | 32GB, with abstract mesh | Failed: Python list layers reject abstract data |
| `/ahmed/vllm-smoke-fsspec-v8` | 32GB, with nnx.List fix | **Succeeded** (342.2s, correct haiku output) |
| `/ahmed/vllm-smoke-abstract-load-v2` | abstract_load, v6e-4, 64GB | Succeeded (240.9s) |
| `/ahmed/vllm-smoke-default-v2` | default (baseline), v6e-4, 64GB | Succeeded (231.0s) |

### Benchmark Jobs (March 22)

| Job | Config | Result |
|-----|--------|--------|
| `/ahmed/vllm-runai-baseline-v7` | 8B, RunAI, v6e-4, 64GB | Succeeded: 135.8s total, 7.0s download |
| `/ahmed/vllm-fsspec-v9` | 8B, fsspec, v6e-4, 32GB | Succeeded: 342.2s total, 126.3s download |
| `/ahmed/vllm-70b-baseline-v4` | 70B, RunAI, v5p-8, TP=4, 200GB | Succeeded: 481s total, 185s download |
| `/ahmed/vllm-70b-fsspec-v3` | 70B, fsspec, v5p-8, TP=4, 200GB | Succeeded: 1482s total, 1174s download |
| `/ahmed/vllm-70b-baseline-32g` | 70B, RunAI, v5p-8, TP=4, 32GB | **OOM killed** |
| `/ahmed/vllm-70b-fsspec-32g` | 70B, fsspec, v5p-8, TP=4, 32GB | **Succeeded**: 1447s, 15.5 GiB peak RSS |

---

## Appendix: Benchmark Results

### 8B (Llama 3.1 8B Instruct, v6e-4, TP=1)

| Metric | RunAI (baseline) | fsspec (fork) |
|--------|-----------------|---------------|
| Weight download | **7.0s** (2.1 GiB/s) | 126.3s (121 MiB/s) |
| Total runtime | **135.8s** | 342.2s |
| Memory needed | 64 GiB | **32 GiB** |
| Peak host RSS | ~15 GiB (full model in RAM) | **12.8 GiB** |

### 70B (Llama 3.3 70B Instruct, v5p-8, TP=4)

| Metric | RunAI (baseline) | fsspec (fork) |
|--------|-----------------|---------------|
| Weight download | **185s** (727 MiB/s) | 1174s (115 MiB/s) |
| Total runtime | **481s** | 1482s |
| Peak host RSS | ~131 GiB (full model) | **15.5 GiB** |
| Works at 32 GiB host RAM | No (OOM killed) | **Yes** |

### Why fsspec Is Slower (and How to Fix It)

The fsspec implementation is **fully sequential**: one HTTP request at a time,
no shard parallelism, no I/O overlap. The concurrency from Levanter's original
code (`asyncio.gather()` + `ThreadPoolExecutor` + semaphore for 4 concurrent
chunks) was stripped during porting to the fork.

| Factor | Current fsspec | RunAI | Potential fix |
|--------|---------------|-------|---------------|
| Chunk concurrency | 1 | 4-16 | Port back Levanter's async gather |
| Shard parallelism | Sequential | All at once | Overlap download + injection |
| I/O + CPU overlap | None | Pipelined | Async pipeline |
| **Expected after fix** | **~500+ MiB/s** | 2.1 GiB/s | Closes ~4x of the gap |

### The Real Value Proposition

fsspec's advantage is **bounded host RAM**, not speed. For 70B models:
- RunAI needs 131+ GiB host RAM (loads all shards simultaneously) → requires
  200 GiB+ workers
- fsspec needs ~15 GiB (one shard at a time) → works on 32 GiB workers
- This enables running 70B inference on cheaper, smaller TPU worker configs

---

## Appendix: Key Files Reference

### Fork (`marin-community/tpu-inference`)

| File | Purpose |
|------|---------|
| `tpu_inference/models/common/model_loader.py` | Bootstrap config, abstract model, architecture routing |
| `tpu_inference/models/jax/streaming_weights.py` | fsspec weight iterator |
| `tpu_inference/models/jax/utils/weight_utils.py` | jax.Array dispatch + lazy torchax init |
| `tpu_inference/runner/tpu_runner.py` | Bootstrap-safe sampling RNG |
| `requirements.txt` | Added fsspec, gcsfs |
| `tests/models/common/test_model_loader.py` | Config parsing, dispatch tests |
| `tests/models/jax/test_streaming_weights.py` | Iterator unit tests |

### Marin (`vllm_load_fast` branch)

| File | Purpose |
|------|---------|
| `lib/marin/src/marin/inference/vllm_async.py` | Async-native backend, startup instrumentation |
| `lib/marin/src/marin/inference/vllm_inprocess.py` | Eligibility, bootstrap staging, shared helpers |
| `lib/marin/src/marin/inference/vllm_server.py` | Backend selection, MODEL_IMPL_TYPE default |
| `lib/marin/src/marin/rl/environments/inference_ctx/inflight/worker.py` | WorkerExtension, timing hooks |
| `lib/marin/pyproject.toml` | `vllm-tpu==0.13.3` dependency |
| `pyproject.toml` (root) | `override-dependencies` for fork wheel |
| `docs/dev-guide/forking-policy.md` | Fork strategy documentation |
| `tests/vllm/test_vllm_inprocess_backend.py` | 13 unit tests |

### External

| Resource | Purpose |
|----------|---------|
| `marin-community/tpu-inference` (GitHub) | The fork repo |
| `vllm-project/tpu-inference` (GitHub) | Upstream |
| `vllm-project/vllm` (GitHub) | vllm source (build with `VLLM_TARGET_DEVICE=tpu`) |
| `vllm-tpu` (PyPI) | Google's pre-built TPU vllm wheel |

### Agent Logbooks and Project Docs

| File | Content |
|------|---------|
| `.agents/logbook/tpu_inference_fork.md` | Fork creation, packaging attempts, all Iris jobs |
| `.agents/logbook/claude_vllm_refactor.md` | Async-native startup hang investigation (13 runs) |
| `.agents/logbook/codex_vllm_refactor.md` | Codex async refactor, timing instrumentation |
| `.agents/projects/vllm_fast_loading.md` | Full project history (695 lines) |
| `.agents/projects/vllm_fast_loading_unified_plan.md` | Detailed fork implementation plan |
| `.agents/projects/tpu_inference_minimal_fast_loading_plan.md` | Minimal fork change spec |
| `.agents/projects/vllm_async_refactor.md` | Async serving architecture design |
| `.agents/projects/vllm_async_monkeypatch_plan.md` | Monkeypatch bootstrap plan |
| `.agents/projects/vllm_serving_analysis.md` | Canonical map of all vLLM paths in Marin |
| `.agents/projects/vllm-docker.md` | Docker sidecar design |
| `.agents/projects/runai_vs_fsspec_benchmark.md` | Benchmark methodology and results |

#### 2026-04-01 22:56 PDT - TinyLlama 1.1B TPU smoke submitted on smallest idle slice

- Goal: validate that the new `main`-branch `vllm` and `tpu-inference` wheels do more than import on CPU by bringing up the smallest practical native TPU vLLM server.
- Chosen compute: `v6e-4` in `us-east1-d` because it was the smallest idle ready slice visible in Iris at submission time.
- Chosen model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as the lightest plausible HF causal LM path for a first native TPU smoke.
- Iris job: `/ahmed/vllm-smoke-tinyllama-main-v3`
- Exact command:

```bash
uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v6e-4 \
  --zone us-east1-d \
  --memory 32GB \
  --disk 100GB \
  --job-name vllm-smoke-tinyllama-main-v3 \
  --extra eval \
  --extra tpu \
  --extra vllm \
  -- python -m marin.inference.vllm_smoke_test \
    --local \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --mode native \
    --max-model-len 1024 \
    --use-completions \
    --prompt "Write a short haiku about TPUs."
```

- Submission result: Iris accepted the job and started dependency sync on the worker.
- Immediate next check: wait for native server startup logs, then capture either the generated completion or the first blocking crash.

#### 2026-04-01 23:01 PDT - TinyLlama 1.1B native TPU smoke succeeded on `v6e-4`

- Final Iris state: `JOB_STATE_SUCCEEDED`
- Job: `/ahmed/vllm-smoke-tinyllama-main-v3`
- TPU: `v6e-4` in `us-east1-d`
- End-to-end smoke timing from the script: `[run 1/1] 245.4s`
- The task exited cleanly with code `0`, so the native `VllmEnvironment` reached readiness, served the request, and shut down normally.

Key evidence from the native vLLM subprocess logs:
- vLLM version in use: `0.0.0.dev20260402+9e3db15a7`
- `tpu-inference` loaders registered successfully, including `tpu_streaming_loader`
- TPU engine initialized on `v6e-4`
- Weight download for `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: `2.115s`
- Total time to load model weights from storage to TPU: `10.85s`
- TPU precompile ran through sampling, gather-logprobs, KV cache init, and backbone shapes up to `num_tokens=1024`

Caveat:
- The final printed completion body was empty. This is not a startup failure, because the job still completed successfully after the request returned, but it is worth a follow-up smoke with a different prompt or `/chat/completions` to confirm output quality rather than just server bring-up.

Conclusion:
- The new `main`-branch Marin forks for `vllm` and `tpu-inference`, together with the JAX `0.9.2` uplift and local packaging changes, are sufficient to boot a native TPU vLLM server and complete a request on a small `v6e-4` slice.

#### 2026-04-01 23:08 PDT - Functional follow-up smoke queued after empty TinyLlama output

- Problem with the first functional smoke: server startup succeeded, but the final completion body was empty, which is not a convincing end-to-end pass.
- Immediate mitigation: make `vllm_smoke_test.py` print `output={output!r}` so whitespace-only or empty generations are explicit in Iris logs.
- New hypothesis: the empty TinyLlama result was caused by the specific model/API combination (`Chat` model with `/completions`), not by the `main`-branch packaging stack.
- Follow-up model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Reason for this choice: open weights, small footprint, `LlamaForCausalLM` architecture in `config.json`, and better alignment with `/chat/completions`.
- Planned command:

```bash
uv run --no-sync iris --config=lib/iris/examples/marin.yaml job run \
  --tpu v6e-4 \
  --zone us-east1-d \
  --memory 32GB \
  --disk 100GB \
  --job-name vllm-smoke-smollm2-main-v1 \
  --extra eval \
  --extra tpu \
  --extra vllm \
  -- python -m marin.inference.vllm_smoke_test \
    --local \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --mode native \
    --max-model-len 1024 \
    --prompt "Reply with exactly TPU_SMOKE_OK and nothing else."
```

#### 2026-04-01 23:12 PDT - SmolLM2 functional TPU smoke passed with visible output

- Job: `/ahmed/vllm-smoke-smollm2-main-v1`
- Final state: `JOB_STATE_SUCCEEDED`
- TPU: `v6e-4` in `us-east1-d`
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- API path: `/chat/completions` (the smoke did not pass `--use-completions`)
- Prompt: `Reply with exactly TPU_SMOKE_OK and nothing else.`
- End-to-end timing from the script: `[run 1/1] 220.5s`
- Logged response: `output='"TPU_SMOKE_OK"'`

Interpretation:
- The `main`-branch `marin-community/vllm` wheel plus the `main`-branch `marin-community/tpu-inference` wheel are not just importable and bootable; they can complete a real native TPU chat request on a small `v6e-4` slice.
- The earlier empty TinyLlama result is best explained by the model/API pairing used in that run rather than a packaging or startup regression.

Useful secondary observation:
- The smoke-script change to print `output={output!r}` turned an ambiguous blank line into a definitive pass/fail signal and should stay.

#### 2026-04-01 23:16 PDT - Consolidated session summary

This section summarizes the full set of actions completed in this session so the logbook is self-contained.

GitHub fork state:
- `marin-community/tpu-inference`
  - reset to track upstream `vllm-project/tpu-inference` `main`
  - default branch set to `main`
  - minimal `marin-release.yml` workflow added on `main`
  - current release wheel:
    - commit/tag lineage: `4cfc17bc`
    - URL: `https://github.com/marin-community/tpu-inference/releases/download/marin-4cfc17bc/tpu_inference-0.0.0.dev20260402+4cfc17bc-py3-none-any.whl`
- `marin-community/vllm`
  - reset to track upstream `vllm-project/vllm` `main`
  - deleted the temporary `marin` branch so `main` stays as close to upstream as possible
  - minimal `marin-release.yml` workflow added on `main`
  - current release wheel:
    - commit/tag lineage: `9e3db15a7`
    - URL: `https://github.com/marin-community/vllm/releases/download/marin-9e3db15a7/vllm-0.0.0.dev20260402+9e3db15a7-py3-none-any.whl`

Local Marin packaging and runtime changes:
- `lib/marin/pyproject.toml`
  - uplifted the JAX baseline from `0.8.0` to `0.9.2`
  - uplifted Torch/Torchvision from `2.9.0`/`0.24.0` to `2.10.0`/`0.25.0`
  - replaced `vllm-tpu` with direct wheel pins to the new `marin-community/vllm` and `marin-community/tpu-inference` `main` releases
  - widened `datasets` from `<4.0.0` to `<5.0.0`
- root `pyproject.toml`
  - added an explicit `override-dependencies` pin for the `marin-community/tpu-inference` `main` release wheel
  - widened the root `datasets` cap to `<5.0.0`
- `lib/levanter/pyproject.toml`
  - widened `gcsfs` and `fsspec[http]` caps to allow the newer TPU stack
  - replaced `jax[tpu]>=0.8.0` with explicit `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38`
- `lib/fray/pyproject.toml`
  - aligned `fray_tpu_test` to the same explicit JAX/TPU versions
- native vLLM launch paths
  - removed the old `VLLM_NATIVE_PIP_PACKAGES = ("vllm-tpu",)` path from `lib/marin/src/marin/inference/vllm_server.py`
  - changed evaluators and smoke jobs to use extras-based environment construction:
    - native mode: include `vllm` extra
    - docker mode: keep plain `eval`/`harbor` + `tpu`
  - touched:
    - `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
    - `lib/marin/src/marin/evaluation/evaluators/simple_evaluator.py`
    - `lib/marin/src/marin/evaluation/evaluators/harbor_evaluator.py`
    - `lib/marin/src/marin/inference/vllm_smoke_test.py`
- smoke instrumentation
  - changed `vllm_smoke_test.py` to print `output={output!r}` so empty or whitespace-only generations are visible in job logs

Locking and environment validation:
- `uv lock` now succeeds with the new stack
- resolved package set includes:
  - `jax==0.9.2`
  - `jaxlib==0.9.2`
  - `libtpu==0.0.38`
  - `vllm==0.0.0.dev20260402+9e3db15a7`
  - `tpu-inference==0.0.0.dev20260402+4cfc17bc`
  - `torch==2.10.0`
  - `torchvision==0.25.0`
  - `triton==3.6.0`

Executed validation runs:
- CPU import smoke:
  - job: `/ahmed/vllm-tpu-main-import-cpu-v1`
  - result: `JOB_STATE_SUCCEEDED`
  - imported `vllm`, `tpu_inference`, and `TpuPlatform`
  - observed versions:
    - `vllm`: `0.0.0.dev20260402+9e3db15a7`
    - `tpu-inference`: `0.0.0.dev20260402+4cfc17bc`
- first TPU smoke:
  - job: `/ahmed/vllm-smoke-tinyllama-main-v3`
  - model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - result: `JOB_STATE_SUCCEEDED`
  - native TPU server booted and served a request, but the logged completion body was empty
- second TPU smoke:
  - job: `/ahmed/vllm-smoke-smollm2-main-v1`
  - model: `HuggingFaceTB/SmolLM2-360M-Instruct`
  - result: `JOB_STATE_SUCCEEDED`
  - logged response: `output='"TPU_SMOKE_OK"'`

Bottom line:
- The upstream-aligned `main` forks are working as the Marin source of truth for TPU `vllm` and `tpu-inference`.
- The local workspace is now wired to those `main` wheels.
- Import on CPU passes.
- Native TPU serving on a small `v6e-4` slice passes with a real generated response.

#### 2026-04-03 15:08 PDT - Why `lm_eval`, `simple`, and `harbor` changed

These evaluator edits were not made to change evaluation semantics. They were
necessary because native TPU eval jobs used to rely on a special ad hoc install
path for `vllm-tpu`, and that path no longer exists in the fork-based setup.

Old behavior:
- native evaluator jobs launched with extras like `("eval", "tpu")` or
  `("harbor", "tpu")`
- those same jobs also passed `pip_packages=("vllm-tpu",)` through
  `VLLM_NATIVE_PIP_PACKAGES`
- this worked only because native workers were separately pip-installing the
  Google TPU wheel outside Marin's normal dependency graph

New behavior:
- `lib/marin/src/marin/inference/vllm_server.py` no longer defines
  `VLLM_NATIVE_PIP_PACKAGES = ("vllm-tpu",)`
- native evaluator jobs now request the repo `vllm` extra directly:
  - `lm_evaluation_harness_evaluator.py`: native uses `("eval", "tpu", "vllm")`
  - `simple_evaluator.py`: native uses `("eval", "tpu", "vllm")`
  - `harbor_evaluator.py`: native uses `("harbor", "tpu", "vllm")`
- docker-mode jobs intentionally do not add the `vllm` extra, because the
  worker does not need in-process vLLM when the server is running in Docker

Why this was required:
- `launch_evaluate_with_ray(...)` builds a remote worker environment from
  `extras` and `pip_packages`
- Fray installs exactly those packages into the remote runtime environment
- once `vllm-tpu` was removed as the special pip-installed package, native TPU
  evaluator jobs needed some other way to get vLLM onto the worker
- the correct replacement is to include Marin's `vllm` extra, which now pins
  the forked `marin-community/vllm` and `marin-community/tpu-inference` wheels

Practical effect:
- native eval/smoke jobs keep working under the new fork-based dependency model
- docker eval jobs avoid an unnecessary in-process vLLM install
- the change is about dependency plumbing for remote workers, not about how the
  evaluators score examples or call model APIs
