# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject manual so the next agent stays aligned.
- Consult the subproject manuals when working in submodule trees:
  * `lib/levanter/AGENTS.md` for Levanter-specific conventions.
  * `lib/marin/AGENTS.md` for Marin-specific conventions
  * `lib/iris/AGENTS.md` for Iris-specific conventions

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- When making significant changes to Grug/Grugformer, follow [change_grug.md](docs/recipes/change_grug.md).
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `./infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Separation of responsibilities: when a change introduces a new subsystem (e.g., serving/inference, data access, evaluation), encapsulate lifecycle/configuration in a dedicated module and have callers depend on the interface rather than re-implementing setup/teardown details.
- Disprefer internal mutation of function arguments, especially config dataclasses; prefer returning a modified copy (e.g., via `dataclasses.replace`) so call sites remain predictable and side effects are explicit.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Do not use `from future import ...` statements.
- Document public APIs with concise Google-style docstrings.

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.

### Deprecation

**NO BACKWARD COMPATIBILITY**: Do NOT add deprecation warnings, fallback paths, or compatibility shims. Update all call sites instead. Only add backward compatibility if the user explicitly requests it.

## Comments

You write detailed comments when appropriate to describe code behavior as a
whole, e.g. at the module or class level, or when describing some subtle
behavior.

You don't generate comments that merely restate the code, e.g.

<bad>
     # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
</bad>

<good>
# We have found that each instance of a FlightServer can provide approximately 1GB/s
# of throughput. As our typical VMs run with 200Gbps NICs, running 16 parallel servers
# should be sufficient to saturate the network.
</good>

## Planning

- When planning, you produce detailed plans including code snippets.
- You ask questions up front when building a plan instead of guessing.
- When a request feels too large for one pass, capture a plan (for example in `.agents/projects/` when the subproject provides one) before pausing.

## Testing

- Always fix tests if you broke them.
- Do not fix tests by relaxing tolerances or hacking around them.
- Avoid “tautological” tests that merely restate implementation logic as asserts; prefer tests that validate externally-observable behavior, integration points, or realistic failure modes.
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject guides for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.

PREFER:

- Integration style tests which exercise behavior and test the output

DO NOT:

- Create tests which validate obvious features: if a type exists, a constant has a value, etc.


## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`

## Ray Run Notes

- For Ray/TPU runs, prefer `TMPDIR=/tmp` and `RAY_TMPDIR=/tmp`.
- Avoid setting `UV_CACHE_DIR` in Ray launch commands unless you have a specific reason (use uv's default cache).
- For a brand new run, omit `--force-run-failed`; add it only when you explicitly want to rerun previously-failed steps.

### Ray Token Auth (Central1)

If you need to authenticate to the Ray dashboard (token auth):

```bash
make get_ray_auth_token
uv run scripts/ray/cluster.py --cluster us-central1 auth
```

### Ray Job Monitoring (Dashboard + Logs)

When a sweep driver job is submitted, the terminal may print `Connection to <ip> closed.` — that is normal (Ray job continues on-cluster). To check status and confirm evals are running:

1) **Open the Ray dashboard tunnel**

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1.yaml dashboard
```

This prints the local dashboard port mapping (commonly `8266`). Use the printed port rather than assuming `8265`/`8266`.

2) **List jobs + find your sweep driver**

Ray dashboard: open `http://127.0.0.1:<PORT>/#/jobs`

3) **Programmatic status checks via Jobs API (includes token auth)**

Ray’s local dashboard uses token auth (usually stored at `~/.ray/auth_token`). These snippets work even when the UI is slow/unreliable.

List recent jobs:

```bash
python - <<'PY'
import json, pathlib, urllib.request
token_path = pathlib.Path.home() / ".ray" / "auth_token"
token = token_path.read_text().strip() if token_path.exists() else None
base = "http://127.0.0.1:8266"  # replace with printed dashboard port
req = urllib.request.Request(base + "/api/jobs/")
if token:
    req.add_header("Authorization", f"Bearer {token}")
with urllib.request.urlopen(req, timeout=10) as r:
    jobs = json.loads(r.read().decode())
for j in sorted(jobs, key=lambda x: x.get("start_time", 0) or 0, reverse=True)[:25]:
    print(j.get("job_id"), j.get("status"), (j.get("entrypoint") or "")[:120])
PY
```

4) **One-command monitoring + optional auto-resubmit**

For long sweeps, it’s useful to have an hourly heartbeat that also summarizes Ray task state (RUNNING vs pending). This helper prints a compact line per job and can optionally re-run a user-provided “resume” command if the job leaves `RUNNING`:

```bash
uv run python scripts/ray/monitor_and_resubmit.py --config infra/marin-us-central1.yaml --job-id 3c010000 --job-id 3d010000 --job-id 47010000 --job-id 68010000 --watch --interval-sec 3600
```

Auto-resubmit example (replace with the exact `uv run python -m marin.run.ray_run ... --resume-from-wandb --force-run-failed` command you want):

```bash
uv run python scripts/ray/monitor_and_resubmit.py --config infra/marin-us-central1.yaml --job-id 68010000 --watch --interval-sec 3600 --resubmit-cmd 'echo \"TODO: paste resume command here\"'
```

Fetch a specific job’s entrypoint/status:

```bash
python - <<'PY'
import json, pathlib, urllib.request
job_id = "e2000000"  # replace
token_path = pathlib.Path.home() / ".ray" / "auth_token"
token = token_path.read_text().strip() if token_path.exists() else None
base = "http://127.0.0.1:8266"  # replace with printed dashboard port
req = urllib.request.Request(base + f"/api/jobs/{job_id}")
if token:
    req.add_header("Authorization", f"Bearer {token}")
with urllib.request.urlopen(req, timeout=10) as r:
    data = json.loads(r.read().decode())
data = data.get("result", data)  # Ray version dependent
print("status:", data.get("status"))
print("entrypoint:", data.get("entrypoint"))
PY
```

Confirm training/evals in logs (look for `Progress on:train`, `Progress on:eval`, `levanter.eval`, and checkpoint saves):

```bash
python - <<'PY'
import json, pathlib, re, urllib.request
job_id = "e2000000"  # replace
token_path = pathlib.Path.home() / ".ray" / "auth_token"
token = token_path.read_text().strip() if token_path.exists() else None
base = "http://127.0.0.1:8266"  # replace with printed dashboard port
req = urllib.request.Request(base + f"/api/jobs/{job_id}/logs")
if token:
    req.add_header("Authorization", f"Bearer {token}")
with urllib.request.urlopen(req, timeout=30) as r:
    logs = json.loads(r.read().decode()).get("logs", "")
tail = logs.splitlines()[-300:]
want = re.compile(r"Progress on:(train|eval)|levanter\\.eval|Saving checkpoint|NaN|diverg|overflow", re.I)
for ln in tail:
    if want.search(ln):
        print(ln)
PY
```

Notes:
- TPU slice instability often shows up as `ActorDiedError` / `scaled to 0 actors`; a driver job can remain `RUNNING` while individual sweep runs crash/retry.
- If you need W&B API queries locally, **use `uv run python`** so you import the real `wandb` package (system Python may shadow it).
- If you hit `ValueError: numpy.dtype size changed` (pandas/numpy binary mismatch during `datasets` import), pass `PIP_IGNORE_INSTALLED=1` in the Ray runtime env (`--env_vars PIP_IGNORE_INSTALLED 1`).
- Prefer passing `--no_wait` to `python -m marin.run.ray_run` so the submit command returns immediately (use the dashboard/logs to follow progress).

#### Interpreting Failures (run_step vs train job)

When debugging “generic” failures, check *where* the stack trace ends:

- **Executor / run_step driver died**: stack traces ending in `marin/execution/executor.py` / `executor_runner.wait()` with messages about the *step runner* dying typically mean the top-level watcher got preempted. This is hard to make perfectly reliable without scheduling the watcher on a non-preemptible node type.
  - Mitigation: resubmit the driver with `--force-run-failed --resume-from-wandb` so it can reacquire step locks and continue. If this is happening frequently, treat it as an infra issue (watcher placement), not a model bug.
- **Training job died**: stack traces mentioning `run_levanter_train_lm` or `fray/cluster/ray/tpu/execution.py:run_on_pod_ray` are “lower down” and are usually fixable/retriable (preemption, slice health checks, transient worker env issues).
  - Example transient env failure: `numpy.dtype size changed, may indicate binary incompatibility` during `datasets` → `pandas` import.

#### Common Failure Signatures Seen in OLMoE Sweeps (v5p-16 multislice)

These are real examples observed in recent OLMoE size sweeps (SwiGLU / bilinear / moe-stab). They often look scary in logs but are frequently *recoverable* if the sweep driver remains alive and is re-submitted with resume enabled.

1) **TPU slice actor pool scaled to 0**

Usually indicates the slice actor died before it started (node heartbeat missed / health check failed / capacity churn):

```text
Exception: v5p-16 slices actor pool wanted to scale to 1 actors, but scaled to 0 actors instead
... ActorDiedError: The actor died unexpectedly before finishing this task.
... The actor died because its node has died. Node Id: ...
... the actor's node was terminated unexpectedly: health check failed due to missing too many heartbeats
```

This comes from `lib/fray/src/fray/cluster/ray/tpu/execution.py` (`scale_multislice` / `_scale_actor_pool` / `_add_members_to_actor_pool`). With the current patches it is treated as transient and the runner backs off and retries.

2) **`OwnerDiedError` while waiting for TPU results**

This is another common “preemption-like” failure mode: a Ray worker that created an `ObjectRef` died.

```text
ray.exceptions.OwnerDiedError: Failed to retrieve object ...
The object's owner has exited. ... Check cluster logs (/tmp/ray/session_latest/logs/*...* at IP address ...) ...
Retrying after preemption in 10s
```

This is not usually a model bug; it indicates churn on TPU worker nodes.

3) **Ray runtime env pip install failed**

This often shows up as a `SubprocessCalledProcessError` from Ray’s runtime-env agent, and can prevent a sweep step from starting (some steps may appear missing or stuck as a result).

```text
ray._private.runtime_env.utils.SubprocessCalledProcessError: Run cmd failed
Command '[.../virtualenv/bin/python', '-m', 'pip', 'install', '-r', '...requirements.txt', ...]' returned non-zero exit status 1.
```

If you see this frequently, check disk pressure (next section). Even though we set `PIP_NO_CACHE_DIR=1` and `PIP_IGNORE_INSTALLED=1`, Ray still has to download/install many wheels repeatedly across nodes.

4) **Ray session disk pressure (`/tmp/ray` over 95% full)**

This is a *big* red flag for “random” failures (runtime env setup, object spilling, worker crashes):

```text
file_system_monitor.cc:116: /tmp/ray/session_... is over 95% full, available space: <~few GB>; capacity: <~100 GB>.
Object creation will fail if spilling is required.
```

If this persists:
- Expect more `OwnerDiedError`, runtime env failures, and actor pool scale-to-0.
- Mitigation usually requires infra action (restart nodes to clear `/tmp/ray`, increase disk size, or configure Ray temp dirs / logs to land on a larger disk).

#### Resume Strategy When Things Flap

When the driver job itself dies or many steps crash due to the failure modes above, the intended workflow is:

- Keep W&B grouping stable (`--wandb-group <GROUP>`) and resubmit the *driver* with `--resume-from-wandb --force-run-failed`.
- This reuses the original `trainer.id`/checkpoint output directories for already-started steps, so the rerun picks up from checkpoints instead of duplicating work.

#### Sweep Resume (OLMoE sizes)

For `experiments.speedrun.olmoe_1b7b_size_speedrun_sweep`:

- A run only *actually* resumes if there is a checkpoint under `gs://marin-us-central1/checkpoints/speedrun/<trainer.id>/checkpoints/step-*/metadata.json`.
- The sweep’s `--resume-from-wandb` maps each sweep base-name (from `build_run(..., run_suffix=None)`) back to an existing `trainer.id` (output dir). If a resume target has no checkpoints, logs will show `No checkpoints found in gs://.../checkpoints` and that specific run restarts from scratch.
- The current resume logic prefers the stable configured tracker name in W&B config (e.g. `trainer.tracker.name`) and falls back to parsing older `osz_...` ids.

Quick confirmation that a driver job resumed instead of restarting:

```bash
python - <<'PY'
import json, pathlib, re, urllib.request
job_id = "e2000000"  # replace
token = (pathlib.Path.home() / ".ray" / "auth_token").read_text().strip()
base = "http://127.0.0.1:8266"  # replace with printed dashboard port
req = urllib.request.Request(base + f"/api/jobs/{job_id}/logs")
req.add_header("Authorization", f"Bearer {token}")
logs = json.loads(urllib.request.urlopen(req, timeout=60).read().decode()).get("logs", "")
want = re.compile(r"Discovered latest checkpoint|Loading checkpoint from|Resuming training from step|No checkpoints found", re.I)
for ln in logs.splitlines():
    if want.search(ln):
        print(ln)
PY
```

If the dashboard port is unclear (tunnels sometimes switch between `8265`/`8266`), probe locally:

```bash
python - <<'PY'
import pathlib, urllib.request
token_path = pathlib.Path.home() / ".ray" / "auth_token"
token = token_path.read_text().strip() if token_path.exists() else None
for port in (8265, 8266, 8267):
    base = f"http://127.0.0.1:{port}"
    req = urllib.request.Request(base + "/api/jobs/")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        urllib.request.urlopen(req, timeout=2).read(50)
        print("OK", port)
    except Exception:
        pass
PY
```

Example (size sweep on Central1):

```bash
TMPDIR=/tmp RAY_TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars RAY_TMPDIR /tmp --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_v5p32 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python experiments/speedrun/olmoe_1b7b_size_speedrun_sweep.py --dataset nemotron_cc --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64
```

### OLMoE 1B7B vs Llama Dense 1.4B (100B DCLM baseline-only)

- W&B:
  - Put both runs in the same project: `WANDB_PROJECT=olmoe_vs_dense_dclm100b`.
  - Use a single explicit group for comparison: `olmoe1b7b_vs_llama1_4b_100b_dclm`.
  - “model supports 8192” in logs refers to the model max; the actual training length is controlled by `--seq-len`.
- Dashboard:
  - The Ray dashboard tunnel can be on `http://127.0.0.1:8265` or `http://127.0.0.1:8266` depending on which tunnel is currently running.
  - Confirm `--seq-len`/`--global-batch-size` via the job entrypoint shown in the “Jobs” tab.

Commands (v5p-32, `seq=4096`, `global_bs=256`, 100B tokens, Levanter-native validation enabled):

```bash
GROUP="olmoe1b7b_vs_llama1_4b_100b_dclm" && SUFFIX="$(git rev-parse --short HEAD)_t$(date +%Y%m%d_%H%M%S)" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "olmoe_vs_dense_dclm100b" --env_vars WANDB_GROUP "$GROUP" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR "gs://marin-us-central1/jax-cache/olmoe_vs_dense_dclm100b_v5p32" --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_nemotron_40b --model olmoe_1b7b --dataset dclm_baseline_only --tpu-type v5p-32 --seq-len 4096 --global-batch-size 256 --token-target 100000000000 --single-checkpoint --checkpoint-save-minutes 60 --use-default-validation --wandb-group "$GROUP" --run-suffix "olmoe1b7b_dclmbo_100b_s4096_b256_v5p32_${SUFFIX}"
```

```bash
GROUP="olmoe1b7b_vs_llama1_4b_100b_dclm" && SUFFIX="$(git rev-parse --short HEAD)_t$(date +%Y%m%d_%H%M%S)" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "olmoe_vs_dense_dclm100b" --env_vars WANDB_GROUP "$GROUP" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR "gs://marin-us-central1/jax-cache/olmoe_vs_dense_dclm100b_v5p32" --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_nemotron_40b --model llama_1_4b --dataset dclm_baseline_only --tpu-type v5p-32 --seq-len 4096 --global-batch-size 256 --token-target 100000000000 --single-checkpoint --checkpoint-save-minutes 60 --use-default-validation --wandb-group "$GROUP" --run-suffix "llama_dense_1p4b_dclmbo_100b_s4096_b256_v5p32_${SUFFIX}"
```

Example (OLMoE sizes sweep, composite mixture, eval during+after, 2 terminals, shared W&B group):

- Uses a deterministic W&B group based on the current git commit so you don't need to coordinate a timestamp across terminals.
- Run uniqueness comes from `--run-suffix`.
- Defaults to 4 sizes × 4 LR multipliers = 16 runs per command.

SwiGLU (regular experts), BS=64:

```bash
GROUP="olmoe_sizes_v5p32_b64_$(git rev-parse --short HEAD)" && SUFFIX="t$(date +%Y%m%d_%H%M%S)_$$" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_swiglu_b64 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_size_speedrun_sweep --dataset nemotron_dclm_fineweb_10b --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64 --eval-suite core --eval-suite-mode both --steps-per-task-eval 500 --wandb-group "$GROUP" --run-suffix "swiglu_${SUFFIX}" --sizes olmoe_1b7b
```

Bilinear experts, BS=64 (same group automatically):

```bash
GROUP="olmoe_sizes_v5p32_b64_$(git rev-parse --short HEAD)" && SUFFIX="t$(date +%Y%m%d_%H%M%S)_$$" && source .venv/bin/activate && TMPDIR=/tmp uv run python -m marin.run.ray_run --cluster "infra/marin-us-central1.yaml" --extra tpu --no_wait --env_vars WANDB_MODE online --env_vars WANDB_API_KEY "${WANDB_API_KEY:-}" --env_vars WANDB_ENTITY "${WANDB_ENTITY:-}" --env_vars WANDB_PROJECT "${WANDB_PROJECT:-}" --env_vars HF_TOKEN "${HF_TOKEN:-}" --env_vars PIP_NO_CACHE_DIR 1 --env_vars PIP_IGNORE_INSTALLED 1 --env_vars TMPDIR /tmp --env_vars JAX_COMPILATION_CACHE_DIR gs://marin-us-central1/jax-cache/olmoe_sizes_bilinear_b64 --env_vars JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS 0 --env_vars JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES -1 --env_vars JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION 1 -- python -m experiments.speedrun.olmoe_1b7b_size_speedrun_sweep --dataset nemotron_dclm_fineweb_10b --tpu-type v5p-32 --seq-len 4096 --global-batch-size 64 --eval-suite core --eval-suite-mode both --steps-per-task-eval 500 --bilinear-mlp --wandb-group "$GROUP" --run-suffix "bilinear_${SUFFIX}" --sizes olmoe_1b7b
```

> This file will be expanded as agent workflows and best practices evolve.
