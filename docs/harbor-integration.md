# Harbor evaluation

Marin's evaluation launcher starts the model, runs Harbor and saves results in FineStore.
It stops inference when the batch ends. The candidate is the model under evaluation.
An optional judge model is shared across the batch.
Iris runs the evaluator worker. Shellbox runs task commands and graders.

Harbor supports `local` (bubblewrap, the default), `daytona`, `qemu` and `shellsim`.
See [Running Evaluations with Marin](tutorials/run-lm-evals.md) to choose a model and suite.
Preflight checks the policy and model settings before allocating evaluation resources.

Before launching:

- Local execution needs prepared images and working bubblewrap namespaces. Shellbox
  also needs a writable part of the worker's cgroup v2 tree to contain task processes.
  Launcher preflight cannot check that permission on a remote worker.
- BixBench and BFCL standalone Pi policies need prepared Pi images. Changed benchmark
  policies need a new approved cohort version. See [verified cohorts](https://github.com/marin-community/marin/blob/main/eval_policy/README.md).
  Historical tasks that install Pi at runtime fail preflight.
- Codex and Claude Code fail Marin preflight because the model relay does not support
  their required protocols.
- Use a new results root if the old root has no recorded evaluator identity.

## Run a benchmark

Choose a model and eval from `experiments/evaluation/models.py` and
`experiments/evaluation/evals.py`:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --evals tb2-lite --dry-run

uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --evals tb2-lite
```

`tb2-lite` runs two Terminal-Bench tasks in Daytona. `--evals agentic` selects the
standard Harbor suite. `--limit N` caps tasks. Policy trial repetitions still apply.
`--no-wait` returns after submission.

Use `--harbor-config` for a policy outside the catalog:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2 --dry-run
```

Remove `--dry-run` to submit. Repeat `--harbor-config` to add policies.
You can also select catalog entries with `--evals` in the same command.

To serve a judge in the same batch:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --platform gpu \
  --judge-model qwen3.5-122b-a10b-fp8 --judge-accelerator H100x8 \
  --harbor-config experiments/evaluation/configs/harbor/simpleqa-hosted-judge.yaml \
  --federated_cluster cw-rno2a --limit 2
```

Run the candidate and judge in the same cluster or region. The run record stores both
model configurations and their hardware. The candidate keeps its own endpoint.
The hosted judge replaces the judge credentials and model names set by the policy.

To resume, use the recorded Marin checkout and the original model, policy, task limit
and judge settings. The launcher records evaluator identity from the launch's source-tree
hash. To resume the AIME example:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2 --resume-results-path "$RESULTS_ROOT"
```

Set `RESULTS_ROOT` to the prior run's `{records_prefix}/{run_id}/results/` directory.
Wait for all earlier drivers using that root to exit. Two live drivers must never share
a results root. Add `--retry-unscored-harbor-trials` to retry known unscored outcomes
while preserving scored outcomes and unknown errors.

## Maintained execution types

Set `environment.type` in the Harbor policy:

| Type | Requirements and limits |
| --- | --- |
| `local` | Needs prepared Linux images pinned by digest and delegated cgroups. Denies external networking. Supports prepared Compose groups. Shares the worker's kernel. |
| `daytona` | Needs backend credentials and an explicit target. Uses a task registry image or Daytona's Dockerfile build service. Rejects Compose tasks. Guest images need Python 3 and `setsid`. |
| `qemu` | Shellbox QEMU image. No shared host filesystem. |
| `shellsim` | Shellbox simulated filesystem and command execution. |

Removed provider names and custom environment import paths fail validation.
Agent import paths load trusted harness code. Harbor cannot select Shellbox's Docker backend.

Daytona accepts `network_policy` (`mode` and optional `value`), `ttl_minutes` and
`create_timeout`. Snapshot names, automatic snapshots, forced rebuilds and policy-level
target overrides are unsupported. Set the target before launching:

```bash
export DAYTONA_TARGET=us
```

The host entry point reads `DAYTONA_API_KEY`, `DAYTONA_API_URL` and `DAYTONA_TARGET`
once. The API URL defaults to `https://app.daytona.io/api`. Missing credentials or target
fail before allocation. SDK `.env` files, JWT credentials and other connection overrides
are not read.

The launcher does not change a policy's provider. Remove unsupported settings from the
policy and prepare any missing task images before launching. Harbor rejects `--no-delete` and
`environment.delete=false` before allocation.

## Policies and datasets

Catalog policies live under `experiments/evaluation/configs/harbor/`.
Each filename matches its `EVALS` key. Suite membership, model selection, hardware, task limits
and secret references belong in `experiments/evaluation/evals.py`.

For a Hugging Face repository containing Harbor task directories:

```yaml
n_concurrent_trials: 4
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: hf://DCAgent2/terminal_bench_2
    ref: "<dataset-commit-sha>"
```

Replace `<dataset-commit-sha>` with the source revision. Use `datasets[].name` for `hf://`
sources. Harbor checks access and counts tasks during preflight, then loads the source
again on the worker. Pin the revision.
A moving ref can change between those calls. Gated sources need `HF_TOKEN`. Parquet rows are not converted
into Harbor task directories.

A registry source uses its native selector, such as `name: aime` and `version: "1.0"`.
A local source uses `datasets[].path`, relative to the policy. The directory must stay
inside the Marin workspace and be included in the Iris source bundle. Absolute, missing and
outside-workspace paths fail preflight.

## Offline execution through Shellbox

Select `local` for prepared tasks and private container groups:

```yaml
n_concurrent_trials: 4
environment:
  type: local
agents:
  - name: terminus-2
datasets:
  - path: prepared-tasks
```

Place the policy beside `prepared-tasks` in the workspace. Submit it with
`--harbor-config path/to/policy.yaml` through the shared launcher.

Install `bubblewrap`, `umoci` and `skopeo` on the trusted evaluator host. For Iris jobs,
put these tools in `iris-task` and rebuild the worker's task image.
The host must permit namespaces and cgroup v2 process containment. See the
[Shellbox setup instructions](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md#local-backend).

Each task must set `allow_internet = false`. Pin images with an immutable `@sha256:` digest.
For a single-container task, set the image and absolute workdir in `task.toml`:

```toml
[environment]
allow_internet = false
docker_image = "registry.example/main@sha256:<64-hex-digest>"
workdir = "/app"
```

Compose tasks provide `environment/shellbox-images.json`:

```json
{
  "main": {"image": "registry.example/main@sha256:<64-hex-digest>", "workdir": "/app"},
  "tau3-runtime": {
    "image": "registry.example/runtime@sha256:<64-hex-digest>",
    "workdir": "/app",
    "agent_logs": "publish"
  }
}
```

Build images on a trusted host. Include the task dependencies, verifier dependencies and
terminal tools. Terminus needs Bash and tmux. Model relays and timed readiness probes
need Python 3. Runtime downloads fail with networking denied. Images must use
root (`USER 0`, `USER 0:0`, or no `USER`).

The importer accepts one Compose file with `main` and prepared sidecars, commands,
environment values, exposed TCP ports, readiness probes and `main` dependencies on started
or healthy services. It rejects unknown fields, host mounts, published host ports, host
networking and dependencies between sidecars. MCP URLs must name a declared service and port.

Containers share a private loopback network. Each has its own filesystem and PID namespace.
External traffic, including DNS, is denied. Sidecars keep their image's ENTRYPOINT/CMD
unless Compose overrides them. The evaluator does not build images or start nested containers.
A sidecar exit closes the whole local task, including the agent. Grading cannot continue
inside that task.

The prepared local path supports host-side agents, image-provided Pi and Pi/ACP, and Tau3's
user relay. Other installed agents and arbitrary agent imports are rejected on this path.

For a reviewed Tau3 task:

```bash
uv run python -m experiments.evaluation.images.tau3.build \
  --task prepared-tasks/tau3-airline-0
```

The helper reuses the pinned Pi/ACP main image. It builds a runtime image with the task's
server and `task_config.json`, then writes `shellbox-images.json`. Set
`allow_internet = false` in `task.toml` and save the prepared snapshot as a Marin artifact.
See the [image recipe](https://github.com/marin-community/marin/tree/main/experiments/evaluation/images/tau3)
for registry setup.

To check Tau3 grading, use a prepared task with a natural-language assertion and a policy
that declares candidate and judge endpoints:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  python vendor/harbor/tests/local_tau3_smoke.py local-tau3.json \
  --api-key-env GLM_BULK_TOKEN
```

The smoke test requires a score and assertion results. A score of zero is valid.
Infrastructure failures fail the test.

Standalone Pi tasks, including single-container tasks, must declare a digest-pinned
`main` image with Pi installed in `environment/shellbox-images.json`.
For Daytona, `task.toml`'s `environment.docker_image` must match it. Pi setup writes
configuration without installing Node or npm packages. If you set `kwargs.version`,
setup checks it against the installed binary. On local execution, the manifest selects
the image. A `docker_image` in `task.toml`, if present, must match. This requirement applies
to the BixBench and BFCL Pi campaign policies and the one-task onboarding ablation.

ACP Pi images need `node`, `pi`, `pi-acp`, and
`/opt/harbor-acp-venv/bin/python` with `agent-client-protocol`. The companion pins are
`@earendil-works/pi-coding-agent@0.80.5` and `pi-acp@0.0.33`. ACP bridges also need
`acpx@0.12.0`. Tau3's simulated-user relay needs `mcp==1.25.0` under `python3`.

For Tau3, supply `--judge-model`. That model serves the simulated user and verifier.
The candidate keeps its own endpoint. The Pi/ACP candidate receives domain tools.
The benchmark relay controls user turns and checks runtime status after each turn.
Runtime MCP failures are infrastructure errors. Candidate process failures keep their
agent classification.

A sidecar declaring `agent_logs: "publish"` has its logs copied into the actor's agent-log
directory after the agent phase, before verification. This supports Tau3's verifier.
There is no shared writable volume. Actor writes do not flow back to the sidecar.

### Local factory settings

The adapter accepts `bwrap`, `umoci`, `image_cache`, `cgroup_parent` and `ttl`.
`registry_authfile` and `signature_policy` configure host-side image acquisition and are
never mounted in guests. An explicit `cgroup_parent` overrides discovery.

Command timeout or cancellation kills that command and its descendants. Task files,
services from earlier successful commands and runtime peers remain available.
Trial cancellation reclaims the whole task. Terminus interrupts its foreground tmux job
on deadline and closes tmux at teardown. Full command cleanup or task teardown also stops
deliberately detached work.

Shellbox attempts numeric CPU and memory limits independently and warns when a controller
cannot enforce one. Process containment remains mandatory. Harbor rejects `ignore`
resource policies. Adjust task values or overrides:

```yaml
environment:
  type: local
  override_cpus: 2
  override_memory_mb: 4096
  kwargs:
    ttl: 3600
```

Disk quotas, GPU allocation and TPU allocation are unsupported. To run the policy directly
on a configured Linux worker:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  harbor run --config local-policy.yaml
```

For offline use, export an image with Skopeo as an OCI layout tagged `image` at
`<image_cache>/sha256-<platform-manifest-digest>`. Use that digest in the task declaration.
Shellbox verifies the manifest, configuration and layers. Explicit signature policies still
require registry verification.

## Credentials

Daytona catalog entries use `DAYTONA_API_KEY` from the launch environment, then fall
back to the approved Google Secret Manager version. The launcher resolves declared secret
references before submission. The Harbor subprocess receives backend credentials, HF access
and storage credentials for result uploads.

Harbor is a trusted host process. `uv --isolated` isolates dependencies.
It does not restrict the process's filesystem or network access. Keep secrets out of policy
files, serialized configuration, images and artifacts. Harbor's `--env-file` configures
the harness. Task templates still need explicitly supplied inputs.

Direct Harbor callers pass host authentication through `HostCapabilities` and template
inputs through a separate `guest_environment` mapping. Entry points read host settings
once. Templates resolve only against the supplied map, which defaults to
empty. Storage, controller and provider credential keys and known secret values are
rejected before entering guests. Keep production credentials out of task definitions,
images and explicit inputs too.

## Sandbox environment isolation

Host-side agents such as Terminus call the candidate endpoint from Harbor. Installed
agents use a loopback relay with a dummy `EMPTY` key. Provider credentials stay on
the host. Candidate and judge credentials are separate.

| Relay | Port | Access |
| --- | --- | --- |
| Candidate | 8788 | Installed candidate agents on local and Daytona backends. |
| Judge | 8787 | Available during verification. Tau3 and Sotopia also use it during execution. |

Both relays support `POST /v1/chat/completions`, `POST /v1/responses` and `POST /tokenize`.
Each relay calls one fixed upstream model. It rejects redirects, limits request and response
sizes, and returns sanitized provider error codes. Responses supports non-streaming text,
output limits and reasoning options. Tools, stored responses and stateful continuations are unsupported.
Streaming chat is buffered: the guest receives SSE frames after the complete response
arrives. Codex and Claude Code need routes or features outside this contract.

`judge_access: verifier` opens the judge relay only during grading. Before opening it,
local execution stops remaining agent command groups. Ordinary grading preserves services
from successful commands. A grader needing both a live agent service and private judge
access needs separate isolation. Daytona cannot confirm that every detached candidate
process has stopped before grading.

`judge_access: runtime` keeps the judge available during execution. Marin selects it for
Tau3 and Sotopia. Direct callers must request it. Every peer sharing the task network can
reach that relay, including the candidate. Tool filtering does not provide network isolation.

Local bubblewrap shares the worker's kernel. Use isolated workers. These contracts do not
certify deployed images or provider-injected settings. See the
[Shellbox backend reference](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md)
for backend-specific isolation limits.

## Endpoint lifecycle

Marin registers each served model under a unique Iris endpoint name. It gives Harbor a
URL that grants access to that model. Expiring leases remove abandoned registrations.
A retry replaces its same-name registration. Saved job and trial JSON redacts these
credential-bearing URLs. Resume supplies new URLs.

`agents[0].model_name` stores the original model location with a `hosted_vllm/` prefix.
`model_alias` routes requests to the served model. SOTOPIA uses
the original location without that prefix for cohort matching. Moving an endpoint or
changing its alias does not change the recorded model identity.

## Agent context limits

Marin derives `model_info.max_input_tokens` from resolved `serve.max_model_len` and
`max_output_tokens` from `generation.max_gen_toks`. A lower policy limit wins.
A policy limit above the serving limit fails preflight. `auto_overrides` clamps model settings to
the checkpoint's native window first. If neither model nor policy states limits, Harbor
defaults to 32,768 input and 8,192 output tokens. `--dry-run` prints the resolved limits.
`record.json` stores them under `eval.harbor`.

Terminus-2 receives the output limit as `llm_call_kwargs.max_tokens`. Lower explicit request
limits win. Higher ones fail preflight. Mini-SWE 2.1.0 uses `/tokenize` to count the full
rendered prompt, including tools, and leaves room for the reply within the total context.
When context runs out, Mini-SWE stops and Harbor can grade the existing work.
Installed harnesses must enforce their limits. Sotopia generation does not yet enforce
`model_info` context limits.

For Terminus, use `llm_backend="openai"` and an OpenAI-compatible `api_base`. For proxies
without `/tokenize`, set `tokenizer_encoding`, such as `cl100k_base`, when tiktoken cannot
identify the model. These are estimates. Exact counts require the serving tokenizer.
Optional `input_cost_per_token` and `output_cost_per_token` are USD per token. Harbor does
not look up limits or prices in a model registry.

LiteLLM is excluded from workspace and standalone evaluation environments. Aider, SWE-agent,
OpenHands and OpenHands SDK installation is blocked. Preinstalled Aider is also blocked.
Copilot CLI is unsupported because it needs GitHub authentication without a guest relay.
Use Mini-SWE, Terminus, or NeMo's OpenAI backend with a compatible endpoint.

## Results

Each evaluation writes `record.json` under `{records_prefix}/{run_id}/` and a FineStore
archive under its `results/` directory. The archive contains original Harbor job and trial
artifacts plus normalized `samples`, `steps` and `rollouts_v1` tables. Evaldash reads samples.
Verifier rewards use `Grading(method="harbor:verifier")`. Samples link trajectories through
`trajectory_uri`.

Harbor saves each attempt's result, available trajectory and a record of whether the
trajectory exists in one transaction. Completion waits for environment cleanup, service
shutdown and required artifact writes. Missing required evidence leaves the attempt
incomplete and retryable. At finalization, Harbor selects the final attempt for each trial.
An incomplete selected attempt prevents sealing, which marks the archive complete.
A later successful retry can complete the run. Resume can recover matching finalized
Harbor files and rebuild derived tables. It keeps the original artifacts from every
attempt. Only selected final attempts contribute normalized rows.

Tau3 ACP trials keep raw ACPX exports at `agent/bridge-trajectory.json`. They do not
produce ATIF `agent/trajectory.json`, derived steps or rollouts. Their scored samples record
that expected absence explicitly.

### Trial error taxonomy

Preflight captures the selected Harbor runtime's error categories. Agent failures are
scoreable model outcomes. Passthrough failures need a verifier result to be scored.
Infrastructure and undecided failures remain unscored. Relay loss, runtime-log collection
failure and teardown failure are retryable infrastructure errors. Unknown error names fail
the evaluation. Marin requires the scored fraction to meet `min_completion_rate` (default
0.9), independently of the archive's evidence requirements for sealing.

## Ownership boundary

`vendor/harbor` is an independent uv project with its own lockfile and editable Rigging and
FineStore dependencies. Marin invokes its isolated CLI for preflight and execution.
Preflight returns policy JSON and benchmark metadata. Execution applies live model routes,
output paths, model kwargs and the runtime task limit. Policy kwargs override model-catalog
kwargs. Explicit runtime fields override both. Runtime overlays do not change the source-policy
digest. Policy `n_tasks` remains part of it.

Reproduce runs from their recorded Marin checkout, dataset revisions and policy. Cache and
resume identity include the launch's tracked Marin source-tree hash. Changes to that tree
invalidate reuse. A missing source-tree hash prevents reuse. The runtime lock is
`vendor/harbor/uv.lock`. See the [vendor guide](https://github.com/marin-community/marin/blob/main/vendor/harbor/README.md)
for direct CLI use and maintenance.

Verified cohort labels fix parsed benchmark policies. To approve a changed policy, add
a new cohort version and its source digests as described in the
[cohort guide](https://github.com/marin-community/marin/blob/main/eval_policy/README.md).
Harbor runtime identity comes from the recorded Marin artifact checkout. A shared cohort
label alone does not establish identical runtime behavior across checkouts.
