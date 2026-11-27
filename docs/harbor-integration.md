# Harbor evaluation

Marin imports Harbor-format packages into the curated evaluation catalog.
The launcher selects those benchmarks by name and runs their TaskSpec corpora
through RolloutEngine. See [Running Evaluations with Marin](tutorials/run-lm-evals.md)
for model, suite and credential settings.

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --evals aime-harbor --limit 2 --dry-run
```

Harbor JobConfig files run through the [vendored Harbor CLI](https://github.com/marin-community/marin/blob/main/vendor/harbor/README.md).
The backend and task settings below apply to direct Harbor execution. To
reproduce an archived Harbor cohort, use its recorded Marin checkout and
dataset revisions. See [verified cohorts](https://github.com/marin-community/marin/blob/main/eval_policy/README.md).

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

Daytona accepts `network_policy` (`mode` and optional `value`), `ttl_minutes`,
`create_timeout` and `image_build`. In snapshot mode, Shellbox derives the snapshot
from the task image or Dockerfile and the task's resource request. Snapshot names,
automatic snapshots, forced rebuilds and policy-level target overrides are not
policy settings. Inventory, preflight and reclamation use `shellbox-daytona-snapshots`; see the
[Daytona operations runbook](https://github.com/marin-community/marin/blob/main/.agents/ops/daytona.md#snapshot-preflight-and-reclamation).

`image_build` selects how each sandbox gets its image. The default, `snapshot`, builds each
distinct task environment once into a named Daytona snapshot and starts sandboxes from it.
Snapshots persist after the run and count against the account's snapshot quota, so a dataset
with more distinct environments than the free quota fails with `Snapshot quota exceeded`.
`declarative` builds the image for every sandbox instead and creates no snapshot; every
trial repeats the build, including the base-image pull, and `snapshot-specs` rejects the
policy. Use it for standalone or inline validation in the Daytona eval account. SkyRL training
rollouts use snapshot mode in the RL account. Select it in the eval policy:

```yaml
environment:
  type: daytona
  kwargs:
    image_build: declarative
```

The host entry point reads `DAYTONA_API_KEY` once. The API URL and target region are
constants (`https://app.daytona.io/api` and `us`). Missing credentials fail before
allocation. SDK `.env` files, JWT credentials and other connection overrides are not read.

Prepare task images for the provider declared in the policy. Harbor rejects `--no-delete` and
`environment.delete=false` before allocation.

## Policies and datasets

Harbor policies use the vendored `JobConfig` schema. Archived policies are
available under `experiments/evaluation/configs/harbor/`. A direct Harbor
policy selects a dataset by registry name and version or by a path relative to
the policy file.

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

Place the policy beside `prepared-tasks` and pass it to the vendored Harbor CLI.

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
  "helper": {
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

The prepared local path accepts Pi, Mini-SWE-Agent, Codex, OpenCode,
Terminus-2, and Terminus-Kira. Preflight rejects unsupported agents and
simulated users before task allocation.

Standalone Pi tasks, including single-container tasks, must declare a digest-pinned
`main` image with Pi installed in `environment/shellbox-images.json`.
For Daytona, `task.toml`'s `environment.docker_image` must match it. Pi setup writes
configuration without installing Node or npm packages. If you set `kwargs.version`,
setup checks it against the installed binary. On local execution, the manifest selects
the image. A `docker_image` in `task.toml`, if present, must match. This requirement applies
to the BixBench and BFCL Pi campaign policies and the one-task onboarding ablation.

A sidecar declaring `agent_logs: "publish"` has its logs copied into the actor's agent-log
directory after the agent phase, before verification.
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

For a direct Daytona policy, supply the eval account credential as
`DAYTONA_API_KEY` in the trusted Harbor host environment.

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
| Judge | 8787 | Available during verification, or during execution when `judge_access: runtime` is set. |

Both relays support `POST /v1/chat/completions`, `POST /v1/responses` and `POST /tokenize`.
Each relay calls one fixed upstream model. It rejects redirects, limits request and response
sizes, and returns sanitized provider error codes. Responses supports non-streaming text,
output limits and reasoning options. Tools, stored responses and stateful continuations are unsupported.
Streaming chat is buffered: the guest receives SSE frames after the complete response
arrives. Some Codex features need routes outside this contract.

`judge_access: verifier` opens the judge relay only during grading. Before opening it,
local execution stops remaining agent command groups. Ordinary grading preserves services
from successful commands. A grader needing both a live agent service and private judge
access needs separate isolation. Daytona cannot confirm that every detached candidate
process has stopped before grading.

`judge_access: runtime` keeps the judge available during execution. Direct callers must
request it. Every peer sharing the task network can
reach that relay, including the candidate. Tool filtering does not provide network isolation.

Local bubblewrap shares the worker's kernel. Use isolated workers. These contracts do not
certify deployed images or provider-injected settings. See the
[Shellbox backend reference](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md)
for backend-specific isolation limits.

## Agent context limits

Archived launcher records store their resolved context limits under
`eval.harbor`. Direct Harbor policies declare their context limits in
`agents[].kwargs.model_info`.

Terminus-2 receives the output limit as `llm_call_kwargs.max_tokens`. Lower explicit request
limits win. Higher ones fail preflight. Mini-SWE 2.1.0 uses `/tokenize` to count the full
rendered prompt, including tools, and leaves room for the reply within the total context.
When context runs out, Mini-SWE stops and Harbor can grade the existing work.
Installed harnesses must enforce their limits.

For Terminus, use `llm_backend="openai"` and an OpenAI-compatible `api_base`. For proxies
without `/tokenize`, set `tokenizer_encoding`, such as `cl100k_base`, when tiktoken cannot
identify the model. These are estimates. Exact counts require the serving tokenizer.
Optional `input_cost_per_token` and `output_cost_per_token` are USD per token. Harbor does
not look up limits or prices in a model registry.

LiteLLM is excluded from workspace and standalone evaluation environments. Use one of
the six supported agents named in the vendored runtime README.

## Archived launcher results

A recorded Harbor evaluation includes `record.json` under `{records_prefix}/{run_id}/` and a FineStore
archive under its `results/` directory. The archive contains original Harbor job and trial
artifacts plus normalized `samples`, `steps` and `rollouts_v1` tables. Evaldash reads samples.
Verifier rewards use `Grading(method="harbor:verifier")`. Samples link trajectories through
`trajectory_uri`.

Harbor saves each attempt's result, available trajectory and a record of whether the
trajectory exists in one transaction. Completion waits for environment cleanup, service
shutdown and required artifact writes. Missing required evidence leaves the attempt
incomplete. At finalization, Harbor selects the final attempt for each trial.
An incomplete selected attempt prevents sealing, which marks the archive complete.
A later successful retry can complete the run. Resume can recover matching finalized
Harbor files and rebuild derived tables. It keeps the original artifacts from every
attempt. Only selected final attempts contribute normalized rows.

### Trial error taxonomy

Preflight captures the selected Harbor runtime's error categories. Agent failures are
scoreable model outcomes. Passthrough failures need a verifier result to be scored.
Infrastructure and undecided failures remain unscored. Relay loss, runtime-log collection
failure and sandbox teardown failure are retryable infrastructure errors. A failed
attempt with unconfirmed teardown retries within its budget, even when a verifier has
returned a reward. The archive requires confirmed teardown for the selected attempt.
Cancellation clears the reward. Unknown error names fail the evaluation. Marin requires
the scored fraction to meet `min_completion_rate` (default 0.9), independently of archive sealing.

## Reproduction

`vendor/harbor` is an independent uv project with its own runtime lock.
Use the recorded checkout, policy and dataset revisions to reproduce an
archived run. See the [vendor guide](https://github.com/marin-community/marin/blob/main/vendor/harbor/README.md)
for direct CLI use and maintenance.

Verified cohort labels fix parsed benchmark policies. To approve a changed policy, add
a new cohort version and its source digests as described in the
[cohort guide](https://github.com/marin-community/marin/blob/main/eval_policy/README.md).
Harbor runtime identity comes from the recorded Marin artifact checkout. A shared cohort
label alone does not establish identical runtime behavior across checkouts.
