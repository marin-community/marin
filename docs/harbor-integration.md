# Harbor evaluation

Marin runs Harbor benchmarks through the shared evaluation launcher. The launcher starts the model
server and, when requested, a second shared judge server; gives Harbor their Iris capability URLs; and normalizes completed trials into v2
`EvalRunRecord`/`EvalSample` artifacts, and tears inference down after the selected evaluations
finish.

Harbor provides containerized agent benchmarks such as Terminal-Bench, SWE-bench Verified, AIME,
GAIA, BFCL, and Aider. Harbor executes tasks only through Marin Shellbox adapters:
`local` (the default), `daytona`, `qemu` and `shellsim`. Removed provider names and
custom environment import paths fail validation.
See [Running Evaluations with Marin](tutorials/run-lm-evals.md) for the model and suite command
matrix. This page documents Harbor-specific configuration and output behavior.

The Marin driver records the model's original location in Harbor's `agents[0].model_name`,
with a `hosted_vllm/` provider prefix, and puts the live serving name in `model_alias`.
Agents route requests through `model_alias`; results and trial resume identity use `model_name`.
SOTOPIA uses the original location, without the provider prefix, to match the comparison cohort.

Harbor uses the OpenAI SDK for inference. In Terminus agent kwargs, set
`llm_backend="openai"` and supply an OpenAI-compatible `api_base` for providers
other than OpenAI. Set `model_info.max_input_tokens` and
`model_info.max_output_tokens` for context and output limits. Optional
`input_cost_per_token` and `output_cost_per_token` fields specify USD per token.
Harbor does not look up limits or prices in a model registry. Agent CLI-reported
costs and built-in prices for Cursor models remain available. Otherwise,
CLI-agent cost estimates remain unset when `model_info` omits prices.

For a proxy without `/tokenize`, set the Terminus agent kwarg
`tokenizer_encoding` to the tiktoken encoding selected for local estimates,
such as `cl100k_base`. This is required for model names tiktoken cannot identify,
including Claude and Gemini names. These estimates do not reproduce the
provider's private tokenizer; use a server token-count endpoint for exact counts.

LiteLLM is excluded from installation in the workspace and standalone evaluation
environments, including as a transitive dependency. Mini-SWE uses Harbor's OpenAI
model adapter. The recording proxy translates Anthropic Messages and OpenAI
Responses tool histories to the configured Chat Completions endpoint. It retains
token IDs and log probabilities when the serving backend returns them.

Aider, SWE-agent, OpenHands, and OpenHands SDK installation is blocked because
those harnesses require LiteLLM. Preinstalled Aider execution is also blocked.
Copilot CLI installation and execution are unsupported because it requires
GitHub authentication without a guest relay. Use Mini-SWE or Terminus instead. NeMo's
LiteLLM backend is also unavailable; use its OpenAI backend with a compatible
endpoint.

## Run a benchmark

Use a model and evaluation name from `experiments/evaluation/models.py` and
`experiments/evaluation/evals.py`:

```bash
# Inspect placement, task limit, and record destination.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals tb2-lite \
  --dry-run

# Serve once and run two Terminal-Bench tasks in Daytona.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals tb2-lite
```

The `agentic` suite expands to the standard Harbor presets:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --evals agentic
```

Use `--limit N` to cap the number of trials and `--no-wait` to return after submission.

For a Harbor verifier that uses the OpenAI API, add a hosted judge to the same launch:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --platform gpu \
  --judge-model qwen3.5-122b-a10b-fp8 \
  --judge-accelerator H100x8 \
  --harbor-config experiments/evaluation/configs/harbor/simpleqa-hosted-judge.yaml \
  --federated_cluster cw-rno2a \
  --limit 2
```

The launcher keeps one judge session alive and shares it across every evaluation in the batch. It
replaces the policy's judge endpoint, credential and declared model roles. Both OpenAI URL aliases
point to the guest relay, and all verifier model roles use the hosted judge. The evaluated agent
continues to use its separate candidate endpoint. Candidate and judge placement must resolve to the
same cluster or region. The durable run record pins the judge model configuration and hardware.

Launch a Harbor `JobConfig` without adding it to the catalog:

```bash
# Validate policy, model overlay, and placement without opening Iris.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2 \
  --dry-run

# Serve the model and run the checked-in policy with a two-task cap.
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml \
  --limit 2
```

`--harbor-config` is repeatable and additive with `--evals`. All selected built-in and supplied
Harbor policies pass schema and placeholder effective-job validation before the launcher
opens an Iris client.

To resume one Harbor evaluation, pass its object-store results root with `--resume-results-path`
and the original model, policy, task limit, and hosted judge settings. Marin rejects the
resume unless the root's saved model-bound identity matches those inputs. Wait for the original
Iris group and any earlier retry to end first: two live Harbor drivers must never share a results
root. Add `--retry-unscored-harbor-trials` to retry known unscored trials while preserving scored
outcomes and unknown errors.

## Maintained execution types

Harbor's CLI and policies select `environment.type` from `local`, `daytona`,
`qemu` and `shellsim`. They default to local bubblewrap. Environment classes cannot be loaded
from a policy or CLI import path. Agent import paths remain a separate trusted
harness extension.

Daytona uses `DaytonaMachineFactory` for creation, execution, transfers and cleanup.
It selects a task's registry image when supplied; otherwise Shellbox prepares the
task's Dockerfile through Daytona's provider image-build service. No Docker daemon
runs inside the task container. Tasks need Python 3 and `setsid` for command
supervision and the host model relay.

The Daytona adapter accepts `network_policy` (`mode` and optional
`value`), `ttl_minutes` and `create_timeout`. Model endpoint settings are supplied
by the trusted launcher. Snapshot names, automatic snapshot toggles, provider target
overrides and forced rebuilds are unsupported. Compose tasks must use prepared
local container groups; Daytona rejects every Compose filename and extra overlay.
The host entry point resolves `DAYTONA_API_KEY`, `DAYTONA_API_URL` and `DAYTONA_TARGET`
once and supplies the connection privately. The public API URL defaults to
`https://app.daytona.io/api`; a live Daytona run requires an explicit target:

```bash
export DAYTONA_TARGET=us  # Select the region approved for this run.
```

Missing credentials or a missing target fail before allocation. The adapter does not
read SDK `.env` authentication, JWT credentials, or environment-based connection overrides.

Checked-in evaluation policies retain their dataset selectors and historical
provider choices. They are not silently rewritten to another environment.
A policy or package needing a removed option, a Compose topology on Daytona,
or an unavailable image fails explicitly. Prepare the task through the supported
path before launching a new run; preserve historical outputs as Marin artifacts.
SkyRL's engine-owned machine path can use Shellbox's Docker backend. Harbor
cannot select it as an execution provider.

## Credentials

Daytona-backed definitions in `experiments/evaluation/evals.py` own their credential references.
They use `DAYTONA_API_KEY` from the launch environment when present, then fall back to the approved
Google Secret Manager version. `DAYTONA_API_KEY` is the only supported credential override.

The generic launcher resolves declared references immediately before Iris submission. The Harbor
subprocess receives the selected backend's credentials, Hugging Face access, and storage credentials
needed for remote trial results and FineStore archives. It remains a trusted process: `uv --isolated`
isolates dependencies. The subprocess retains its user identity, filesystem and network access.
Task containers must not inherit that authority. Do not put resolved credentials in model YAMLs,
runner configs, or evaluation artifacts. Harbor's `--env-file` configures its trusted harness;
it does not grant the file's variables to task templates. The Marin launcher binds explicit
task capabilities and host model authentication. A missing capability fails instead of
prompting to inherit a host variable. Use the Marin launcher for hosted judge authentication.

## Offline execution through Shellbox

For prepared tasks and private container groups, select Harbor's local bubblewrap adapter:

```yaml
n_concurrent_trials: 4
environment:
  type: local
agents:
  - name: terminus-2
datasets:
  - path: prepared-tasks
```

Install `bubblewrap`, `umoci`, and `skopeo` on the trusted evaluator host. For an
evaluation running as an Iris task, these tools belong in `iris-task`; they do
not belong in the inner rollout images. The Iris task Dockerfile includes them,
but workers need a rebuilt task image to use that addition. Namespace permissions
and cgroup delegation are separate host requirements.

Place this policy beside `prepared-tasks` inside the Marin workspace, then use the shared launcher:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --harbor-config path/to/policy.yaml --limit 1 --dry-run
```

Shellbox discovers the worker's delegated cgroup hierarchy. Launcher preflight does not
validate a distant worker's delegation. Remove `--dry-run` to submit.
Each task must set `allow_internet = false`. A single-container task
names a prebuilt `docker_image` with an immutable `@sha256:` digest and an absolute `workdir`.
Compose tasks instead provide `environment/shellbox-images.json`, mapping each service to its prepared
image and working directory:

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

Build and push these images during trusted task preparation, preserving each task's inputs. Tau3's
runtime image includes its task-specific `task_config.json`; one image cannot silently substitute
another task's configuration. Dockerfiles and Compose build declarations may remain in the source
artifact. Evaluation reads the prepared references and never executes their build recipes.

The importer produces one Shellbox `TaskSpec` containing named containers and an
explicit primary named `main`. A task without peers has that same representation
with one container. Both use `factory.create_task`; startup, readiness, artifact
transfer and cleanup share the same path.

Build images separately with the task's dependencies, verifier dependencies and terminal utilities
already installed. Terminus needs Bash and tmux; runtime downloads cannot succeed on this path.
Tasks using the model relay need Python 3 in their main image.
Containers with timed readiness checks need Python 3 in their own images to supervise
individual probes without stopping the container on the first timeout.
Prepare a reviewed Tau3 task snapshot with the shared image helper:

```bash
uv run python -m experiments.evaluation.images.tau3.build --task prepared-tasks/tau3-airline-0
```

The helper reuses the dependency-pinned Pi/ACP main image, builds the task-specific
runtime, and writes `shellbox-images.json`. See the
[image recipe](https://github.com/marin-community/marin/tree/main/experiments/evaluation/images/tau3) for registry setup.
Set `allow_internet = false` in the prepared task's `task.toml` and retain the
prepared snapshot as a Marin artifact. Image builds run on trusted preparation
infrastructure.
The primary stays alive for agent commands and verification. Sidecars retain their prepared
image's ENTRYPOINT/CMD unless Compose explicitly overrides them. The adapter does not build
images or install a nested container runtime.
Shellbox adapters reclaim environments at completion. `--no-delete` and
`environment.delete=false` are rejected before backend allocation.

Host-side agents such as Terminus make model calls in the trusted Harbor process.
Installed agents use the loopback relay described below. Shellbox runs task commands and
verification in persistent bubblewrap containers. Members share private loopback and retain
separate filesystems and PID namespaces. External traffic is denied, including DNS.
Provider credentials stay in the trusted harness.
Task environment templates resolve only against explicit task capabilities. They never read the
driver's ambient environment, including through differently named aliases.
Image contents and explicit task inputs must themselves be free of production credentials.

The runtime has no independent Docker or other provider execution backend. The prepared local path
supports image-provided Pi and Pi/ACP agents and the benchmark-owned Tau3 user relay. Other installed
agents, arbitrary agent imports and custom host mounts are rejected. Its Compose importer supports `main` plus prepared
sidecars, commands, environment values, exposed TCP ports, readiness checks and `main` dependencies
on started or healthy services. MCP URLs must name an exposed endpoint inside the container group.
A task may contain one Compose file. Unknown fields, published host ports, host
networking and dependencies between sidecars fail explicitly.

Containers share only the group's loopback network. External egress remains denied. Tau3's user model
and verifier receive a loopback OpenAI endpoint and an `EMPTY` key. The relay fixes the upstream
model name and keeps provider authentication in the host process. Its supported routes are listed below.
Tau3 requires an explicit `--judge-model` in the shared launch command. That hosted model serves
both the simulated user and verifier assertions; the candidate agent keeps its own model endpoint.
For Tau3's Pi/ACP candidates, Marin excludes conversation-control tools from the candidate's
MCP extension. Domain tools remain available; the benchmark relay advances user turns.
After each candidate turn, the relay checks the runtime's canonical status before submitting
another message, so a conversation that ended during the turn proceeds to native grading.
HTTP failures from the runtime's MCP task group are infrastructure errors; candidate
process failures retain their agent classification.
This tool filtering controls the harness interface; it does not isolate callers sharing
the runtime's task-local network.
Without a hosted model, the adapter rejects the task before allocating its container group. A direct
Harbor caller supplies `model_endpoint` and `model_name`, an explicit `HostCapabilities` value for
host authentication, and a separate `guest_environment` mapping for task templates. Executable
entry points capture their host environment once; adapters never read an ambient model key.

The relay starts one Python HTTP helper and exchanges bounded request/response
frames over a persistent process session. The session uses owned local process
pipes or Daytona's SDK transport; provider credentials remain on the host.

Installed candidate agents use a separate loopback relay on port 8788 on both local
and Daytona backends; the simulated-user and verifier relay uses port 8787. Marin resolves installed
agents through Harbor's agent classes, including Pi, OpenCode, Mini-SWE, and ACP Pi.
Their provider configuration and environment contain a loopback URL and an `EMPTY`
key. The candidate endpoint and its authentication remain in the trusted driver;
the executor supplies the served endpoint's API key only through the driver environment.
Host-side agents such as Terminus and the AIME agent call their candidate endpoint directly.
Persisted job and trial JSON redacts Iris bearer URLs; live configurations retain their routes.
Marin supplies a fresh model capability when a run resumes.

Both relays authorize only `POST /v1/chat/completions`, `POST /v1/responses`, and
`POST /tokenize`. They fix the upstream model, reject redirects, and bound request
and response bodies. Upstream authentication and context-limit errors retain sanitized SDK
error codes; provider error messages are never forwarded. The Responses route supports non-streaming text input with
an output-token limit and reasoning options; tools, stored responses, and stateful
continuations are unsupported.
Codex and Claude Code require protocols beyond this relay's supported contract. Marin's launch
preflight rejects them. A text-only Responses route does not provide their full tool protocol.
The tokenization route uses the configured upstream's server root. For installed
SDKs requesting streaming chat, the host obtains one non-streaming completion and
returns bounded SSE frames containing its text, reasoning, tool calls, finish reason,
and usage. This adds no streaming transport: the guest receives the complete buffered
response after generation finishes. Tasks using either relay need Python 3.

The recognized Harbor agent-log bind is lowered only when the prepared sidecar declares
`agent_logs: "publish"`. After the agent phase and before verification, the adapter copies sidecar logs
into the actor's agent-log directory through the control plane. This preserves tau3's runtime-state
handoff to its verifier. It does not implement a shared writable volume; actor writes do not flow
back into the sidecar.

Unsupported task requirements
fail without falling back to Docker, public networking or a privileged profile. Existing remote
backend policies remain explicit choices; this adapter does not migrate their tasks automatically.

This boundary protects the trusted driver from task code. Harbor and its dependencies still execute
in the trusted driver and retain its storage authority. Moving orchestration to TaskCompendium, or
isolating that driver behind model and artifact brokers, is separate work.

### Local factory settings

Use an isolated Linux worker with working bubblewrap namespaces, `skopeo` and `umoci`.
The prepared images must use numeric root (`USER 0` or `USER 0:0`) or omit `USER`.
The adapter accepts `bwrap`, `umoci`, `image_cache`, `cgroup_parent` and `ttl` factory settings.
Optional `registry_authfile` and `signature_policy` files configure host-side image acquisition;
they are never mounted into guests. Host mounts are unsupported. The guest has no external
network route.

The Harbor local adapter requires a writable delegated cgroup v2 hierarchy, discovered
by Shellbox. Command
timeouts kill that command's descendants without destroying the task filesystem or runtime peers.
Before opening a verifier-only judge relay, the adapter stops all agent command groups, including detached processes;
failure to confirm cleanup leaves the trial unscored. The same delegation enables CPU and memory limits
across each container's processes and descendants. Shellbox attempts the task's
numeric CPU and memory declarations independently, warning when a controller
cannot enforce them. The worker must run within a delegated
hierarchy; see the [Shellbox delegation instructions](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md#local-backend).
Terminus reuses a tmux terminal: an agent deadline interrupts its foreground process group while
preserving earlier background services for ordinary grading. This does not claim to stop deliberately
detached descendants of a terminal command. Full local quiescence before private judge access, or
final task teardown, stops those remaining command groups.
For example:

```yaml
environment:
  type: local
  override_cpus: 2
  override_memory_mb: 4096
  kwargs:
    ttl: 3600
```

The local adapter rejects `ignore` resource policies. Adjust numeric task values or
overrides to change limits. Missing CPU or memory controllers does not disable
process containment; a worker without containment delegation cannot run this adapter.
Storage quotas, GPU and TPU allocations remain unsupported.

Run the policy through the isolated Harbor binary on that worker:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  harbor run --config local-policy.yaml
```

Model and judge settings use the explicit host capabilities described above. A local command
timeout or cancellation stops its descendant processes while retaining the task for collection
and grading. Trial cancellation closes the whole task. Cleanup preserves the originating
interruption during startup, commands and file transfers. Confirmed cleanup of detached command
descendants is currently a local-backend guarantee; the other backends retain their existing
process cancellation behavior. Local execution shares the host kernel;
use it for trusted tasks on isolated workers. Iris remains the default production boundary.

The local backend uses the same Docker/OCI image contents; it unpacks the layers
instead of starting a Docker container. A prepared image can remain local.
Export the host-built image with Skopeo into an OCI layout tagged `image`, put
that layout at `<image_cache>/sha256-<platform-manifest-digest>`, and reference
the same platform manifest digest in `shellbox-images.json`. Shellbox verifies
the manifest, configuration and every layer before using that cache offline.
An explicit signature policy still requires registry verification.

The Tau3 smoke runner requires a prepared task with an NL assertion, such as
`tau3-retail-2` from Tau3 1.0.1, and a policy declaring the candidate and judge
endpoints. It uses the task's native verifier:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  python vendor/harbor/tests/local_tau3_smoke.py local-tau3.json \
  --api-key-env GLM_BULK_TOKEN
```

The runner retains trial artifacts and requires a scored result with native NL
assertion results. Reward zero is valid; infrastructure failures fail the smoke.

## Sandbox environment isolation

Keep provider credentials in the harness process. An actor sandbox should receive
only its image's configuration, backend runtime defaults and explicitly declared
task or command variables. Treat task definitions, images, build contexts and
explicit mounts as trusted inputs: they can deliberately supply credentials.
Task, agent, solution and verifier environment templates resolve against an explicit capability
mapping whose default is empty. They cannot resolve arbitrary host variables. Storage,
control-plane and model-provider credential keys and known credential values are rejected before
sandbox commands run. Model clients receive inert keys for a declared relay. Host authentication
files, GitHub tokens and AWS/Bedrock credentials cannot be forwarded to installed agents.

The template mapping is a resolution source, not an environment forwarded to every agent.
Candidates receive only their declared agent variables; installed candidates use the separate
candidate relay on port 8788. Judge access defaults to `judge_access: verifier`: the relay on port
8787 opens at verification and closes when verification ends. This restricts when the service
exists; it does not create a separate verifier network. Ordinary grading preserves services left
by successful agent commands. Before verifier-only judge access, local execution stops candidate command
cgroups. A grader that needs both a live agent-started service and private judge access requires
separate isolation. Daytona has no equivalent confirmation for every detached descendant,
so an independently surviving candidate process could reach the relay during grading.

Tau3's simulated user/runtime and Sotopia's partner/evaluator need their model during execution.
The Marin driver declares `judge_access: runtime` for those consumers and explicitly configures
the Tau3 user environment. Standalone task runtimes must request that mode themselves. Runtime
model access shares the task group's loopback network: the candidate can also reach that relay
while the runtime uses it. These tasks do not provide isolation between model consumers inside
one task group.

Below, persistent variables are explicit `persistent_env` / `--ae` settings;
Harbor infrastructure values name images, resource limits and log mount paths.

| Backend | Environment and configuration supplied by the launcher |
| --- | --- |
| Shellbox local / bubblewrap | Fixed `PATH`, `HOME`, `LANG`, optional hash seed and explicit task/command variables. Task variables take effect inside bubblewrap and reach it through a private inherited descriptor, keeping values out of host process listings. Private `/etc` copies allowlisted runtime files; host startup files, password hashes, private keys, `/opt` contents and sysfs are hidden. Shared program directories and explicit mounts remain trusted inputs. |
| Shellbox Daytona | Image/snapshot configuration and explicit task/command variables. The Daytona client keeps its API credentials in the harness; creation forwards only `MachineSpec.env`. |
| Shellbox QEMU | Image configuration and explicit task/command variables reach the guest over its serial protocol. The QEMU host process uses host runtime settings; these do not become guest variables. No host filesystem is shared with the guest. |
| Shellbox ShellSim | Simulator defaults and explicit task/command variables in a private virtual filesystem. |

See the [Shellbox backend reference](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md)
for each backend's isolation and resource limits.

These contracts describe repository launch paths. They do not certify deployed
images, existing snapshots, provider-side injected settings or Docker daemon
configuration. Keep those inputs free of credentials. Bubblewrap shares the
worker's kernel and should run on an isolated worker. Its `ALLOW` network policy
shares the host network. Use denied networking and the host relay for task code.

## Endpoint lifecycle

Harbor receives a candidate `RunningModel` whose base URL is an Iris link endpoint. A hosted judge
uses a separate capability URL injected into the verifier environment. The inference runner
chooses the opaque endpoint name, registers either the direct server or broker proxy with Iris, and
mints the capability URL. Daytona never receives a worker address.

Each inference session chooses an opaque endpoint name. Iris endpoint leases remove abandoned
registrations, and a retried task attempt atomically replaces its own same-name registration.

## Policies and datasets

Harbor policy belongs in YAML. This example uses a Hugging Face repository whose root contains
Harbor task directories:

```yaml
n_concurrent_trials: 4
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - name: hf://DCAgent2/terminal_bench_2
    ref: main
```

Use `datasets[].name`, not `datasets[].path`, for `hf://org/repository`. Harbor loads the repository's
task directories during preflight to count tasks and on the submitted worker to run them. The
repository must already contain Harbor tasks; Parquet rows are not converted. A Harbor registry source
uses its native selector, such as `name: aime` plus `version: "1.0"`. A local source uses a relative path:

```yaml
environment:
  type: daytona
agents:
  - name: terminus-2
datasets:
  - path: tasks
```

Local paths resolve against the directory containing the policy. The resulting directory must remain
inside the Marin workspace and must be included in the Iris workspace bundle. The launcher stores its
workspace-relative path so the submitted worker resolves it under the unpacked workspace. Absolute,
outside-workspace, and missing local directories fail before Iris submission. Harbor checks Hugging Face
repository access during preflight and loads the snapshot again on the worker. Gated repositories need
`HF_TOKEN` in the launch environment; Marin forwards it to the isolated worker. A moving ref such as
`main` can resolve to different commits between preflight and execution.

Every catalog policy lives under `experiments/evaluation/configs/harbor/` and shares its filename
with its `EVALS` key. Keep suite membership, runtime task caps, model and hardware selection, and
secret source declarations in `experiments/evaluation/evals.py`.

## Ownership boundary

Harbor lives in `vendor/harbor` as an independent uv project with its own lockfile. It uses editable
local Rigging and FineStore sources. Marin has no Harbor dependency; its launcher invokes Harbor
through an isolated subprocess CLI for two calls:

1. Preflight parses YAML or JSON with Harbor's Pydantic models, rejects unsupported launch shapes,
   validates a placeholder model/endpoint overlay, loads the dataset through Harbor to count tasks,
   and emits opaque deterministic policy JSON plus Marin-owned metadata.
2. Execution reparses the opaque policy, applies the real endpoint, served model, output directory,
   workspace-local dataset path when applicable, model kwargs, and task limit, then validates the
   complete typed job before calling Harbor. Hugging Face and registry selectors stay intact.

Runtime values do not change the source-policy digest. Policy kwargs override model-catalog kwargs;
the served endpoint/model, output paths, local source path, and explicit `--limit` override both.
For a Harbor target or simulated user with `name: pi`, the prepared image supplies
Pi. An explicit `kwargs.version` is checked against the installed binary; Marin
does not replace it with a separate version pin. Pi setup does not install Node or
npm packages. Historical tasks that relied on runtime installation need a prepared
harness image.

Marin preflight requires standalone Pi tasks to declare a digest-pinned `main`
image in `environment/shellbox-images.json`. Build that image from the task's
canonical image with Pi installed, preserving its benchmark dependencies. On
Daytona, `task.toml`'s `environment.docker_image` must match the declared image.
The declaration records the preparation contract; Pi setup checks the installed
binary. This applies to the BixBench and BFCL standalone Pi campaign policies,
including the one-task onboarding ablation.

ACP Pi uses its own companion pin, `@earendil-works/pi-coding-agent@0.80.5`, with
`pi-acp@0.0.33`. Its image must expose `node`, `pi`, and `pi-acp` on `PATH` and provide
`/opt/harbor-acp-venv/bin/python` with `agent-client-protocol`. Every ACP bridge target,
including Claude Code and Gemini, also requires image-provided `acpx@0.12.0` on `PATH`.
The Tau3 simulated-user relay requires `python3` with `mcp==1.25.0`. Setup stages launchers,
endpoint credentials, provider configuration, and ACP session configuration; it does not
install these dependencies. The same Tau3 image can run standalone Pi with its
image-provided version.

For Qwen3, set the model's `agent.agent_kwargs.thinking_format` to
`qwen-chat-template` and `serve.reasoning_parser` to `qwen3`. The latter keeps reasoning
out of simulated-user messages and structured judge answers. Serve the context window
declared by the policy; Tau3's ACP policy uses 131,072 tokens.

Temporary policy and overlay files are owner-readable and removed after each isolated call.

The runtime uses `vendor/harbor/uv.lock`. Reproduce runs from their recorded Marin checkout
and artifact provenance, with pinned dataset revisions and policies.
The evaluation cache and resume record include the evaluator source and lockfile, adapter,
scoring, storage, and execution source identities. Candidate and judge identities describe the
registered models; a replacement serving endpoint or alias does not change the evaluation.
The vendored package evaluates prebuilt task datasets; dataset generators are omitted.
See the [vendor maintenance guide](https://github.com/marin-community/marin/blob/main/vendor/harbor/README.md)
for dependency updates and direct evaluation CLI commands.

### Trial error taxonomy

Preflight snapshots Harbor's infrastructure, agent, passthrough, and undecided error categories from
the selected Harbor runtime in the isolated environment. Marin uses that snapshot to classify trial results.
Undecided errors are known to Harbor but have no default scoring
policy; Marin leaves them ungraded and counts them against the completion gate. Names absent from all
four categories fail the run as an infrastructure failure, so taxonomy changes cannot silently alter
scoring or completion coverage.
Required relay loss, runtime-log collection failure and environment teardown failure use named
infrastructure errors and remain retryable on resume. They leave the affected trial unscored;
the evaluation's completion gate still applies.

## Agent context limits

The agent's `model_info.max_input_tokens` comes from the model's resolved `serve.max_model_len` and its
`model_info.max_output_tokens` from `generation.max_gen_toks`, so the agent compacts against the
window the server actually offers. With `auto_overrides`, explicit context limits are clamped to the
checkpoint's native window before Harbor preflight; the batch retains that resolved serving configuration.
A Terminus-2 policy also receives the resolved output limit as `llm_call_kwargs.max_tokens`, which applies
the budget to ordinary chat-completion requests. An explicit lower request limit in the policy wins; a
request limit above `model_info.max_output_tokens` fails preflight.
A policy may state a lower limit to keep headroom under that window, and the lower limit wins:
`grug-opencode-id.yaml` asks for 64512 input tokens against a
model serving 65536. A policy limit above the served one fails preflight, before Iris opens, with
both values named. A limit neither the model nor the policy states falls back to Harbor's own
default, 32768 input and 8192 output tokens. `--dry-run` prints the resolved pair per Harbor eval
and `record.json` keeps it under `eval.harbor`.

Installed harnesses must enforce these limits themselves. Harbor's native Mini-SWE
adapter consumes `model_info` for the Mini-SWE 2.1.0 harness with hosted vLLM. The input limit
denotes the total context, including the reply. It requires the serving endpoint's
`/tokenize` route, counts the rendered prompt with tools and the generation prefix,
and caps the reply to the remaining context and the configured output limit.
Context exhaustion uses Mini-SWE's native limit handler so Harbor can verify the
existing patch. No Marin agent subclass is required.

Sotopia uses `harbor.agents.installed.sotopia:SotopiaAgent`, packaged with its
episode runner in Harbor. Judge API failures and exhausted missing-grade retries
become `VerifierRuntimeError`. Sotopia's native generation still does not enforce
`model_info` context limits.

## Results

Each Harbor evaluation writes:

- `{records_prefix}/{run_id}/record.json`
- a FineStore archive under `{records_prefix}/{run_id}/results/`, including normalized `samples`
  and `steps` tables plus the evaluator-neutral `rollouts_v1` conversation table
- Harbor-native job metadata, trial results, and trajectories preserved as FineStore objects

Harbor preserves each attempt's native result and trajectory as it finishes. At job finalization,
it writes normalized rows only for the selected final attempts, then adds job metadata and seals
the archive. Retried attempts retain separate source artifacts and do not contribute sample or
step rows. A resumed invocation rebuilds the selected tables, including derived rollouts, while
retaining earlier native attempt artifacts. Every final trial becomes an agentic `EvalSample`. The
verifier reward is stored as `Grading(method="harbor:verifier")`, and the trajectory is referenced
by `trajectory_uri`. Marin derives the `rollouts_v1` table from `steps`, normalizing each trajectory's
participants and ordered message, reasoning, tool-call, and observation parts. Evaldash reads the
normalized sample table directly. `record.json` stores the
deterministic source-policy digest and any Marin runtime task cap. A source policy's own `n_tasks`
remains part of the policy digest.

Each attempt commits its native result, trajectory, and trajectory-presence metadata in one
FineStore transaction. A result is terminal only after the trial's service task group and
environment cleanup finish. Resume can repair a partial archive from terminal native files;
missing required evidence fails the run and leaves the archive unsealed.
An attempt with unconfirmed completion retains native evidence with unknown trajectory presence,
but contributes no normalized rows. It remains retryable on resume and does not prevent a later
complete selected attempt from sealing the archive.
Queued trajectory writes drain before trajectory presence becomes authoritative. Terminus commits
its final trajectory through the required artifact lane; persistence failure leaves the attempt
incomplete. A selected incomplete attempt prevents sealing and fails the evaluation until it is
retried successfully.

Tau3's ACP bridge retains `agent/bridge-trajectory.json` as a raw ACPX export in the trial
directory. It does not currently convert that export to Harbor's ATIF `agent/trajectory.json`.
Those trials have scored samples and explicit trajectory absence in FineStore, with no derived
step or rollout rows. This is separate from recovering a missing declared ATIF trajectory.
