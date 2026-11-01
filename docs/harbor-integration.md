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
overrides only the verifier's `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MODEL_NAME`; the evaluated
agent continues to use the candidate endpoint. Candidate and judge placement must resolve to the
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
Shellbox's Docker backend remains available to callers such as SkyRL's engine-owned
machine path; it is not a Harbor execution provider.

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

Place this policy beside `prepared-tasks` inside the Marin workspace, then use the shared launcher:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-8b --harbor-config path/to/policy.yaml --limit 1 --dry-run
```

Remove `--dry-run` to submit. Each task must set `allow_internet = false`. A single-container task
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
For example, prepare a reviewed tau3 task snapshot on a trusted image builder. These commands
preserve the runtime's per-task configuration and add Terminus's missing terminal dependencies:

```bash
TASK_DIR=prepared-tasks/tau3-airline-0
MAIN_TAG=registry.example/harbor/tau3-airline-0-main:prepared
RUNTIME_TAG=registry.example/harbor/tau3-airline-0-runtime:prepared
cp "$TASK_DIR/environment/Dockerfile" "$TASK_DIR/environment/Dockerfile.prepared"
cat >> "$TASK_DIR/environment/Dockerfile.prepared" <<'DOCKERFILE'
RUN apt-get update && apt-get install -y --no-install-recommends bash tmux \
    && rm -rf /var/lib/apt/lists/*
DOCKERFILE
docker buildx build --platform linux/amd64 --push --tag "$MAIN_TAG" \
  --file "$TASK_DIR/environment/Dockerfile.prepared" "$TASK_DIR/environment"
docker buildx build --platform linux/amd64 --push --tag "$RUNTIME_TAG" \
  "$TASK_DIR/environment/runtime-server"
MAIN_DIGEST=$(docker buildx imagetools inspect "$MAIN_TAG" --format '{{.Manifest.Digest}}')
RUNTIME_DIGEST=$(docker buildx imagetools inspect "$RUNTIME_TAG" --format '{{.Manifest.Digest}}')
cat > "$TASK_DIR/environment/shellbox-images.json" <<JSON
{
  "main": {"image": "$MAIN_TAG@$MAIN_DIGEST", "workdir": "/app"},
  "tau3-runtime": {
    "image": "$RUNTIME_TAG@$RUNTIME_DIGEST",
    "workdir": "/app",
    "agent_logs": "publish"
  }
}
JSON
```

Use an existing authenticated registry namespace in place of `registry.example`. Review the source
Dockerfiles before building, set `allow_internet = false` in the prepared task's `task.toml`, and
retain the prepared snapshot and manifest as Marin artifacts. These preparation commands require
trusted build infrastructure; they are never run by the evaluation sandbox.
The primary stays alive for agent commands and verification. Sidecars retain their prepared
image's ENTRYPOINT/CMD unless Compose explicitly overrides them. The adapter does not build
images or install a nested container runtime.

The agent makes model calls in the trusted Harbor process. Shellbox runs task commands and
verification in persistent bubblewrap containers. Members share private loopback and retain
separate filesystems and PID namespaces. External traffic is denied, including DNS.
Provider credentials stay in the trusted harness.
Task environment templates resolve only against explicit task capabilities. They never read the
driver's ambient environment, including through differently named aliases.
Image contents and explicit task inputs must themselves be free of production credentials.

The runtime has no independent Docker or other provider execution backend. The prepared local path also rejects installed agents,
custom agent imports and custom host mounts. Its Compose importer supports `main` plus prepared
sidecars, commands, environment values, exposed TCP ports, readiness checks and `main` dependencies
on started or healthy services. MCP URLs must name an exposed endpoint inside the container group.
Unknown Compose fields, published host ports, host networking, extra Compose files and dependencies
between sidecars fail explicitly.

Containers share only the group's loopback network. External egress remains denied. Tau3's user model
and verifier receive a loopback OpenAI endpoint and an `EMPTY` key. A trusted host relay forwards only
chat completions and tokenization to the selected model, fixes its model name, and rejects redirects;
the upstream endpoint capability and provider authentication stay in the host process.
Tau3 requires an explicit `--judge-model` in the shared launch command. That hosted model serves
both the simulated user and verifier assertions; the candidate agent keeps its own model endpoint.
Without a hosted model, the adapter rejects the task before allocating its container group. A direct
Harbor caller supplies `model_endpoint` and `model_name` and may grant endpoint authentication with
`harbor.environments.model_relay.host_model_authentication`; the adapter never reads an ambient model key.

The relay starts one Python HTTP helper and exchanges bounded request/response
frames over a persistent process session. The session uses owned local process
pipes or Daytona's SDK transport; provider credentials remain on the host.

Installed candidate agents on Daytona use a separate loopback relay on port 8788;
the simulated-user and verifier relay uses port 8787. Marin resolves installed
agents through Harbor's agent classes, including Pi, OpenCode, Mini-SWE, and ACP Pi.
Their provider configuration and environment contain a loopback URL and an `EMPTY`
key. The candidate endpoint and its authentication remain in the trusted driver;
the executor supplies the served endpoint's API key only through the driver environment.
Host-side agents such as Terminus and the AIME agent call their candidate endpoint directly.
Persisted job and trial JSON redacts Iris bearer URLs; live configurations retain their routes.
Resumed runs receive a fresh model capability from Marin rather than reusing one from saved JSON.

Both relays authorize only `POST /v1/chat/completions`, `POST /v1/responses`, and
`POST /tokenize`. They fix the upstream model, reject redirects, and bound request
and response bodies. The Responses route supports non-streaming text input with
an output-token limit and reasoning options; tools, stored responses, and stateful
continuations are unsupported.
The tokenization route uses the configured upstream's server root. For installed
SDKs requesting streaming chat, the host obtains one non-streaming completion and
returns bounded SSE frames containing its text, reasoning, tool calls, finish reason,
and usage. This adds no streaming transport: the guest receives the complete buffered
response after generation finishes. Tasks using either relay need Python 3.

The recognized Harbor agent-log bind is lowered only when the prepared sidecar declares
`agent_logs: "publish"`. Before actor commands and artifact downloads, the adapter copies sidecar logs
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

An explicitly delegated cgroup v2 `cgroup_parent` enables CPU and memory limits
across each container's processes and descendants. Use `limit` or `auto` to
apply the task's declarations. The worker must run within that delegated
hierarchy; see the [Shellbox delegation instructions](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md#local-backend).
For example:

```yaml
environment:
  type: local
  cpu_enforcement_policy: limit
  memory_enforcement_policy: limit
  kwargs:
    cgroup_parent: /sys/fs/cgroup/path/to/delegated-worker
    ttl: 3600
```

Without delegation, explicitly select `ignore` for declared CPU or memory values;
`auto` does not silently discard them. Storage quotas, GPU and TPU allocations
remain unsupported. For a worker without delegation:

```yaml
environment:
  type: local
  cpu_enforcement_policy: ignore
  memory_enforcement_policy: ignore
  kwargs:
    ttl: 3600
```

Run the policy through the isolated Harbor binary on that worker:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  harbor run --config local-policy.yaml
```

Model and judge settings follow the host bindings described above. A command timeout or
cancellation closes the entire local machine or group. Local execution shares the host kernel;
use it for trusted tasks on isolated workers. Iris remains the default production boundary.

The local backend uses the same Docker/OCI image contents; it unpacks the layers
instead of starting a Docker container. A prepared image can remain local.
Export the host-built image with Skopeo into an OCI layout tagged `image`, put
that layout at `<image_cache>/sha256-<platform-manifest-digest>`, and reference
the same platform manifest digest in `shellbox-images.json`. Shellbox verifies
the manifest, configuration and every layer before using that cache offline.
An explicit signature policy still requires registry verification.

To exercise the complete local Tau3 path with a real candidate, simulated user
and assertion judge, use a prepared task whose canonical reward includes an
NL assertion, such as `tau3-retail-2` from Tau3 1.0.1. Keep its runtime
`task_config.json`, instructions and verifier unchanged. Select Terminus 2,
configure a local tokenizer encoding for a hosted endpoint without `/tokenize`,
and put the upstream `model_endpoint` and `model_name` in the local environment
kwargs. The candidate's OpenAI client uses the host endpoint; the runtime and
verifier receive the loopback relay and an inert key. Supply CPU/memory policies
as above and omit the unsupported storage declaration from the prepared task.

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime \
  python vendor/harbor/tests/local_tau3_smoke.py local-tau3.json \
  --api-key-env GLM_BULK_TOKEN
```

The command retains normal Harbor trial artifacts and checks that Tau2 evaluated
a real runtime trajectory and returned NL assertion results. A model reward of
zero is a scored outcome; an infrastructure error or skipped evaluator fails
the smoke run. Task images contain the same MCP client and terminal dependencies
used by the candidate. No oracle solution is uploaded.

To inventory a local task snapshot without executing it:

```bash
uv run python -m experiments.evaluation.audit_harbor_tasks path/to/tasks \
  --source organization/dataset --revision SNAPSHOT_COMMIT > inventory.json
```

The report counts task manifests and Compose services, lists fields requiring review, and records
locations of Docker command references. Source and revision are supplied by the caller. A Compose
file can be a single-container wrapper; a command reference can be an unused helper. Neither
establishes a requirement for nested Docker. The inventory does not certify image contents,
external-access requirements or compatibility with offline execution.

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
Below, persistent variables are explicit `persistent_env` / `--ae` settings;
Harbor infrastructure values name images, resource limits and log mount paths.

| Backend | Environment and configuration supplied by the launcher |
| --- | --- |
| Shellbox local / bubblewrap | Fixed `PATH`, `HOME`, `LANG`, optional hash seed and explicit task/command variables. Task variables take effect inside bubblewrap and reach it through a private inherited descriptor, keeping values out of host process listings. Private `/etc` copies allowlisted runtime files; host startup files, password hashes, private keys, `/opt` contents and sysfs are hidden. Shared program directories and explicit mounts remain trusted inputs. |
| Shellbox Docker / local gVisor | Image configuration and explicit task/command variables; no host directory mounts or implicit forwarding of host variables in Shellbox. Docker client configuration, including automatic proxy settings, remains an operator-controlled input. `runsc` changes the execution boundary, not credential policy. |
| Shellbox Daytona | Image/snapshot configuration and explicit task/command variables. The Daytona client keeps its API credentials in the harness; creation forwards only `MachineSpec.env`. |
| Shellbox QEMU | Image configuration and explicit task/command variables reach the guest over its serial protocol. The QEMU host process uses host runtime settings; these do not become guest variables. No host filesystem is shared with the guest. |
| Shellbox ShellSim | Simulator defaults and explicit task/command variables in a private virtual filesystem. |

See the [Shellbox backend reference](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md)
for each backend's isolation and resource limits.

These contracts describe repository launch paths. They do not certify deployed
images, existing snapshots, provider-side injected settings or Docker daemon
configuration. Keep those inputs free of credentials. Bubblewrap shares the
worker's kernel and should run on an isolated worker. Its `ALLOW` network policy
shares the host network. Docker/gVisor bridge networking can also reach host or
cloud metadata services. Use denied networking and the host relay for task code.

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
For a Harbor target or simulated user agent with `name: pi`, Marin always sets `kwargs.version` to `0.87.0`, overriding
versions supplied by the policy or model configuration. Preflight includes this pin in the saved
policy and the saved policy's digest; execution applies the same pin. Pi evaluations install
version `0.87.0` without resolving npm's `latest` tag.
Temporary policy and overlay files are owner-readable and removed after each isolated call.

The runtime uses `vendor/harbor/uv.lock`. Reproduce runs from their recorded Marin checkout
and artifact provenance, with pinned dataset revisions and policies.
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
