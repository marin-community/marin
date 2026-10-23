# Harbor evaluation

Marin runs Harbor benchmarks through the shared evaluation launcher. The launcher starts the model
server and, when requested, a second shared judge server; gives Harbor their Iris capability URLs; and normalizes completed trials into v2
`EvalRunRecord`/`EvalSample` artifacts, and tears inference down after the selected evaluations
finish.

Harbor provides containerized agent benchmarks such as Terminal-Bench, SWE-bench Verified, AIME,
GAIA, BFCL, and Aider. Trials can run in Daytona or another Harbor-supported sandbox environment.
See [Running Evaluations with Marin](tutorials/run-lm-evals.md) for the model and suite command
matrix. This page documents Harbor-specific configuration and output behavior.

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
those harnesses require LiteLLM. Use Mini-SWE or Terminus instead. NeMo's
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
and the original model, policy, runtime, task limit, and hosted judge settings. Marin rejects the
resume unless the root's saved model-bound identity matches those inputs. Wait for the original
Iris group and any earlier retry to end first: two live Harbor drivers must never share a results
root. Add `--retry-unscored-harbor-trials` to retry known unscored trials while preserving scored
outcomes and unknown errors.

## Credentials

Daytona-backed definitions in `experiments/evaluation/evals.py` own their credential references.
They use `DAYTONA_API_KEY` from the launch environment when present, then fall back to the approved
Google Secret Manager version. `DAYTONA_API_KEY` is the only supported environment override.

The generic launcher resolves declared references immediately before Iris submission. The isolated
Harbor subprocess receives the Daytona key without inheriting the orchestrator's other credentials.
Do not put resolved credentials in model YAMLs, runner configs, or evaluation artifacts.

## Sandbox environment isolation

Keep provider credentials in the harness process. An actor sandbox should receive
only its image's configuration, backend runtime defaults and explicitly declared
task or command variables. Treat task definitions, images, build contexts and
explicit mounts as trusted inputs: they can deliberately supply credentials.
An explicit task environment declaration such as `API_KEY = "${HOST_KEY}"`
resolves against the harness environment and grants that access. A `${HOST_KEY}`
reference appearing only in a Compose file does not request host forwarding.
Below, persistent variables are explicit `persistent_env` / `--ae` settings;
Harbor infrastructure values name images, resource limits and log mount paths.

| Backend | Environment and configuration supplied by the launcher |
| --- | --- |
| Shellbox local / bubblewrap | Fixed `PATH`, `HOME`, `LANG`, optional hash seed and explicit task/command variables. Task variables take effect inside bubblewrap and reach it through a private inherited descriptor, keeping values out of host process listings. Private `/etc` copies allowlisted runtime files; host startup files, password hashes, private keys, `/opt` contents and sysfs are hidden. Shared program directories and explicit mounts remain trusted inputs. |
| Shellbox Docker / local gVisor | Image configuration and explicit task/command variables; no host directory mounts or implicit forwarding of host variables in Shellbox. Docker client configuration, including automatic proxy settings, remains an operator-controlled input. `runsc` changes the execution boundary, not credential policy. |
| Shellbox Daytona | Image/snapshot configuration and explicit task/command variables. The Daytona client keeps its API credentials in the harness; creation forwards only `MachineSpec.env`. |
| Shellbox Iris | The `SANDBOX` profile excludes parent environment inheritance, workspace bundles, cluster secrets, task tokens, shared caches and service-account tokens. Kubernetes service links are disabled. `secret_env` is an explicit exception for trusted graders and must never be used for actors. |
| Shellbox QEMU | Image configuration and explicit task/command variables reach the guest over its serial protocol. The QEMU host process uses host runtime settings; these do not become guest variables. No host filesystem is shared with the guest. |
| Shellbox ShellSim | Simulator defaults and explicit task/command variables in a private virtual filesystem. |
| Harbor Docker Compose | Explicit task/persistent variables and Harbor infrastructure values. The local Docker client receives only allowlisted connection, login-path and OS runtime variables from the host. Other host variables are unavailable for Compose interpolation. |
| Harbor Daytona / Modal Compose | Explicit task/persistent variables and Harbor infrastructure values. Compose references alone do not copy host variables into the remote sandbox. |

Iris's legacy `CONTAINER_PROFILE_GVISOR` retains cluster credentials and is for
trusted workloads. Model-controlled work must use `CONTAINER_PROFILE_SANDBOX`.
See the [Iris profile reference](https://github.com/marin-community/marin/blob/main/lib/iris/docs/container-profiles.md)
and the [Shellbox backend reference](https://github.com/marin-community/marin/blob/main/lib/shellbox/README.md).

These contracts describe repository launch paths. They do not certify deployed
images, existing snapshots, provider-side injected settings or Docker daemon
configuration. Keep those inputs free of credentials. Local bubblewrap is for
trusted commands; its `ALLOW` network policy shares the host network. Local
Docker/gVisor bridge networking can also reach host or cloud metadata services.
Use denied networking when unnecessary, or Iris's filtered sandbox egress for
model-controlled workloads on production infrastructure.

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

The runtime uses `vendor/harbor/uv.lock`. Its runtime descriptor records the upstream
revision from `UPSTREAM.toml`, the lock digest, and a digest of the packaged source and local Marin
libraries. The upstream revision identifies the import base; execution always uses the vendored
source. Published September cohort labels remain validated against their original runtimes and
cannot label new vendored runs. Reproduce historical runs from their recorded Marin checkout.
The vendored package evaluates prebuilt task datasets; dataset generators are omitted.
See the [vendor maintenance guide](https://github.com/marin-community/marin/blob/main/vendor/harbor/README.md)
for dependency updates and direct evaluation CLI commands.

### Trial error taxonomy

Preflight snapshots Harbor's infrastructure, agent, passthrough, and undecided error categories from
the selected Harbor runtime in the isolated environment. Marin uses that snapshot to classify trial results and records
the Harbor commit SHA in `record.json`. Undecided errors are known to Harbor but have no default scoring
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

Harbor writes each verifier-scored trial as it finishes, then adds surviving ungraded attempts and
job metadata before sealing the archive. Every final trial becomes an agentic `EvalSample`. The
verifier reward is stored as `Grading(method="harbor:verifier")`, and the trajectory is referenced
by `trajectory_uri`. Marin derives the `rollouts_v1` table from `steps`, normalizing each trajectory's
participants and ordered message, reasoning, tool-call, and observation parts. Evaldash reads the
normalized sample table directly. `record.json` stores the
deterministic source-policy digest and any Marin runtime task cap. A source policy's own `n_tasks`
remains part of the policy digest.
