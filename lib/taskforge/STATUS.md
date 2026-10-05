# Taskforge status

The single account of Taskforge after phase 2 (RolloutEngine rewrite): what works, what is broken
or not run, the upstream changes it needs, the PR stack, and what must hold before phase 3
(`review/`, `loop/`, `queue/`). Every claim about a model-facing or cluster-facing component names
its evidence under `.evidence/<package>/`. Evidence is gitignored and exists only on the laptop
that ran the check (see `.evidence/README.md`). Last full verification: 2026-10-05, 20:39 UTC.

## Summary

Taskforge is a standalone uv project in `lib/taskforge` of the `taskforge/base` worktree
(`/Users/k3sc0re/openathena/branches/marin-taskforge`), stacked on PR 9623 (`rollout-engine`,
now at `7bbe4b9cd7`). It builds on TaskCompendium 0.22, `lib/rolloutengine`, `lib/shellbox` and
`lib/verifyit`. Silo and the Harbor lowering are deleted (DESIGN.md decisions 1 and 2).

Phase 2 built `llm.agent`, `llm.web`, `llm.rollout_model`, `sandbox.factories`,
`sandbox.images` with an Iris image builder, `spec.draft`, `spec.controls`, `validate` and
`build`. All of them pass their unit tests and their live checks on the GLM-5.3 interactive
endpoint, with these failures and gaps:

- **Broken: `llm.rollout_model` breaks the exact-token contract when GLM samples a
  non-canonical tokenization.** The chat endpoint re-tokenizes the replayed text, so sampled ids
  such as `")," "("` come back as the canonical `"),("`. It failed 2 of 7 reruns of the d01 build
  draft. No client-side message construction can fix it; it needs a tokens-in path the GLM router
  does not offer. See Validate.
- **Broken: the shipped shellbox Iris backend fails every `create`.** Docker tasks run on Iris
  only with the readiness-poll fix from `docs/upstream/shellbox/iris-machine.patch` (unapplied),
  which the cluster probe applied in-process. With it, a docker task graded 3/3 on cw-rno2a.
- **Broken: model-controlled sandboxes see credentials.** On cw-rno2a every sandbox carries the
  cluster's object-store keys (`AWS_*`, `CW_KEY_*`) from Iris `task_env`, and Iris child jobs
  inherit any `-e` secret of the parent unless the parent scrubs it.
- **Open security action: rotate `capability-registry-publisher`.** The first image builder put
  it in an exec argv that the cw-us-east-02a controller logged twice. The builder no longer does
  this; the credential has not been rotated.
- **Not run:** Docker on the laptop (no daemon), an Iris image built by Taskforge running in an
  Iris sandbox (workers cannot pull from envreg), hundreds-wide concurrency (20 tested), a cluster
  Finelog write, and the template's model-driven build steps.

Verification on 2026-10-05, from `lib/taskforge`, against the committed stack head:

- Unit: `uv run --group test pytest tests`: **212 passed, 25 skipped** in 3.9s.
  `uv run --group test pytest tests/build`: **23 passed, 1 skipped** in 0.7s. Pytest's default
  `norecursedirs` skips directories named `build`, so `pytest tests` never collects
  `tests/build`; run it explicitly. Log: `.evidence/status/unit-20261005T203919Z.txt`.
- Live, GLM-5.3 interactive tier (router `/health`: `high=4, bulk=33`):
  `uv run --group test pytest tests -m live_glm`: **25 passed** in 909.8s (agent 8, llm 6,
  rollout_model 2, endpoint 1, proposal 1, triage 1, validate 6), and
  `pytest tests/build -m live_glm`: **1 passed** in 32.4s, 20:39-20:55 UTC. Log:
  `.evidence/status/live_suite-20261005T203930Z.txt`; this run's proposal and triage evidence is
  `.evidence/proposal/run-20261005-164056` and `.evidence/triage/run-20261005-164654` (local
  time names). A scan of `.evidence/status` and this file for the GLM token and Parallel key
  found nothing. The d01 rollout that breaks the prefix contract is not part of the suite.
- Types: `uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check`: 0 errors (1 suppressed, 6 warnings
  not shown). `tests/` and `scripts/` are not in `project-includes`.
- Lint: `./infra/pre-commit.py --files` on all 118 tracked `lib/taskforge` files: OK.

## How to run

Run every command from `/Users/k3sc0re/openathena/branches/marin-taskforge/lib/taskforge`
unless it says otherwise. The package has its own `uv.lock` and `.venv`; `uv run` there uses
them, not the root workspace. Do not add flags such as `-m "not slow"`: the package `addopts`
already sets the marker expression.

Unit tests (live tests skip without the env below):

```bash
uv run --group test pytest tests
uv run --group test pytest tests/build   # not collected by `pytest tests` (norecursedirs)
```

Live tests on the GLM-5.3 interactive tier. The router port-forward must be up
(`curl -s http://127.0.0.1:18000/health` returns `"status":"ok"`; DESIGN.md has the restart
command). The proposal and triage live tests need the gitignored capability catalog, and the
agent and build live tests need the Parallel key (`PARALLEL_KEY=` line in
`~/openathena/build_envs/.parallel_key`, override with `TASKFORGE_PARALLEL_KEY_FILE`); each
skips with a reason when its input is missing.

```bash
export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export TASKFORGE_GLM_TOKEN_FILE=~/openathena/glm-infer/glm_api_token.txt
export TASKFORGE_CAPABILITY_CATALOG=/Users/k3sc0re/openathena/branches/marin-autoenv/experiments/post_training/capability_env_gen/new_catalog.json
uv run --group test pytest tests -m live_glm
uv run --group test pytest tests/build -m live_glm
```

`-m live_glm` replaces the package's default marker expression. Live tests take the session
fixtures `glm_settings` (the token is excluded from `repr`), `capability_catalog` and
`parallel_key`. The package pytest config sets `timeout = 60` and `asyncio_mode = "auto"`; long
live tests carry `@pytest.mark.timeout(<seconds>)`. The triage live test scores every proposal
under `.evidence/proposal/`, so its input and cost grow with each proposal run.

Types (the package has its own `[tool.pyrefly]`; it checks `src` only):

```bash
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

Lint, from the worktree root, with paths relative to it:

```bash
./infra/pre-commit.py --fix --files lib/taskforge/<path> ...
```

In zsh, pass a file list with `xargs` or `${=files}`; an unquoted `$files` does not word-split.

Cluster scripts (Iris, no GLM unless stated; usage is in each docstring):
`scripts/build_image_job.py` (image builder), `scripts/iris_machine_probe.py` (shellbox Iris
backend probe), `scripts/cluster_rollout_probe.py` (validate trials inside an Iris task with GLM
resolved in-task; submit command in its docstring and under Cluster probe below).

## Packages

| Package | State | Evidence |
|---|---|---|
| llm (client, structured, store, endpoint) | Works on the laptop endpoint. `GlmClient` reports a tool-call reply that spent its whole budget as `length`. Continuation is lossy at seams. `resolve_glm_base_url` worked inside an Iris task on cw-rno2a. | `.evidence/llm/`, `.evidence/standalone/`, `.evidence/cluster/` |
| llm.agent, llm.web | Works: 8/8 live checks. Concurrency tested at 20 agents only. ShellSim only. | `.evidence/llm/agent/` |
| llm.rollout_model | Works on short and reasoning-heavy ShellSim rollouts. **Breaks the token-prefix contract when GLM samples a non-canonical tokenization (2 of 7 d01 reruns); needs a tokens-in path.** | `.evidence/validate/rollout_model/`, `.evidence/build/run-2/` |
| ledger | Works locally (JSONL, Finelog round trip on finelog's embedded server). Validate and build emit spans. Cluster Finelog write not run. | `.evidence/ledger/` |
| sandbox | Factory selection and up-front refusal work. Iris image builder works end to end (build, push, adversarial and failing builds). Iris DOCKER reported unavailable: shipped backend broken. Laptop Docker not run. | `.evidence/sandbox/` |
| spec | Assembled specs run end to end on ShellSim through `ShellboxRolloutEngine` (unit tests). Docker and Iris specs round-trip only. No model calls. | `tests/spec/`; phase 1 in `.evidence/phase1-spec/` |
| proposal | Works. Latest live run 20/20 parsed. Null-slot path unit-tested only. | `.evidence/proposal/` |
| triage | Works. 103 proposals x 3 samples, 0 output errors, 31 ACCEPT / 72 REPAIR. REJECT never seen live. | `.evidence/triage/` |
| validate | Trials, classification, retries and control replay work live on null and ShellSim tasks (8/8), and on cw-rno2a for null and docker tasks (with the readiness fix). Staged-task control replay not built. | `.evidence/validate/`, `.evidence/cluster/` |
| build | Two real ACCEPT proposals built to validated TaskSpecs and rebuilt from cache. Model-driven template steps never ran live. Imports two private RolloutEngine names. | `.evidence/build/` |
| review, loop, queue | Not started (empty packages). | |

## LLM

`taskforge.llm.client.GlmClient` is the one GLM-5.3 transport (async, streaming httpx, 512
pooled connections, 429/5xx retry, router holds, stall timeouts, continuation on `length`).
`llm.structured` owns `StructuredTool` (forced strict tool call, one repair) and
`complete_structured`; `StructuredTool.parse` takes the completion's `ToolCall`s. `CallStore`
caches calls by request hash. `llm.endpoint` holds `GLM_MODEL` and `resolve_glm_base_url`
(ported from `experiments/post_training/glm.py`), so nothing imports `experiments` or
`marin-core`. `httpx` is bounded `<1`: with `prerelease = "allow"` uv picked a 1.0 dev release
without `AsyncClient`.

Since review: a completed segment with `finish_reason: tool_calls` and `completion_tokens >=
max_tokens` is reported as `FinishReason.LENGTH`, because vLLM's GLM tool parser reports a
budget-cut tool call as `tool_calls` (`.evidence/llm/agent/probe_tool_call_cut-20261005T200437Z.json`:
raw `tool_calls`, reported `length`, 300/300 tokens). The raw events keep what vLLM sent.

Live checks (`tests/llm/test_live.py`, 6 checks; latest pass in the status run above; promoted
files under `.evidence/llm/`, every run under `.evidence/llm/runs/`):

- `a_plain_completion.json`: `max_tokens=131072`, `stop`.
- `b_context_overflow.json`: a 250,055-token prompt at `max_tokens=131072` gets a 400; a one-token
  probe measures the prompt and the retry at the remaining context succeeds. The server enforces
  prompt + max_tokens <= 262,144, not a separate output cap.
- `c_continue_on_length.json`: continues to `END`. **Seams are lossy** (duplicated phrases,
  dropped spaces). The test checks only that it continued and ended in `END`. It failed once in 5
  runs when a segment ended exactly on `END` with `finish_reason: length` and the continuation
  appended meta prose (`.evidence/standalone/live_suite-20261005.txt`). Not fixed.
- `c_continue_mid_reasoning.json`: continued mid-reasoning, but **the answer was wrong** (397
  against 401 uncut). The test checks only for a non-empty answer.
- `d_structured.json`, `d_structured_repair.json`: forced tool call validated; the repair
  recovered a validator failure. GLM sometimes ignores a forced `tool_choice` and answers in
  prose; the repair recovers it.
- Iris: `resolve_glm_base_url` resolved `http://10.168.192.79:8010/v1` inside an Iris task on
  cw-rno2a through relay `/muchanem/glm53-relay-rno2a` (`.evidence/cluster/probe-01-logs.txt`).

Open: `CallStore` does not record failed calls or deduplicate concurrent identical requests, and
stores every raw SSE event (the triage evidence is about 550 MB). `decode_tokens_per_second` is
meaningless for very short replies. The default relay in `experiments/post_training/glm.py`
(`/muchanem/glm53-relay-08a`) is dead; callers must name a live relay per cluster.

## Agent loop

DESIGN.md decision 4: the Taskforge-owned loop. `llm/agent.py` (`run_agent`, `shell_tool`) and
`llm/web.py` (`web_tools`: Parallel search and extract over a caller-owned `httpx.AsyncClient`,
retrying 408/429/5xx and transport errors with `ExponentialBackoff`). Tool arguments are checked
against each tool's JSON schema; every rejected call (`INVALID_ARGUMENTS`, `UNKNOWN_TOOL`,
`TRUNCATED_CALL`) goes back to the model as an error result. One `LLM_CALL` span per turn and one
`STEP` span per tool call. Non-object arguments are replayed wrapped as `{"invalid_arguments":
...}`, because the server returns HTTP 400 for them raw. Decision record:
[docs/agent_loop_decision.md](docs/agent_loop_decision.md).

Live, 2026-10-05 20:04 UTC (`.evidence/llm/agent/live_suite_review-20261005T200349Z.txt`, 15
passed), all under `.evidence/llm/agent/`:

| Check | Result | Evidence |
|---|---|---|
| T1 package and tests in ShellSim | answered, 6 turns, 10.0s; independent pytest rerun green | `t1_shellsim-20261005T200400Z.json` |
| T1b seeded failing package, fix loop | answered, 8 turns, 7 shell calls, 9.2s; rerun green | `t1b_fix_loop-20261005T200409Z.json` |
| Web search through Parallel | answered, 4 turns, 1 search; answer holds PyPI's 3 latest versions, checked against PyPI's JSON API before and after | `web_parallel-20261005T200424Z.json` |
| Malformed tool call injected by a relay | `invalid_arguments` returned, replay accepted, recovered in 3 turns | `malformed_tool_call-20261005T200426Z.json` |
| 20 concurrent agents on one `GlmClient` | 20/20 correct, 1.6s total; tiny task, not a load test | `concurrent_20-20261005T200427Z.json` |
| Cut inside a tool call (`max_tokens=400`) | turn 1 `length`, call returned as `truncated_call`; answered after 6 turns, 300 lines | `length_cut_tool_call-20261005T200435Z.json` |
| Probe: tool call cut at `max_tokens=300` | vLLM sent `tool_calls`, `GlmClient` reported `length` | `probe_tool_call_cut-20261005T200437Z.json` |
| Probe: replaying non-object arguments | raw malformed and `[1]` both HTTP 400; wrapped forms accepted | `probe_replay_arguments-20261005T200437Z.json` |

The first web check (`web_parallel-20261005T184045Z.json`) failed: the model went straight to
`web_fetch` without searching; the prompt now says to start with `web_search`.

Open: `jsonschema` is imported by `agent.py` but reaches the venv only through
`verifyit[schema]`; declare it in `pyproject.toml`. `shell_tool` copies RolloutEngine's shell tool
definition and observation format and imports `SHELL_TOOL_NAME` from non-public
`rolloutengine.task_session`. Not tested: hundreds-wide concurrency, a real stream stall, a
GLM-produced malformed call (only injected), Docker or Iris machines.

## Ledger

`taskforge.ledger.records.span(...)` times a block and records a `LedgerEntry` on success or
failure, with a caller-set `cause`. `JsonlLedger` writes one file per item with single `O_APPEND`
writes; `FinelogLedger` writes table `taskforge.ledger`; `run_ledger` picks JSONL off-cluster and
JSONL plus Finelog under Iris. `scripts/ledger_summary.py` reports counts, failures and tokens.
`entry_from_json` was retyped for pyrefly 1.0; behavior is unchanged.

Works (13 unit tests; `.evidence/ledger/`): Finelog round trip on finelog's embedded server, 4
processes x 200 concurrent appends with no torn lines, off-cluster JSONL fallback. Emitters now
exist: `llm.agent` (`LLM_CALL`, `STEP`), `validate.trials` (`TRIAL`), `build` (`LLM_CALL` per
author call and per step model call).

Not run: a write to the cluster Finelog from inside an Iris task. Nothing computes
`code_hash`/`input_hash`/`output_hash`.

## Sandbox

`sandbox/factories.py` and `sandbox/images.py`; the image builder is `scripts/build_image_job.py`
with its push job `scripts/push_image_task.py`.

- `machine_factories(MachineHost)` returns the `EnvironmentKind -> MachineFactory` mapping
  `ShellboxRolloutEngine` takes: ShellSim always; on LAPTOP, Docker when a daemon and Skopeo are
  found (probed once); on IRIS, `IrisMachineFactory` on `$IRIS_CONTROLLER_URL` (fails fast if
  unset).
- `factory_capabilities` and `task_refusals` refuse a task up front with typed `Refusal`s (no
  factory, image source, network, execution user, resource limits, GPUs). They cover the task
  machine, agent, stage and collect users, shell-verifier grading environments, and the root
  commands RolloutEngine runs itself (fetching AUTO, SKIP or excluded-directory artifacts;
  removing stage graders' private files). A test calls the real shellbox factories and checks they
  refuse what the table says; it trips when the Iris patch lands.
- Iris DOCKER is reported unavailable, so every docker task is refused on Iris today. Its row
  describes the patched backend: registry images only, `NetworkPolicy.DENY` only, no execution
  users, no GPUs.

Image builder, live on Iris (marin hub federated to cw-us-east-02a, DEFAULT profile): a build job
runs kaniko `--no-push` in a pinned bash image and writes `image.tar` to `$IRIS_OUTPUT_DIR`, which
Iris's uploader archives with a sha256; a separate push job in a pinned `python:3.14-slim` image
downloads and checks the archive, fetches a sha256-checked `crane`, and pushes to a tag unique to
the publish. The credential reaches only the push job, as env `REGISTRY_AUTH`, which Iris redacts
in job status. The submitter reads the manifest back by tag and checks linux/amd64.

| Check | Result | Evidence (`.evidence/sandbox/`) |
|---|---|---|
| Build and push, split builder | works: build 49.1 s, total 79.2 s, `...taskforge-image-build-smoke@sha256:4429731d...`; `REGISTRY_AUTH` redacted in the stored request | `build_image_job-split-run1.txt` |
| Adversarial Dockerfile (writes into `/kaniko`, spoofs state, watches `/proc` for the credential) | works: published; RUN env held only HOME, PATH, PWD; no credential observed | `build_image_job-split-adversarial.txt`, `-adversarial2.txt` |
| Failing Dockerfile (RUN exit 7) | works: `BuildFailed` with `exit status 7`, no push job | `build_image_job-split-failing.txt` |
| Shipped `IrisMachineFactory.create` | **broken**: `AttributeError: 'TaskStatus' object has no attribute 'error'` on every create | `iris_machine_probe-06.txt` |
| Iris create with the patched readiness poll (us-west4-a) | partial: sequential creates 5.7-65.3 s, 8 concurrent 10.5-17.8 s; uploads over about 96 KiB fail with 128 KiB chunks; TTL enforced 1-2 min late | `iris_machine_probe-05.txt`, `-06.txt` |
| Iris sandbox from a Taskforge-built envreg image | **broken**: GCP workers cannot pull (no credential) | `iris_machine_probe-05.txt` |
| gVisor network | marin GCP: no DNS or egress (effectively DENY); cw-rno2a: DNS works; v4 us-central2-b: gVisor does not start; cw-us-east-02a: pods hang in PodInitializing | `explore/`, `.evidence/cluster/gvisor_check_rno2a.txt` |
| Rootless alternatives (buildah) | broken on CW DEFAULT; kaniko image cannot be an Iris task image (no bash) | `explore/` |
| `machine_factories(LAPTOP)` with Docker | not run: no Docker on this laptop | |

Open:

- **Rotate `capability-registry-publisher`.** The old single-job builder delivered it through an
  exec argv; the cw-us-east-02a controller log has 2 such lines (`credential_exposure_check.txt`,
  counted, not printed). Rotation needs the registry owner and a new Secret Manager version;
  `SECRET_VERSION` in `build_image_job.py` stays `"1"` until then.
- The push job's pod spec carries the credential in plain env (readable by cluster operators);
  the registry offers only Basic auth. Every CoreWeave task pod, including the build job, carries
  the cluster's object-store keys, which RUN steps can read. Archives over 2 GiB cannot be
  published. Builds are not reproducible (different digests per run); references are always
  pinned by digest.
- Killed or expired Iris sandboxes raise untyped `RuntimeError`.
- Phase-1 `.evidence/sandbox/silo_*` files remain; they describe deleted code.

## Spec

`spec/draft.py` assembles a TaskCompendium 0.22 `TaskSpec`: `environment` (with a `Resources`
dataclass), `file`, `shell_command`, `reward_file`, `shell_verifier`, `answer_verifier` (verifyit
exact, numeric, mcq, predicted_action; it is the build SDK's name for
`taskcompendium.grading.verifier_descriptor`), `staged`, `stage` and `assemble`. `assemble`
validates every grader once and returns the JSON round-tripped spec. It rejects a shell verifier on
a null environment, private grader content that is agent-visible (environment files, DockerBuild
context, any stage's `workdir_files`), grader files that overwrite environment paths, answer
verifiers on file or state answers, and stage `minimum_rewards` keys other than `reward` unless
the grader writes JSON reward files.

`spec/controls.py` holds fixed controls (`Transcript` or `Workspace` payload with an
`Expectation`). `validate_controls` checks the set against the task: required categories per stage
(known_correct, empty_or_malformed, plausible_wrong, shortcut or reward_hack), negatives at most
0.2, positives demanding more than 0.2, partial controls only on JSON reward-file graders (the
only graders whose `GradeResult.rewards` carry components), and payloads the task can replay.

Validation (offline only; no model calls): `pytest tests/spec`, 37 passed. Shell verifiers
(stdout, exit code, JSON reward file) and a staged task with `minimum_rewards` gating run on
ShellSim through `ShellboxRolloutEngine` and grade right and wrong answers as expected; answer
verifiers grade after a JSON round trip; a partial control's components (`reward 0.5, format 1,
value 0`) were checked once against the real engine with a scratch script.

Open: Docker and Iris execution of an assembled spec has not run (round-trip only). Math, judge,
composite and verifyit execution modes are not reachable from a TaskSpec (upstream U-V1, U-T4,
U-T5, U-T6). A `Transcript` cannot express malformed wire data (non-JSON tool arguments).
`spec/extensions/` is empty: no gap needed an interim module.

## Proposal

`proposal.model` parses and renders the TaskProposal document (YAML front matter plus six
required sections) and digests its canonical form. `CapabilitySource.propose(idea, n)` returns a
`ProposalBatch` with the planning call and one `SlotProposal` or `SlotFailure` per slot; a failed
slot does not drop its siblings; the plan allows at most ceil(n/3) slots per environment and
verification pairing.

Live (`tests/proposal/test_live.py`, d01.algebra.linear-transformations and
d27.reporting.close_measurement, n=10 each): `run-20261005-131404` 20/20 parsed, 1 plan repair,
1 document repair, all `stop`, 70-383 s and 12.5k-83.9k completion tokens per slot;
`run-20261005-142014` (after the standalone move) and `run-20261005-130516` also 20/20. Evidence:
`.evidence/proposal/`.

Open: GLM drops or mangles the front matter in about 5 of 100 documents, so the document repair is
load-bearing. The server does not enforce strict tool schemas. No live slot has come back null.
`RepoIdea` is a placeholder. The loop must decide what a `SlotFailure` does to its idea.

## Triage

Deterministic checks, then `GlmRubric` (seven flat integer axes, independent samples), then a
decision in code: ACCEPT or REJECT needs a strict majority of the per-sample rule, else REPAIR.
A FATAL check or a null proposal rejects with no model call. `GlmRubric.repair` makes one rewrite
call; the caller re-evaluates.

Live (`.evidence/triage/`): `run-20261005-132121` scored 103 proposals x 3 samples at temperature
0.7: 311 calls, 0 output errors, 31 ACCEPT / 72 REPAIR / 0 REJECT; 2 of 3 repairs reached ACCEPT.
Majority decisions agreed with an earlier run on 57 of 63 shared proposals.
`run-20261005-142618` repeated the live test after the standalone move.

Open: REJECT never seen live (unit-tested only). `resources_in_build_plan` has false positives.
Triage and build import `taskforge.proposal.model`; DESIGN.md says stage packages never import
each other. Either the seam type moves down a layer or DESIGN.md changes.

## Validate

`llm.rollout_model.GlmRolloutModel` is RolloutEngine's model callable over `GlmClient`. It
requests `return_token_ids` and logprobs, takes the exact served ids from the stream, cross-checks
them against usage, and requires `max_continuations == 0`: a continuation re-renders the prompt
and would break the exact-token contract, so a cut turn ends the rollout with stop reason
`length`. This is an exception to DESIGN.md decision 6 that DESIGN.md does not record yet.

`validate.trials.run_trials(task, plan, settings, model)` runs k trials through
`ShellboxRolloutEngine`. A task `task_refusals` rejects fails up front as `MACHINE_UNSUPPORTED`
with no machine. Each attempt is a `TRIAL` span and a rollout JSON. `classify` is the only
classifier: the task's own setup, healthcheck or capability failure is `TASK_SETUP` (not retried;
matched on RolloutEngine's error-message prefixes until it raises typed errors); retryable causes
back off with `ExponentialBackoff`; `agent_timeout` is a budget stop, graded with the engine's
partial grade or 0. `validate.controls.replay` replays controls as scripted `CONTROL` trials with
server-tokenized turns (2 one-token requests per scripted turn); a workspace control is written by
shell calls after setup, as an agent would. `validate.evidence` marks an item `Incomplete` with
counts by cause.

Live, 2026-10-05 20:21 UTC (`pytest tests/validate tests/llm/test_rollout_model.py -m live_glm`,
8 passed), under `.evidence/validate/`:

| Check | Result | Evidence |
|---|---|---|
| Multi-turn ShellSim rollouts keep the served prefix | 3 rollouts, prefix kept every turn, graded 1.0 | `rollout_model/20261005T202152Z.json` |
| Reasoning-heavy rollouts | 3 x 6 turns, 3,150-5,504 reasoning tokens on the long turn, reasoning replayed, multi-line heredoc arguments; prefix kept 18/18 | `rollout_model/reasoning-20261005T202218Z.json` (two earlier runs failed only the test's reasoning-token threshold: `-201951Z`, `-202039Z`) |
| Null-environment math task, k=3 | 3/3 graded 1.0; uses the numeric verifier because `math` mode is unreachable | `a_math_trials-20261005T202146Z/` |
| ShellSim task with a private shell grader, k=3 | 3/3 graded 1.0 | `b_shellsim_trials-20261005T202146Z/` |
| Control replay on both tasks | every control MET, including the workspace control written by shell turns | `c_controls_math-20261005T202147Z/`, `c_controls_shellsim-20261005T202147Z/` |
| Forced start failures | 2 classified `machine_start` and retried, 3/3 graded; `UnsupportedMachineSpec` not retried | `d_forced_failure-20261005T202148Z/`, `d_forced_unsupported-20261005T202150Z/` |

**Broken, not fixable in the client:** a rollout breaks the exact-token contract whenever GLM
samples a token sequence that is not the tokenizer's canonical encoding of its text. Chat
completions take messages, so each turn's prompt is the re-rendered conversation re-tokenized, and
the replayed text comes back with different ids. Recorded live on the d01 build draft
(`.evidence/validate/rollout_model/prefix-bug/`): `record-1.json` turn 3 sampled `"1" ")," "(" "1"`
(ids 16, 701, 7, 16) inside `smul(2**(n-1),(1,0))`, and turn 4's prompt carries the canonical
`"1" "),(" "1"` (16, 23482, 16) at index 6,689; `record-5.json` turn 7 sampled `' "^' "(...)"`
(39698, 47235) where the canonical encoding is `' "' "^(" "...)"` (330, 13260, 32425).
`tokenize_probe.json` has the server's canonical encodings. That was the only divergence in each
failing prompt. The other candidates are ruled out: across 29 recorded turns, assistant turns
replayed with empty reasoning, content beside tool calls, two tool calls, and multi-line JSON
arguments all re-rendered token for token. Streaming is not involved: the run-1 non-streaming
adapter passed by chance, as did 5 of 7 streamed reruns here.

No message the client builds can restore sampled ids that differ from the canonical encoding. A
fix needs a tokens-in path: the router (SMG 1.10 in front of vLLM 0.28) rejects integer prompts on
`/v1/completions`, ignores `prompt_token_ids`, and serves no `/generate`, `/inference/v1/generate`
or `/tokenize`. `GlmRolloutModel` now checks the served prompt itself and raises
`RolloutContractError` naming the divergent index and the ids on both sides, in place of the
engine's generic message. Live after the change: the d01 repro (`rollout_only.py`) passed twice
(`prefix-bug/rollout_only-{1,2}.json`), and `pytest tests/llm/test_rollout_model.py tests/validate
-m live_glm` passed 8/8 (`prefix-bug/live_pytest.txt`).

Open: staged-task control replay raises `ValueError` (not built). Docker factories not exercised
on the laptop. `GenerationLimitReached` on a context overflow carries the served prefix, not the
rendered prompt ids (unit-tested only). Scripted control turns carry no logprobs. `TASK_SETUP` and
the agent-timeout zero score are not in DESIGN.md.

## Build

`build/step.py`, `sdk.py`, `author.py`, `run.py`, `template/standard.py`. Steps are memoized
async functions with a mandatory `StepRole`; the key covers step code (transitively, including
same-module helpers and every data global read), arguments, `SDK_VERSION` (`taskforge.build/2`),
the proposal and the model policy. `author()` makes one structured GLM call that writes a builder
program against the `Build` SDK (`b.llm`, `b.research`, `b.machine`, `b.shell_tool`,
`b.try_grader`, `b.spec`, `b.controls`); `Revision(source, failure)` continues it. Programs run in
a restricted module (import allowlist, no `exec`/`eval`/`open`); this is a guardrail, not a
security boundary. `run_build` enforces: the verifier is a GRADER step's output, the controls a
CONTROLS step's output, `validate_controls` passes, and no control reuses a candidate the program
graded with `try_grader` other than a reference or empty answer.

Live (GLM-5.3 interactive), under `.evidence/build/`:

| Check | Result | Evidence |
|---|---|---|
| `pytest tests/build -m live_glm` (author, build, all-hit rebuild, positive control's transcript replayed through RolloutEngine) | passed in 40.7 s | `live_pytest-20261005-1605.txt`, `live-test/20261005-160550/` |
| run-1: author and build two ACCEPT proposals | both built to validated TaskSpecs: d27 after 4 revisions, d01 after 2; rollouts with an inline adapter graded 1.0 | `run-1/` |
| run-2 (after SDK reference, prompt and replay-rule changes) d27.reporting.close_measurement/8 | built on its first program (52,813 output tokens); `GlmRolloutModel` rollout graded 1.0; GRADER patch missed only `grader`, 6 steps hit | `run-2/summary-d27*.json`, `run-2/rollout-d27*.json` |
| run-2 d01.algebra.linear-transformations/5 | built on its third program (4 author calls, 290,549 output tokens); **rollout broke the prefix contract** (see Validate); patch and rebuild not run | `run-2/summary-d01*.json`, `run-2/error-d01*.txt`, `run-2/rollout-d01-rerun.json` |
| Control-replay check on the run-1 d27 program | names 4 of its 6 controls | `replay-check/` |
| Agent and web research inside a step | partial: only an authored d01 program used them (web search and fetch, 7 shell calls) | `run-1/ledger/d01.algebra.linear-transformations--5.jsonl` |

Open:

- `sdk.py` imports `rolloutengine.machines._task_machine` and `rolloutengine.grading._shell_grade`
  until upstream exports them (U-R2). The build PR should not merge before that.
- The template's model-driven steps (research agent, `structured_until` loops) never ran live:
  every authored program replaced them with deterministic code. Its stdout grader reports no
  reward components, so it writes no partial controls.
- The author-to-build revision loop exists only in `.evidence/build/live_build.py`; in the library
  it belongs to review/loop.
- Reasoning tasks must run in ShellSim to get a script grader, because RolloutEngine refuses a
  shell verifier on a null environment (U-R5). The ShellSim python shim rejects
  `pattern.match(s, pos)` and lacks `traceback`; an authored grader scored 0.0 there and 1.0 under
  CPython (U-S6).
- Docker environments never ran. Memo keys miss changes to library code outside the program
  module unless `SDK_VERSION` is bumped. `sys.modules` keeps one module per compiled program.

## Cluster probe

`scripts/cluster_rollout_probe.py` runs `validate.trials.run_trials` inside an Iris task with GLM
resolved in-task. Run from the worktree root (the token is read into a variable, never echoed):

```bash
GLM="$(sed -n 's/^GLM_API_TOKEN=//p' ~/openathena/glm-infer/glm_api_token.txt | head -1 | tr -d '[:space:]')"
lib/taskforge/.venv/bin/iris --cluster=marin job run --no-wait --no-sync \
  --job-name taskforge-cluster-rollout-probe-NN --target-cluster cw-rno2a --priority interactive \
  --cpu 2 --memory 3GB --timeout 5400 -e GLM_API_TOKEN "$GLM" -- \
  bash -c 'cd lib/taskforge && uv run --frozen python scripts/cluster_rollout_probe.py --relay-job /muchanem/glm53-relay-rno2a'
```

Rebuild the results from the job logs with `--reassemble <logs> --results-dir <dir>`.

Result (`/muchanem/taskforge-cluster-rollout-probe-02`, cw-rno2a, 2026-10-05 20:14 UTC; evidence
in `.evidence/cluster/probe-02-*`, first run in `probe-01-*`):

- GLM resolved in-task to `http://10.168.192.79:8010/v1` through relay
  `/muchanem/glm53-relay-rno2a`, interactive token.
- Null-environment math task, k=3: Complete, 3/3 graded 1.0, 0.7 s.
- Docker task (`python:3.12-slim` by digest, network on, private shell grader) on
  `machine_factories(MachineHost.IRIS)` as shipped: **Incomplete, `machine_start` x3** (6 failed
  creates), the shellbox readiness bug, classified and retried as designed.
- The same task after `apply_readiness_fix` (the patch's readiness hunk, applied in-process by the
  probe only): Complete, 3/3 graded 1.0, 18.8 s phase wall; in probe-01 concurrent creates took
  15.9 s. No sandbox job leaked.
- Credentials: the probe scrubs `GLM_API_TOKEN`, `HF_TOKEN` and `WANDB_API_KEY` from `os.environ`
  and `IRIS_JOB_ENV` before creating sandboxes (0 token bytes seen in a sandbox). The sandboxes
  still carry `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CW_KEY_ID` and `CW_KEY_SECRET` from
  cw-rno2a `task_env`, so the job **exits 1 by design** with `SANDBOX_SECRETS`. Do not run
  untrusted models with `machine_factories(MachineHost.IRIS)` on CoreWeave until this is fixed.
- Results were uploaded to `s3://marin-us-east-02a/marin/taskforge/cluster_rollout_probe/20261005T201432Z`
  with the task's own credentials; not read back from the laptop.

Only cw-rno2a has been observed to run the full path: it has both a live GLM relay and working
gVisor with DNS. GLM relays are reachable only on CoreWeave peers; marin GCP gVisor works in
us-west4-a but has no network and no relay; cw-us-east-02a gVisor hung (not re-tested). Egress
beyond DNS on cw-rno2a was not tested. The math task drew no real reasoning (19 tokens) and the
docker task took 2 turns, so this shows the path works, not long-rollout behavior.

## Upstream changes needed

Consolidated from every phase-2 report. IDs are referenced above. "Filed" means an issue or draft
PR exists in `marin-community/marin` (index: [docs/upstream/README.md](docs/upstream/README.md));
all are open as of 2026-10-05.

### rolloutengine (`lib/rolloutengine`, stacks on PR 9623)

| ID | Change | Files | Needed by | Filed |
|---|---|---|---|---|
| U-R1 | Grade a supplied final state (transcript or installed workspace) without model inference | `grading.py`, `engine.py` | build `try_grader`; staged control replay | #9763 (no PR) |
| U-R2 | Export the task-machine lifecycle and in-machine shell grading (`_task_machine`, `_shell_grade`) | `machines.py`, `grading.py` | build `sdk.py` (private imports block the build PR) | draft `docs/upstream/rolloutengine/public-task-machine.md`, not filed |
| U-R3 | Export the shell tool contract: `SHELL_TOOL_NAME`, the tool definition built in `session_start`, the observation JSON from `advance` | `task_session.py` | `llm.agent.shell_tool` (copies it today) | no |
| U-R4 | Raise typed setup and healthcheck errors instead of `RuntimeError`/`TimeoutError` with message prefixes | `machines.py`, `task_session.py` | `validate.classify` (matches message prefixes) | no |
| U-R5 | Allow a `ShellVerifierSpec` with its own grading `environment` on a null task environment | `grading.py` (`_grade_rollout`) | build: reasoning tasks without a solver shell | no |
| U-R6 | Document that `GenerationLimitReached` may carry the served prefix, or make its prompt ids optional (a server that rejects an over-context prompt cannot render it) | `contracts.py` | `llm.rollout_model` | no |
| U-R7 | Optional: grade the partial state on an ADVANCE agent timeout (Taskforge scores such a trial 0) | `engine.py` | `validate.classify` | no |
| U-R8 | Document that a model adapter must report a budget-cut tool call as `length` (vLLM's GLM parser reports `tool_calls`; `GlmClient` already normalizes it) | `contracts.py` or `docs/references/task-rollouts.md` | any non-Taskforge adapter | no |

### shellbox (`lib/shellbox`)

| ID | Change | Files | Filed |
|---|---|---|---|
| U-S1 | Fix the Iris readiness poll: compare with `iris.client.workload.TaskState`, report `status.error_message`. Proven necessary and sufficient on cw-rno2a | `src/shellbox/backends/iris/machine.py` | patch `docs/upstream/shellbox/iris-machine.patch` (unapplied; `git apply --check -p1` passes; 4 new tests) |
| U-S2 | Declared network policy, `IrisMachineFactory(network: NetworkPolicy)`, chosen per cluster (marin GCP gVisor is DENY, cw-rno2a has DNS) | same | in the patch (single policy; per-cluster choice not yet) |
| U-S3 | Sandbox jobs must not inherit the parent's env (`IRIS_JOB_ENV`); the patch blanks only `HF_TOKEN`/`WANDB_API_KEY` and misses `-e` secrets such as `GLM_API_TOKEN` | same | partly in the patch |
| U-S4 | `TRANSFER_CHUNK_BYTES` 64 KiB (128 KiB exceeds `MAX_ARG_STRLEN`) | same | in the patch |
| U-S5 | Raise a typed exception for a killed or expired sandbox instead of `RuntimeError('Iris exec failed: ...')` | same | no |
| U-S6 | ShellSim python shim: support `pos` in compiled-regex `match`/`search`, add `traceback`, document the supported stdlib | `src/shellbox/backends/shellsim/` | no |

### taskcompendium (`lib/taskcompendium`)

| ID | Change | Files | Filed |
|---|---|---|---|
| U-T1 | Keep verifier status and detail in `GradeResult` (reader in `rolloutengine/grading.py`) | `grading.py` | #9757, draft PR #9769 |
| U-T2 | Grade fenced JSON and numeric JSON answers | `submission.py` | #9758, draft PR #9766 |
| U-T3 | Pass verifier files to verifyit candidate graders | `grading.py` | #9764 |
| U-T4 | Make judge reachable from a TaskSpec: a judge verifier kind with a `JudgeSpec` table, and let `ShellboxRolloutEngine` take a `JudgeConnection` so credentials stay out of the spec (also `rolloutengine/grading.py`) | `models.py`, `grading.py` | no |
| U-T5 | Reach verifyit aggregation (`aggregate_rewards`) from a TaskSpec instead of a shell script. Needs an ownership decision: the upstream README says TaskCompendium gains no generic verifier kinds | `models.py`, `grading.py` | no (aggregation itself: #9762) |
| U-T6 | Run verifyit execution modes (pytest, stdio, junit, gotest, script) from a TaskSpec without installing the verifyit CLI in every image; engine-run verifier kind. Same ownership question | `models.py`, `grading.py`, `rolloutengine/grading.py` | no |
| U-T7 | One typed error for "unsupported verifier mode" and "invalid parameters" (today `NotImplementedError` vs `ValueError`) | `grading.py` (`verifier_descriptor`, `resolve_verifier`) | no |
| U-T8 | Declare `reasoning_content` on `ChatAssistantMessage`; prefix-preserving replay passes today only because pydantic ignores extra keys | `chat.py` | no |

### verifyit (`lib/verifyit`)

| ID | Change | Files | Filed |
|---|---|---|---|
| U-V1 | Grade extracted text candidates for every answer mode (`math` first; also `ifeval`, `json-schema`, `xml-elements`, `csv-columns`), so `math` resolves through TaskCompendium's registry | `src/verifyit/candidate.py` | #9759, draft PR #9765 |
| U-V2 | Output budgets for judge reference and checklist rubrics | `src/verifyit/modes/grade_judge.py` | #9760, draft PR #9768 |
| U-V3 | Per-criterion checklist grades and repeated judgments | `src/verifyit/modes/grade_judge.py` | #9761, draft PR #9770 (on #9768) |
| U-V4 | Weighted and gated reward aggregation | `src/verifyit/grade.py` | #9762, draft PR #9767 |

### Outside the four libraries

- Iris owners: GCP workers need a pull credential for `envreg.208261-marin-gpu.coreweave.app`;
  gVisor fails on the v4 us-central2-b workers; cw-us-east-02a gVisor pods hang in
  PodInitializing; CoreWeave `task_env` injects object-store keys into every pod, including gVisor
  sandboxes; child jobs inherit the parent's explicit env, including secrets.
- `experiments/post_training/glm.py` (marin-autoenv tree): `DEFAULT_GLM_RELAY_JOB =
  /muchanem/glm53-relay-08a` is killed. Live endpoints: `/muchanem/glm53-relay` (served by
  `/muchanem/glm53-relay-e`, cw-us-east-02a) and `/muchanem/glm53-relay-rno2a` (cw-rno2a).

## Stack

Six layers on `rollout-engine` (PR 9623 head `7bbe4b9cd7`). Each layer is one commit (06 has a
second, this file); its PR body is committed at `docs/prs/taskforge/<NN-name>.md` (first line
`# <title>`). Nothing is pushed. Unit tests and pyrefly pass on every layer at its own head.

| Branch | Head | PR title | Contains | Unit tests at head |
|---|---|---|---|---|
| `taskforge/01-foundation` | `7dffa43a15` | [taskforge] Add the package skeleton, GLM client, agent loop and ledger | project files, DESIGN.md, STATUS.md, `llm/*` (client, structured, store, endpoint, policy, agent, web, rollout_model), `ledger/*`, empty `review`/`loop`/`queue`, agent-loop decision docs; 43 files | 60 passed, 15 skipped |
| `taskforge/02-spec-sandbox` | `5ca5a15fef` | [taskforge] Assemble TaskSpecs and select shellbox machine factories | `spec/draft.py`, `spec/controls.py`, `sandbox/factories.py`, `sandbox/images.py`, image builder and probe scripts, `docs/upstream/`; 28 files | 111 passed, 15 skipped |
| `taskforge/03-proposal` | `8d0daa3968` | [taskforge] Generate task proposals from capability ideas | `proposal/*`; 9 files | 128 passed, 16 skipped |
| `taskforge/04-triage` | `76fa9ca061` | [taskforge] Triage proposals with structural checks and a GLM rubric | `triage/*`; 9 files | 169 passed, 17 skipped |
| `taskforge/05-validate` | `e081b0e7a1` | [taskforge] Run validation trials and control replay on RolloutEngine | `validate/*`, `tests/llm/test_rollout_model.py` (it needs `spec`), `scripts/cluster_rollout_probe.py`; 15 files | 212 passed, 25 skipped |
| `taskforge/06-build` | this commit's parent `8a09e72eb4` | [taskforge] Build TaskSpecs from proposals with authored programs | `build/*` and `tests/build/*`; 14 files; plus this STATUS.md | 212 passed, 25 skipped; `tests/build` 23 passed, 1 skipped |

Known stack issues:

- `llm/rollout_model.py` lives in 01 (the agent imports it) but its tests live in 05, so 01 covers
  it only through `test_agent`.
- The d01 token-prefix break is a transport limit (no tokens-in path), not a bug in 01's
  `llm/rollout_model.py`; that layer now reports the divergent ids.
- `tests/build` is not collected by `pytest tests`; the fix is `norecursedirs` in
  `lib/taskforge/pyproject.toml` (01). `lib/taskforge/.gitignore` un-ignores `tests/build/`
  (the root `.gitignore` ignores `build/`).
- `jsonschema` is undeclared in `pyproject.toml` (01).
- PR 06 imports private RolloutEngine names (U-R2).
- Three stray untracked files in `lib/taskforge` (`closed_form.txt`, `diagonalization.txt`,
  `growth.txt`, eigenvalue and closed-form math output written at 15:10 local) are probably from a
  live agent run that used the cwd. Delete them; they are not committed.
- Live GLM tests were run on the stack head only, not per layer, and the trunk moved two commits
  (`0d18c93185` failed-stage mean rewards, a SkyRL pin) after most live evidence was recorded.

Publish (from `/Users/k3sc0re/openathena/branches/marin-taskforge`; `--auto` creates draft PRs
with generated titles, then the loop sets each title and body from `docs/prs/`):

```bash
cd /Users/k3sc0re/openathena/branches/marin-taskforge
gh stack submit --auto
for f in lib/taskforge/docs/prs/taskforge/*.md; do
  branch="taskforge/$(basename "$f" .md)"
  pr="$(gh pr list --head "$branch" --json number -q '.[0].number')"
  tail -n +3 "$f" | gh pr edit "$pr" --title "$(head -1 "$f" | sed 's/^# //')" --body-file - --add-label agent-generated
done
gh stack view --json
```

## Phase 3 entry conditions

Each condition names the package it gates. A package starts when its conditions hold.

### review/

1. **The rollout model keeps the token contract on real tasks.** The d01 prefix break is
   diagnosed (non-canonical sampled tokenization re-tokenized by the chat endpoint, see Validate)
   but not fixed: it needs a tokens-in generation path on the GLM router. Review decides from validate evidence; a transport that fails on
   7-turn tasks makes that evidence `Incomplete` for the wrong reason.
2. **Validate produces the evidence review consumes.** DESIGN.md lists `validate/solver.py`,
   `adversary.py` and `calibration.py`; none exists. Decide the solver k, the adversary roles (as
   RolloutEngine `TaskSession` factories) and the solve-rate band that counts as calibrated, and
   build them with live evidence.
3. **The `Decision` contract is settled.** `Accept | Reject | Repair(patch, invalidate) |
   Retry(cause)`, with `Repair` producing a builder-program patch and the step names it
   invalidates, matching `run_build(..., invalidate=...)`.
4. **The seam-type rule is decided.** `proposal.model` is imported by triage and build; review
   will import `build` and `validate` outputs. Either seam types move to a shared layer or
   DESIGN.md allows seam imports.
5. **DESIGN.md matches the code**: decision 6's exception for rollout turns, `TASK_SETUP`, the
   agent-timeout zero score, the Reuse-first list (now `taskforge.llm.structured.StructuredTool`
   and `taskforge.llm.endpoint.resolve_glm_base_url`), and the package layout (no `silo.py`,
   `lowering.py`; not a root workspace member).

### loop/

1. **Review exists** and returns a `Decision` for a built and validated item.
2. **The build revision loop moves into the library.** Today it lives in
   `.evidence/build/live_build.py` (up to 3 author revisions on a build failure).
3. **Loop policies are decided**: maximum triage repair rounds; maximum build revisions and token
   budget per item (d01 needed 3 programs and 290,549 output tokens); what a `SlotFailure` does to
   its idea; whether triage's cross-run agreement (57/63) is acceptable.
4. **The event log is designed and emitted**: append-only per-item events over the ledger, with
   status derived from it, and a cluster Finelog write from an Iris task read back.
5. **The template's model-driven build steps run live at least once**, or the template is
   replaced by authoring only.

### queue/

1. **One cluster runs the whole path with the shipped libraries.** U-S1 (readiness poll) is
   merged, so `machine_factories(MachineHost.IRIS)` works without the probe's in-process fix, and
   the probe passes on that cluster. Today only cw-rno2a has a relay plus working gVisor.
2. **Sandboxes carry no credentials.** U-S3 lands (no env inheritance into sandbox jobs), and
   Iris stops injecting object-store keys into gVisor sandbox pods, or the user accepts the
   exposure in writing. `cluster_rollout_probe.py` passes without `SANDBOX_SECRETS`.
3. **`capability-registry-publisher` is rotated**, and either Iris workers can pull from envreg or
   built images go to a registry the workers can pull from.
4. **Concurrency is measured at the intended width.** Hundreds of concurrent agents and trials on
   one `GlmClient` against the interactive and bulk tiers, with no throttle added before an
   observed failure (DESIGN.md decision 6).
5. **Relay selection is explicit configuration** per cluster, not the dead default in
   `experiments/post_training/glm.py`.
