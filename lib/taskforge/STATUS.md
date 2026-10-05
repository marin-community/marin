# Taskforge status

**Phase 2 tree change (2026-10-05).** Taskforge now lives in the worktree `taskforge/base`
(`/Users/k3sc0re/openathena/branches/marin-taskforge`, cut from PR 9623 head `7eb1dff26f`) and is a
standalone uv project like `lib/rolloutengine`: its own `pyproject.toml`, `uv.lock` and `.venv`
in `lib/taskforge`, path dependencies on the sibling `lib/*` packages, TaskCompendium 0.22, and
no root-workspace membership. GLM endpoint discovery moved into `taskforge.llm.endpoint` and
`StructuredTool` into `taskforge.llm.structured`, so nothing imports `experiments` or
`marin-core`; `httpx` is bounded `<1` because `prerelease = "allow"` otherwise picks a 1.0 dev
release without `AsyncClient`. The Silo sandbox and the Harbor spec lowering are deleted (see
Sandbox and Spec). Verified on that tree: unit 101 passed, 9 skipped; live (GLM-5.3 interactive)
9 passed in 784s (`.evidence/standalone/live_suite-20261005-run2.txt`); an earlier live run
failed `test_truncated_answer_is_continued` once (a segment ended exactly on `END` with
`finish_reason: length`, and the continuation appended prose; 3 reruns passed) and the proposal
and triage live tests before they took the catalog from `TASKFORGE_CAPABILITY_CATALOG`
(`.evidence/standalone/live_suite-20261005.txt`). The rest of this file is the phase 1 account
on `mark/autoenv`; its commands and counts predate this change except "How to run".

Single account of phase 1 (foundations): what works, what is broken or not run, and what must be
true before phase 2 starts. Every claim about a model-facing component names its evidence under
`.evidence/<package>/` (gitignored; it exists only on the machine that ran the check, see
`.evidence/README.md`). Last full verification: 2026-10-05.

## Summary

Phase 1 is done for `llm`, `ledger`, `spec`, `proposal` and `triage`, each with stated gaps.
`sandbox` is **not** done: it is unit-tested only, because no Silo broker is running, so it has
never executed a command in a real sandbox. `build`, `validate`, `review`, `loop` and `queue` have
not started (empty packages). The agent loop is decided (in-house, over `GlmClient`) but not
written.

Verification run on 2026-10-05, from the repository root:

- Unit: `uv run pytest lib/taskforge/tests`: **137 passed, 9 skipped** in 15s. Per package:
  llm 29 (+6 live), ledger 13, sandbox 19, spec 18, proposal 17 (+1 live), triage 41 (+1 live),
  plus the skeleton endpoint test (live).
- Live, GLM-5.3 interactive tier (router `/health`: `high=11, bulk=26`): `uv run pytest
  lib/taskforge/tests -m live_glm`: **9 passed** in 861s. Log:
  `.evidence/status/live_suite-20261005.txt`.
- Lint: `./infra/pre-commit.py --files` on all 89 untracked `lib/taskforge` files: OK.
- Types: `uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check --baseline .pyrefly-baseline.json`:
  0 errors (424 suppressed). `lib/taskforge/tests` is not in pyrefly `project-includes`.
- Working tree: `pyproject.toml` and `uv.lock` modified (taskforge added as a workspace member,
  root dependency, pyrefly include and search paths); `lib/taskforge/` entirely untracked. Nothing
  is committed.

A passing live test is weaker than it sounds in two places, both stated below: the continuation
tests do not check the seams or the answer, and the triage test does not check decision
stability.

## How to run

Run every command from `lib/taskforge` in the `taskforge/base` worktree
(`/Users/k3sc0re/openathena/branches/marin-taskforge/lib/taskforge`) unless it says otherwise.
The package has its own `uv.lock` and `.venv`; `uv run` there uses them, not the root workspace.

Unit tests (live tests skip without the env below):

```bash
uv run --group test pytest tests
```

Live tests against GLM-5.3 on the interactive tier. The router port-forward must be up
(`curl -s http://127.0.0.1:18000/health` returns `"status":"ok"`; see `DESIGN.md` to restart it).
The proposal and triage live tests also need the capability catalog, which is gitignored and
absent from this tree; point `TASKFORGE_CAPABILITY_CATALOG` at a local copy:

```bash
export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export TASKFORGE_GLM_TOKEN_FILE=~/openathena/glm-infer/glm_api_token.txt
export TASKFORGE_CAPABILITY_CATALOG=/Users/k3sc0re/openathena/branches/marin-autoenv/experiments/post_training/capability_env_gen/new_catalog.json
uv run --group test pytest tests -m live_glm  # live only
uv run --group test pytest tests              # unit + live
```

Live tests carry `@pytest.mark.live_glm` and take the session fixture `glm_settings`
(`base_url`, `model`, `token`; the token is excluded from `repr`); the proposal and triage live
tests also take `capability_catalog`. The package pytest config sets `timeout = 60` and
`asyncio_mode = "auto"` (`pytest-asyncio` is in the `test` group); give long live calls
`@pytest.mark.timeout(<seconds>)`.

Types, from `lib/taskforge` (the package has its own `[tool.pyrefly]`, checking `src` against the
sibling `lib/*/src` trees and `.venv/bin/python`; `tests` is not included):

```bash
uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check
```

Lint for the files you touched, from the worktree root
(`/Users/k3sc0re/openathena/branches/marin-taskforge`), with paths relative to it:

```bash
./infra/pre-commit.py --fix --files lib/taskforge/<path> ...
```

In zsh, pass a file list with `xargs` or `${=files}`; an unquoted `$files` does not word-split.

## Packages

| Package | State | Evidence |
|---|---|---|
| skeleton | Works. Package installs; `glm_settings` reaches the interactive endpoint and `/v1/models` lists `glm-5.3`. | `.evidence/skeleton/` |
| llm | Works on the laptop endpoint: 6/6 live checks pass. Continuation is lossy at seams. Not validated inside an Iris task. Undeclared dependencies. | `.evidence/llm/` |
| ledger | Works locally (JSONL; Finelog round trip against finelog's embedded server). Cluster Finelog write not run. Nothing emits entries yet. | `.evidence/ledger/` |
| sandbox | **Not validated live.** 19 unit tests against an in-process fake. No Silo broker exists to test against. No image builder. | `.evidence/sandbox/` |
| spec | Phase 2 `draft.py` and `controls.py`: assembled specs run end to end on shellsim through `ShellboxRolloutEngine` (unit tests). Docker and Iris only round-trip tested. No control replay yet (needs a public RolloutEngine grading entry point). No model calls, so no live evidence. | `tests/spec/`; phase-1 Harbor evidence in `.evidence/phase1-spec/` |
| proposal | Works. Latest live run: 20/20 parsed, 1 document repair, 1 plan repair. Null-slot path unit-tested only. | `.evidence/proposal/` |
| triage | Works. Latest live run: 103 proposals x 3 samples, 0 output errors, 31 ACCEPT / 72 REPAIR. REJECT never seen live. | `.evidence/triage/` |
| build | Builds real proposals to a validated TaskSpec live (d27 on its first program, d01 after 2 revisions). Production rollout of the d01 draft fails the token-prefix contract in `llm.rollout_model`. Imports two private RolloutEngine names. | `.evidence/build/` |
| validate | `outcome`, `classify`, `trials`, `controls`, `evidence` and `llm.rollout_model` work live on ShellSim and null tasks (8/8 live checks). Docker and Iris factories not exercised. Staged-task control replay not built. | `.evidence/validate/` |
| review | not started | |
| loop | not started | |
| queue | not started | |

## LLM

`taskforge.llm.client.GlmClient` is the one GLM-5.3 transport (async, streaming httpx, 512 pooled
connections). `complete_structured` forces one strict tool call (marin `StructuredTool`) and
allows one repair. `CallStore` caches calls by request hash under
`items/<stage>/<hash>/{request,response,result}.json`. Each live run writes
`.evidence/llm/runs/<check>-<utc>.json`; a run whose assertions pass is copied to
`.evidence/llm/<check>.json`.

Live, 2026-10-05 17:13 UTC (the promoted files are this run):

- `a_plain_completion.json`: `max_tokens=131072`, effort high. `stop`, 493 completion tokens,
  ttft 0.20s, 208 tok/s decode.
- `b_context_overflow.json`: a 250,055-token prompt at `max_tokens=131072` gets a 400; a
  one-token probe measures the prompt and the retry at `max_tokens=12089` succeeds (`stop`).
  Attempts: `[0 context_overflow 131072 400] [-1 completed 1 200] [0 completed 12089 200]`;
  249,984 tokens served from the prefix cache. The server enforces prompt + max_tokens <= 262,144,
  not a separate 131,072 output cap.
- `c_continue_on_length.json` (`max_tokens=150`): 2 continuations, `stop`, ends in `END`. **The
  seams are lossy**: this run duplicated a phrase (`thirteen - widely considered an
  unluckythirteen - widely considered an unlucky number`) and dropped a space (`aday`). Earlier
  runs dropped spaces, punctuation and newlines at 2 of 3 seams. The test asserts only that it
  continued and ended in `END`.
- `c_continue_mid_reasoning.json` (`max_tokens=32`, temperature 0): cut off mid-reasoning, the
  client reopens the assistant turn as `<think>` plus the partial reasoning with
  `continue_final_message`; 2 continuations, `stop`. **The answer is wrong**: 397, against 401
  from an uncut run (recorded as `expected_answer`). The test asserts only a non-empty answer.
  A user "continue" turn was tried first and looped for 32 continuations without an answer
  (`runs/c_continue_mid_reasoning-20261005T170944Z.json`, `-170955Z.json`).
- `d_structured.json`: one forced `record_capital` call validated (Tokyo, 13.96) with no repair.
  In an earlier promoted run GLM ignored the forced `tool_choice` and answered in prose; the
  repair recovered it. GLM reports `finish_reason: stop` for a forced tool call.
- `d_structured_repair.json`: a validator the schema cannot express (uppercase city). `Paris`
  failed, the repair returned `PARIS`; 2 calls.

Unit-only (fake router): 429/5xx retry with Retry-After, route-404 and empty-pool holds,
hold timeout, non-retryable 4xx, stall timeout, aborted or malformed streams, a continuation that
no longer fits the context (returns the partial output with `finish_reason: length`).

Broken or not run:

- `endpoint_in_task` (`resolve_glm_base_url` inside an Iris task) and the relay's `/health` shape
  there: not run. With the wrong shape, retryable failures spend attempts instead of holding.
- `lib/taskforge/pyproject.toml` declares neither `marin-core` (`marin.inference.structured_output`,
  imported by llm, proposal and triage) nor the root package (`experiments.post_training.glm`,
  imported by `llm/client.py`). The second is a lib-to-experiments layering violation.
- CallStore does not record failed calls or deduplicate concurrent identical requests. It stores
  every raw SSE event in `response.json` (about 130k events for a 131k-token reply).
- `decode_tokens_per_second` is meaningless for very short replies (one-chunk arrival).

## Ledger

`taskforge.ledger.records.span(ledger, kind, *, item_id, round, step, ...)` times a block and
records a `LedgerEntry` on success or failure; a caller-set `cause` wins over the exception class
name, and a record failure during an in-flight exception is attached as a note, never replacing
it. `JsonlLedger` writes one file per item with single `O_APPEND` writes. `FinelogLedger` writes
table `taskforge.ledger`. `run_ledger(root, run_id)` yields JSONL alone off-cluster and
`CompositeLedger` (JSONL first, authoritative) under Iris. `scripts/ledger_summary.py` reports
count, failures, busy, elapsed and tokens per kind and kind/step.

Works (`uv run pytest lib/taskforge/tests/ledger`, 13 pass; `.evidence/ledger/unit_tests_after_review.txt`):
round trip through finelog's embedded native server with every column checked; 4 processes x 200
concurrent appends to one item file with no torn lines; off-cluster fallback to JSONL
(`laptop_finelog_resolution.txt`); summary script on a demo ledger (`summary_script_demo.txt`).

Not run: a write to the cluster Finelog from inside an Iris task (`telemetry.resolve()` returns
None off-cluster). No GLM validation applies. Open: nothing emits entries yet (llm, sandbox and
build must wrap their work in `span`), and nothing computes `code_hash`/`input_hash`/`output_hash`.

## Sandbox

Phase 2, 2026-10-05. Silo is gone (DESIGN.md decision 2). The package is `factories.py` (factory
selection and up-front refusal) and `images.py` (DockerBuild contexts and the `ImageBuilder`
protocol); the Iris builder is `scripts/build_image_job.py` with its push job
`scripts/push_image_task.py`.

- `machine_factories(MachineHost)` returns the `EnvironmentKind -> MachineFactory` mapping
  `ShellboxRolloutEngine` takes: ShellSim always; on LAPTOP, local Docker when a daemon and Skopeo
  are found (probed once); on IRIS, `IrisMachineFactory` on `$IRIS_CONTROLLER_URL`.
- `factory_capabilities` and `task_refusals` refuse a task up front with typed `Refusal`s (no
  factory, image source, network, execution user, resource limits, GPUs). They cover the task
  machine, stage and collect users, shell-verifier grading environments, and the root commands
  RolloutEngine itself runs on the task machine (fetching AUTO, SKIP or excluded-directory
  grading artifacts; removing stage graders' private files).
- Iris DOCKER is reported unavailable: shipped `IrisMachineFactory.create` always fails. Its row
  describes the patched backend (`docs/upstream/shellbox/iris-machine.patch`): registry images
  only, NetworkPolicy DENY only, no execution users, no GPUs.

Unit tests: `uv run --group test pytest tests/sandbox` (14 passed). One test calls the real
shellbox factories and checks they refuse what the capability table says they refuse; it also
trips when the Iris patch lands.

Image builder, live on Iris (marin hub, federated to cw-us-east-02a, DEFAULT profile):

- Two jobs. The build job runs kaniko (`--no-push`) in a pinned bash image and writes
  `image.tar` to `$IRIS_OUTPUT_DIR`; Iris's uploader sidecar archives it to
  `s3://marin-us-east-02a/tmp/ttl=7d/iris/task-outputs/...` and records its sha256. The push job
  runs in a fresh pinned `python:3.14-slim` image, downloads that archive (SigV4 GET, sha256
  checked), fetches a sha256-checked `crane`, and pushes under a tag unique to the publish. The
  credential reaches only the push job, as env `REGISTRY_AUTH`, which Iris redacts in job status
  responses. The submitter reads the manifest back by tag, derives the digest from its bytes, and
  checks linux/amd64.
- Split run 1: build 49.1 s, total 79.2 s, `...taskforge-image-build-smoke@sha256:4429731d...`;
  `REGISTRY_AUTH` shows as redacted in the stored job request. Evidence:
  `.evidence/sandbox/build_image_job-split-run1.txt`.
- Adversarial Dockerfile (RUN writes into `/kaniko`, spoofs the old `/app/.state`, starts a
  watcher scanning `/proc/*/environ` and `cmdline` for the credential, scans the filesystem):
  published normally; the RUN env held only HOME, PATH and PWD; the watcher reported no hit; the
  filesystem hits were the kaniko and crane binaries and the Dockerfile itself, which contain the
  search strings. Evidence: `build_image_job-split-adversarial.txt`, `-adversarial2.txt`.
- Failing Dockerfile (RUN exit 7): `BuildFailed` carrying kaniko's `exit status 7`; no push job
  is submitted. Evidence: `build_image_job-split-failing.txt`.

Open:

- The old single-job builder put the credential in an `ExecInContainer` argv, which the
  CoreWeave controller logs at INFO: 2 such lines are in the cw-us-east-02a controller log
  (`credential_exposure_check.txt`, counted, not printed). `capability-registry-publisher` must
  be rotated by the registry owner.
- The push job's pod spec still carries the credential (readable by cluster operators); the
  registry speaks only Basic auth, so there is no short-lived scoped token. Every CoreWeave task
  pod carries the cluster's object-store keys, which RUN steps can read. Archives over 2 GiB
  (Iris's output limit) cannot be published.
- Iris sandboxes (`docs/upstream/shellbox/iris-machine.md`): shipped create always fails, DENY is
  refused, uploads over about 96 KiB fail, submitter HF_TOKEN/WANDB_API_KEY leak into sandboxes;
  all fixed by the unapplied patch. GCP workers cannot pull from envreg (no pull credential), so
  built images cannot run on Iris. gVisor starts only in some zones (us-west4-a observed); the
  job that creates sandboxes must be placed there. Killed or expired sandboxes raise untyped
  `RuntimeError`.
- `machine_factories(LAPTOP)` with Docker and `machine_factories(IRIS)` inside an Iris task have
  not been exercised (no Docker on this laptop; taskforge is not a root workspace member).

## Spec

Phase 2 (current). `taskforge.spec.draft` assembles a TaskCompendium 0.22 `TaskSpec` from builder
outputs: `environment`, `file`, `shell_command`, `reward_file`, `shell_verifier`, `answer_verifier`
(verifyit exact, numeric, mcq and predicted_action only), `staged`, `stage` and `assemble`.
`assemble` validates every grader once and returns the JSON round-tripped spec. It rejects a
grader that does not fit the answer type or environment, private grader content that is also
agent-visible (environment files, DockerBuild context, or any stage's `workdir_files`), and stage
`minimum_rewards` keys other than `reward` unless the stage grader writes JSON reward files.
`taskforge.spec.controls` holds fixed controls (a `Transcript` or a `Workspace` payload with an
`Expectation`). `validate_controls` checks the control set against the task: required
categories per stage, payloads the task can replay, positive controls demanding more than 0.2,
and partial controls only on graders that write JSON reward files, which are the only ones whose
`GradeResult.rewards` carry components.

Validation, 2026-10-05, offline only: `uv run --group test pytest tests/spec` (37 passed). Shell
verifiers (stdout, exit code, JSON reward file) and a staged task with `minimum_rewards` gating
run on shellsim through `ShellboxRolloutEngine` and grade right and wrong answers as expected.
Answer verifiers grade through `grade_answer` after a JSON round trip. A partial control's
expected components (`reward 0.5, format 1, value 0`) were checked once against the real engine
on shellsim; the test suite only validates them. The package makes no model calls, so there is no
GLM evidence; `.evidence/spec/` is empty.

Open: Docker and Iris execution of an assembled spec has not been run. Control replay belongs to
`validate/` and needs public RolloutEngine entry points that grade a given transcript or an
installed workspace (DESIGN.md decision 2). Math, judge, composite and verifyit execution modes are
not reachable from a TaskSpec on TaskCompendium 0.22.

Phase 1 (superseded; Harbor lowering and control replay are gone per DESIGN.md decision 1). Its
evidence moved to `.evidence/phase1-spec/`; the paths below are relative to that directory.

`taskforge.spec.lowering.lower(spec, out, *, convention)` lowers a main-schema `TaskSpec` to a
Harbor package through `taskcompendium.lowering.lower_to_harbor` and returns a `HarborPackage`
with a per-file SHA-256 manifest held in memory. `taskforge.spec.controls.replay` grades fixed
controls through `taskcompendium.verifier_registry.grade_answer` on the conversation a Harbor trial
would record. Anything main cannot do raises `taskforge.spec.gaps.Unsupported` naming an issue body
in `docs/upstream/`: action interfaces, shell or process capabilities, file and state answers,
workspace candidates, and every `criterion_mutation` (partial-credit) control.

Offline (`uv run lib/taskforge/scripts/spec_make_evidence.py`, writes `lower_and_replay.json`): the
JSON-convention bolts package round-trips through TaskCompendium's readers and its instruction does
not contain the reference answer; gold, plausible-wrong, shortcut, and malformed controls grade as
labeled, and a mislabeled gold control is reported as a violation.

Live, 2026-10-05, GLM-5.3 interactive tier (`lib/taskforge/scripts/spec_live_harbor_trial.py`; how
to run is in its docstring, it needs TaskCompendium's `harbor` extra): 3 Harbor trials per
convention (`live_harbor_trial.json`, raw trials under `harbor-live/`) plus one direct call per
convention with the identical request body (`live_direct_calls.json`: request, full response with
`usage` and `finish_reason`, wall time).

- plain: 3/3 graded 1.0 (reply `42`); direct call `finish_reason: stop`, 28 completion tokens.
- answer-call: 3/3 graded 1.0 (`submit_answer` `{"answer": "42"}`); direct call
  `finish_reason: tool_calls`, 43 completion tokens.
- json: 0/3. Every reply was a fenced `{"answer": 42}`, which main's `extract_answer` rejects
  (`extraction_error`); the direct call returned the same reply with `finish_reason: stop`. Use the
  plain or answer-call convention until `docs/upstream/json-submission-extraction.md` lands.

Broken or open: Harbor's `ChatAgent` keeps only the assistant message (no `usage`,
`finish_reason`, or token counts) and sends no `max_tokens`
(`docs/upstream/harbor-chat-usage.md`). No judge verifier exists on main, so none was exercised.
The blocking gaps for real pipeline output are `container-verifier-runtime`, `resources-with-roles`,
`workspace-state`, and `shellsim-binding`; no `spec/extensions/` module exists yet. None of the
`docs/upstream/` issues has been filed.

## Proposal

`taskforge.proposal.model` parses and renders the TaskProposal document (YAML front matter plus six
required `##` sections; headings inside fenced code are ignored); `digest` hashes the canonical
render. `CapabilitySource.propose(idea, n)` is `async` and returns a `ProposalBatch`: planning
request and completions, then one `SlotProposal` or `SlotFailure` per slot in slot order. A failed
slot does not drop its siblings; any other error cancels the outstanding slots. Planned-null slots
make no model call. The plan allows at most ceil(n/3) slots per environment x verification pairing.

Live, GLM-5.3 interactive, `tests/proposal/test_live.py` on d01.algebra.linear-transformations
and d27.reporting.close_measurement, n=10 each, default `LLMPolicy` (131072 tokens, effort high,
all slots concurrent):

- `run-20261005-131404` (verification run): 20/20 parsed, 0 failures, 0 null slots, all `stop`,
  0 continuations. d01's plan needed the structured repair (2 calls, 55s); d27's validated in 1
  call. d27 had one document repair, this time for invalid YAML (a `resources:` key run onto the
  previous line), not the usual missing key. Max pairing count 3 (d01: 6 distinct pairings; d27: 5).
  Per slot 70-383s and 12.5k-83.9k completion tokens; d01 slot 4 spent 79k tokens reasoning. Wall
  time 437s (d01) and 295s (d27).
- `run-20261005-130516`: 20/20 parsed, 1 document repair (missing `null_reason`), plans in 1 call.
- Earlier runs (`run-20261005-121949`, `-122436`, `-125551`) predate the batch return and the
  pairing cap; `propose-smoke/` (n=3, d43.culinary.scaling) records ids, pairings and digests only.

Broken or open: GLM drops or mangles front matter in about 5 of 100 live documents, so the document
repair is load-bearing. The server does not enforce strict tool schemas (missing required fields,
stringified nested objects), so every structured call relies on pydantic validation plus repair.
Long-tail reasoning reaches 75-80k tokens on one proposal. No live slot has come back null; the
null path is unit-tested only. `RepoIdea` is a placeholder until a build_envs source exists. The
loop owner must decide what a `SlotFailure` does to the idea.

## Triage

`taskforge.triage` runs deterministic checks (`checks.py`), then `GlmRubric` (`program.py`), then
decides in code. A FATAL check failure or a null proposal rejects with no model call. `GlmRubric`
takes `samples` independent rubric samples (CallStore stage `triage.rubric.<i>`), each scoring seven
flat integer axes through `record_review`; `rubric_decision` takes the majority of the per-sample
accept rule (ACCEPT or REJECT needs a strict majority, else REPAIR). `GlmRubric.repair` makes one
rewrite call and returns `Repair(proposal, call)`; the caller re-evaluates and bounds rounds. Seam:
`async evaluate(p, checks, rubric, ctx: CheckContext) -> Verdict`.

Live, GLM-5.3 interactive, temperature 0.7, all calls concurrent, every call `stop`:

- `run-20261005-132121` (verification run): every proposal under `.evidence/proposal/` (103) x 3
  samples, 311 calls, 216s wall, 2.77M completion tokens, median 131s per proposal. 0 rubric output
  errors. 31 ACCEPT / 72 REPAIR / 0 REJECT; 67 of 103 unanimous. reward_validity is the blocking
  axis (104 of 309 samples scored 3, 10 scored 2). Structural: all FATAL checks pass on all 103;
  advisory `resources_in_build_plan` fails on 25. Repairs: 2 of 3 reached ACCEPT (3/3 votes); the
  d43 scaling_2 repair stayed REPAIR (114s, 27.4k completion tokens).
- Stability: majority decisions agree with `run-20261005-130451` on 57 of 63 shared proposals, and
  `run-20261005-125939` vs `-130451` agreed on 38 of 43
  (`run-20261005-130451/stability_vs_run-20261005-125939.json`). With one sample the agreement was
  31 of 43. Some disagreements are unanimous within each run, so the noise is partly correlated.
- `run-20261005-124428` failed: GLM sent the nested `scores` object as a JSON string twice. The
  axes are now top-level fields.

Open: the rubric has never recommended REJECT live; the REJECT branch and the no-model-call
reject paths are unit-tested only. `resources_in_build_plan` has known false positives. The
`.evidence/triage` runs total about 550 MB (raw SSE events in CallStore).

## Build

`taskforge.build` (`step.py`, `sdk.py`, `author.py`, `run.py`, `template/standard.py`): memoized
async steps keyed on step code, arguments, `SDK_VERSION`, proposal, and policy; the `Build` SDK;
one structured GLM call that authors a builder program; and `run_build`, which enforces the
library rules. Unit: `uv run --group test pytest tests/build` passes (24 tests, 1 live skipped).

What `run_build` enforces: the verifier is a GRADER step's output; the controls are a CONTROLS
step's output; `validate_controls` passes. It also fails a build whose controls include a
candidate `b.try_grader` graded, unless that candidate is a grader's reference answer or the
empty answer. Every `try_grader` call records its candidate as a `try_grader/<digest>.json`
resource, and these resources are replayed on cache hits. On the run-1 d27 program, which graded
all its control candidates in its GRADER step, the check names 4 of its 6 controls
(`.evidence/build/replay-check/`). Memo keys include every data global a step reads, including
dicts, lists, `Fraction`s, and dataclass instances. A global with no JSON form raises
`TypeError`. `SDK_VERSION` is `taskforge.build/2`.

Live (GLM-5.3 interactive, 2026-10-05):

- `pytest tests/build -m live_glm`: 1 passed in 40.7s
  (`.evidence/build/live_pytest-20261005-1605.txt`, `live-test/20261005-160550/`). The authored
  program prototyped its grader on two wrong answers that are not controls, and the build
  passed. The positive control's whole transcript was replayed through RolloutEngine.
- `.evidence/build/live_build.py run-2` on the two ACCEPT proposals of run-1, after the SDK
  reference, author prompt, and control-replay rule changed. d27.reporting.close_measurement/8
  built on its first program (1 author call, 52,813 output tokens). Its rollout through
  `GlmRolloutModel` with the laptop factories took 1 turn, graded 1.0, and kept the prefix.
  A synthetic GRADER patch missed only `grader`; the other six steps hit. In run-1 this proposal
  needed 4 revisions. d01.algebra.linear-transformations/5 built on its third program
  (4 author calls, 290,549 output tokens). Both revisions came from the program's own ShellSim
  compatibility probe in its environment step: the first failed with reward 0.0, the second
  failed the heredoc read-back check.
- **Broken:** the d01 rollout through `taskforge.llm.rollout_model.GlmRolloutModel` raised
  `RolloutContractError: Model transport changed the served token prefix`. A rerun reproduced
  it: turns 1 to 4 kept the prefix, and turn 5 (prefix 9,733 tokens) did not
  (`.evidence/build/run-2/rollout-d01-rerun.json`, script `.evidence/build/rollout_only.py`).
  The run-1 inline non-streaming adapter kept the prefix on all 7 d01 turns. The bug is in
  `llm.rollout_model` (streamed token ids or reasoning replay), not in build.

Open:

- `sdk.py` imports `rolloutengine.machines._task_machine` and `rolloutengine.grading._shell_grade`.
  This blocks the build PR until upstream exports them
  ([docs/upstream/rolloutengine/public-task-machine.md](docs/upstream/rolloutengine/public-task-machine.md),
  not filed).
- Build imports `taskforge.proposal.model` (the seam type). DESIGN.md says stage packages never
  import each other, and triage does the same. Either move the seam type down a layer or amend
  DESIGN.md.
- The template's model-driven steps (research agent, `structured_until` loops) are covered by
  fake-GLM tests only. Authored programs replace them with deterministic code.
- The template's stdout grader reports no reward components, so the template writes no partial
  controls. The author prompt tells programs to use a JSON reward file when the proposal needs
  partial controls. No live program has done so yet.
- The author-to-build revision loop exists only in `.evidence/build/live_build.py`; in the
  library it belongs to review/loop. Docker environments were not exercised.

## Agent loop

Use a Taskforge-owned loop, `taskforge.llm.agent.run_agent`, over the existing `GlmClient`, with
shell tools routed through `shellbox.machine.Machine.run` and Parallel web search as plain tools
in `taskforge.llm.web`; vanilla pi was second, and all four candidates (in-house, pi, omp,
LangChain `create_agent`) ran web search successfully on GLM-5.3, so web search did not decide
it, while GLM-specific handling (reasoning replay, retries, holds, ledger records) and in-process
concurrency did. Decision, evidence and the list of what no candidate achieved:
[docs/agent_loop_decision.md](docs/agent_loop_decision.md).

### llm.agent

`llm/agent.py` (`run_agent`, `shell_tool`) and `llm/web.py` (`web_tools`, Parallel search and
extract over a caller-owned `httpx.AsyncClient`, retrying 408/429/5xx/transport errors) are written
and pass live on the interactive endpoint: `uv run --group test pytest tests/llm -m live_glm`
gave 15 passed in 70.7s (`.evidence/llm/agent/live_suite_review-20261005T200349Z.txt`). Evidence,
all under `.evidence/llm/agent/`:

| Check | Result | Evidence |
|---|---|---|
| T1 package and tests in ShellSim | answered, 6 turns, 10.0s; independent pytest rerun green | `t1_shellsim-20261005T200400Z.json` |
| T1b seeded failing package, fix loop | answered, 8 turns, 7 shell calls, 9.2s; rerun green | `t1b_fix_loop-20261005T200409Z.json` |
| Web search through Parallel | answered, 4 turns, 15.5s, 1 search; answer holds PyPI's 3 latest versions (checked against PyPI's JSON API before and after the run) | `web_parallel-20261005T200424Z.json` |
| Malformed tool call injected by a relay | `invalid_arguments` returned, replay accepted, recovered in 3 turns | `malformed_tool_call-20261005T200426Z.json` |
| 20 concurrent agents on one `GlmClient` | 20/20 correct, 1.6s total; tiny task, not a load test | `concurrent_20-20261005T200427Z.json` |
| Cut inside a tool call (`max_tokens=400`) | turn 1 finished `length`, its call returned as `truncated_call`; answered after 6 turns, 300 lines written | `length_cut_tool_call-20261005T200435Z.json` |
| Probe: tool call cut at `max_tokens=300` | vLLM sent `tool_calls`, `GlmClient` reported `length` (300/300 tokens) | `probe_tool_call_cut-20261005T200437Z.json` |
| Probe: replaying non-object arguments | raw malformed and `[1]` both HTTP 400; wrapped by `replay_arguments` both accepted | `probe_replay_arguments-20261005T200437Z.json` |

`GlmClient` reports a `tool_calls` segment that spent its whole `max_tokens` as `length`, so the
agent loop and `rollout_model` (which then ends the rollout with stop reason `length`) both see
the cut. The agent replays assistant turns through `rollout_model.assistant_wire_message`, wrapping
non-object arguments; `rollout_model` keeps arguments as served, because RolloutEngine ends a
rollout on them and the token prefix must not change.

Open: `jsonschema` is imported by `agent.py` but reaches the venv only through `verifyit[schema]`;
declare it in `pyproject.toml`. `shell_tool` copies RolloutEngine's tool definition and observation
format (only `SHELL_TOOL_NAME` is imported, from non-public `rolloutengine.task_session`) until
rolloutengine exports them. Not tested: hundreds-wide concurrency, a stream stall, a GLM-produced
malformed call, Docker or Iris machines. DESIGN.md names the Parallel key as
`~/openathena/build_envs/stage0/.parallel_key`; the file is `~/openathena/build_envs/.parallel_key`
(`PARALLEL_KEY=` line, override with `TASKFORGE_PARALLEL_KEY_FILE`). The phase-1 probe file
`probe_replay_and_tool_call_cut-20261005.json` was an ad-hoc probe without requests; the two
`probe_*` files above replace it.

## Validate

`llm.rollout_model.GlmRolloutModel` is the RolloutEngine model over `GlmClient` (exact served ids
from `return_token_ids`; no continuation on length, see below). `validate.trials.run_trials` runs k
trials through `ShellboxRolloutEngine`, refuses up front any task `sandbox.factories.task_refusals`
rejects (`MACHINE_UNSUPPORTED`, no machine started), retries retryable causes after a
`rigging.timing.ExponentialBackoff`, and records one ledger span and one rollout JSON per attempt.
`classify` maps a task's own setup, healthcheck or capability failure to `TASK_SETUP` (not
retried), and treats `agent_timeout` as a budget stop: `Graded` with the engine's partial grade, or
reward 0 (`passed=False`, stop reason `agent_timeout`) when the engine graded nothing.
`validate.controls.replay` checks the set with `validate_controls` and `max_turns`, then replays
each control as a `CONTROL` trial; a workspace control is written by scripted shell calls (base64
into the path, then `chmod`) after setup, as the agent would.

Live, 2026-10-05 20:21 UTC (`pytest tests/validate tests/llm/test_rollout_model.py -m live_glm`,
8 passed):

- `rollout_model/20261005T202152Z.json`: 3 multi-turn ShellSim rollouts, served prefix preserved
  on every turn, graded 1.0.
- `rollout_model/reasoning-20261005T202218Z.json` (and `-202125Z`): 3 rollouts x 6 turns with one
  puzzle per turn. 3,150 / 5,504 / 5,163 reasoning tokens on the long turn (7,398 max in the
  earlier run), reasoning replayed on the following turns, multi-line tab-indented heredoc tool
  arguments; prefix preserved on 18/18 turns, graded 1.0.
- `a_math_trials-20261005T202146Z`, `b_shellsim_trials-20261005T202146Z`: 3/3 graded 1.0 each.
- `c_controls_math-20261005T202147Z`, `c_controls_shellsim-20261005T202147Z`: every control MET,
  including the workspace control delivered by a shell turn (18 tokenize requests on ShellSim).
- `d_forced_failure-20261005T202148Z`: 2 forced start failures classified and retried, 3/3 graded.
  `d_forced_unsupported-20261005T202150Z`: `MACHINE_UNSUPPORTED`, not retried.

Open: a rollout turn never continues on `finish_reason == "length"` (a continuation re-renders the
prompt and breaks the exact-token contract), an exception to DESIGN.md decision 6 that DESIGN.md
does not yet record. Task-setup failures are recognized by RolloutEngine's error messages until it
raises typed setup errors. Agent-timeout zero scores and `TASK_SETUP` are not in DESIGN.md yet.
TaskCompendium 0.22's verifier registry does not reach verifyit `math` mode (the math task uses its
numeric verifier).

## Upstream issue drafts

Ready-to-file TaskCompendium issue bodies in `docs/upstream/` (index and construct-003 feature
tally in [docs/upstream/README.md](docs/upstream/README.md)). None has been filed.

| Draft | Blocks |
|---|---|
| `container-verifier-runtime.md` | 15 of 16 construct-003 tasks (container `script` verifiers) |
| `resources-with-roles.md` | 10 of 16 tasks carry verifier-role resources, 2 agent-role |
| `workspace-state.md` | `process` capability tasks (raised as `Unsupported`) |
| `shellsim-binding.md` | ShellSim tasks (raised as `Unsupported`) |
| `composite-verifier.md` | composite verification proposals |
| `judge-verifier-budgets.md` | judge verification (main has no judge verifier) |
| `multi-step-success-policy.md` | multi-step tasks (`all_required_steps` in 2 of 16) |
| `file-and-state-submissions.md` | file and state answers (raised as `Unsupported`) |
| `grading-detail.md` | partial-credit `criterion_mutation` controls (raised as `Unsupported`) |
| `action-interface-providers.md` | action interfaces (raised as `Unsupported`) |
| `json-submission-extraction.md` | JSON convention: GLM-5.3 0/3 (fenced, numeric answer) |
| `harbor-chat-usage.md` | Harbor chat trials drop usage, finish_reason; send no max_tokens |
| `invalid-task-outcome.md` | telling an invalid task from an infra error |
| `structured-answer-verifiers.md` | structured-answer and instruction-constraint verifiers |
| `package-manifest.md` | package identity and provenance |
| `task-metadata.md` | coverage tags and difficulty |

## Phase 2 entry conditions

Each condition names the phase 2 package it gates. A package starts when its conditions hold.

1. **Agent loop exists and is live-validated** (build, validate). `taskforge/llm/agent.py` and
   `taskforge/llm/web.py` are written and pass live GLM-5.3 tests for: multi-turn tool dispatch
   with `reasoning_content` replay, Parallel search and extract, an unparseable tool call returned
   to the model as an error result, and a shell tool run through a `Machine`.
2. **A real sandbox has run a command** (build, validate). Either a Silo broker and hosts are up
   (needs the user's approval) and `scripts/silo_smoke.py` passes every check in an Iris job, or
   the shellboxified Silo has merged and the same checks pass against it. Until then nothing in
   build or validate can be validated live.
3. **An image builder exists** (build). An `ImageBuilder` that builds an `EnvironmentRecipe` in
   an Iris CPU task, pushes to `envreg.208261-marin-gpu.coreweave.app` and reads back the manifest
   digest, and Silo hosts can pull the result.
4. **Executable verifiers have a home** (validate, review). `container-verifier-runtime` and
   `resources-with-roles` are either merged upstream or present as interim
   `spec/extensions/<gap>.py` modules with tests; without them 15 of 16 accepted construct-003
   shapes cannot be lowered or graded.
5. **The GLM path works inside Iris** (loop, queue; also any build or validate run on the
   cluster). `GlmClient` completes a call from an Iris task via `resolve_glm_base_url`, the relay's
   `/health` shape is confirmed, and a `taskforge.ledger` row is written to and read back from the
   cluster Finelog.
6. **Dependencies are declared** (all). `lib/taskforge/pyproject.toml` declares `marin-core`, and
   `resolve_glm_base_url`/`GLM_MODEL` move into a lib package (or the experiments dependency is
   declared and accepted) so taskforge does not import `experiments`.
7. **DESIGN.md matches the code** (all). The triage seam is updated (async `evaluate` with
   `ctx`, `Repair`, `Verdict.rubric` as a tuple of samples, `TriageDecision`), `proposal/model.py`
   is named as a seam module stage packages may import, and decision 4 records the agent loop.
8. **Loop policies are decided** (loop). The maximum number of triage repair rounds, what a
   `SlotFailure` does to its idea, and whether the triage rubric's residual noise (57/63 cross-run
   agreement) is acceptable or needs more samples or a lower temperature.
9. **Ledger is emitted** (loop, for STAGE-TIMING replacement). `GlmClient` calls and sandbox
   operations are wrapped in `span`, so `ledger_summary.py` reports real timings.

Not entry conditions, but known before phase 2: continuation on `length` is lossy at seams and
did not reproduce an uninterrupted answer mid-reasoning, so long structured outputs need
validate-and-repair; and the JSON submission convention should not be used with GLM-5.3.
