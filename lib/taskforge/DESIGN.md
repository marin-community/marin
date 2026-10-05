# Taskforge design charter

Taskforge is the rewrite of the `experiments/post_training/capability_env_gen` task-generation
pipeline as an encapsulated library. This file is the working contract for everyone building it.
The long-form proposal with diagrams is the "Taskforge Rewrite Proposal" artifact; this file
supersedes it where they differ.

Status: phase 2 (RolloutEngine rewrite) in progress, 2026-10-05. This tree is the worktree
`taskforge/base`, branched from the head of PR 9623 (`rollout-engine`), because Taskforge builds
on TaskCompendium 0.22 and `lib/rolloutengine`, which exist only there. Phase 1 (STATUS.md) was
built against TaskCompendium 0.8 on `mark/autoenv`; its `spec/` package is obsolete.

## Decisions already made

1. **TaskSpec is upstream's, and the task's own scripts carry task-specific grading.** Taskforge
   owns no spec model. A built task is a TaskCompendium 0.22 `TaskSpec`: `environment` (null,
   shellsim, or docker with `RegistryImage` or `DockerBuild`, files, setup, healthcheck), a
   `verifier`, optional `stages`. Task-specific checks are `ShellVerifierSpec` scripts the task
   ships (private files, reward on stdout, exit code, or reward files, optionally in a separate
   grading environment). Generic verifier TYPES (math, mcq, judge, pytest, aggregation) belong to
   `lib/verifyit`, which TaskSpec reaches through its verifier registry; VerifyIt is used as a
   library, not a CLI, and extended upstream where that is missing. Anything beyond both becomes
   an upstream issue (drafts in `docs/upstream/`, sorted by owner). Interim code lives in
   `taskforge/spec/extensions/<issue>.py`, one module per gap. No tarballs, no source patches,
   and no Harbor lowering: RolloutEngine executes a TaskSpec directly.
2. **Execution is RolloutEngine's; Silo is gone.** `rolloutengine.engine.ShellboxRolloutEngine`
   runs a task: it creates one shellbox `Machine` per attempt from the caller's
   `MachineFactory` per `EnvironmentKind`, installs files, runs setup and healthchecks, drives
   the shell tool, grades, and closes. Taskforge supplies the model callable (a GLM adapter
   over `taskforge.llm` honoring the exact-token contract: `return_token_ids`), the factories
   (`shellbox.backends.iris` on the cluster, docker locally, shellsim), and optional
   `TaskSession` factories for adversary roles. Nothing in Taskforge talks to a sandbox except
   through a `Machine` the engine or a builder step created. The `sandbox/` package shrinks to
   image building (`DockerBuild` recipe to a pushed `RegistryImage` digest) and factory
   selection. Changes the engine needs for our use (a public grade-this-transcript entry point
   for control replay, `NetworkPolicy.DENY` on the Iris backend, machine reuse across a build
   session) are PRs against `lib/rolloutengine` and `lib/shellbox`, stacked on PR 9623.
3. **Every other recommendation in the proposal stands:** builder programs with memoized steps,
   typed trial outcomes with one classifier, review as a program that returns a Decision, an
   append-only event log as the only item state, a lease queue on a replicated worker gang.
4. **Agent loop: the Taskforge-owned loop** (`taskforge.llm.agent.run_agent` over `GlmClient`,
   shell tool through `Machine.run`, Parallel web search in `taskforge.llm.web`), per
   `docs/agent_loop_decision.md`. Approved by the user on 2026-10-05. The builder's agent runs
   on this loop; solver and adversary rollouts run on RolloutEngine, whose model callable is the
   same `GlmClient`.
5. **Deliverables are PRs that can be filed today**, stacked with the `gh-stack` skill on top of
   PR 9623: one Taskforge PR per seam, plus PRs against `lib/rolloutengine`, `lib/taskcompendium`
   (draft) and `lib/verifyit` (draft). Each PR is small, tested, and lint-clean on its own.
6. **Inference defaults are the model maximum.** `max_tokens` starts at the model's output
   limit (131,072 for GLM-5.3) or remaining context; never a 32k ladder. On `finish_reason ==
   "length"` keep the output and continue. Timeouts are stall timeouts on the stream, not fixed
   wall-clock caps. Concurrency defaults to hundreds; throttle only after an observed failure.

## Package layout and ownership

```
lib/taskforge/
  pyproject.toml            name: marin-taskforge ; hatchling ; workspace member of the root
  DESIGN.md  STATUS.md      this file ; running account of what works and what is broken
  src/taskforge/
    llm/        client.py  policy.py  store.py  structured.py
    ledger/     records.py  jsonl.py  finelog.py
    sandbox/    silo.py (SiloMachineFactory)  _silo_client.py (vendored)  images.py  fake.py
    spec/       lowering.py  controls.py  extensions/
    proposal/   model.py  source.py  sources/capability.py  sources/build_envs.py
    triage/     checks.py  program.py  verdict.py
    build/      sdk.py  step.py  author.py  run.py  template/standard.py
    validate/   outcome.py  classify.py  trials.py  controls.py  solver.py  adversary.py  calibration.py  evidence.py
    review/     decision.py  rules.py  adjudicate.py  program.py
    loop/       program.py  events.py  patch.py
    queue/      coordinator.py  worker.py
  tests/        one test module per source module ; live tests gated by env (see Testing)
  .evidence/    gitignored ; raw request/response samples from live validation, by package
```

Dependency direction inside the library: `queue, loop` → `proposal, triage, build, validate,
review` → `llm, sandbox, spec, ledger` → external (`shellbox`, `taskcompendium`, `rigging`,
`finelog`, `iris`). Stage packages never import each other.

## Seam contracts

```python
# proposal/model.py
@dataclass(frozen=True)
class ProposalHeader:         # the YAML front matter; validated by code
    id: str; source: SourceRef; environment: Environment; verification: Verification
    grounding: Grounding; research: tuple[ResearchItem, ...]; build: tuple[BuildItem, ...]
    resources: tuple[str, ...]; null_reason: str | None
@dataclass(frozen=True)
class TaskProposal:
    header: ProposalHeader; body: str                 # markdown with required section headings
    @property
    def digest(self) -> str: ...                      # sha256 of the canonical bytes
# proposal/source.py
class ProposalSource(Protocol):                       # async: GlmClient is async-only
    async def propose(self, idea: Idea, n: int) -> ProposalBatch: ...
@dataclass(frozen=True)
class ProposalBatch:                                  # everything the event log records
    planning_request: tuple[Message, ...]; planning: tuple[Completion, ...]
    slots: tuple[SlotProposal | SlotFailure, ...]     # one per slot, in slot order; each carries
                                                      # its request, completions, repair error

# triage
class Check(Protocol):
    name: str; severity: Severity                     # FATAL | ADVISORY
    def run(self, p: TaskProposal, ctx: CheckContext) -> CheckResult: ...   # PASS | FAIL | SKIP
def evaluate(p, checks, rubric) -> Verdict            # structural fatal → reject, no model call

# build
@step                                                 # memo key = sha(source, arg digests, sdk version, model policy)
def build(b: Build) -> TaskDraft: ...

# validate
Outcome = Graded | Ungraded                           # Ungraded carries Cause (StrEnum) and retryable
def classify(failure) -> Cause                        # the ONLY classifier; UNCLASSIFIED is counted, never hidden
Evidence.status: Complete | Incomplete                # Incomplete carries Counter[Cause]

# review
Decision = Accept | Reject | Repair(patch, invalidate) | Retry(cause)

# loop
def task_loop(item, cfg) -> Terminal                  # state = append-only EventLog; status is derived
```

## Conventions

Root `AGENTS.md` and `TESTING.md` apply in full. The ones that bite here:

- No backward compatibility, no shims, no `hasattr` probing. Update call sites.
- Dataclasses and `StrEnum`, not dicts and string keys. Replace boolean flags with meaningful
  parameters or separate classes.
- Let exceptions propagate. Catch only to add context or to classify into a typed `Cause`.
- Separate computation from I/O. Resolve environment-dependent defaults once at the boundary.
- No `*_utils.py`. Function names reflect return types.
- Reuse first: `rigging.timing` (`retry_with_backoff`, `ExponentialBackoff`, `TokenBucket`,
  `RateLimiter`, `Deadline`), `marin.inference.structured_output.StructuredTool`,
  `shellbox.machine`, `shellbox.agent.BashAgent`, `taskcompendium.lowering`,
  `tasktrove_verify.modes.grade_judge`, `experiments/post_training/glm.py::resolve_glm_base_url`,
  the Finelog pattern in `lib/marin/src/marin/rollouts/catalog.py`.
- Lint and types: `./infra/pre-commit.py --files <paths> --fix` and `uv run pyrefly check` must
  pass before a package is called done.

## Live validation with GLM-5.3 (interactive tier)

Everything that talks to a model must be exercised against the real endpoint before it is
reported as working. From this laptop:

```bash
# router port-forward (already running in this session; restart if /health fails)
KUBECONFIG=~/.kube/open-athena kubectl -n open-athena port-forward svc/glm53-router 18000:8000 &
curl -s http://127.0.0.1:18000/health          # {"status":"ok","workers":{"high":4,"bulk":33},...}

export TASKFORGE_GLM_BASE_URL=http://127.0.0.1:18000/v1
export TASKFORGE_GLM_TOKEN_FILE=~/openathena/glm-infer/glm_api_token.txt   # line: GLM_API_TOKEN=...
```

- Model id `glm-5.3`, OpenAI-compatible chat completions, streaming supported. The bearer token
  binds the tier: the file above is the interactive (`high` pool) token. Do not use the bulk
  token for validation. Never write a token into the repo, a log, or an evidence file.
- Responses carry a `reasoning` field on the assistant message and `finish_reason`. Context
  window 262,144. Output limit 131,072; verify this by requesting it.
- Inside an Iris task the endpoint is resolved with `resolve_glm_base_url`, not the port-forward.
- Live tests live beside unit tests and skip with a clear reason when `TASKFORGE_GLM_BASE_URL`
  is unset. Give them a `live_glm` marker registered in this package's pytest config. Do not
  change the repo's default marker expression.
- Record every live validation as raw evidence: request, full response (including usage and
  finish_reason), wall time, and what was being checked, under `lib/taskforge/.evidence/<package>/`.
  Summarize the result and the evidence path in `STATUS.md`. "It should work" is not a result.

Web search for agent-loop evaluation: Parallel's API key is the `PARALLEL_KEY=...` line of
`~/openathena/build_envs/.parallel_key` (tests read another file from
`TASKFORGE_PARALLEL_KEY_FILE`); the MCP endpoint is `https://search.parallel.ai/mcp?mode=advanced`.
See `build_envs/stage0/run_area.sh` and `EXPERIMENT.md` for the three routes that have been tried.

## Silo access

Silo source is on branch `mark/silo` at
`/Users/k3sc0re/openathena/branches/marin-silo/experiments/silo/src/silo/` (`client.py`,
`http.py`, `errors.py` are stdlib-only, 557 lines; vendor them into
`taskforge/sandbox/_silo_client.py` with a header naming the source commit). The broker is
reachable only from inside the cluster (`SILO_BROKER_RESOLVE_URL` goes through the Iris
controller proxy; credentials in
`~/openathena/capability_env_gen/build/construct-003/silo_env.sh`). Unit-test the adapter against
a fake broker/host that mirrors the HTTP API (`broker/server.py`, `host/server.py`); live-test it
with a small Iris job (follow the `use-iris` skill; never stop or restart any cluster or job).

## Reference code map (what exists today, for porting or for evidence)

- Pipeline under rewrite: `experiments/post_training/capability_env_gen/capability_pipeline/`.
  Keep conceptually: `provider_retry.py`, `composite_policy.py`, `native_judge_protocol.py`,
  `inference.StageStore` (content-addressed call cache), `schema.py` + `validation.py` (proposal
  semantics), `conveyor.STATE_TABLE` (as a list of states to cover, not as code).
- Timing and bug evidence: `~/openathena/capability_env_gen/build/construct-003/STAGE-TIMING.md`,
  `~/openathena/capability_env_gen-dev/docs/audits/construct003_pipeline_bugs.md`.
- build_envs claim checks and gate philosophy: `~/openathena/build_envs/grade/check_claims.py`,
  `grade/gate.py`, `grade/prepare_shard.py --redact`.
- Closest existing Marin pipeline shape: `experiments/post_training/tasktrove/mcqa_routing_pipeline.py`.
