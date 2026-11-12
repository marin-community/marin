# forge

> **Interim README.** This is the operator's guide for running the pipeline from the
> `mark/forge-integration` branch while it is being built and piloted. It is deliberately more
> detailed than a package README should be and will be replaced before the PR stack lands.
> If you are reading this on `main`, it is stale: open an issue.

`forge` builds verified tasks from proposals. One agentic **author** session writes a `TaskSpec`
and its controls, **validation** runs solver trials, control replays and adversarial review
against the lowered task, and a pure **review** rule turns that evidence into accept, modify or
abandon. Every effect (model call, shell command, file read, grade) is recorded in an event log,
so a crashed workpiece resumes by replay: model calls come back from the log, commands are
re-executed on a fresh machine.

Design: `agent-docs/forge-implementation-plan-20261105.md` in the `openathena` checkout. This
file covers how to run it.

## Layout

| path | what |
|---|---|
| `src/forge/effects/` | effect types, the event log, replay, observers, `Hold`/`Restart`/`Transient` |
| `src/forge/resources/` | executors: inference (per-role waiting backend), environments, grading, rollouts, the round-owned verifier machine |
| `src/forge/harness/` | the agentic loop and its tools (`shell`, `read_file`, `write_file`, `grade`, `submit`, `validate`, `report`, `finish`, web) |
| `src/forge/stages/` | `triage`, `author`, `validate`, `review`; the review rules; prompt summaries |
| `src/forge/model/` | frozen data: proposal, workpiece, task spec, verdicts, decisions |
| `src/forge/policy/` | the policy record: `defaults/policy.json` and `defaults/prompts/<role>/*.md` |
| `src/forge/lowering.py` | `TaskSpec` → `TaskRolloutSpec` for the host; the only place that chooses machines |
| `src/forge/arms.py` | the S and T transition tables |
| `src/forge/coordinator.py` | stage → exit → edge loop, budgets, bounded stage replay |
| `src/forge/driver.py` | CLI: runs a shard of proposals, publishes to S3, resumes |
| `src/forge/wiring.py` | builds resources and config from the policy for a host |
| `experiments/forge/run_forge.py` | submits one Iris driver job per shard |

## The two arms

Both arms run the same stages; they differ in one transition table.

- **S, separate sessions.** author → submit → validate → review. On *modify*, a new author
  conversation starts, seeded with the findings and the previous submission's files.
- **T, validate as a tool.** The author's tool set gains `validate`. Calling it suspends the
  session, runs the validate stage, and returns a compact summary as the tool's reply; the full
  result is written to `.validation/<n>.json` in the workspace. Submit then goes through review
  as in S.

## Running on the cluster

Everything runs as Iris jobs on `cw-rno2a`. The entry point is the launcher, which submits one
driver job per shard; each driver works through its share of the proposals with `--concurrency`
workpieces in flight, one shellbox Iris sandbox per machine.

### Credentials

Export these in the same shell as the launch (never commit or print them):

| variable | source | used for |
|---|---|---|
| `CW_KEY_ID`, `CW_KEY_SECRET` | your personal CoreWeave key (`CW_STORAGE_KEY_ID/SECRET` in `~/openathena/.env`) | reading proposals, publishing outputs |
| `GLM_API_KEY` | the **high-pool** GLM token: the value after `GLM_API_TOKEN=` in `glm-infer/glm_api_token.txt` (source the file; the driver refuses the whole line), not the bulk token | author, solver and adversary model calls through the relay |
| `GLM_RELAY_JOB` | the Iris relay job that tunnels to the model routers, e.g. `/muchanem/glm53-relay-rno2a-g-20261027T0414Z` | endpoint resolution; one relay serves both `glm-5.3` and `deepseek-v4.1-flash` |
| `PARALLEL_API_KEY` | `TASKFORGE_PARALLEL_KEY` in `~/openathena/.env` | the author's web tools (optional) |

`GLM_BASE_URL` may replace `GLM_RELAY_JOB` for a direct endpoint; set exactly one.

### Inputs

- **Proposals**: a sample file such as
  `s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261103/sample.json` (192
  proposals, ranked; wave 1 is ranks 0–31). Each entry points at a proposal JSON whose header
  carries `environment` (`reasoning`, `shellsim`, `container`). The driver skips environments not
  in `proposals.environments` (default: reasoning and shellsim) and prints `SKIP <id>: environment=container`.
- **Policy overlay**: a JSON object of `key: value` pairs layered over
  `src/forge/policy/defaults/policy.json`, stored on S3. The pilots use
  `{"adversary.roster": ["adversary"]}` because the default roster is empty.
- **Ranks**: `--ranks 0,1,2` selects proposals by rank; no trailing comma (BSD `seq -s,` adds
  one, strip it with `sed 's/,$//'`).

### Launch

From the worktree root (`uv` needs the workspace `pyproject.toml`):

```bash
cd ~/openathena/branches/marin-forge-integration
export CW_KEY_ID=... CW_KEY_SECRET=... GLM_API_KEY=... PARALLEL_API_KEY=...
export GLM_RELAY_JOB=/muchanem/glm53-relay-rno2a-g-20261027T0414Z
ROOT=s3://marin-us-east-02a/marin/forge/pilot-<date>
uv run --frozen fsutil cp policy.json $ROOT/policy.json
for arm in S T; do
  uv run --frozen experiments/forge/run_forge.py \
    --proposals s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261103/sample.json \
    --ranks "$(seq -s, 0 31 | sed 's/,$//')" --arm $arm \
    --output $ROOT/$arm --policy $ROOT/policy.json \
    --job-prefix forge-pilot-<date>-$(echo $arm | tr A-Z a-z) \
    --drivers 7 --concurrency 4 --cpu 8 --memory 32GB --disk 100GB
done
```

Each proposal maps to `int(sha256(str(proposal_key)), 16) % 4096`; drivers process their
half-open `--bucket-range a:b` in manifest order. The launcher splits buckets evenly
across `--drivers N`; `--driver-range a:b` launches only those driver indices. To replace
drivers 24–31 of a 32-driver fleet with 16 drivers, cancel them, then launch with
`--drivers 64 --driver-range 48:64` and a new job prefix. This covers the same 3072:4096
bucket span without changing the other drivers, output paths, or resume behavior.
Keep `--buckets 4096` fixed across relaunches. The launcher passes `--max-retries 3`,
`--sync-package marin-forge`, pins the build-parallelism variables to `--cpu`, and leaves jobs
preemptible; a preempted driver resumes from its published checkpoints. Job names are
`/muchanem/<prefix>-<driver-index>`.

The driver prints every policy value it resolves next to the declared one (`POLICY key
declared=… resolved=…`) and the model each role uses (`MODEL role=author model=… endpoint=…`).
Read those lines first when a run looks wrong.

### Watching

`iris job logs` works only while a federated job is running, and `job list` is dominated by
the sandbox child jobs, so watch drivers by name:

```bash
uv run --frozen iris --cluster=marin job describe /muchanem/forge-pilot-<date>-s-0 | grep '^State:'
uv run --frozen iris --cluster=marin job logs /muchanem/forge-pilot-<date>-s-0 --max-lines 400
```

Sandboxes are child jobs named `/muchanem/<driver>/shellbox-<id>`; `job describe` on one shows
why it died (`OOMKilled`, preempted, did not start).

### Outputs

Each workpiece publishes to `<output>/<capability>/<slot>/`:

| file | contents |
|---|---|
| `events.jsonl` | the full effect log: every request, result, tool call, retry, scope; uploaded every 25 events or 60 s, and at every stage boundary |
| `workpiece.json` | the frozen workpiece: proposal, submission, evidence per round, decisions, terminal |
| `status.json` | `terminal` (`accepted`, `abandoned`, `held`, or `null` while running) and when the upload started |

A driver restarted against the same output resumes each workpiece from these files and prints
`SKIP <id>: published terminal=<state>` for every published terminal, `held` included. Held
workpieces are resumed only when the launch passes `--resume-held`; use it under a new job prefix
after an infrastructure outage (sandbox admission, a provider gap longer than
`provider.unavailable_wait`). Model calls replay from the event log; shell commands are
re-executed on fresh machines, so a resumed author session takes minutes before new events appear.
**Cancelling a driver publishes its in-flight workpieces as `held` with the reason "parent job …
has terminated"**; those resume the same way.

While a driver is working on a workpiece, its `status.json` reads `terminal: null` with
`upload_started_at`; the hub's `job describe` for a finished federated driver can keep saying
`running` indefinitely, so judge progress by these files and by the driver's pods, not by job state.

Run `uv run --frozen python -m forge.driver report <output-root> [--since-hours N]`
with the storage credentials above. It reads published snapshots and event parts concurrently,
then prints terminal/reason counts, accepted/hour over 1/6/24 hours, tokens by role,
stage durations, and consecutive-event latency proxies. Replayed results do not add token
spend or latency samples. Solver usage comes from recorded rollout steps; other roles follow
their scopes. Missing stage scopes (currently review) print `n/a`. Percentiles use nearest rank.
`--since-hours` includes workpieces active in that window and filters metric events by time;
acceptance windows are clipped to it, with rates divided by the clipped duration. Acceptance
uses the terminal decision timestamp, not file upload time. Empty roots and unreadable artifacts
fail instead of silently producing an incomplete tally.

## What the policy controls

The keys in `defaults/policy.json`, by group. Override any of them in the overlay.

- **Models**: `author.model` (DeepSeek V4.1 Flash), `solver.model` and `adversary.model` (GLM-5.3),
  each with `<role>.context_tokens`. There is no output cap; the prompt limit is the model's window.
- **Review band**: `band.low`/`band.high` on the solver pass rate; `solver.k` trials. Placement
  tags a task, never filters it: a *too easy* task gets one hardening round, a *too hard* one easing
  rounds within `budget.max_modify_rounds`, and a task still out of band is accepted with
  `accepted_with: too_easy` or `too_hard`. A revision that regresses (a defect, a placement on the
  other side, or an author that stops) reverts to the last clean submission and accepts that.
- **Budgets**: `budget.max_modify_rounds`, `budget.max_validation_retries`, `budget.max_triage_repairs`.
- **Adversary**: `adversary.roster` (names of adversary harnesses; empty by default),
  `adversary.image`. A finding is a demonstrated claim with a candidate and an expected grade;
  the review counts only claims the grader confirms, re-graded on a fresh machine.
- **Sandboxes**: `author.image`, `author.workdir`, `sandbox.memory_mb`, `grader.timeout`,
  `solver.*` timeouts (the turn and attempt timeouts are raised to cover `provider.unavailable_wait`).
- **Provider**: `provider.unavailable_wait` (a busy provider is waited out, with
  `provider.backoff_*`), `provider.stall_timeout`, `provider.max_connections`.
- **Tools**: `tools.output_chars`; longer output is saved to `.tool-output/<digest>.txt` in the
  workspace and the model gets the head plus the path.

Prompts live in `defaults/prompts/<role>/`; the author's system prompt renders the live
`TaskSpec` and control schemas, and the adversary's user prompt renders the task without file
payloads (they are in the workdir).

## Environments

Lowering accepts three task environments: none (the solver answers in the conversation),
ShellSim, and the one pinned python container (`GRADER_BASE_IMAGE`). Any other image is refused
at the author's `validate`/`submit` with a message naming the fix. The author and adversary
always work in the pinned container with the task's resources hydrated.

## Running locally

Only the tests run locally; the pipeline does not.

```bash
uv run --frozen --package marin-forge --group test pytest lib/forge/tests -q -p no:cacheprovider
cd lib/forge && uv run --frozen pyrefly check
```

`tests/test_end_to_end.py` drives both arms through fakes (inference, environments, grading);
`tests/resources/` covers each executor. Any change that touches the sandbox backend must also
pass a live probe on `cw-rno2a` before it is relaunched into a pilot: the fakes have passed while
the real backend failed three times. The verifier restructure adds that probe as a marked live
test (`test_verifier_live.py`, excluded by the default marker expression and run as an Iris job).

## Known limits on this branch

- `container` proposals are skipped until image support exists beyond the pinned image.
- Bubblewrap cannot run unprivileged in an Iris Kubernetes task (#10692); the verifier machine
  is a long-lived shellbox Iris sandbox.
- A cancelled driver leaves `held` records that need `--resume-held` (see Outputs).
- `lib/taskforge` is frozen and untouched; `forge` does not import it.
