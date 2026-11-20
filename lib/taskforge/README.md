# taskforge

`taskforge` builds verified tasks from proposals. One agentic **author** session writes a `TaskSpec`
and its controls, **validation** runs solver trials, control replays and adversarial review
against the lowered task, and a pure **review** rule turns that evidence into accept, modify or
abandon. Every effect (model call, shell command, file read, grade) is recorded in an event log,
so a crashed workpiece resumes by replay: model calls come back from the log, commands are
re-executed on a fresh machine.

This file covers how to run it.

## Layout

| path | what |
|---|---|
| `src/taskforge/effects/` | effect types, the event log, replay, observers, `Hold`/`Restart`/`Transient` |
| `src/taskforge/resources/` | executors: inference (per-role waiting backend), environments, grading, rollouts, the round-owned verifier machine |
| `src/taskforge/harness/` | the agentic loop and its tools (`shell`, `read_file`, `write_file`, `grade`, `submit`, `report`, `finish`, web) |
| `src/taskforge/stages/` | `author`, `validate`, `review`; the review rules; prompt summaries |
| `src/taskforge/model/` | frozen data: proposal, workpiece, task spec, verdicts, decisions |
| `src/taskforge/llm/` | the GLM transport kept from the previous Taskforge: client, policy, wire helpers |
| `src/taskforge/policy/` | the policy record: `defaults/policy.json` and `defaults/prompts/<role>/*.md` |
| `src/taskforge/lowering.py` | `TaskSpec` → `TaskRolloutSpec` for the host; the only place that chooses machines |
| `src/taskforge/coordinator.py` | author → validate → review, decisions and budgets, bounded stage replay |
| `src/taskforge/driver.py` | CLI: runs a shard of proposals, publishes to S3, resumes |
| `src/taskforge/wiring.py` | builds resources and config from the policy for a host |
| `experiments/taskforge/run_taskforge.py` | submits one Iris driver job per shard |

## The workflow

author → submit → validate → review. On *modify*, a new author conversation starts, seeded with the
findings and the previous submission's files; on *retry*, the same submission is validated again.

On every author entry, including revision rounds, the workspace holds the proposal, the previous
submission's `task.json`, `controls.json` and resources, and any validation results. A caller can add
its own files, such as reference data, with `TaskforgeConfig.workspace` (a `WorkspaceHydration`; the
default adds none). Only the author sees them; a file reaches solvers or graders only through the task.

## Running on the cluster

Everything runs as Iris jobs on `cw-rno2a`. The entry point is the launcher, which submits one
driver job per shard; each driver works through its share of the proposals with `--concurrency`
workpieces in flight, one shellbox Iris sandbox per machine.

### Credentials

Export these in the same shell as the launch (never commit or print them):

| variable | used for |
|---|---|
| `CW_KEY_ID`, `CW_KEY_SECRET` | your personal CoreWeave object-storage key: reading proposals, publishing outputs |
| `GLM_API_KEY` | the interactive-tier model token: author, solver and adversary model calls through the relay |
| `GLM_RELAY_JOB` | the Iris relay job that tunnels to the model routers: endpoint resolution; one relay serves both `glm-5.3` and `deepseek-v4.1-flash` |
| `PARALLEL_API_KEY` | the author's web tools (optional) |

`GLM_BASE_URL` may replace `GLM_RELAY_JOB` for a direct endpoint; set exactly one. The launcher
passes the four keys as `--secret NAME env:NAME`, so their values stay off the `iris job run`
command line; the driver reads them from Iris secret files.

### Inputs

- **Proposals**: a sample file such as
  `s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261103/sample.json` (192
  proposals, ranked; wave 1 is ranks 0–31). Each entry points at a proposal JSON whose header
  carries `environment` (`reasoning`, `shellsim`, `container`). The driver skips environments not
  in `proposals.environments` (default: reasoning and shellsim) and prints `SKIP <id>: environment=container`.
- **Policy overlay**: a JSON object of `key: value` pairs layered over
  `src/taskforge/policy/defaults/policy.json`, stored on S3. The pilots use
  `{"adversary.roster": ["adversary"]}` because the default roster is empty.
- **Ranks**: `--ranks 0,1,2` selects proposals by rank; no trailing comma (BSD `seq -s,` adds
  one, strip it with `sed 's/,$//'`).

### Launch

From the repository root (`uv` needs the workspace `pyproject.toml`):

```bash
export CW_KEY_ID=... CW_KEY_SECRET=... GLM_API_KEY=... PARALLEL_API_KEY=...
export GLM_RELAY_JOB=/<user>/<relay-job>
ROOT=s3://marin-us-east-02a/marin/forge/pilot-<date>
uv run --frozen fsutil cp policy.json $ROOT/policy.json
uv run --frozen experiments/taskforge/run_taskforge.py \
  --proposals s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261103/sample.json \
  --ranks "$(seq -s, 0 31 | sed 's/,$//')" \
  --output $ROOT/out --policy $ROOT/policy.json --job-prefix forge-pilot-<date> \
  --drivers 7 --concurrency 4 --cpu 8 --memory 32GB --disk 100GB
```

Each proposal maps to `int(sha256(str(proposal_key)), 16) % 4096`; drivers process their
half-open `--bucket-range a:b` in manifest order. The launcher splits buckets evenly
across `--drivers N`; `--driver-range a:b` launches only those driver indices. To replace
drivers 24–31 of a 32-driver fleet with 16 drivers, cancel them, then launch with
`--drivers 64 --driver-range 48:64` and a new job prefix. This covers the same 3072:4096
bucket span without changing the other drivers, output paths, or resume behavior.
Keep `--buckets 4096` fixed across relaunches. The launcher passes `--max-retries 3`,
`--sync-package marin-taskforge`, pins the build-parallelism variables to `--cpu`, and leaves jobs
preemptible; a preempted driver resumes from its published checkpoints. Job names are
`/<user>/<prefix>-<driver-index>`.

The driver prints every policy value it resolves next to the declared one (`POLICY key
declared=… resolved=…`) and the model each role uses (`MODEL role=author model=… endpoint=…`).
Read those lines first when a run looks wrong.

### Watching

`iris job logs` works only while a federated job is running, and `job list` is dominated by
the sandbox child jobs, so watch drivers by name:

```bash
uv run --frozen iris --cluster=marin job describe /<user>/forge-pilot-<date>-0 | grep '^State:'
uv run --frozen iris --cluster=marin job logs /<user>/forge-pilot-<date>-0 --max-lines 400
```

Sandboxes are child jobs named `/<user>/<driver>/shellbox-<id>`; `job describe` on one shows
why it died (`OOMKilled`, preempted, did not start).

### Outputs

Each workpiece publishes to `<output>/<capability>/<slot>/`:

| file | contents |
|---|---|
| `journal.json` | the head: journal format (2), the owning writer, the chunks that make up the event stream and the bytes they cover, `status` (`terminal` is `accepted`, `abandoned`, `held`, or `null` while running), `published_at`, and once the workpiece ends the frozen `workpiece` (proposal, submission, evidence per round, decisions, terminal) |
| `chunks/<writer>-<offset>.jsonl` | immutable slices of the effect log: every request, result, tool call, retry, scope; published after 25 events or 60 s, whichever comes first |

Events are recorded by stable kinds (`inference.request`, `env.run_command`, ...; the table is in
`effects/events.py`), never by Python module path. Each driver claims `journal.json` by
compare-and-swap when it starts and advances it the same way after each chunk. A second driver on
the same workpiece (a relaunch racing a live driver, a preempted worker's replacement) takes the
head; the first then fails with `Superseded` at its next checkpoint and publishes nothing more.
Readers follow only the chunks the head names. A root in any other journal format, including the
format-1 layout (`events.jsonl`, `events.jsonl.parts/`, `status.json`), fails with an error naming
the format found and is left untouched; such roots are converted once by a separate migration.

A driver restarted against the same output resumes each workpiece from its journal and prints
`SKIP <id>: published terminal=<state>` for every published terminal, `held` included. Held
workpieces are resumed only when the launch passes `--resume-held`; use it under a new job prefix
after an infrastructure outage (sandbox admission, a provider gap longer than
`provider.unavailable_wait`). Model calls replay from the event log; shell commands are
re-executed on fresh machines, so a resumed author session takes minutes before new events appear.
Replay reuses recorded model results only while the re-executed commands return the output that
was recorded; after the first difference, the remaining model calls run again.
Cancelling a driver sends SIGTERM to each workpiece process, which exits without a final snapshot:
its head keeps `terminal: null`, and the next launch resumes it without `--resume-held`.

While a driver is working on a workpiece, its head reads `terminal: null` with
`published_at`; the hub's `job describe` for a finished federated driver can keep saying
`running` indefinitely, so judge progress by these files and by the driver's pods, not by job state.

Run `uv run --frozen python -m taskforge.driver report <output-root> [--since-hours N]`
with the storage credentials above. It reads each journal head and its chunks concurrently,
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
  with `author.context_tokens` and `adversary.context_tokens` as their prompt limits. There is no output cap;
  the prompt limit is the model's window.
- **Review band**: `band.low`/`band.high` on the solver pass rate; `solver.k` trials. A task
  that is *too easy* gets one hardening round and is then accepted with `accepted_with: too_easy`,
  reverting to the earlier submission if the hardening broke something.
- **Budgets**: `budget.max_modify_rounds`, `budget.max_validation_retries`.
- **Adversary**: `adversary.roster` (names of adversary harnesses; empty by default),
  `adversary.image`. A finding is a demonstrated claim with a candidate and an expected grade;
  the review counts only claims the grader confirms, re-graded on a fresh machine. An adversary
  whose context window or output budget runs out before it reports is `exhausted`: the task can
  still be accepted, and the Accept's reasons say the search budget ran out, not that it validated the task.
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

A task runs with no machine (the solver answers in the conversation), on ShellSim, or, when it
names a `docker_image`, on an Iris gVisor machine. The grader's environment is restricted: its
image must be the one pinned python container (`GRADER_BASE_IMAGE`) or null for ShellSim, and the
author's `submit` refuses any other grader image with a message naming the fix. The author and
adversary always work in the pinned container with the task's resources hydrated.

## Running locally

Only the tests run locally; the pipeline does not.

```bash
uv run --frozen --package marin-taskforge --group test pytest lib/taskforge/tests -q -p no:cacheprovider
cd lib/taskforge && uv run --frozen pyrefly check
```

`tests/test_end_to_end.py` drives the workflow through fakes (inference, environments, grading);
`tests/resources/` covers each executor. Any change that touches the sandbox backend must also
run on `cw-rno2a` before it is relaunched into a pilot: the fakes have passed while the real
backend failed three times.

## Known limits

- `container` proposals are skipped until image support exists beyond the pinned image.
- Bubblewrap cannot run unprivileged in an Iris Kubernetes task (#10692); the verifier machine
  is a long-lived shellbox Iris sandbox.
- A cancelled driver publishes no final snapshot for its in-flight workpieces; the next launch
  resumes them from their last checkpoint (see Outputs).
- S3 output roots (`marin/forge/...`), Iris job prefixes (`forge-pilot-*`), sandbox paths
  (`.forge-candidates/`, `/tmp/forge-hydration-*`) and the event-log encoding markers
  (`__forge_*__`) keep the `forge` name of the pilot runs. Roots written by the `forge` pilots
  use journal format 1 and need the one-time migration before `taskforge` reads them.
