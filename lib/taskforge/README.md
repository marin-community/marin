# taskforge

`taskforge` builds verified tasks from proposals. One agentic **author** session writes a `TaskSpec`
and its controls, **validation** runs solver trials, control replays, adversarial review and a
machine-less **solve-review** of the recorded trials against the lowered task, and a pure **review**
rule turns that evidence into accept, modify or abandon. Every effect (model call, shell command, file read, grade) is recorded in an event log,
so a crashed workpiece resumes at its last stage boundary: paid results come back from the log by key, and an
unfinished session starts again on a fresh machine.

This file covers how to run it.

## Layout

| path | what |
|---|---|
| `src/taskforge/effects/` | effect types, the event log, replay, observers, `Hold`/`Restart`/`Transient` |
| `src/taskforge/resources/` | executors: inference (per-role waiting backend), environments, grading, rollouts, the round-owned verifier machine |
| `src/taskforge/harness/` | the role runner (`role.py`: machine, hydration, scope, session, close for every model role), the agentic loop and its tools (`shell`, `read_file`, `write_file`, `grade`, `submit`, `report`, `finish`, web) |
| `src/taskforge/stages/` | `author`, `validate`, `review`, `confirm`; the review rules; prompt summaries |
| `src/taskforge/model/` | frozen data: proposal, workpiece, task spec, verdicts, decisions |
| `src/taskforge/llm/` | the model client (the one retry owner), the `Provider` protocol and its GLM default, model profiles, wire helpers |
| `src/taskforge/policy/` | the policy record: `defaults/policy.json` and `defaults/prompts/<role>/*.md` |
| `src/taskforge/lowering.py` | `TaskSpec` → `TaskRolloutSpec` for the host; the only place that chooses machines |
| `src/taskforge/coordinator.py` | author → validate → review, decisions and budgets, stage boundaries, bounded stage restarts |
| `src/taskforge/driver.py` | CLI: runs a shard of proposals, publishes to S3, resumes |
| `src/taskforge/wiring.py` | builds resources and config from the policy for a host |
| `experiments/post_training/capability_driven_envs/` | our launcher: one Iris driver job per bucket range |

## The workflow

author → submit → validate → review → confirm → review. On *modify*, a new author conversation starts,
seeded with the findings and the previous submission's files; on *retry*, the same submission is validated
again with a fresh cohort of solver trials under the new round's attempt; the retried round's trials stay in its
own record. A limited slot is never bought again when the same visit or cohort is replayed, but a retry's
fresh cohort buys every slot of the new round, that one's included. A limit never causes a retry. On *accept*, the confirmation step runs before the acceptance is final (see **Confirmation** below).

On every author entry, including revision rounds, the workspace holds the proposal, the previous
submission's `task.json`, `controls.json` and resources, and any validation results. A caller can add
its own files, such as reference data, with `TaskforgeConfig.workspace` (a `WorkspaceHydration`; the
default adds none). Only the author sees them; a file reaches solvers or graders only through the task.

## Running on the cluster

The pipeline runs as Iris jobs. Each `taskforge.driver run` job works through its share of the proposals
with `--concurrency` workpieces in flight, one shellbox Iris sandbox per machine. The capability-driven runs
submit their drivers with the launcher in
[`experiments/post_training/capability_driven_envs`](../../experiments/post_training/capability_driven_envs/README.md).

### Credentials

Export these in the same shell as the launch (never commit or print them):

| variable | used for |
|---|---|
| `CW_KEY_ID`, `CW_KEY_SECRET` | your personal CoreWeave object-storage key: reading proposals, publishing outputs |
| `GLM_API_KEY` | the interactive-tier model token: author, solver and adversary model calls through the relay |
| `PARALLEL_API_KEY` | the author's web tools (optional) |

The launcher passes the four keys as `--secret NAME env:NAME`, so their values stay off the
`iris job run` command line; the driver reads them from Iris secret files. The model gateway is a
driver flag, not a variable: `--glm-relay-job` names the Iris relay job that tunnels to the model
routers (one relay serves both `glm-5.3` and `deepseek-v4.1-flash`), and `--glm-base-url` names a
direct endpoint ending in `/v1` instead; pass exactly one. A blank relay job, or a base URL that
does not end in `/v1`, is a usage error before anything is claimed. Inside an Iris job the driver
reaches the controller that Iris names in `IRIS_CONTROLLER_URL`.

### Inputs

- **Proposals**: a sample file such as
  `s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261103/sample.json` (192
  proposals, ranked; wave 1 is ranks 0–31). Each entry points at a proposal JSON whose header
  carries `environment` (`reasoning`, `shellsim`, `container`). The driver skips environments not
  in `proposals.environments` (default: reasoning and shellsim) and prints `SKIP <id>: environment=container`.
- **Policy overlay**: a JSON object of `key: value` pairs layered over
  `src/taskforge/policy/defaults/policy.json`, stored on S3. The default roster runs the two
  adversary briefs; `{"adversary.roster": ["adversary"]}` selects the single general
  adversary.
- **Ranks**: `--ranks 0,1,2` selects proposals by rank; no trailing comma (BSD `seq -s,` adds
  one, strip it with `sed 's/,$//'`).

### Launch

The launcher's README has the command. Each proposal maps to `int(sha256(str(proposal_key)), 16) % --buckets`
(4096), whatever prefix holds its capability directory; a driver processes its half-open `--bucket-range a:b` in manifest order.
Keep `--buckets` fixed across relaunches, so that replacing some drivers with more or fewer covering the same
span changes no other driver, output path, or resume behaviour. A preempted driver resumes from its published
checkpoints. `uv run --frozen python -m taskforge.driver --help` lists the exit codes of a `one` child and when `run`
fails.

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

| path | contents |
|---|---|
| `journal.json` | the head: `format` (the JSON integer 16); `writer`, the owning writer's random id; `chunks`, the chunk names in stream order; `size`, the bytes they hold; `status` (`stage`, `round`, `terminal`, which is `accepted`, `rejected`, `abandoned`, `held`, or `null` while running); `workpiece`, the terminal workpiece's object name or `null`; and `published_at`, the writer's clock at the swap |
| `chunks/<sha256>.jsonl` | an immutable slice of the effect log (every request, result, tool call, retry, scope), ending at a line boundary and named by the SHA-256 of its bytes; published after 25 events or 60 s, whichever comes first |
| `workpieces/<sha256>.json` | the immutable terminal workpiece (proposal, submission, evidence per round, decisions, terminal), named by the SHA-256 of its bytes; read it at `workpieces/<head.workpiece>` |

The event stream is the head's `chunks`, concatenated in order, each read from `chunks/<name>`. A
reader checks every chunk and workpiece against the digest in its name and fails with
`CorruptJournal`, naming the path, when one is missing or altered or the head is malformed,
including a negative `size`, a chunk or workpiece name that is not a SHA-256 with its suffix, and a
terminal `status` with a null `workpiece`. A finished root is recognised from the head
alone, with one GET. Chunks and workpieces that no head names (a writer that lost the head, or one
that died before its swap) stay behind; nothing collects them.

Events are recorded by stable kinds (`inference.request`, `env.run_command`, ...; the table is in
`effects/events.py`), never by Python module path. Each driver claims `journal.json` by
compare-and-swap when it starts and advances it the same way after each chunk. A second driver on
the same workpiece (a relaunch racing a live driver, a preempted worker's replacement) takes the
head; the first then fails with `Superseded` at its next checkpoint and publishes nothing more.
Readers follow only the chunks the head names. A root in any other journal format, including the
format-1 layout (`events.jsonl`, `events.jsonl.parts/`, `status.json`), format 2 (sandbox rows
replayed by content, without owners) and format 3 (adversary findings with an expected reward band, not a
defect direction with observed and deserved grades), format 4 (trials hydrated without a detail, a too-easy
baseline in place of a digest-bound fallback), format 5 and format 6 (no solve-review: its acceptances name no
missing evidence although none ran, and a resumed format-6 round would buy a review its recorded history lacks),
format 7 (no confirmation step: its acceptances end at the validation round and name no missing confirmation, and a
resumed format-7 round would buy one its history lacks),
fails with an error naming both formats and exits 4
with the root untouched, before anything is claimed, whether the root is running, held or finished; `report`
lists it as `skipped unsupported_format <root>/journal.json has journal format N, not 16` and reports the
rest. Formats 8 to 15 differ: their rows decode at format 16, so `report` reads them (their rows' `opened` and
`attempt` fields are read and discarded), but the driver refuses
to claim a running or held root of those formats, exiting 4 with its head untouched and naming the writer of its
format or a fresh output prefix as the way on; a finished root of those formats is skipped untouched. Format 16
resumes at stage boundaries and runs sandbox operations live, where format 15 rebuilt each sandbox by replaying its
recorded operations in order. Format 15 rejects unsupported judge settings during an author's
`grade` or `submit`. A format-14 author could receive a grade for the same draft and take a different next action.
Format 14 records a solver generation still running at
`solver.time_limit` as stopped at the time limit (`time_limit`), which a format-13 reader cannot decode; format 13 recorded it as a generation limit, under another
solver clock. Format 13 commits reviewed visits, which a format-12 root lacks, so its resumed rounds would
decide again on re-observed evidence. Format 12 words two author requests otherwise: a final revision's request ends
with its ordinal, and a too-easy placement's line no longer calls the round the one hardening attempt, so a resumed
format-11 author would send requests its recorded calls did not. Formats 8 to 10 embedded a schema reference in their
author calls that this writer does not reproduce, so a resumed author would buy different answers that the recorded
sandbox history then refuses. A format-8 workpiece omits author responses to findings, a final revision's notes, a
confirmation's shipped notes and a finding's citation groups at their defaults (`OMIT_*` in `model/review.py`), and a
fault revision's null placement; no reader that predates the omitted placement read a root outside tests. Format 11
adds the `schema.author_reference` row (below), which a format-10 reader cannot decode; at 11 that reader refuses the
root as another format. Format 10 adds a row's `billed` field, outside the event, naming the role that
paid for a physical cached resolution; a format-9 reader refuses a format-10 root as another format. Format 9 adds the `rollouts.solver_generation` row (a solver generation awaiting its
decision-model grade, below), which a format-8 reader cannot decode; at 9 that reader refuses the root as another
format, untouched, instead of reporting it corrupt. Such roots are converted once by a separate migration, or rebuilt. Format 5 places a round over its scored solver slots and holds a slot
without one; it covers two writers that its header cannot tell apart: one whose readers cannot decode a `decision_model.read` row, and one that could write
`decision_model.read` rows but had no `effect.invalidated` row (the marker
naming the refused slot result an operator-authorised replacement supersedes); a bare request row in either cannot
establish an operator's intent. Format 6 refuses both alike, and a reader of the first kind refuses a format-6 root
the same way instead of misreading its new rows. Before claiming, the reader
validates every row and exits 5 (`CorruptJournal`) on any violation: strictly typed fields, an ordered sequence,
and `billed` only on a physical cached resolution. A best-effort check for accidental corruption also requires each
trial receipt in a stage boundary to name an earlier physical solver resolution of that trial's slot; a deliberately
rewritten journal is out of scope, and reading a transcript at its receipt checks the full request again (a mismatch
there holds the workpiece).
A storage hiccup never costs paid model work; only an explicit refusal stops the writer
(`checkpoint.is_transient_storage_error`). A refusal is an HTTP 4xx other than 408 and 429 (denied
access, a bad key or signature, a missing bucket) or a local filesystem refusal (permission, a missing
path, no space). Every other storage error, including resets, short reads, timeouts, throttling and
any 5xx, is transient: a checkpoint's lines stay pending and are resent from the head's size on the
next timer tick, and the final publication and the history reads on resume retry for several
minutes. After any inconclusive conditional write (412, 409, or a transient error) the writer reads
the target instead of trusting the error: its own bytes mean the write applied; a target still absent,
or a head still at the version this writer observed, means it did not, and it is retried against that
same version; other bytes under a content name are `CorruptJournal`, and a head naming another writer
is `Superseded`. The claim at start retries the same way before any model call. A refusal, or final retries running out, stops the
writer with `PublicationFailed`: the process fails, the head keeps its last checkpoint, and a relaunch
replays that history and re-executes the effects after it. A collision found while publishing is
`CorruptJournal`. Neither publishes a hold.

A driver restarted against the same output resumes each workpiece from its journal and prints
`SKIP <id>: published terminal=<state>` for every published terminal, `held` included. Held
workpieces are resumed only when the launch passes `--resume-held`; use it under a new job prefix
after an infrastructure outage (sandbox admission, a provider gap longer than
`provider.unavailable_wait`). The one exception is a workpiece held only until the decision model answers again:
its last decision is `Incomplete(..., awaits="decision_model")` and its `one` child exits 75. Every later launch
enters it without the flag, and `run` parks such proposals until the decision model's health route reports healthy,
or, on a gateway without that route, until a `one --probe` child exits 10 because its own reader's last physical
outcome in that process was a valid answer. A hold that mixes the outage with any other cause stays manual.

**Generation checkpoints.** When a solver trial (validation or confirmation) is graded by the decision model and
its generation finished but the grade did not come back, because the decision model gave no read within
`decision_model.unavailable_wait` or returned a level off its scale, the generated rollout (every conversation, the
graded chat, native actions, stop reason, captured outputs and usage) is journalled as a `rollouts.solver_generation`
row before the slot holds. It is keyed by the slot's own identity (task spec, trial, attempt) under the solver's
rollout fingerprint. The next execution of that slot (an unflagged relaunch after an outage, `--resume-held` after
an off-scale level) dispatches no solver request: it grades the recorded generation again through the same
TaskCompendium decision-model grade and recorded read, so an equal answer reads the decision model once. `report`
charges each generation once, reading the journal in order: a checkpoint row charges its generation (including one
still awaiting its grade), the slot results after it were graded from it and add nothing, and a slot result before it
(a refused slot's paid turns, a format-8 off-scale result) paid for an earlier generation and keeps its charge. The
order is read over the whole journal, so `--since-hours` charges only rows inside the window. Limits: a generation interrupted before it finished (a machine or model failure, a cancelled process) is
generated again, as is a slot whose solver fingerprint changed (solver model, sampling, execution settings or the
grader identities it records). A trial with no answer, or a refused slot, is no generation awaiting a grade and
keeps its existing replay. Graders that run in a sandbox or through the LLM judge keep their existing path. The
row reaches the published journal with the next checkpoint and with the held publication; if the process dies
abruptly between the read's failure and that publication, or at any time while the read is still waiting, the
next launch generates the answer again. A gateway refusing the resumed read (an HTTP 4xx other than 404, 408 and
429), or answering it with an HTTP 200 that is no distribution over the levels (a body that is not JSON, a missing
field or level, another shape, a boolean, a number too large for a float, a non-finite or negative probability, a
non-finite sum, or all zero), is no outage: the reader raises a hold whose diagnostic (cause and body together) is
capped at 300 characters and records no read, the solver batch lets every admitted slot settle and starts no
further one, and the workpiece holds for a person (exit 0, as any held workpiece) instead of restarting the
stage. A refusal of a slot's first read still restarts the stage, as before.

**Stage boundaries.** Each launch resumes from the last `stage.boundary` row in its journal: the workpiece the next
stage starts from, and the plan it was recorded under. A boundary is recorded when the author submits or stops into a
continuing decision, and when a review continues (a business `Retry` included); validation and confirmation
evidence stays provisional until the review after it continues, so a launch cut or held between them runs that stage
again, its paid solver slots and grades hitting their keys. A review that decides `Incomplete`, or a stage that
raises a hold, leaves the previous boundary in force: an explicit resume re-enters the stage that produced the
evidence, never the review that held on it. A boundary carries each trial's `receipt`, the journal row its result
was bought at (none for a slot whose output budget ran out twice, the only case with no row; a slot-local refusal
keeps its row). Boundaries ride the ordinary checkpoints. A hard kill can lose an author submission and paid
effects after the last published chunk. A graceful cancellation attempts to publish its buffered rows.

The stock driver can reuse a boundary when its proposal and policy match the published plan. The journal
format versions the wiring that builds its roles, workspaces and machines from that policy. Custom wiring
starts at the author on every launch and still reuses eligible paid requests.
A launch whose plan differs from the latest boundary's records a new initial boundary and starts at the author;
it never searches back for an earlier boundary of its own plan, so switching back to an earlier plan starts that
plan's author again. Its paid results still hit their own keys, so it buys only what no row holds. An edit to the
policy, even of an operational setting such as the solver heartbeat, is such a change. A new plan's decisions
follow its settings, so lowering `budget.max_missing_trials` below a round's missing slots turns that round into a
hold. The author reads each round's evidence as `.validation/<n>.json` in ordinary JSON with sorted keys and without
receipts (a nonfinite reward is `null`), so a cold launch hydrates the bytes the first one wrote.

Model calls replay from the event log by their content; shell commands and file operations run live on a fresh
machine, so a resumed author session takes minutes before new events appear. A restarted session reopens its
conversation and runs its tools again: with the tool output its first run saw, each recorded model turn hits the
cache. A changed PID, file or errno can produce a new request and another purchase. Each turn checks its full key.
The author's system prompt embeds the `TaskSpec` and control JSON schemas, generated from the installed
TaskCompendium models at a root's first fresh author session and recorded as a `schema.author_reference` row
(about 29 KB); every later author session of the root, in this launch or a resumed one, replays that text, so a
wording-only change in TaskCompendium's schemas does not affect a root that recorded it. `wiring.install` binds
this effect; a caller that passes `run_workpiece` its own installer must bind `AuthorReference` to
`stages.author.author_reference` (or its own executor) to run the stock author stage.

The judge callback requires greedy sampling. The author reference omits `temperature` and `chat_template_kwargs`.
The `submit` and `grade` tools ask the author to revise a draft that changes either default (0 and `{}`), selects the
`answers` rubric, or requests an unsupported reasoning effort. The reference also omits VerifyIt's HTTP timeout and
retry settings. Existing tasks may supply them. Taskforge's provider policy controls the callback transport.

Each command or file operation is journalled as its request and then its outcome: a resolution under the sandbox's
fingerprint, or a handled guest failure (an `OSError` such as a missing file, the only failure tools hand to the model)
recorded by errno name and text; the outcome row's `answers` names its request row. These rows feed the usage report
and inspection. With concurrent sandboxes, the link identifies the command that produced an outcome, but
does not identify which sandbox issued that command. Recovery reads these rows as part of the journal and
dispatches sandbox operations live. A transfer archive the backend cannot unpack, a lost machine or an unknown
command outcome restarts the stage on a fresh machine within its restart budget (`STAGE_REPLAY_ATTEMPTS`, three
attempts), and holds after it.
SIGTERM (an Iris cancel, a preemption, a lost node's grace period) is a crash. The command line's `driver.main`,
the process entry point and not an embedding API (it leaves its handler installed), handles it by exiting 143 at
once: no cancellation, no final snapshot, no flushed output. `one` and `run` called as library functions install
nothing. Where the driver is its container's PID 1 the handler is what lets it exit at all, since the kernel drops a
signal PID 1 has no handler for; so the immediate exit holds only from the moment `main` installs the handler, and a
single SIGTERM delivered while the interpreter is still importing is lost (a later one exits). `run` starts each `one` child with `--die-with-parent FD`, the
read end of one pipe that the parent alone holds open and never writes (the write end is not inheritable and the
driver never forks without exec while holding it). The child checks the pipe for end of file before it reads or
claims anything, then watches it from a thread, and exits 143 when it reads end of file, however its parent died,
SIGKILL and OOM included. This holds under normal interpreter progress only: the watcher must take the GIL, so a
stopped child or a native call holding the GIL can outlive its parent, and a parent death after the first check is
asynchronous, so the child may still start effects. A worker's own local subprocesses and remote sandboxes receive
no end of file. A strict subtree deadline belongs to the runtime: on Iris, the container's PID namespace and the
pod's grace period. SIGINT keeps its asyncio behaviour (the run is cancelled and stops its children). An unsignalled
parent still counts a child that exits 143 or dies by a signal as failed.

What a crash promises is that a relaunch with the same manifest, partition and output root can reopen each root's
last committed valid head, not that one relaunch completes it. An old claim still in flight can win against the first
replacement, and a hold resumed with `--resume-held` whose refusal's invalidation had not committed may need the flag
again. A head whose conditional write committed before the driver heard back is the head the relaunch reads: a
terminal or held workpiece is skipped and an outage-only hold is entered. Paid external work that finished or was
running when the process died may have no journal row and can be bought again; the checkpoint triggers (25 events,
60 seconds) bound neither. Pending output is not delivered. Sandbox jobs are cancelled by the controller
asynchronously, so a driver's exit does not certify their deletion. Exit 143 alone promises neither an Iris retry nor
a particular budget: Iris charges a pod failure with an authoritative disruption condition to the preemption budget,
an ordinary failure to `--max-retries`, and ends an explicitly cancelled job. Object-store heads are conditional
PUTs, so a crash leaves the previous head or the new one; on a local root a kill can land inside the head's in-place
rewrite and leave it unreadable (`CorruptJournal`).
A workpiece process cancelled in-process publishes a terminal head only with the workpiece that
reached it; one cancelled after its terminal event was written but before that workpiece came back
publishes `terminal: null` over the full history, and the next launch replays it to the terminal
workpiece without new model calls.

**Rollout rule: never launch a root with an earlier driver.** A context-window refusal of an agent's request
(author, adversary or reviewer) is recorded as a `context_exceeded` completion, so a resumed visit replays the
stop. This revision and later ones read every row earlier drivers wrote; earlier drivers reject that row. The
earlier driver of this journal format raises while it reads the journal, before it claims or publishes anything,
so the root stays as published: `CheckpointSink.claim` decodes the history before it swaps the head. No
format number fences the drivers before this layout. So roots written by this revision or later must never be
launched by an earlier driver, and a rollback of the binary uses a fresh output prefix.

While a driver is working on a workpiece, its head reads `terminal: null` with
`published_at`; the hub's `job describe` for a finished federated driver can keep saying
`running` indefinitely, so judge progress by these files and by the driver's pods, not by job state.

Run `uv run --frozen python -m taskforge.driver report <output-root> [--since-hours N]`
with the storage credentials above. It reads each journal head and its chunks concurrently,
then prints terminal/reason counts, the outcomes of the solver trials in the completed validation and confirmation
records of terminal workpieces (`solver_trial_outcomes scope=terminal_records outcome=<o> count=<n>`, one line per
outcome; each bought slot once, though a confirmation record repeats the trials it keeps; a
running root and a stage that did not finish add nothing), accepted/hour over 1/6/24 hours, tokens by role,
stage durations, and consecutive-event latency proxies. Replayed results do not add token
spend or latency samples. Solver usage comes from recorded rollout steps. Each physical model call and solver slot
is charged to the role its row names as payer (`billed`), set by the visit, judge callback or stage scope that made
it, so overlapping roles never take each other's calls. The confirmation step reports apart, once: its solver slots
as `role=confirmation` and its final review as `role=final_review`, never also under `solver` or `solve_review`; a
judge call during it is `role=judge`. A row that names no payer, written before format 10, and a call from a visit
built without one (`Visit(..., bills=...)` is keyword-only) are charged to `role=unattributed`, a line printed only
when some charged row falls there; total spend is unchanged. A grade that follows its recorded generation names its
payer but is not charged again. An interrupted solver attempt's partial usage is not reported (no journalled result holds it). Missing stage scopes (currently review) print `n/a`. Percentiles use nearest rank.
`--since-hours` includes workpieces active in that window and filters metric events by time;
acceptance windows are clipped to it, with rates divided by the clipped duration. Acceptance
uses the terminal decision timestamp, not file upload time. Empty roots and unreadable artifacts
fail instead of silently producing an incomplete tally.

## What the policy controls

The keys in `defaults/policy.json`, by group. Override any of them in the overlay.

- **Models**: `author.model` (DeepSeek V4.1 Flash), `solver.model`, `adversary.model` and
  `solve_review.model` (GLM-5.3), with `author.context_tokens`, `adversary.context_tokens` and
  `solve_review.context_tokens` as their prompt limits. There is no output cap;
  the prompt limit is the model's window.
- **Review band**: `band.low`/`band.high` on the solver pass rate; `solver.k` trials. A task
  out of band gets one band-only revision round within the shared budget and is then accepted at its own
  placement, tagged `accepted_with: <band>`; only a fault or the author stopping reverts to the stored fallback.
  `band.accept_too_easy` (default true) false refuses a too-easy placement on every path: a round asks for revision
  rounds within the shared budget, then restores a stored fallback (which only an accepted band can be) or is
  abandoned; a final revision's round goes on to its confirmation, which places it again; and a confirmation
  placed too easy asks for a final revision while `budget.final_revisions` has one left, else abandons the task
  (`Confirmation revision failed`, or `Confirmation failed` when no revision was allowed, naming the placement,
  the setting and the spent budget; a final revision's finding is listed unresolved only when the final review
  reports it again or the author declined it).
  The coordinator enforces it for any review rule (`TaskforgeConfig.review`): a terminal acceptance at a band
  `config.band` refuses raises `ValueError` naming the setting, while a pending one still goes to its confirmation.
  In this order: a clean revision placed outside its fallback's band is accepted at its new
  placement, tagged; a round with review findings is a defect round, not the band-only one, even out of band;
  and the band-only round is spent only by a round revised for its placement alone.
- **Budgets**: `budget.max_modify_rounds`, `budget.max_validation_retries`, and `budget.final_revisions` (default 1,
  at least 0): the revisions the final review may ask for, outside `budget.max_modify_rounds` (below). Each
  final revision request ends with its ordinal (`This is final revision N of M.`, counted from the recorded
  decisions and appended after `author.revision` renders, so a custom template cannot drop it): two identical
  requests are two author attempts, and a restart of either replays it.
- **Adversary**: `adversary.roster` (brief names, each run as one adversary session; default
  `false_accept` and `false_reject`), `adversary.image`. The default user prompt renders `{brief}`, so each
  name needs a prompt `adversary.brief.<name>` (`prompts/adversary/brief/<name>.md`); a user prompt
  that renders only `{name}` (`Review strategy: {name}`) uses the roster name itself as the strategy
  and runs any name. The default briefs: `false_accept` looks for a submission
  the grader accepts but should not, `false_reject` for a correct submission the grader rejects, and
  `adversary` is a single general strategy line. An overlay may rewrite a brief's text;
  a new brief is a new prompt entry in the policy (`Policy({**policy.entries, "adversary.brief.mine":
  Prompt("...")})`) and a name on the roster. Every brief reports through one schema: a defect
  direction (`false_accept`, `false_reject`, `grader_crash`), the candidate, the grade the grader
  OBSERVED and the reward the candidate DESERVES. Validation keeps a finding only when the replayed
  grade is wrong in the stated direction on the round's verifier and again on a fresh one. The report
  tool refuses a finding whose observed grade differs from the adversary's own latest grade of that
  exact candidate in its visit. When findings of a nonempty report fail reproduction, the replay's
  grades answer the pending report call and the adversary may report again in the same session,
  `adversary.max_revisions` times (default 1); reproduced findings are kept. A search
  that ends without a reproduced finding, by an empty report or because its context window or output
  budget ran out first (`exhausted`), counts as evidence of a strong task: the task can still be
  accepted, and its result and the Accept's reasons say it ran out of search budget without a finding,
  never that it validated the task. Each result carries a `continuation` (the search's name by visit and occurrence,
  why it stopped: `reported`, `budget_exhausted` or `held`, and the digest of its conversation at the
  stop) for a workflow that continues an exhausted search; the default workflow does not.
- **Solve-review**: once a round's solver trials and controls are clean and its placement is
  reviewable, one session with no machine reads the recorded trials and the task's files through
  `list_evidence` (resumable from a source number and a character offset into its summary, so a long
  address is never cut off), `read_evidence` (pages addressed by source, part and character offset) and
  `search_evidence`, and calls `report`. Sources have their own namespaces: `trial/<n>` (each message, an
  interruption's marker, then the grade), `task/task.json`, `task/prompt.md`, `task/controls.json`, and
  `resource/<role>/<path>` for every file the task carries, unresolved build inputs (`actor_build`,
  `grader_build`) included; a file's summary is the signature TaskCompendium's task review signs (role,
  path, mode, mtime, sha256 and size), so a mode change or an equal-length edit changes the evidence (bytes
  that are not UTF-8 are named opaque, with no parts). A reply to a session without a machine, a loop
  diagnostic as much as a tool's output, is clipped to `tools.output_chars` where the harness delivers it,
  so nothing spills to a file it cannot read; a cap below one evidence page (240) is refused when the
  config is built. Everything the reviewer sees, the opening's task manifest, the trial summaries and the
  task prompt's historical tool calls included, renders canonically (sorted JSON, integral floats as integers, tool-call ids kept), and the
  opening always ends with the index's digest outside the editable template, so a resumed round, cold or
  warm, replays its review only when the evidence is unchanged. Findings have one of four kinds (`ambiguous_prompt`, `grader_defect`,
  `answer_leak`, `environment_problem`), a description and edit with text beyond whitespace, and citations
  `{source, start, end, quote}`. A finding is kept only when every quote occurs, whitespace collapsed, in
  the parts it addresses; any other is recorded as dropped with its reason. The report must account for
  every trial that did not pass, by a real trial number, and a failure attributed to a task defect needs a
  finding that survived its citations and cites that trial (task files may support it alongside);
  otherwise the report goes back to the reviewer. While the shared
  budget has a round left, kept findings send the task back for a revision (the author may decline a
  finding by number, with a reason, on `submit`), which is validated again and restores the reviewed
  version through the fallback on a fault; after that they are the acceptance's `notes`. A restore carries
  the review, missing evidence, placement denominator and adversary exhaustion of the round it restores, and
  labels what triggered it: `Revision fault:` for a fault, `Revision placement:` for a placement
  the policy does not accept. An acceptance names what it went without in
  `missing`: no review ran or it ended without a report (`solve_review`), or trials lacked a complete
  transcript (`transcripts`; an interrupted trial is reviewed by its prefix, and only a round where no
  trial recorded a message skips the review). Missing evidence never holds a task. The review runs over
  any set of trials, so a later step can pass its own prompts and report tool.
- **Confirmation**: `solver.confirmation_k` (5) more solver trials of the submission an acceptance ships,
  then a final review over all of them; 0 makes the acceptance final at its validation round. The shipped
  round is the one the acceptance chose (a restored fallback's included), and its scored trials and
  transcripts are read from the journal, never bought again. That holds for completed slots whose results the
  journal can reuse, and, for a decision-model grader, for an answer generated in full whose grade did not come back
  (see "Generation checkpoints"). The new slots number past that round's
  indices under their own attempt, through the same solver batch, concurrency, heartbeat and
  `budget.max_missing_trials` allowance (a refused slot beyond it holds). The batch and its grade observers
  settle as validation's do, so an outage beside another hold is a manual hold, and only a pure
  decision-model outage parks for automatic relaunch. Controls and adversaries do not run again. The
  confirmation record is assessed as a validation round first, before the final reviewer is asked: an invalid
  task or a grader fault in the new trials is the author's to repair and an exhausted trial budget rejects, never a
  denominator entry, and a record that rejects or is incomplete is decided without the final review. Every
  trial's outcome (`TrialResult.outcome`, read from its recorded cause and grade) is one of `graded` (a reward,
  a submission failure scoring zero), `generation_limit` (it ran out of context or output), `time_limit` (still
  generating at `solver.time_limit`), `budget_exhausted` (a configured turn budget spent: `max_turns`),
  `faulted` (a grader or task fault, a finding) or `unscored` (held, ungraded,
  grader infrastructure; its cause says which). Graded
  trials and limits, and only they, are placed: a limit is an unsolved trial whatever its grade says (a grader failure
  on its partial output, returned or raised at the verifier's deadline, is kept in its grade, never holds validation
  and replays on resume; a decision-model outage still checkpoints the generation for regrading), which the acceptance
  names (`Counted unsolved without a gradable answer: ...`), and an unscored trial is named (`confirmation_trials`) and excluded. The final review
  (`solve_review.final.*` prompts, with its own nudge) uses the same reviewer, evidence index, citation
  checks and replay identity (the confirmation builds one index for its final review and its shipped-note check; its
  sources are a read-only mapping and its version is computed once, so a `solve_review(report=...)` factory that
  serializes the index converts `dict(evidence.sources)`, a `dict[str, Source]`, with the fields it needs:
  `dataclasses.asdict` and a pydantic dump of `Evidence` refuse the read-only mapping), is shown every earlier report with what became of each finding (the
  author's response recorded against the review that gave it: revised, declined with its reason, stopped,
  a note, or not acted on; a revision a restore discarded says so), and sorts its new findings into `revise` and `note`. An account may blame a
  trial on an earlier finding without repeating it while that finding's quotes still occur in the trial. A
  finding given both dispositions is refused. In every report, round or final, each submitted version of a
  finding is checked against the evidence first, and the versions that survive become one finding carrying
  their distinct citations, so a stale copy never hides a valid one and the author answers one number. The
  finding keeps each surviving version's citations as its own group (`groups`, stored only when there are
  several): shipped notes, an earlier finding an account blames and the earlier-claim recheck keep a finding
  while any one group's citations all still occur, so a revision that changes one cited trial never takes down
  the version citing another.
  The placement over every placed trial re-tags the acceptance in any band the policy accepts (a refused band:
  above); `note` findings are the acceptance's notes. A new `revise`
  finding (or one an earlier review gave whose latest response on the shipped round's history is a revision:
  a restore discards the revisions after the round it restores, and a later decline overrides) or a fault buys
  a revision outside the shared modification budget while `budget.final_revisions` has one left, validated as
  any round; the final review's notes go to the
  author beside it as optional. A finding repeated from an earlier report buys nothing and is kept once.
  A final revision keeps no fallback: a clean revision is accepted at its placement whatever the band and
  confirmed again, its round findings are notes, and a fault or an author stop abandons the task
  (`Confirmation revision failed`, naming the unresolved finding). With no final revision left, any `revise`
  finding is a note and a confirmation fault abandons the task (`Confirmation failed` when none was allowed). A
  final revision's finding is named unresolved only when the last final review reports it again or the author's
  latest answer to it on the kept history is a decline.
  The acceptance ships every earlier note, optional note and new note the shipped trials still support,
  each once, in a version whose quotes occur in them (the confirmation record's `notes`).
  `accepted_with` is published only on a final acceptance. Missing confirmation evidence
  (`confirmation_trials`, `transcripts`, `solve_review`) is named once, never a hold.
- **Sandboxes**: `author.image`, `author.workdir`, `sandbox.memory_mb`, `grader.timeout`,
  `solver.*` timeouts. `solver.time_limit` (4200 s, 70 minutes; it replaces `solver.turn_timeout`, which measured
  the engine's model-turn wall from before admission) bounds one solver generation, from its first generated token:
  a generation still running past it is cut, and the trial is recorded at the time limit (`time_limit`): unsolved,
  never retried. The engine has no model-turn wall: the model client owns a turn, its time limit, stalls and
  availability window. `solver.attempt_timeout` (8400 s; the run refuses one that is not greater than `solver.time_limit`) is a watchdog on
  one physical attempt, machine preparation and grading included. Once the time limit is proven, an attempt it stops
  is still the settled unsolved trial. An attempt it stops before that restarts the stage within the existing restart
  bound (`STAGE_REPLAY_ATTEMPTS`, three attempts in all), after which the slot is held: at the default cap that can
  spend about seven hours of solver time on one slot. Two availability waits around a long cut stream can reach it,
  and an operator who wants such recoveries sets a larger cap. A judge grade's deadline is its normal minimum or `provider.unavailable_wait` plus
  `provider.stall_timeout`, whichever is larger; a decision-model grade's is its normal minimum plus
  `decision_model.unavailable_wait`, so its typed outage comes first. Neither depends on the solver settings.
- **Provider**: `provider.unavailable_wait` (a busy provider is waited out, with
  `provider.backoff_*`), `provider.stall_timeout`, `provider.max_connections`. The run refuses a
  `solver.time_limit` that is not positive and finite, and a `solver.attempt_timeout` that is not finite and
  greater than it. That is a configuration check, not a promise about which clock wins: the cap should leave room
  above the time limit for machine preparation, tool turns and grading, and a cap close to the limit can interrupt
  a trial the limit would have failed, which then restarts as above.
- **Tools**: `tools.output_chars` (at least 240); longer output is saved to `.tool-output/<digest>.txt` in
  the workspace and the model gets the head plus the path, or, for a session without a machine, is clipped.

Prompts live in `defaults/prompts/<role>/`; the author's system prompt renders the root's
recorded `TaskSpec` and control reference (above), and the adversary's user prompt renders its brief and the task without
file payloads (they are in the workdir).

## Models and providers

Every model call goes through one `ModelClient` (`llm/client.py`), the only place that retries.
It sends each attempt through a `Provider` and decides from the attempt's outcome. A solver trial is
charged as failed only on positive evidence that the model itself ran out; anything the fleet, the
relay, the transport or local admission may have caused is an outage, and no failed trial is recorded.

| Failure | Evidence checked | What happens |
|---|---|---|
| 400 context overflow (recognised wording or `context_length_exceeded`) | the rejection itself | `ContextExhausted`: a solver trial ends at the generation limit (counted, never solved); the author and adversary get `ContextExceeded` |
| solver generation past `solver.time_limit` | the same stream still generating: a second generated payload at least `CONFIRMATION_INTERVAL` (1 s) after its first one past the bound, with no finish, abort, cut or stall in between; counted from each attempt's first token; no pool state is consulted | the trial ends at the time limit (`time_limit`; counted, never solved) |
| other rejection (401, unknown 400 wording, ...) | the status | `ModelRequestRejected`, never retried; a solver rollout holds at once with the status and reason |
| 408, 429, any 5xx, 404 `no route`, no endpoint, registry lookup failure, refused connection, full local pool, an error body not whole within `provider.stall_timeout` of its headers, stall (also inside a tool call), cut or aborted stream (also past the bound) | `/health` while waiting | waited out with backoff, polling `/health` while the pool reports no workers, for `provider.unavailable_wait`; then `ModelUnavailable`, which restarts the stage (a solver rollout is interrupted, never a failed trial) |

Generated text is nonempty content, reasoning, a function name or an argument fragment; a role, an index or
an empty string is not (`progressed`), and the bound, stall detection and window renewal all use that one
test. The first failure in a call after its attempt generated text, other than a stall, restarts the
availability window; that happens once per call, and later failures count against the restarted window. A `/health` read that is undecodable, or
that outlasts `HEALTH_TIMEOUT` or the remaining window, counts as unknown availability.

Known limit: the generation bound is measured where the client reads the stream. Output still queued on the path
when its worker exited can keep arriving for longer than `CONFIRMATION_INTERVAL` and is then charged as a failed
trial, also while `/health` reports the pool empty. A wall-clock bound also makes the failed-trial rate depend on
fleet decode speed; a server-enforced output-token limit (`finish_reason=length`) would not. Whether a client-side
cutoff may be charged as a failed trial at all is an open decision for the project owner.

The defaults are `GlmProvider` (the GLM router, `llm/glm_provider.py`) for every role and the
`GLM_5_3` profile (`llm/policy.py`: sampling defaults, context window, no output limit, tool-call
argument repair). A caller overrides either per role through the public wiring:

```python
from dataclasses import replace
from taskforge.llm.policy import GLM_5_3
from taskforge.wiring import runtime

profiles = {"author": replace(GLM_5_3, name="glm-5.3/output-32k", max_output_tokens=32_768)}
providers = {"adversary": MyProvider()}  # a `Provider` subclass: `model` and `stream(body, progress)`
async with runtime(policy, source, parallel_api_key, providers=providers, profiles=profiles) as (config, resources):
    ...
```

`build_config(policy, factories, profiles)` and `build_resources(policy, providers, factories, web, profiles)` take the
same mappings when the caller builds the bindings itself. A provider must serve its role's `<role>.model`, which the
replay keys record; a mismatch is a `ValueError` naming both. The policy's `sampling.*` keys still apply on top of
the profile. A provider reports admission and every payload to `progress` and inherits a `no_workers()` that
knows of no outage. The profile's output limit is a ceiling on any token-limit field a request sets. Replay keys carry
the model, the role, the request, and for any profile but `GLM_5_3` its `name` and output limit, so a changed profile
never replays a result it did not shape. A solver rollout's and a candidate grade's own cache keys also
carry a non-default solver or judge profile, since their replay skips the inference inside them. A callable
has no stable hash: a profile with its own `replay_arguments` needs its own name, and a new one when the
formatter changes.

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
run on `cw-rno2a` before it is relaunched: the fakes can pass while the real backend fails.

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
