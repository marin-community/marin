# taskforge

`taskforge` builds verified tasks from proposals. One agentic **author** session writes a `TaskSpec`
and its controls, **validation** runs solver trials, control replays, adversarial review and a
machine-less **solve-review** of the recorded trials against the lowered task, and a pure **review**
rule turns that evidence into accept, modify or abandon. Every effect (model call, shell command, file read, grade) is recorded in an event log,
so a crashed workpiece resumes by replay: model calls come back from the log, commands are
re-executed on a fresh machine.

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
| `src/taskforge/coordinator.py` | author → validate → review, decisions and budgets, bounded stage replay |
| `src/taskforge/driver.py` | CLI: runs a shard of proposals, publishes to S3, resumes |
| `src/taskforge/wiring.py` | builds resources and config from the policy for a host |
| `experiments/taskforge/run_taskforge.py` | submits one Iris driver job per shard |

## The workflow

author → submit → validate → review → confirm → review. On *modify*, a new author conversation starts,
seeded with the findings and the previous submission's files; on *retry*, the same submission is validated
again. On *accept*, the confirmation step runs before the acceptance is final (see **Confirmation** below).

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
  `src/taskforge/policy/defaults/policy.json`, stored on S3. The default roster runs the two
  adversary briefs; the earlier pilots' `{"adversary.roster": ["adversary"]}` still selects their
  single legacy adversary.
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

| path | contents |
|---|---|
| `journal.json` | the head: `format` (the JSON integer 8); `writer`, the owning writer's random id; `chunks`, the chunk names in stream order; `size`, the bytes they hold; `status` (`stage`, `round`, `terminal`, which is `accepted`, `rejected`, `abandoned`, `held`, or `null` while running); `workpiece`, the terminal workpiece's object name or `null`; and `published_at`, the writer's clock at the swap |
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
lists it as `skipped unsupported_format <root>/journal.json has journal format N, not 8` and reports the
rest. Such roots are converted once by a separate migration, or rebuilt. Format 5 covers two writers that its
header cannot tell apart: Taskforge before decision-model grading, whose readers cannot decode a `decision_model.read` row,
and the first decision-model writers, which could write `decision_model.read` rows but had no `effect.invalidated` row (the marker
naming the refused slot result an operator-authorised replacement supersedes); a bare request row in either cannot
establish an operator's intent. Format 6 refuses both alike, and a reader from before decision-model grading refuses a format-6 root
the same way instead of misreading its new rows. Before claiming, the reader
validates every row and exits 5 (`CorruptJournal`) on any violation: strictly typed fields and at most
one envelope field; an open row carries a scope entry, a request a sandbox (`env.*`) event, an outcome a
resolution under re-executed sandbox fingerprints or a failure with an errno-name category, and no other
row carries any of these, so cached results never enter an owner's history; references name an earlier
open row or unanswered request of the same sandbox definition; and an answered request matches the
request and outcome its owner already established at that position. A request left unanswered may differ
from the recording, which is how a held divergence looks.
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
outcome in that process was a valid answer. A hold that mixes the outage with any other cause stays manual. Model calls replay from the event log; shell commands are
re-executed on fresh machines, so a resumed author session takes minutes before new events appear. A held
format-8 root read by the same revision, with the same effective inputs, resumes on its unfinished turns alone; the
author's system prompt embeds the pinned reference, so a wording-only change in TaskCompendium's schema does not
affect it.
Each sandbox attempt opens a row naming its owner (stage, round, visit, role and roster occurrence) and
its initial sandbox definition; each command or file operation it issues references that row, takes the
next position in the owner's history, and its outcome references the request. A resumed attempt is
served the outcome recorded at each position of the longest prefix its owner established, and runs a
request at a frontier (one whose outcome was never recorded) live once; a request beyond the history
returns its fresh result. A served command returns its recorded output, so later model calls replay
even when its text drifts (a PID, a timestamp, the truncation flag); each drift is logged at INFO as a
`Replay of <kind> diverged` note naming digests only. A handled guest failure (an `OSError` such as a
missing file, the only failure tools hand to the model) is recorded by errno name and text and served
again when the guest fails the same way; a transfer archive the backend cannot unpack is a stage
restart. A recorded outcome that an attempt dispatched but could not verify (the machine failed first)
is no frontier: the stage restarts within its replay budget and holds after it, never counting the
attempt as evidence. The workpiece holds as `Unreproducible` when an owner reopens on another sandbox definition, issues
another request at a recorded position, changes an outcome (success, errno, exit status or read
bytes), or returns normally before its recorded history ends.
Cancelling a driver sends SIGTERM to each workpiece process, which exits without a final snapshot:
its head keeps `terminal: null`, and the next launch resumes it without `--resume-held`.
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
then prints terminal/reason counts, accepted/hour over 1/6/24 hours, tokens by role,
stage durations, and consecutive-event latency proxies. Replayed results do not add token
spend or latency samples. Solver usage comes from recorded rollout steps; other roles follow
their scopes. The confirmation step reports apart, once: its solver slots as `role=confirmation` and its final
review as `role=final_review`, never also under `solver` or `solve_review`; a judge call during it is
`role=judge`. An interrupted solver attempt's partial usage is not reported (no journalled result holds it). Missing stage scopes (currently review) print `n/a`. Percentiles use nearest rank.
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
  As in live Forge, in this order: a clean revision placed outside its fallback's band is accepted at its new
  placement, tagged; a round with review findings is a defect round, not the band-only one, even out of band;
  and the band-only round is spent only by a round revised for its placement alone.
- **Budgets**: `budget.max_modify_rounds`, `budget.max_validation_retries`.
- **Adversary**: `adversary.roster` (brief names, each run as one adversary session; default
  `false_accept` and `false_reject`), `adversary.image`. The default user prompt renders `{brief}`, so each
  name needs a prompt `adversary.brief.<name>` (`prompts/adversary/brief/<name>.md`); a user prompt
  that renders only `{name}` (`Review strategy: {name}`) uses the roster name itself as the strategy
  and runs any name. The default briefs: `false_accept` looks for a submission
  the grader accepts but should not, `false_reject` for a correct submission the grader rejects, and
  `adversary` is the earlier single adversary's strategy line. An overlay may rewrite a brief's text;
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
  never that it validated the task. Each result carries a `continuation` (the search's sandbox owner,
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
  journal can reuse: an answer generated in full whose grading then failed is generated again today, so a grading
  outage can repeat a fully generated answer (a follow-up will checkpoint the generation). The new slots number past that round's
  indices under their own attempt, through the same solver batch, concurrency, heartbeat and
  `budget.max_missing_trials` allowance (a refused slot beyond it holds). The batch and its grade observers
  settle as validation's do, so an outage beside another hold is a manual hold, and only a pure
  decision-model outage parks for automatic relaunch. Controls and adversaries do not run again. The
  confirmation record is assessed as a validation round first, before the final reviewer is asked: an invalid
  task or a grader fault in the new trials is the author's to repair and an exhausted trial budget rejects, never a
  denominator entry, and a record that rejects or is incomplete is decided without the final review. Only
  valid scores and generation limits are placed: a generation limit is an unsolved trial, and any other
  slot without a score is named (`confirmation_trials`) and excluded. The final review
  (`solve_review.final.*` prompts, with its own nudge) uses the same reviewer, evidence index, citation
  checks and replay identity, is shown every earlier report with what became of each finding (the
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
  As in live Forge, the placement over every scored trial re-tags the acceptance in any band, and
  `band.accept_too_easy` does not refuse it; `note` findings are the acceptance's notes. A new `revise`
  finding (or one an earlier review gave whose latest response on the shipped round's history is a revision:
  a restore discards the revisions after the round it restores, and a later decline overrides) or a fault buys
  one revision
  outside the shared modification budget, validated as any round; the final review's notes go to the
  author beside it as optional. A finding repeated from an earlier report buys nothing and is kept once.
  That revision keeps no fallback: a clean revision is accepted at its placement whatever the band and
  confirmed again, its round findings are notes, and a fault or an author stop abandons the task
  (`Confirmation revision failed`, naming the unresolved finding). After it any `revise` finding is a note,
  and the acceptance ships every earlier note, optional note and new note the shipped trials still support,
  each once, in a version whose quotes occur in them (the confirmation record's `notes`).
  `accepted_with` is published only on a final acceptance. Missing confirmation evidence
  (`confirmation_trials`, `transcripts`, `solve_review`) is named once, never a hold.
- **Sandboxes**: `author.image`, `author.workdir`, `sandbox.memory_mb`, `grader.timeout`,
  `solver.*` timeouts (the turn and attempt timeouts are raised to cover `provider.unavailable_wait`).
- **Provider**: `provider.unavailable_wait` (a busy provider is waited out, with
  `provider.backoff_*`), `provider.stall_timeout`, `provider.max_connections`. The run refuses a
  `solver.turn_timeout` shorter than the window plus a stall, or too short for a first attempt to reach its
  first token inside the solver's generation bound (the turn less 60 s).
- **Tools**: `tools.output_chars` (at least 240); longer output is saved to `.tool-output/<digest>.txt` in
  the workspace and the model gets the head plus the path, or, for a session without a machine, is clipped.

Prompts live in `defaults/prompts/<role>/`; the author's system prompt renders the pinned
`TaskSpec` and control reference (`defaults/author-reference-v2.json`, with the decision-model grader kind from
`defaults/author-reference-v2-decision-model.json` inserted where a decision model is configured), and the adversary's user prompt renders its brief and the task without
file payloads (they are in the workdir).

## Models and providers

Every model call goes through one `ModelClient` (`llm/client.py`), the only place that retries.
It sends each attempt through a `Provider` and decides from the attempt's outcome. A solver trial is
charged as failed only on positive evidence that the model itself ran out; anything the fleet, the
relay, the transport or local admission may have caused is an outage, and no failed trial is recorded.

| Failure | Evidence checked | What happens |
|---|---|---|
| 400 context overflow (recognised wording or `context_length_exceeded`) | the rejection itself | `ContextExhausted`: a solver trial ends at the generation limit (counted, never solved); the author and adversary get `ContextExceeded` |
| solver generation past `solver.turn_timeout` less 60 s | the same stream still generating: a second generated payload at least `CONFIRMATION_INTERVAL` (1 s) after its first one past the bound, with no finish, abort, cut or stall in between; counted from each attempt's first token; no pool state is consulted | the trial ends at the generation limit |
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
