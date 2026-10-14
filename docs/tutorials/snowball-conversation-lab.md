# Run a Snowball conversation lab

The conversation lab runs sixteen comparison sessions concurrently
and replaces each session when it finishes. A session runs every checkpoint in
the selected roster; the default roster has two candidates. GLM-5.3, hosted on the Ortet cluster,
writes realistic user requests and followups inspired by a small WildChat
sample. Every candidate receives the same opening request and initial files.
GLM then follows each candidate's conversation independently. Random scenarios alternate
between an isolated ShellSim workspace and ordinary chat. Queued topics take
priority and can select either workload. GLM separately reviews
the saved traces for strengths and concrete failures.

GLM uses English by default. It uses another language when a queued idea or task
requests or clearly calls for it, or when the WildChat inspiration example is not
in English. Followups retain the language of the opening user request. Exact
queued prompts and evidence quotes keep their original language.
Ortet constrains GLM's scenario, followup, and review output to their JSON schemas.
The chat-only scenario schema requires its `files` object to be empty.
The worker escapes angle brackets in JSON sent to GLM so quoted tool markers
do not activate Ortet's output parser. Decoding that JSON restores the exact
source text; candidate requests and saved candidate traces retain their original text.
The worker separately validates review evidence against the saved traces.
It accepts exact command text decoded from tool arguments. When GLM attaches an
exact quote to the wrong message, the worker corrects the link only if that quote
matches one message from the same candidate. The viewer records these corrections;
invented quotes, ambiguous matches, and links to another candidate remain rejected.
When a review fails validation, the worker offers GLM short excerpts from the
affected candidate's saved text. The repair schema keeps each exact quote paired
with its source message and candidate and requires exactly one assessment for
every candidate. GLM must recheck the claims and remove failures unsupported by
the saved conversation traces. The worker validates the repaired review against
those traces. The viewer shows the earlier validation error alongside the
accepted review.

For each followup, GLM sees only that candidate's public answers and tool results.
It responds to that conversation and can end each branch independently. The
reviewer sees all candidate branches under blinded labels and evaluates each against its
own user requests. Different followups can change task difficulty; aggregate
scores describe these exploratory conversations. The viewer labels older runs
that used shared followups.

GLM and the candidate assistants receive a ShellSim capability note for agentic
tasks. It recommends small Python scripts using the supported standard-library
modules and local text files. Basic shell commands and other simulated tools
have incomplete coverage. Python `sqlite3` and `threading`, the `sqlite3`
executable, Ruby, and pandas are unavailable. Simple NumPy operations work;
tasks should not assume full NumPy API coverage. Tools remain enabled throughout
each agentic turn; the runner does not impose a per-turn call cutoff.

The private [Marina viewer](https://applets.marina.oa.dev/a/25f504c6-c6fb-4e14-8401-07dd8e6fbfba/)
shows full checkpoint names and pinned revisions, paired answers, expandable
thinking, tool calls and results, and GLM reviews. Prose and thinking render as
Markdown; raw text remains available beside each message and in the trace export.
Thinking starts collapsed for both candidates. **Expand thinking** opens all
thinking blocks in the current view and becomes **Collapse thinking**. Individual
blocks have their own **Expand thinking** / **Collapse thinking** labels and
retain their state through live refreshes. The toolbar shows **Collapse thinking**
whenever any block is open, including when newly arriving blocks are closed.
Tool output and evidence quotes retain their literal text. Select a run to inspect its
immutable candidate roster. Breakage filters select the model identified by a GLM finding and
failure category. Each evidence quote links to a saved message. **Export trace**
downloads the conversation together with its run configuration.
Selecting a conversation updates its URL with the run and case. Copy that URL to
reopen the same trace and checkpoint roster, including after newer runs are added.

GLM judgments describe exploratory tasks. A matching quote proves that the
cited text exists; it does not independently establish the judge's factual
claims. Check the evidence before using an aggregate score as a model-selection
signal.

The **Workload** selector beside **Where each model bends** filters both
category score charts and **Breakages** to chat, agentic, or their
combined sample. Whole-run status totals remain global. Clicking a breakage
group also selects that workload in the conversation browser. The summary
workload is preserved in copied links and after reload; the conversation
browser's separate workload filter can be changed independently.

## Start a comparison

Run these commands from the Marin checkout after `uv run iris --cluster=marin login`.
The default candidates in `experiments/evaluation/snowball_probes/config.py` are:

| Candidate | Checkpoint | Pinned revision | GPUs |
| --- | --- | --- | --- |
| Snowball · Dr. Doom | `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` | `1dfdbb34b6bc72f7249605c0e2e88a854c9f343a` | 2 H100 |
| Qwen 3.6 · 35B A3B | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 2 H100 |

Choose unique serving and gateway job names for each active comparison. The
gateway resolves its Iris target cluster's private Ortet relay for each request; that cluster must
have a registered GLM-5.3 relay. Candidate servers can run on another peer with
available GPUs. The `cw-rno2a` relay was verified on October 10, 2026. Check
its current registration with `uv run iris --cluster=marin endpoints list
/muchanem/glm53-relay-rno2a`.

```bash
uv run iris --cluster=marin job run --job-name snowball-lab-serve \
  --target-cluster cw-rno2a --cpu 1 --memory 2GB --disk 5GB \
  --priority batch --no-wait --sync-package marin-core -- \
  python -m experiments.evaluation.snowball_probes.serve --cluster cw-rno2a

uv run iris --cluster=marin job run --job-name snowball-lab-glm \
  --target-cluster cw-rno2a --cpu 1 --memory 1GB --disk 2GB \
  --priority batch --no-wait --sync-package marin-core -- \
  python -m experiments.evaluation.snowball_probes.gateway --cluster cw-rno2a
```

Create a dotenv file containing `GLM_API_TOKEN`. Keep it outside the workspace
bundle. The launcher reads this file and passes its token to the remote worker, which sends it through a scoped
gateway link; the gateway sets Ortet's downstream authorization header. Obtain
the token through the Ortet service owners using the access instructions in the
private [Ortet team brief](https://github.com/Open-Athena/ortet-inference/blob/1bf336ea4620299e577dce0e7db68aadd40c49bc/notes/team_brief.md).

Use the existing viewer's UUID, or publish another private applet:

```bash
uv run marina validate infra/marina/applets/snowball_probes --build
uv run marina publish infra/marina/applets/snowball_probes --mode private --build --json
```

Run the driver on an Iris CPU worker so it can continue when a developer's
machine sleeps or disconnects. New drivers use an applet-issued session key for
one run. Your Marina login creates the run, then calls
`POST /runs/<run-id>/create-session-key` before submitting the job. The response
contains a random 256-bit key; the applet stores only its SHA-256 hash. Issuing
another key for that run immediately replaces the previous key.

The launcher passes the key through the explicit `SNOWBALL_SESSION_KEY` job
environment parameter. It is absent from command arguments, run configuration,
and saved conversation traces. Each API request supplies `X-Lab-Session-Key`
for application authorization and a renewable machine IAP token for admission
to Marina. The existing IAP access in
[#10050](https://github.com/marin-community/marin/pull/10050) remains necessary;
a session key alone cannot bypass IAP. No Cloud SQL worker login or roles are
required. The proposed direct-database grant in
[#10105](https://github.com/marin-community/marin/pull/10105) was closed without
applying its resources.

The worker's IAP condition admits only
`/a/25f504c6-c6fb-4e14-8401-07dd8e6fbfba/api/` on `applets.marina.oa.dev`.
Keep this API-only restriction: Marina's generic applet SQL route sits outside
the custom backend and does not apply its session-key checks. Static assets,
versioned APIs, and the generic SQL route are excluded from worker admission.

The configured Marin CPU pools use
`iris-worker@hai-gcp-models.iam.gserviceaccount.com`. If you select another pool,
check its service account in `lib/iris/config/marin.yaml` and verify its IAP access
before launching there.

A key can save results, recover state, and claim queued ideas only in its run.
It cannot create runs, issue or revoke keys, change ownership or checkpoints,
queue topics, or change pause, stop, concurrency, or token-cap controls. Reads
are scoped to the same run. Invalid or revoked keys return HTTP 401 even when
the request's IAP identity belongs to the owner. Keys remain valid until the
owner rotates or revokes them, including while a stopped worker flushes its
final traces. Job retries retain the same key.

The run owner can revoke a key through the authenticated applet API:

```python
from marina.client import marina_request

marina_request(
    "https://applets.marina.oa.dev",
    "POST",
    "/a/25f504c6-c6fb-4e14-8401-07dd8e6fbfba/api/runs/<run-id>/revoke-session-key",
)
```

Revocation blocks subsequent requests and causes the driver to retain its
local outbox and pause new chats. For a planned replacement, stop the old driver
while its key is still valid and wait for `stopped` with zero reserved tokens;
then revoke its key. If a key was revoked before uploads finished, preserve the
old task disk: its unuploaded journal needs explicit owner recovery before the
run can be settled. A replacement on another host can recover only uploaded
records. Start the replacement with a new run ID and `--previous-run` to carry
the settled token total. Driver configuration version 21 records this transport;
older run configurations remain immutable.

Replace the example user namespace, job IDs, and run ID with your launch values:

```bash
uv run python -m experiments.evaluation.snowball_probes.submit_worker \
  --user your-user --job-name snowball-lab-worker \
  --coordinator /your-user/snowball-lab-serve \
  --glm-gateway /your-user/snowball-lab-glm/ortet \
  --applet 25f504c6-c6fb-4e14-8401-07dd8e6fbfba \
  --run-id dr-doom-vs-qwen36-new-run \
  --glm-env /tmp/glm.env --concurrency 16 --token-cap 10000000000
```

The launcher submits to the `marin` controller. Its scheduler selects a
non-preemptible worker. The driver requests 4 CPUs, 4 GB memory, and 8 GB disk;
Iris retries up to ten failed attempts. The job renews scoped inference links
through Iris and obtains IAP tokens from the machine's service account. The
run-scoped key distinguishes application sessions even when jobs share that
account. The GLM token travels through an explicit Iris environment parameter
and is written to a mode-`0600` temporary file in the task. No user login refresh
credential or database password is copied into the job.

To replace a healthy existing driver, select **Stop run** in the viewer and
wait for its state to become `stopped` with zero reserved tokens. This flushes
its traces and terminates its serving coordinator and gateway; launch new
serving and gateway jobs before submitting the replacement driver. Add `--previous-run <stopped-run-id>` to carry its generated
token total into the new run. `--inspiration-cache <wildchat.json>` reuses an
existing small WildChat sample; otherwise the launcher fetches the pinned
sample. Both the sample and launch arguments travel with the job so retries
can start on another host.

Two current runs use the patched `grug` reasoning parser:

| Run | Checkpoint revision | CPU driver | Serving coordinator | GLM gateway |
| --- | --- | --- | --- | --- |
| `antidoom-patched-grug-vs-qwen36-20261013-session-v1` | Antidoom `88f4d8b8c2a6083b9d107836b2a3a028e2465a8c` | `/dlwh/snowball-probes-worker-20261013-session-v1` | `/dlwh/snowball-probes-serve-20261013-v1` | `/dlwh/snowball-probes-glm-20261013-v1/ortet` |
| `merge-patched-grug-vs-qwen36-20261013-session-v1` | Shared100 `81d4cd7d39ddb7499b1f85d7a1f39fc6658b2a01` | `/dlwh/snowball-probes-worker-20261013-session-v2` | `/dlwh/snowball-probes-serve-20261013-v3` | `/dlwh/snowball-probes-glm-20261013-v3/ortet` |

Antidoom is `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02`.
Shared100 is `open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09`;
its newer revision includes the patched Antidoom chat template. The replacement
reuses cached weight revision `462ad9af071055da39ef4bfea1d4ba2c9429a223` with
the latest tokenizer and template. Startup verifies that every runtime artifact
size and HF hash matches between revisions, allowing only README and chat-template
changes. The viewer records both revisions and the verification. Both compare
against `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0` and use sixteen
sessions each. Their rosters are `antidoom_patched_grug.json` and
`merge_patched_grug.json` in `experiments/evaluation/snowball_probes/`.

The run-specific GPU release is `marin-vllm-gpu-20261012-3a054c458ca8`, recorded
next to the rosters in `patched_vllm_release.json`. Its compiled wheel predates
[vLLM PR #94](https://github.com/marin-community/vllm/pull/94), so the coordinator
loads the Grug parser from source commit
`ac3b5ea5e8510c002f3d349ea5c0f8b3f20e74ac` through vLLM's parser-plugin interface.
That commit changes only the parser among runtime package files relative to the
wheel's source. The run records both source revisions and the installed plugin
hash in the viewer checkpoint cards and exported run configuration.
The repository-wide GPU release pin remains unchanged.

Supply the roster and release to the serving entrypoint:

```bash
python -m experiments.evaluation.snowball_probes.serve --cluster cw-rno2a \
  --candidates experiments/evaluation/snowball_probes/antidoom_patched_grug.json \
  --vllm-release experiments/evaluation/snowball_probes/patched_vllm_release.json
```

The serving coordinators and gateways run on `cw-rno2a`. Each run queues the same
thirty [focused topic families](../reports/snowball-focused-probes-2026-10-11.md#results-by-workload),
with five comparisons per topic. Chat-only ideas explicitly tell GLM that the
candidates have no execution tools. The previous laptop systemd service is disabled.
Inspect a driver with:

```bash
uv run iris --cluster=marin job describe /dlwh/snowball-probes-worker-20261013-session-v1
uv run iris --cluster=marin task logs /dlwh/snowball-probes-worker-20261013-session-v1/0 --tail --max-lines 50
```

## Pause, stop, and recover

The viewer's **Pause** control stops new rounds and conversations after in-flight requests finish.
It retains the candidate servers. **Resume** continues the same run.
**Stop run** cancels in-flight requests, saves their partial traces, and
terminates that run's serving coordinator and GLM gateway. Reaching the token
cap performs the same cleanup. A stopped run is terminal; start a new comparison
with new job names and a new run ID. Each remote driver keeps its local journal
in `/tmp/snowball-probes-state` inside its task.

The worker checks endpoint availability every 15 seconds. When a serving task
is preempted, the viewer shows **waiting for servers** and the worker leaves
queued ideas unclaimed until all candidates are available. Serving-job
coordinates may change during recovery; checkpoint identities and evaluation
settings remain pinned to the run.

The lab serving entrypoint allows three hours for startup through
`--startup-timeout 10800`; this covers checkpoint mirroring before vLLM starts.
The old merged run exceeded its 40-minute readiness deadline while mirroring
weights, which cancelled the server and exhausted driver retries. Use the
replacement run in the table above; the old run retains its failure explanation
and queued topics.

A running driver can also show **waiting for servers** during initial model
staging. An uncached checkpoint is copied from Hugging Face into the serving
cache before its inference server starts. With desired state `running`, the
driver starts conversations automatically when all candidate endpoints become
ready; no **Resume** action is needed. Inspect the candidate task logs to
distinguish staging from a failed serving job.

List the coordinator's child jobs, choose an `inference-*` job, and replace
`<candidate-job-id>` below with that job's full ID. Its first task is `/0`:

```bash
uv run iris --cluster=marin job list --prefix /dlwh/snowball-probes-serve-20261013-v1
uv run iris --cluster=marin task logs <candidate-job-id>/0 --tail
```

The cap counts generated completion tokens from every candidate and from GLM's
scenario, followup, review, and review-repair calls. Completion counts include
thinking tokens. Thinking counts are displayed separately when the server
reports them and are not added twice. Missing usage and interrupted requests
are charged at their reserved output maximum. Reservations prevent concurrent
requests from spending the same remaining budget.

The worker journals compressed raw stream chunks in `chunks/<message-id>.jsonl.gz`, stores compressed current traces
and the token ledger in `state.sqlite`, and queues uploads in a durable outbox.
The applet API stores compressed trace payloads in its existing Postgres schema;
the viewer reads those same tables. An
upload failure retains the local data and pauses new conversations until
uploads recover. A fresh heartbeat with zero active chats and **Trace upload failed**
can therefore mean an upload backlog. Inspect the worker's exception before
restarting it, using `uv run iris --cluster=marin task logs <worker-job-id>/0 --tail`.
Evidence quotes preserve NUL characters through storage and export. Once
uploads recover, the same worker resumes automatically.
A worker restart charges abandoned reservations conservatively
and marks unfinished conversations as interrupted. If a cluster retry loses
its disk, it restores the token total and unfinished chats through the applet API, charging
the last saved reservations conservatively. Recovery fetches only streaming
messages through the run-scoped case-read endpoint; completed
messages remain in Postgres. This bounds recovery responses even when a
conversation has a large saved history. It starts fresh conversations using
the remaining budget. See the [large-history recovery incident](https://marina.oa.dev/echo/wiki/707)
for the observed failure and response. Raw chunk journals survive only while that task disk
survives; uploaded thinking, answers, and tool traces remain in Marina.

Current Grug candidates use the patched server `grug` reasoning parser with
Hermes tool-call parsing. The parser scopes its initial phase to the current
assistant response, including responses after tool results. It holds an unmarked prefix until a closing thinking marker,
a tool call, or the end of the response. A prefix before a closing marker becomes
reasoning. A prefix before a tool call becomes reasoning and the call reaches
Hermes; a marker-free call has no reasoning prefix. Marker-free prose becomes
public content at response completion. An unclosed explicit thinking block
remains reasoning.
The worker saves API reasoning separately and retains it as `reasoning_content`
in Grug candidate history, which the patched chat template serializes with
thinking markers before public content and tool calls. When reasoning is absent,
the template omits that block. GLM's user role receives only public answers and tool results;
its separate reviewer receives saved thinking.

Thinking counts use `usage.completion_tokens_details.reasoning_tokens` when
available. Missing Grug counts
are estimated with the candidate server's tokenizer and labeled accordingly.
The run's token ledger charges completion usage, which includes all generated
tokens; the separate thinking count adds no second charge.
Both models' thinking starts collapsed and can be expanded or collapsed together.

Historical `grug_optional` runs retain their original traces and show a warning
that their reasoning parser could prevent native tool calls from executing.
The later native-Hermes runs preserve original API text in candidate histories
and separate explicit markers in the worker. Their earliest configuration,
before version 18, omitted `skip_special_tokens=false` on requests without tools,
so detokenization removed thinking markers and joined reasoning to the answer.
Their chat scores retain that confound. Native-Hermes configuration 18 preserves
markers; **Native API content** exposes its raw assistant text. The current
server-parser runs use API reasoning fields instead.

Canceling only the driver job is a maintenance action: it saves partial work
and leaves the serving jobs available. Use the viewer's **Stop run** control
for complete teardown. If the driver cannot run, cancel its two owned serving
jobs manually. The driver runs independently of the machine that submitted it.
For local debugging, the `worker` module also accepts the original command-line
arguments; that local process requires a machine that stays awake.

Every inference stream has a 20-minute total deadline, including waiting for
headers and receiving its final chunk. HTTP keepalives do not extend that
deadline. A timed-out request preserves its partial thinking and answer, charges
the reservation conservatively, and fails the pair without a GLM score. The
worker then frees the session for another conversation. Override this deadline
with `--inference-timeout <seconds>` when submitting a driver; it is recorded in
the run configuration. Control requests use a 60-second transport timeout.

```bash
uv run iris --cluster=marin job cancel --exact /your-user/snowball-lab-serve
uv run iris --cluster=marin job cancel --exact /your-user/snowball-lab-glm
```

Use **Task status → Completed** to show only conversations with a saved,
validated review. **Generating**, **Reviewing**, **Failed**, **Review failed**,
and **Interrupted** filter the other lifecycle states. The filter is applied
before pagination and is preserved in copied links as `state=complete`.

## Queue a topic or prompt

Select the active run in the viewer, then use **What should GLM try next?**.
For example, enter “Check ability to follow JSON schemas: nested objects,
strict field types, missing fields, and changes across turns.” Choose
**Topic · GLM invents chats**, a workload, and the number of conversations,
then click **Queue for this run**. GLM creates distinct tasks around that
focus and picks the relevant task category. WildChat remains loose inspiration.

Choose **Exact opening prompt** to give both candidates your text verbatim
as their first user message. GLM still prepares the task, writes subsequent
user turns, and reviews the traces. **Mix of chat + agentic** uses the regular
alternating workload schedule; choose **Chat** or **Agentic · ShellSim** to
require one mode. Each conversation runs against every checkpoint in the
selected run and consumes the same shared token budget.

Queued work takes priority when a session becomes free; it does not interrupt
an existing conversation. The queue shows how many chats started, received a
review, or failed. **Cancel remaining** cancels unclaimed chats while allowing
started chats to finish. **View chats** filters the viewer to an idea's saved
conversations. The topic or exact prompt appears above each paired trace and
in the exported provenance. Queues persist in Marina across worker restarts;
claims are atomic and retrying the same case cannot consume another chat.

## Add another checkpoint

Export the candidate list, then edit or append entries with a unique `key`, a
clear `label`, the checkpoint repository, and its exact revision:

```bash
uv run python - <<'PY'
import json
from dataclasses import asdict
from pathlib import Path
from experiments.evaluation.snowball_probes.config import CANDIDATES
Path('experiments/evaluation/snowball_probes/candidates.json').write_text(
    json.dumps([asdict(candidate) for candidate in CANDIDATES], indent=2)
)
PY
```

The other fields specify GPU count, tensor and data parallelism, reasoning
parser, and tool-call parser. Keep at least two candidates. Submit the edited
file inside the checkout's workspace bundle with a new coordinator name:

```bash
uv run iris --cluster=marin job run --job-name snowball-lab-serve-next \
  --target-cluster cw-rno2a --cpu 1 --memory 2GB --disk 5GB \
  --priority batch --no-wait --sync-package marin-core -- \
  python -m experiments.evaluation.snowball_probes.serve --cluster cw-rno2a \
  --candidates experiments/evaluation/snowball_probes/candidates.json
```

The worker reads their metadata from the coordinator, and the viewer adds their
checkpoint cards and comparison selectors automatically. The first two
candidates appear in the aggregate score chart; breakage groups include the
whole roster. Existing run configurations cannot be overwritten with a new
checkpoint or revision.

## Opening checks and judgments

GLM declares up to three deterministic checks before either candidate answers
and before their follow-ups diverge. Both candidates receive the same opening
request. Checks can validate a JSON schema, compare exact public text, or run a
read-only command against an agentic artifact. Shell checks run in an isolated
copy of the workspace, so they cannot change the candidate's files or tool budget.
The checks run after the first user round's tool calls and final answer,
before GLM generates a follow-up. Verifier commands run separately from
candidate tool calls. GLM is instructed to make these commands read-only; the
workspace copy protects candidate files even if a generated command writes.

The viewer separates **Verified opening task · by category** from
**GLM-rated conversation · by category**. An opening task passes when all its
checks pass. Only pairs with conclusive results for both candidates enter the
verified chart; tasks without checks and inconclusive pairs remain visible but
unscored. Conclusive checks pass or fail: malformed JSON, schema violations,
text mismatches, and nonzero checker exits fail. Simulator errors, unsupported
operations, and terminal harness limits make a check inconclusive. The category
is the task category GLM assigns when creating the scenario. Inspect check definitions under **Task & workspace** and results under
**Strengths & breakages**.

GLM authors these checks, so inspect their coverage and expected answers before
using them as benchmark evidence. Its separate conversation ratings cover
adaptive follow-ups that can differ in difficulty between candidates. They are
qualitative judgments rather than the deterministic opening-task pass rate.

## Conversation limits

Current Grug candidates use the patched `grug` reasoning parser and Hermes with
thinking enabled. The [parser fix](https://github.com/marin-community/vllm/pull/94)
handles historical thinking markers and tool calls that precede a closing
thinking marker. Earlier `grug_optional` runs could prevent those calls from
executing; their agentic scores should not be compared as clean checkpoint pass
rates. Some detected unparsed calls are marked as infrastructure failures,
with the original GLM judgment retained.

Each conversation has at most four user rounds. Candidate rounds have a 16,384
completion-token budget, including tool-call generations and thinking. Tools
remain enabled on every agentic request. GLM
calls have an 8,192-token output budget, including thinking. ShellSim gives each candidate an
independent virtual filesystem with 64 MiB memory and 16 MiB disk. Each shell
tool request gets 500 million CPU fuel units and a combined 2 MiB stdout/stderr
budget, including workspace reconstruction and the new command;
stdout and stderr are each capped at 4000 characters per tool response, with
truncation recorded in the trace. Native input-token counts bound each request
to the model's context window. Each candidate's workspace supports 32 shell tool
invocations and 128 KiB of command input history across the conversation;
a tool invocation can contain multiple shell commands, and outputs do not count
toward this history bound. Exhausting the token,
context, or workspace history budget produces an explicit harness limit event;
context and workspace limits end that candidate's branch. GLM may ask a candidate
to continue after a turn exhausts its completion-token budget; that followup uses
one of the four user rounds. When a terminal context, CPU, or workspace limit prevents judging the pair,
the worker excludes both candidates from aggregate scores. Unsupported ShellSim
operations also exclude the pair, with the original review retained. A single
turn exhausting its completion-token budget can still receive a follow-up. Historical traces
from runs that disabled tools at a per-turn cutoff carry a viewer notice that
literal tool-call text in those requests was not executed.
Each request creates a fresh simulator and replays its command history before
executing the next command. Its CPU budget covers that reconstruction and the
new command together, rather than resetting between replayed commands.
Fuel counts virtual-machine instructions, rather than wall-clock seconds.
`--shellsim-cpu-limit` configures it for the run, and the viewer shows the recorded
cap. Increase it when valid tasks need more simulated execution; no conversion
from fuel to host CPU time is assumed. CPU exhaustion immediately fails that candidate's chat: the runner saves
the tool result and a terminal limit event, then sends no more commands or GLM
followups for that candidate. The other candidate continues under the same
limits until its own chat ends. The pair is then marked as an error and receives
no GLM review or scores.
ShellSim has no host filesystem or network access. Unsupported simulator operations,
malformed calls, token truncation, transport failures, and review-validation
errors remain visible in the traces. The reviewer is instructed to separate
infrastructure and simulator failures from model failures.

Grug continuations retain API reasoning, public content, and tool results. Qwen continuations
use its public API content and tool results. GLM's user role sees only public conversations;
its separate reviewer receives the thinking traces. Reviews must cite exact
text from the affected candidate's messages. One failed review-validation
attempt can be repaired by GLM; a second failure is retained as `review_error`.
