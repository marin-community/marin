# [taskforge] Add the package skeleton, GLM client, agent loop and ledger

Stacks on PR 9623 (`rollout-engine`). This is the bottom of the Taskforge stack: every later
layer imports the packages added here.

Add `lib/taskforge` as a standalone uv project with its own `pyproject.toml`, `uv.lock` and
`.venv`, and path dependencies on `taskcompendium`, `rolloutengine`, `shellbox`, `verifyit`,
`rigging`, `finelog` and `iris`. It is not a root workspace member, so it builds on TaskCompendium
0.22 and RolloutEngine, which exist only on PR 9623. Taskforge rewrites the
`experiments/post_training/capability_env_gen` task-generation pipeline as a library. `DESIGN.md`
records the decisions it follows: tasks are upstream TaskCompendium `TaskSpec`s, RolloutEngine
executes them, and task-specific grading lives in each task's own shell verifier scripts.
`STATUS.md` is the running account of what works and the evidence for it.

`taskforge.llm` is the only GLM-5.3 transport. `GlmClient` streams over httpx with 512 pooled
connections, retries 429 and 5xx responses, holds while the router has no capacity, applies stall
timeouts, and continues a reply cut at `max_tokens`. `complete_structured` forces one strict tool
call and allows one repair, and `CallStore` caches calls by request hash. `llm.endpoint` resolves
the GLM relay inside an Iris task. `llm.rollout_model` adapts the client to RolloutEngine's model
contract using the token ids the server returns. `llm.agent` is the Taskforge-owned agent loop
chosen in `docs/agent_loop_decision.md` (four candidates were run live; the reports are in
`docs/agent_loop/`): shell tools run through a shellbox `Machine`, and `llm.web` adds Parallel
search and extract. `taskforge.ledger` records timed spans to per-item JSONL files, and to Finelog
on Iris; `scripts/ledger_summary.py` summarizes them. `review`, `loop` and `queue` are empty
packages here.

Unit tests on this layer alone: 60 passed, 15 skipped (live tests skip without the endpoint
environment). Live runs on the GLM-5.3 interactive endpoint passed the client checks (a
131,072-token completion, a 250k-token context overflow measured and retried, continuation on
length, a structured call and its repair) and the agent checks (a package written and tested in
ShellSim, a fix loop on a seeded failing package, Parallel web search, recovery from an injected
malformed tool call, 20 concurrent agents, a tool call cut by `max_tokens`). The raw requests and
responses are in `lib/taskforge/.evidence/`, which is gitignored; `STATUS.md` summarizes each run.
The Finelog ledger was checked against finelog's embedded server, not the cluster.

Known gaps: text continuation on length is lossy at the splice (dropped spaces and punctuation)
and a reasoning cut can change the answer; the tests check only that the reply continued.
`llm.agent` imports `jsonschema`, which arrives only through `verifyit[schema]`. The agent copies
RolloutEngine's shell tool definition until `rolloutengine` exports it.

`httpx` is bounded below 1 because `prerelease = "allow"` otherwise resolves a 1.0 dev release
without `AsyncClient`. The `llm.rollout_model` tests need `taskforge.spec` and land in
`taskforge/05-validate`.
