# [taskforge] Generate task proposals from capability ideas

Stacks on PR 9623 (`rollout-engine`) through `taskforge/02-spec-sandbox`. It uses only
`taskforge.llm` from the layers below.

`taskforge.proposal.model` parses and renders the TaskProposal document: YAML front matter
validated by code, plus six required sections, and a digest of its canonical form.
`CapabilitySource.propose(idea, n)` plans `n` slots in one structured GLM call, allows at most
ceil(n/3) slots per environment and verification pairing, and writes the proposals concurrently.
It returns a `ProposalBatch` with the planning request and completions and one `SlotProposal` or
`SlotFailure` per slot, so a failed slot does not drop its siblings and the event log can record
every call. Planned null slots make no model call.

Unit tests on this layer: 128 passed, 16 skipped. Live on the GLM-5.3 interactive endpoint, two
ideas at n=10 parsed 20 of 20 proposals with one document repair and one plan repair, every call
finishing `stop`. GLM drops or mangles the front matter in about 5 of 100 live documents, so the
one-call document repair is required. The live test reads the capability catalog from
`TASKFORGE_CAPABILITY_CATALOG` because the catalog is not in the repository.

Out of scope: no live slot has come back null, so that path is unit-tested only. `RepoIdea` is a
placeholder until a repository-environment source exists. What a `SlotFailure` does to its idea is
for the loop layer to decide.
