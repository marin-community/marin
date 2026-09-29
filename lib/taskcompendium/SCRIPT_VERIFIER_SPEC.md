# Script verifiers and private resources

Status: proposal. [#9216](https://github.com/marin-community/marin/pull/9216)
imports TaskTrove MCQ tasks into a generic `mcq_answer` verifier. Script grading
and verifier-only resources are not implemented.

## Verifier boundary

A `TaskSpec` names a grading method, not the dataset that supplied the task.
The implemented direct-answer kinds are `exact_answer` and `mcq_answer`;
[#9540](https://github.com/marin-community/marin/pull/9540) adds
`numeric_answer`. A proposed `script` kind would run a pinned grader in an
isolated verifier runtime. The source dataset and importer revision remain in
`TaskSpec.source`.

Importers translate source grading records into these source-independent
contracts. They reject tasks whose grading behavior cannot be represented.
The TaskTrove MCQ importer parses `tests/verifier.toml` and retains the
expected letter and option count, without storing the original TOML. The
current MCQ, exact, and numeric implementations call candidate scorers in
`tasktrove-verify` to preserve behavior during migration. A candidate scorer
compares an already-extracted answer with private grading parameters; it does
not parse the model's submission format or read an answer file. The package is
an implementation dependency, not a serialized verifier kind. Its scoring code
can move into TaskCompendium as the remaining modes are migrated.

## Existing executable graders

[`tasktrove-verify`](../tasktrove-verify/README.md) supports math, schema,
structured-output, instruction-following, Reasoning Gym, code-test, judge, and
`script` modes in addition to direct answers. Its file-based graders expect
private files under `/tests`, an agent workspace under `/app`, and often an
answer file at `/app/answer.txt`. Its `script` mode runs a task-supplied shell
or Python grader with a timeout. It reads a reward from `reward.json`,
`reward.txt`, or the final stdout line and writes a `verdict.json` status.

These are source behaviors for importers to inspect. A TaskCompendium
submission convention extracts a direct answer before invoking a candidate
scorer. It does not create a synthetic `Answer: X` line or answer file.
Executable graders instead need private resources and a verifier-side view of
the completed environment.

## Proposed script contract

The private `script` verifier configuration identifies an executable, fixed
arguments, a timeout, a pinned runtime image, private resource digests, and a
result protocol. The source importer translates its verifier description into
that configuration. It need not retain the source `verifier.toml` unless a
specific task needs it as an input artifact. The executable and every private
file are pinned by embedded bytes or URI plus SHA-256 digest. The runtime
image is pinned by digest and declares its interpreter and dependencies.

After the agent's final action, the Harbor environment selected during
lowering captures an independent snapshot of the relevant workspace for the
verifier runtime. The runtime stages the snapshot at `/app` and private files
at `/tests`, verifies digests, and applies an explicit network policy. It
exposes a normalized submission record at `/verifier/submission.json`: the
answer type, convention ID, and extracted answer when there is a direct
answer. File and workspace-state tasks supply their final artifacts through
`/app`. The grader writes one result at `/verifier/result.json` with a status
and, only for a scored result, a reward. The adapter can translate an existing
TaskTrove script's reward files into this result during migration.

Private files stay invisible to the agent. The grader may modify its workspace
copy without changing the agent's state or another trial. Imported scripts and
submitted programs never run in the host verifier process. State held by an
external service needs an explicit snapshot or read-only bridge before a script
can grade it.

The result protocol returns a reward only for a scored verdict. Invalid task
configuration and infrastructure failures carry no model reward.
TaskCompendium can map `scored` to `GRADED` and `infra_error` to
`INFRA_ERROR`; it needs a distinct `INVALID_TASK` outcome for failures
discovered at execution. Import-time validation should reject invalid
contracts before export when possible. The initial TaskTrove adapter must
preserve its current reward channels and timeout behavior in tests, then
record any intentional semantic changes in the generic contract.

## Implementation sequence

1. Keep `mcq_answer`, `exact_answer`, and `numeric_answer` source-independent.
   Candidate scorers may temporarily live in `tasktrove-verify`.
2. Add private resource packaging, a pinned verifier runtime, and a
   verifier-side workspace snapshot path.
3. Add the generic `script` verifier and import one TaskTrove script task
   into it. Check that private files, process output, and grader mutations
   cannot reach the agent or another trial.
4. Move the required scoring modes into TaskCompendium or another
   source-independent package, then remove the `tasktrove-verify` dependency.
