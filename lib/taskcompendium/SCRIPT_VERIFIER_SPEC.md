# Script verifiers and private resources

Status: proposal. Direct-answer verifier kinds are implemented in
[#9216](https://github.com/marin-community/marin/pull/9216) and
[#9540](https://github.com/marin-community/marin/pull/9540). Script grading
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
The serialized verifier holds normalized grading parameters, not a source
manifest or dataset-specific mode name. A candidate scorer compares an
already-extracted answer with those parameters; it does not parse the model's
submission format or read an answer file. Shared scorer code is an
implementation detail.

A submission convention extracts a direct answer before invoking its scorer.
Executable graders instead need private resources and a verifier-side view of
the completed environment.

## Proposed script contract

The private `script` verifier configuration identifies an executable, fixed
arguments, a timeout, a pinned runtime image, private resource digests, and a
result protocol. The source importer translates its verifier description into
that configuration. It need not retain a source grading manifest unless a
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
and, only for a scored result, a reward. An importer can translate another
grader's result files into this protocol during migration.

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
contracts before export when possible. Each imported executable grader needs
behavioral tests for its reward channels and timeout rules. Record any
intentional semantic changes in the generic contract.

## Implementation sequence

1. Keep `mcq_answer`, `exact_answer`, and `numeric_answer` source-independent.
2. Add private resource packaging, a pinned verifier runtime, and a
   verifier-side workspace snapshot path.
3. Add the generic `script` verifier and import one executable task into it.
   Check that private files, process output, and grader mutations
   cannot reach the agent or another trial.
4. Consolidate shared scoring code in a source-independent package as
   importers move onto the generic verifier kinds.
