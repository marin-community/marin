# Reuse TaskTrove verifiers in TaskCompendium

Status: MCQ reuse is implemented in [#9216](https://github.com/marin-community/marin/pull/9216);
executable modes remain proposed. TaskCompendium parses the TaskTrove
`tests/verifier.toml`, stores its validated MCQ answer and option count, and
uses `tasktrove-verify` to score an extracted candidate. This proposal extends
that reuse to modes that need private files or an isolated verifier runtime.

## Existing contract

[`tasktrove-verify`](../tasktrove-verify/README.md) parses a flat
`tests/verifier.toml` into a typed mode specification. It already implements
MCQ, math, numeric, exact, schema, structured-output, instruction-following,
Reasoning Gym, code-test, judge, and `script` modes. Its `script` mode runs a
task-supplied shell or Python grader with a timeout and reads a reward from
`reward.json`, `reward.txt`, or the final stdout line. The CLI writes
`verdict.json` with `scored`, `invalid_task`, or `infra_error` status. Only a
scored verdict produces Harbor reward files.

These graders expect a TaskTrove task layout: private files under `/tests`, an
agent workspace under `/app`, and often an answer file at `/app/answer.txt`.
The current TaskCompendium slice instead extracts a plain or JSON final answer
from a direct-chat response. A direct call to `tasktrove_verify.grade()` would
therefore miss its input file; the MCQ mode also expects an `Answer: X` line.
The input adaptation must be explicit.

## TaskCompendium boundary

TaskCompendium owns source import, `TaskSpec`, submission conventions, answer
extraction, and target lowering. `tasktrove-verify` owns TaskTrove mode
validation and scoring. Use one `tasktrove` verifier kind in the TaskCompendium
registry. For MCQ, its private configuration stores the validated expected
letter and option count; the source URI, revision, and row identify the
original archive. Executable modes may require the full `verifier.toml`,
verifier-only resources, and a runtime image. A reproducible task bundle must
also pin the `tasktrove-verify` version used to grade it. Do not add a
TaskCompendium kind for each TaskTrove mode or dataset.

The importer validates the original TOML and retains the fields needed by its
adapter. It rejects a source task if its instructions, answer, private files,
or runtime cannot be represented faithfully. A `TaskSpec` can use a plain or JSON
submission convention only when the selected mode has a defined adaptation
from the convention's extracted answer to that mode's grader input.

For direct answers, factor a candidate-oriented scoring function out of each
supported `tasktrove-verify` mode. The existing file-based grader keeps its
source-specific extraction, then calls that function. The TaskCompendium
adapter applies its submission convention and calls the same function. For
MCQ, this means the source grader still parses its `Answer: X` line, while the
adapter passes the extracted letter. Both paths use one option-range and
correctness check. The first implementation needs only MCQ; other modes can
be added when their candidate representation is clear. Avoid reconstructing
`Answer: X` text or a temporary answer file merely to reuse the old parser.

TaskCompendium currently reports a malformed submission as
`EXTRACTION_ERROR` with no reward. TaskTrove's file-based MCQ grader scores a
missing `Answer:` line as zero. This difference is deliberate in the present
TaskCompendium slice and must be visible in tests and training admission; the
shared candidate scorer begins only after successful extraction. Decide on a
single malformed-submission policy before claiming full parity across the two
harnesses.

## Script and workspace modes

For an executable mode, including TaskTrove's existing `script` mode, retain
the original `verifier.toml` and private test files. Pin every resource by
embedded bytes or URI plus SHA-256 digest. Pin an isolated verifier image by
digest with the required interpreter and dependencies. Verify resource digests
before execution. Keep all `/tests` files invisible to the agent.

After the agent's final action, a compatible lowering makes an independent
snapshot of its workspace available at `/app` inside that verifier runtime.
The adapter stages private files at `/tests`, invokes the pinned
`tasktrove-verify` CLI, and reads its existing `verdict.json`. The grader may
modify the verifier-side workspace copy without changing the agent's state or
another trial. Imported scripts and submitted programs never run in the host
verifier process. A direct-chat lowering can use a verifier runtime without
granting the agent shell access; a workspace task requires a lowering that can
capture and rebase its final workspace. Provider-backed state needs an
explicit verifier-side snapshot or read-only bridge before these modes apply.

Preserve TaskTrove's script result channels and timeout behavior initially.
Its `script` mode currently treats a reported reward as scored even if the
script exits nonzero; no reported reward is an infrastructure error. Its
script timeout and out-of-range reward score zero. Any change to those rules
belongs in `tasktrove-verify` and should be tested there before TaskCompendium
adopts it.

Map `scored` to TaskCompendium `GRADED` with the reported reward, and
`infra_error` to `INFRA_ERROR` with no reward. Reject `invalid_task` before
export when validation can detect it. If it occurs at execution, record it as
an invalid task with no model reward; the current TaskCompendium `Outcome`
needs a distinct value for that case. Do not convert either failure to a
graded zero.

## Implementation sequence

1. #9216 extracts the MCQ candidate scorer in `tasktrove-verify`, uses it from
   both its file-based mode and the TaskCompendium adapter, and stores the
   validated MCQ fields under one `tasktrove` registry kind.
2. Add private resource packaging, pinned verifier runtimes, and a workspace
   snapshot path for executable modes. Exercise a TaskTrove `script` task and
   confirm that private files, process output, and grader mutations cannot
   reach the agent or another trial.

`tasktrove-verify` selects its known mode modules internally by import. It
does not accept an arbitrary Python class named in `TaskSpec`. Use its `script`
mode for an externally supplied grader executable. Add reflective class
loading only if a concrete grader needs the live Python environment object
and the script interface cannot express its inputs.
