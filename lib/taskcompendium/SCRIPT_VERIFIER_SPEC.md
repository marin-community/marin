# Script verifiers for TaskCompendium

Status: proposed. The current TaskCompendium slice supports registered Python
verifiers and direct-chat Harbor tasks. It does not package or run grader scripts.

## Purpose

A task may already have a grader script whose behavior cannot be expressed by a
built-in answer comparison. `script` should be one reusable verifier kind. A
`TaskSpec` selects that kind and pins the script, its private inputs, and the
runtime needed to execute it. Importing another source should not require a
new verifier kind solely to call that source's script.

Submission conventions still control how the agent is asked to deliver an
answer and how that answer is extracted. The script checks correctness. A task
can reuse the same script across conventions that extract the same semantic
answer if the script grades `answer`. A grader that inspects `raw_response`
may intentionally make delivery format part of the score.

## Private task contract

Add `script` to `VerifierKind` and a Pydantic-validated configuration selected
by `VerifierSpec`. The configuration contains:

| Field | Contract |
| --- | --- |
| `protocol_version` | Version of the input and result files described below. Reject unknown versions. |
| `entrypoint` | Relative path of an executable file in the private verifier resources. Its interpreter comes from a shebang or the runtime image. |
| `args` | Fixed argument vector. Execute without shell interpolation. |
| `runtime_image` | Immutable image digest containing the interpreter and grader dependencies. |
| `timeout_seconds` | Positive outer limit on the grader process. The runner enforces a separate maximum. |
| `resources` | Private files needed by the grader, each with a normalized relative path, executable bit, and embedded bytes or a URI plus SHA-256 digest. |

The entrypoint must name a pinned resource. Resource paths cannot be absolute,
contain `..`, collide, or escape their staging directory through symlinks.
Validate the configuration and resource references before exporting a Harbor
task. Verify fetched bytes against their digest before execution. The runtime
image and resource digests are part of the task's reproducible verifier
identity; a mutable tag or unpinned remote file is insufficient.

For example, the private part of a task could name `VerifierKind.SCRIPT` with
parameters equivalent to:

```json
{
  "protocol_version": 1,
  "entrypoint": "grade.py",
  "args": [],
  "runtime_image": "registry.example/grader@sha256:<digest>",
  "timeout_seconds": 60,
  "resources": [
    {"path": "grade.py", "uri": "s3://example/grade.py", "sha256": "<digest>", "executable": true},
    {"path": "reference.json", "uri": "s3://example/reference.json", "sha256": "<digest>", "executable": false}
  ]
}
```

This is a proposed shape, not an implemented JSON schema. A common resource
model can replace the inline `resources` list when TaskCompendium gains private
task resources; it must retain the same visibility and pinning rules.

## Execution contract

The Harbor verifier adapter performs these steps for one submission:

1. Apply the selected submission convention. If extraction fails, return
   `EXTRACTION_ERROR` without starting the grader.
2. Materialize the pinned verifier resources under `/verifier/resources` in an
   isolated verifier runtime. They are never placed in the agent's workspace.
3. After the agent's final action and before grading, capture the agent-visible
   workspace roots. Make an independent copy available at `/workspace` with
   paths rebased beneath that root. The script may modify this verifier-side
   copy while testing; those changes cannot affect the agent's workspace or
   another trial. Direct-chat tasks use an empty workspace.
4. Write `/verifier/input.json`, invoke the entrypoint with the fixed argument
   vector, and enforce the outer timeout. The working directory is `/workspace`.
5. Read `/verifier/result.json` and convert it to `GradeResult`.

The input file has one versioned shape:

```json
{
  "protocol_version": 1,
  "raw_response": "the agent's final response, or null",
  "answer": "the convention's extracted text answer, or null",
  "workspace": "/workspace",
  "resources": "/verifier/resources"
}
```

The runner also sets `TASKCOMPENDIUM_INPUT` and `TASKCOMPENDIUM_RESULT` to the
input and result paths. A file convention supplies a path within the workspace
snapshot as `answer`; a workspace-state task supplies null. The script reads
workspace state from `/workspace` and private references from
`/verifier/resources`. No `BaseEnvironment` Python object is serialized into
the input file. A provider-backed task needs a separate verifier-side snapshot
or read-only provider bridge before it can use `script`. Answer graders should
use `answer`; scripts that use `raw_response` can enforce intrinsic output
format requirements, which belong to the task's private correctness contract.

The result file contains a finite reward in `[0, 1]` and may contain a short
diagnostic string:

```json
{"reward": 0.0, "reason": "tests failed"}
```

A valid result means `GRADED`, including reward zero. A missing entrypoint,
missing or invalid result, out-of-range reward, or outer timeout means
`INFRA_ERROR` with no reward. A script may report a valid result after an
internal test command fails; the result file, not the process exit code,
determines the grade. The outer timeout overrides any result file written
before the process was killed. A nonzero exit without a valid result is an
infrastructure error. Keep bounded stdout and stderr as verifier-only
diagnostics; never add private paths, references, or test output to the
agent-visible response.

## Isolation and compatibility

Run imported scripts in the pinned isolated runtime, never in the host verifier
process. Disable network access by default. Mount private resources read-only
and give each run fresh input, output, and workspace directories. Bound process
time, memory, and log size. The outer timeout and runtime failures retain the
existing distinction between verifier failure and a graded wrong answer.

The verifier runtime is independent of the agent environment. A direct-chat
task can therefore use a script without granting the agent a shell. A task
that needs to inspect the final workspace must use a lowering that can capture
that state. The lowering must reject file and workspace-state submissions when
its environment cannot capture and rebase the final workspace. It must reject
symlinks or mount points that would expose paths outside the captured roots.
Stateful tool providers need an explicit snapshot or bridge; this proposal
does not infer one from `TaskRequirements`.

`tasktrove-verify` already has a `ScriptSpec` that runs a task script with a
timeout and reads a reward. Its grader conventions can be adapted behind this
protocol, but the TaskCompendium result file has one JSON channel and its own
failure mapping. Existing in-process verifiers remain useful for simple
checks. Loading a verifier class by Python import path can be considered when
a concrete verifier needs the live environment object and the process
interface cannot represent its inputs.

## Acceptance cases for the implementation PR

- One imported direct-chat task uses a private script and grades the same
  answer through plain and JSON submission conventions.
- One workspace task grades its final state from an isolated snapshot. Grader
  writes do not change the agent workspace.
- A wrong answer produces `GRADED` with reward `0`; missing resources,
  digest mismatch, malformed result, and outer timeout produce `INFRA_ERROR`.
- The agent cannot read the grader script, reference files, input file, or
  verifier logs through its environment.
- Repeated runs with the same task, submission, captured workspace bytes,
  runtime image, and resource digests select the same grader bytes and produce
  the same result for a deterministic script.
