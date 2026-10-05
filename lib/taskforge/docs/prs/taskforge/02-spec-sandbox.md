# [taskforge] Assemble TaskSpecs and select shellbox machine factories

Stacks on PR 9623 (`rollout-engine`) through `taskforge/01-foundation`.

`taskforge.spec.draft` assembles a TaskCompendium 0.22 `TaskSpec` from builder outputs:
environment, files, shell verifiers, verifyit answer verifiers (exact, numeric, mcq,
predicted_action) and stages. `assemble` validates every grader once and rejects a grader that does
not fit the answer type or environment, private grader content that the agent can also see, and
stage reward gates the grader cannot produce. `spec.controls` defines fixed controls (a transcript
or a workspace payload with an expectation) and checks a control set against its task. Taskforge
adds no spec model of its own; gaps go upstream (below).

`taskforge.sandbox.factories` builds the `EnvironmentKind -> MachineFactory` mapping that
`ShellboxRolloutEngine` takes: ShellSim always, local Docker when a daemon and Skopeo are present,
and `IrisMachineFactory` inside an Iris task. `task_refusals` refuses a task up front, with typed
reasons, when no factory supports its image source, network policy, execution user, resource
limits or GPUs. The Iris docker row describes the patched backend in
`docs/upstream/shellbox/iris-machine.patch`, because the shipped factory fails every create.
`sandbox.images` and `scripts/build_image_job.py` turn a `DockerBuild` context into a pushed
registry digest in two Iris jobs: kaniko builds without pushing, and a separate job pushes the
archive, so the registry credential reaches only the push job.

`docs/upstream/` holds the issue bodies filed on 2026-10-05 for gaps in TaskCompendium (#9757,
#9758), verifyit (#9759 to #9762) and RolloutEngine (#9763), one unfiled RolloutEngine issue, the
review comment posted on PR 9623, and the shellbox Iris patch with its write-up.

Unit tests on this layer: 111 passed, 15 skipped. Assembled shell-verifier and staged tasks run on
ShellSim through `ShellboxRolloutEngine` and grade right and wrong answers as expected. One test
calls the real shellbox factories and checks they refuse what the capability table says. The image
builder ran live on Iris (cw-us-east-02a): a build and push in 79 s, an adversarial Dockerfile that
searched the build for the credential and found nothing, and a failing Dockerfile reported as
`BuildFailed` without a push. That evidence is in the gitignored `.evidence/sandbox/`.

Out of scope: Docker and Iris execution of an assembled spec (the cluster run is in
`taskforge/05-validate`), and applying the shellbox patch, which belongs in a shellbox PR. The push
job's pod spec still carries the registry credential, readable by cluster operators.
