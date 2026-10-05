# [taskforge] Run validation trials and control replay on RolloutEngine

Stacks on PR 9623 (`rollout-engine`) through `taskforge/04-triage`. It uses `llm`, `ledger`,
`spec` and `sandbox` from the layers below.

`taskforge.validate.trials.run_trials` runs k trials of a `TaskSpec` through
`ShellboxRolloutEngine` with `llm.rollout_model.GlmRolloutModel`, and records one ledger span and
one rollout JSON per attempt. A task the machine factories refuse fails up front as
`MACHINE_UNSUPPORTED` with no machine started. Each trial is `Graded` or `Ungraded` with one typed
`Cause`, and `classify` is the only classifier: the task's own setup or healthcheck failure is
`TASK_SETUP` and is not retried, retryable causes back off and retry, and an agent timeout keeps
the engine's partial grade. `validate.controls.replay` replays fixed controls as scripted trials
and writes a workspace control through shell calls after setup, as an agent would.
`validate.evidence` marks an item incomplete with counts by cause. The `llm.rollout_model` tests
land here because they assemble tasks with `taskforge.spec`.

Unit tests on this layer: 212 passed, 25 skipped. Live on the GLM-5.3 interactive endpoint, 8
checks passed: multi-turn ShellSim rollouts kept the served token prefix on every turn, including
3 rollouts of 6 turns with up to 5,504 reasoning tokens replayed; math and ShellSim trials graded
3/3; every control met its expectation; forced start failures were classified and retried.

`scripts/cluster_rollout_probe.py` ran the same trials inside an Iris task on cw-rno2a with GLM
resolved in-task. A null-environment task graded 3/3. A docker task on the shipped
`IrisMachineFactory` failed 3/3 as `machine_start` (`'TaskStatus' object has no attribute
'error'`). With the readiness poll from `docs/upstream/shellbox/iris-machine.patch` applied in the
probe process it graded 3/3. A sandbox's environment then listed the cluster's object-store keys
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CW_KEY_ID`, `CW_KEY_SECRET`), names only, so a
model-controlled machine with network can read them. Logs and results are in the gitignored
`.evidence/cluster/`.

Out of scope: staged-task control replay, which needs a public RolloutEngine entry point that
grades a supplied state (#9763); Docker on a laptop; `NetworkPolicy.DENY` on Iris; and removing
the credentials from Iris sandboxes, which belongs in shellbox or Iris.
