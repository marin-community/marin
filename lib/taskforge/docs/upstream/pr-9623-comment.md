🤖 Notes from building a task-generation pipeline (Taskforge, `lib/taskforge`, stacked on this branch) against the 0.22 schema and `ShellboxRolloutEngine`. Two filed issues target code added here, and two staged-grading behaviors are worth a decision before merge.

**Issues against this PR's code**

- #9757: a task grading script that writes a verifyit `verdict.json` with `status: invalid_task` (or `infra_error`) is reported as `GradingFailure.MISSING_REWARD` when `FileReward` finds no reward file, so a task defect and an infrastructure failure are indistinguishable to the caller. The issue proposes `Outcome.INVALID_TASK` and a verdict-file reward source. A draft PR stacked on this branch is in preparation.
- #9763: a public entry point to grade a supplied transcript or workspace without model inference, for replaying reference solutions and negative controls. In the meantime the pipeline does this with a scripted model callable, so this is not blocking.

**Staged grading, `engine._run_stages` and `grading._combined_stage_grade`**

- With `StageRewardStrategy.MEAN`, a later stage that ends ungraded (for example `INFRA_ERROR` from a grader timeout) stops the loop, and the aggregate is the mean over the earlier graded stages only, with status `GRADED`. The reference notes that the mean includes only valid grades, so this is intended, but it means an infrastructure failure in stage 2 of 3 yields a training reward for an incomplete task, and nothing in `GradeResult.failure` marks the aggregate as partial. Should an ungraded stage make the aggregate ungraded, or at least set `failure` so error policies can mask it?
- With `StageRewardStrategy.FINAL` and `minimum_rewards`, a stage whose reward is below its minimum stops execution and the aggregate is that stage's own reward (0.3, say), not 0. For binary stage rewards this gives conjunctive success; for partial-credit stages it rewards a failed gate. Is that the intended reading of `minimum_rewards`?

**Iris machines and `environment.network`**

`EnvironmentSpec.network` defaults to `False`, which `_task_machine` maps to `NetworkPolicy.DENY`, and `shellbox.backends.iris.IrisMachineFactory.create` refuses `DENY` because an Iris job's network is not configurable. So every docker task with the default network setting fails at machine creation on the Iris backend. Is the intended fix a factory-declared capability (the gVisor job profile satisfies denial) that the engine consults, or an explicit per-task opt-in? A small patch for the factory side is being drafted and can be adjusted to whichever you prefer.
