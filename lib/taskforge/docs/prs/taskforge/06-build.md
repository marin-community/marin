# [taskforge] Build TaskSpecs from proposals with authored programs

Stacks on PR 9623 (`rollout-engine`) through `taskforge/05-validate`. It builds on proposals
(`taskforge/03-proposal`), spec assembly and controls (`taskforge/02-spec-sandbox`), and its live
test replays controls with `taskforge.validate`.

`taskforge.build` runs builder programs made of memoized async steps. A step's key covers its
code, arguments, the data globals it reads, `SDK_VERSION`, the proposal and the model policy, so a
revised program reruns only the steps it changed. `build.author` makes one structured GLM call that
writes a program against the `Build` SDK. `build.template.standard` is the default step sequence
taken from the `capability_env_gen` builds: research, fixtures, environment, grader, instructions,
assembly and controls. `run_build` requires the verifier and controls to come from GRADER and
CONTROLS steps, runs `validate_controls`, and fails a build whose controls reuse a candidate the
program already graded with `try_grader`, other than a reference or empty answer.

Unit tests: `pytest tests/build` gives 23 passed, 1 skipped. Pytest's default `norecursedirs`
skips directories named `build`, so `pytest tests` does not collect them; run the directory
explicitly. Live on the GLM-5.3 interactive endpoint, the build test passed in 40.7 s. Two ACCEPT
proposals built to validated TaskSpecs: d27.reporting.close_measurement on its first authored
program, and d01.algebra.linear-transformations on its third, after the program's own ShellSim
probe failed twice. Changing only the GRADER step reran that step and hit the cache for the other
six.

Known failure: a rollout of the d01 draft through `GlmRolloutModel` raised `RolloutContractError`
(the served token prefix changed at turn 5, prefix 9,733 tokens), and a rerun reproduced it. The
fault is in `llm.rollout_model`, not in build.

`build.sdk` imports `rolloutengine.machines._task_machine` and `rolloutengine.grading._shell_grade`
until RolloutEngine exports a task machine and in-machine shell grading
(`docs/upstream/rolloutengine/public-task-machine.md`, not filed). Out of scope: the
author-to-build revision loop, which belongs to the review and loop layers, and Docker
environments, which no build has used.
