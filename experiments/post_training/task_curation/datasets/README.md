# Dataset declarations

Every RL dataset is one `RlDataPipeline` declared here, and every declaring module
exports `pipelines()`, which [sources.py](../sources.py) collects into the catalog.
A declaration names the pinned source, the converter that builds each task with
its grader, the agent environment, and an optional review rubric and grader
controls. See the [experiment overview](../README.md) for the fields and the
artifact each declaration produces.

## Adding a dataset

1. Copy the closest declaration. [skyrl/math.py](skyrl/math.py) is the smallest
   complete example: a pinned `HfSource`, a converter built on
   `math_answer_task`, `ShellSim()`, a rubric string and `MATH_CONTROLS`.
2. Set the source pin and files, write the converter, and choose the
   environment. Use a helper from `taskcompendium.convert` when one fits; keep
   field mapping, prompt rewrites, rubrics and constants in the declaring module.
3. If the grader needs a script, put it next to the module as `<name>_grade.py`;
   vendor an upstream scorer under `<family>/scorers/` and list that directory
   in the declaration's `ships`. Build the grader with `grade_script`,
   `shipped_files` and `script_package` from
   `taskcompendium.convert.script_grader`, which ship the script, the scorer
   files and the row's `config.json` in the task's verifier resources. Set
   `grader=GRADER_PACKAGES` from [environments.py](environments.py), the
   packages every grade script in the catalog imports, and use
   `required_grader_environment(context)` as its environment. A script that
   needs more declares its own `Environment` (see
   [environment.py](../environment.py)), such as `COMPILER_GRADER_PACKAGES`,
   which adds a C++ toolchain. The packages live in [grader.in](grader.in) and
   its compiled lock [grader.lock](grader.lock); regenerate the lock after
   changing `grader.in` (see [images/](../images/README.md)).
4. Add a representative raw row to the family test's `ROWS` and add the module's
   `pipelines()` to [sources.py](../sources.py).

A rubric and controls can come later: without a rubric rows are kept
unreviewed, and without controls sandbox-graded rows other than judge-graded
ones stay out of `final/`.

## Families

| Module | Sources |
| --- | --- |
| [skyrl/math.py](skyrl/math.py) | MarinSkyRL math and numeric-answer sets, graded in process. |
| [skyrl/code.py](skyrl/code.py) | APPS, Eurus-2 code, verifiable coding problems and Gretel text-to-SQL, graded by `apps_grade.py`, `lcb_grade.py` and `sql_grade.py`. |
| [skyrl/ifeval.py](skyrl/ifeval.py) | Nemotron IF and RLVR IFEval, graded by `ifeval_grade.py` with the vendored SkyRL IFEval scorer; the only control is an empty reply, since the sources publish no passing replies. |
| [skyrl/mcq.py](skyrl/mcq.py) | GPQA and OpenScience multiple choice. |
| [skyrl/preference.py](skyrl/preference.py) | HH-RLHF and KTO preference components; no runnable grader. |
| [nemotron_ultra/components.py](nemotron_ultra/components.py) | Nemotron RL Ultra blends: one `COMPONENTS` table over the mopd, rlvr1 and rlvr2 blends. [graders.py](nemotron_ultra/graders.py) converts the rows; a graded component ships a `*_grade.py` script with the vendored NeMo Gym scorer it calls, except the math components, which the verifyit math comparator grades in process. |
| [arc/arc.py](arc/arc.py) | TaskTrove ARC-AGI tasks and the Ultra NVARC components' converter, graded by the vendored NVARC scorer through `arc_grade.py`. |
| [reasoning_gym/tasks.py](reasoning_gym/tasks.py) | TaskTrove Reasoning Gym tasks, and entries generated from the pinned `reasoning_gym` wheel, which `reasoning_gym_grade.py` regenerates with the grader packages before scoring. |
| [tasktrove/code.py](tasktrove/code.py) | TaskTrove competitive programming (Code Contests, Codeforces, Nemotron competitive coding, TACO), graded by stdio cases with `COMPILER_GRADER_PACKAGES`. |
| [tasktrove/python_tests.py](tasktrove/python_tests.py) | TaskTrove Python unit-test sources as one table, graded by pytest. |
| [tasktrove/nl2bash.py](tasktrove/nl2bash.py) | TaskTrove shell tasks, graded by an output checker. |
| [tasktrove/repositories.py](tasktrove/repositories.py) | TaskTrove SWE repositories; no agent image covers their per-task repositories. |
| [tasktrove/structured_outputs.py](tasktrove/structured_outputs.py), [tasktrove/instruction_following.py](tasktrove/instruction_following.py) | Structured-output and instruction-following tasks, graded in process. |
| [tasktrove/math.py](tasktrove/math.py) | TaskTrove math, graded by each archive's SymPy scorer and `test.sh`. |
| [tasktrove/judged.py](tasktrove/judged.py), [tasktrove/qa.py](tasktrove/qa.py) | Judged responses and open QA (verifyit judge, admitted without controls), and knowledge MCQA. |
| [tasktrove/calendar.py](tasktrove/calendar.py), [tasktrove/multichallenge.py](tasktrove/multichallenge.py), [tasktrove/puzzles.py](tasktrove/puzzles.py) | Calendar scheduling (the archive's checker), multi-turn challenges (verifyit judge) and puzzles. |

## Judge-graded sources

The judged, open-QA and MultiChallenge sources grade with verifyit's judge mode.
Verification cannot run their graders yet: the grader packages have no judge client
(`openai`), and grading machines deny network access, so a sandboxed judge cannot
reach an endpoint. Their verification stage is therefore empty: no task is sampled,
`verify/report.json` records `skipped` with reason `judge grader; no control path
yet`, and kept rows are admitted to `final/`. Choosing where the judge runs and how
it receives the endpoint and credentials is follow-up work.

## Sources not declared

Seven Atlas sources read `open-athena/task-trove`, the output of the retired
TaskTrove cleanup, and have no archive in `open-thoughts/TaskTrove`; they are not
declared: AweAI-Team__CalibForge, GAIR__OpenSWE__openswe_oss,
GAIR__OpenSWE__openswe_other, R2E-Gym__R2E-Gym-V1, SWE-Gym__SWE-Gym,
XiaomiMiMo__MiMo-V2.6-RL-oss__code and XiaomiMiMo__MiMo-V2.6-RL-oss__music.
