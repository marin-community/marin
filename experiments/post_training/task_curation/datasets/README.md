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
   in the declaration's `ships`. Read the bytes in the converter and ship them in
   the task's verifier resources. A sandboxed grader runs in the grader image:
   set `grader_image=GRADER` from [images/recipes.py](../images/recipes.py) and
   use `required_grader_environment(context)` as its environment.
4. Add a representative raw row to the family test's `ROWS` and add the module's
   `pipelines()` to [sources.py](../sources.py).

A rubric and controls can come later: without a rubric rows are kept
unreviewed, and without controls sandbox-graded rows stay out of `final/`.

## Families

| Module | Sources |
| --- | --- |
| [skyrl/math.py](skyrl/math.py) | MarinSkyRL math and numeric-answer sets, graded in process. |
| [skyrl/code.py](skyrl/code.py) | APPS, Eurus-2 code, verifiable coding problems and Gretel text-to-SQL, graded by `apps_grade.py`, `lcb_grade.py` and `sql_grade.py` in the grader image. |
| [skyrl/ifeval.py](skyrl/ifeval.py) | Nemotron IF and RLVR IFEval, graded by the pinned SkyRL IFEval scorer. |
| [skyrl/mcq.py](skyrl/mcq.py) | GPQA and OpenScience multiple choice. |
| [skyrl/preference.py](skyrl/preference.py) | HH-RLHF and KTO preference components; no runnable grader. |
| [nemotron_ultra/](nemotron_ultra/) | Nemotron RL Ultra blends: one `COMPONENTS` table over the mopd, rlvr1 and rlvr2 blends, with the scorer calls in `graders.py`. |
| [arc/](arc/), [reasoning_gym/](reasoning_gym/) | ARC-AGI and Reasoning Gym tasks from TaskTrove and the generated set, with the grader scripts the Ultra components also use. |
| [tasktrove/code.py](tasktrove/code.py) | TaskTrove competitive programming (Code Contests, Codeforces, Nemotron competitive coding, TACO), graded by stdio cases. |
| [tasktrove/python_tests.py](tasktrove/python_tests.py) | TaskTrove Python unit-test sources as one table, graded by pytest. |
| [tasktrove/nl2bash.py](tasktrove/nl2bash.py) | TaskTrove shell tasks, graded by an output checker. |
| [tasktrove/repositories.py](tasktrove/repositories.py) | TaskTrove SWE repositories; no agent image covers their per-task repositories. |
| [tasktrove/structured_outputs.py](tasktrove/structured_outputs.py), [tasktrove/instruction_following.py](tasktrove/instruction_following.py) | Structured-output and instruction-following tasks, graded in process. |
| [tasktrove/math.py](tasktrove/math.py) | TaskTrove math, graded by the source scorers through `math_grade.py`. |
| [tasktrove/judged.py](tasktrove/judged.py), [tasktrove/qa.py](tasktrove/qa.py) | Judged responses and open QA (verifyit judge; not admitted until a judge runs), and knowledge MCQA. |
| [tasktrove/calendar.py](tasktrove/calendar.py), [tasktrove/multichallenge.py](tasktrove/multichallenge.py), [tasktrove/puzzles.py](tasktrove/puzzles.py) | Calendar scheduling, multi-turn challenges and puzzles. |

## Sources not declared

Seven Atlas sources read `open-athena/task-trove`, the output of the retired
TaskTrove cleanup, and have no archive in `open-thoughts/TaskTrove`; they are not
declared: AweAI-Team__CalibForge, GAIR__OpenSWE__openswe_oss,
GAIR__OpenSWE__openswe_other, R2E-Gym__R2E-Gym-V1, SWE-Gym__SWE-Gym,
XiaomiMiMo__MiMo-V2.6-RL-oss__code and XiaomiMiMo__MiMo-V2.6-RL-oss__music.
