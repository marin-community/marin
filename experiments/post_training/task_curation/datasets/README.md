# Dataset declarations

Every RL dataset is one `RlDataSource` declared here, and every declaring module
exports `sources()`, which [sources.py](../sources.py) collects into the catalog.
Each source combines source information, its typed `config` and one optional
pipeline callable. The callable owns ingestion and stages. The reusable
`process_rows` callable accepts `CurationRecipe` for pinned row conversion,
optional review and grader controls. See the [reference](../../../../docs/references/task-curation.md#declarations)
for declaration fields and artifact outputs.

## Adding a dataset

1. Add an `RlDataSource` to the closest family module for runnable sources; for
   inventory-only entries use [unconverted.py](unconverted.py) when the conversion is
   unfinished or excluded, or [campaign_releases.py](campaign_releases.py) when the
   entry records an externally published campaign release whose admission lives in the
   release itself and must not be re-converted.
   Set `SourceInfo`'s stable `id`, display `title`, `origin`, family and tags.
   Leave `count=None` unless the pinned input population has been counted.
   Keep an existing ID when updating a source so its Atlas reviews remain linked.
2. For a runnable source, supply `config` and a callable accepting the source
   and `PipelineOptions`. To use the common row processor, set
   `config=CurationRecipe(...)` and `pipeline=process_rows`;
   [skyrl/math.py](skyrl/math.py) is an example. Reuse
   `taskcompendium.convert` helpers where they fit. Inventory-only sources use
   `info.dataset=SourceReference(...)` and no pipeline. Use the `excluded` tag
   when the entry is deprecated or excluded from use.
3. If adding a module, import it and include its `sources()` results in
   [all_sources()](../sources.py). Add a behavioral conversion test in the family
   test module, and run it with [test_catalog.py](../tests/test_catalog.py).
4. Run a SAMPLE or FULL campaign to publish the updated inventory. Any campaign
   includes all source declarations; select the new source to also curate and
   assess it. Inspect its records through `finestore.rl.curation.CurationCatalog`.
   Commit the Python declarations. Update the dashboard with the
   [Atlas loader](../../../../docs/references/rl-data-atlas.md#updating-atlas).

See [offline parquet counts](../README.md#declaring-a-dataset) for count regeneration.

## Script graders

If the grader needs a script, put it next to the module as `<name>_grade.py`;
vendor an upstream scorer under `<family>/scorers/`. Build the grader with `grade_script`,
`shipped_files` and `script_package` from
`taskcompendium.convert.script_grader`, which ship the script, the scorer
files and the row's `config.json` in the task's verifier resources. Set
`grader=GRADER_PACKAGES` from [environments.py](environments.py), the
packages every grade script in the catalog imports, and use
`required_grader_environment(context)` as its environment. A script that
needs more declares its own `Environment` (see
[environment.py](../environment.py)), such as `COMPILER_GRADER_PACKAGES`,
which adds a C++ toolchain. The pinned requirements live in
[environments.py](environments.py), grouped by the scorers that need them; the
build compiles them to a hash lock. For a new buildable environment, follow the
[registration instruction](../images/README.md#building).

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
| [structure_closure.py](structure_closure.py) | Snowball v0.1 structural train and development-validation splits. Assistant text prefills use the serving model's tokenizer; verifyit grades the first completed outer region in process. |
| [thoughtcrime.py](thoughtcrime.py) | Evaluation prefixes mined from Marin pi trajectories, one per benchmark task, before a reply that closed its reasoning span twice or opened a second span; one-turn tool-calling tasks that verifyit's `reasoning_markup` mode grades in process. |
| [tasktrove/code.py](tasktrove/code.py) | TaskTrove competitive programming (Code Contests, Codeforces, Nemotron competitive coding, TACO), graded by stdio cases with `COMPILER_GRADER_PACKAGES`. |
| [tasktrove/python_tests.py](tasktrove/python_tests.py) | TaskTrove Python unit-test sources as one table, graded by pytest. |
| [tasktrove/nl2bash.py](tasktrove/nl2bash.py) | TaskTrove shell tasks, graded by an output checker. |
| [tasktrove/repositories.py](tasktrove/repositories.py) | TaskTrove SWE repositories; no agent image covers their per-task repositories. |
| [tasktrove/structured_outputs.py](tasktrove/structured_outputs.py), [tasktrove/instruction_following.py](tasktrove/instruction_following.py) | Structured format and grounding checks; instruction-following constraints. |
| [tasktrove/math.py](tasktrove/math.py) | TaskTrove math, graded by verifyit's `math` mode in the grader sandbox; source scorer parity is not guaranteed. |
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

## Native Harbor inspection releases

`native_harbor.py` preserves original Harbor task files, their permission bits and timestamps,
including separately shipped solution archives. The TaskSpec keeps all execution and verifier
files private and delegates grading to the native Harbor Trial. The original `task.toml` retains
its resource shape, network policy, setup timeout and grading boundary. This path does not lower
workspace state into a file-capture contract or substitute an actor environment.

`native_quick.py` runs the full-population QUICK pipeline over a named entry in
`retired-inventory-only.json`, publishes a private `open-athena` TaskCompendium release, and
reads the upload back to verify file hashes. For example, inside a production-priority CPU Iris
job with a three-day timeout:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python -m \
  experiments.post_training.task_curation.native_quick \
  --source 'TaskTrove:DCAgent__exp_rpt_nemotron-cpp' \
  --root /tmp/native-quick-UNIQUE_VERSION \
  --evidence-root s3://marin-us-east-02a/marin/rl-data/atlas-disposition/UNIQUE_VERSION/results
```

Choose a fresh output version. Hub inputs must first be acquired with `harbor_hub.py`; pass the
completed acquisition root with `--hub-input`. The reader uses only its frozen selected task
references. Archive links that inline TaskCompendium files cannot represent, missing environments,
and grading-boundary mismatches remain mechanical exclusions. Quality warnings never exclude a
package from this QUICK conversion.

`native_releases.py` registers executed private releases from their ledger records. Its callable
imports them through QUICK, verifies every serialized TaskSpec hash, and writes a normalized view
without a `final/` view. These inspection releases require separate SAMPLE/FULL qualification,
full-population oracle filtering when solutions exist, and an exact-pin lexical-decontamination
receipt before admission. The backend capability pass supplies actor-only Daytona measurements,
fingerprints, examples and decontamination evidence.

Entries that cannot preserve an executable contract remain retired with a per-entry reason in the
ledger. Their old artifacts and review history stay archived.
