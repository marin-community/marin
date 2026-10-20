# Skill2Env conversion

The existing `Harbor Hub:skill2env` catalog entry has a conversion recipe for
network-disabled tasks whose final Docker working directory is within `/workspace`, native programmatic tests,
and a reference solution. Acquisition resolves the pinned release through Harbor
and verifies each downloaded task's content digest using the pinned client in
`config/external/harbor`. The resolved task list is retained in the export
manifest. Acquisition does not edit upstream tasks.

Acquire a bounded selection for QUICK conversion; Docker and container-registry access are not required:

```bash
uv run --isolated --project config/external/harbor --frozen python \
  -m experiments.post_training.task_curation.harbor_hub \
  --release skill2env/skill2env@sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c \
  --task skill2env/task_api-design-reviewer_jdkbai64@sha256:acfc393449fdad22ae5a2da63d16d27605c02a983059fac0c5dd49384bf6c260 \
  --output /tmp/skill2env-input
```

Repeat `--task` for more tasks; `--all` selects the release's complete population.
Each native task archive is cached at `tasks/<task-reference-hash>.tar.gz`, using a SHA-256
hash of the full immutable task reference. Its pipeline identity stays the same
across selections, including when differently named tasks share content.
Earlier build-time working directories are retained; actor and verifier execution
use the final literal absolute working directory.
Conversion retains the native `environment/` files as an
unresolved `DockerBuildContext`. The verifier context adds workspace clearing
so submitted deletions are preserved. Private tests and reference solutions
remain separate from both build contexts. The runtime rejects unresolved
contexts before execution.

Convert the prepared selection with the existing driver:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver --mode quick --run \
  --source skill2env --input-root /tmp/skill2env-input \
  --output-root /tmp/skill2env-converted
```

QUICK writes normalized TaskSpecs for inspection. It does not build images,
run graders or admit training tasks. Retained build recipes do not freeze the
results of package downloads during a later build.

Building and pinning these images, configuring native resource limits, and
validating grading in those containers are follow-up work. The converted tasks
cannot execute in SAMPLE/FULL while their build contexts remain unresolved.

Without an input override, the normal source-download step acquires and caches
all 7,496 packages before curation sampling. Use the bounded acquisition above for
local inspection. The `--input-root` option applies to every selected source,
so run this selection separately from sources using their normal downloads.
A selected acquisition is a subset, not evidence about the entire release.

The grading definition invokes the unchanged `tests/test.sh` in a separate sandbox. Private
tests and reference solutions are excluded from agent resources. It retains
native components in grading details and reports their arithmetic mean, following
the outcome-reward method in [the paper, section 3.2.2](https://www.alphaxiv.org/abs/2609.reinforcing-agents-collective-skills.pdf#page=6).
It uses the existing scalar reward interface without a separate pass/fail verdict.
The optional written LLM rubric is preserved but not executed.

Workspace capture preserves file contents, including changes to preexisting
files and deletions. It does not reproduce file modes, symlinks, empty
directories, running processes, or changes outside `/workspace`. Tasks that
depend on these properties need another execution contract. A passing reference
solution alone does not establish grader equivalence.
