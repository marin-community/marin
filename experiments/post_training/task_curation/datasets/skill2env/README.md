# Skill2Env conversion

The existing `Harbor Hub:skill2env` catalog entry has a conversion recipe for
network-disabled tasks with a single `/workspace`, native programmatic tests,
and a reference solution. Preparation verifies the release membership and each
task's content digest through the pinned client in `config/external/harbor`.
The Hub release ID and the current SDK's membership hash differ for this
release, so preparation also checks the separately pinned 7,496-task membership
hash. It does not edit upstream tasks.

Export a bounded selection for QUICK conversion; Docker and registry access are not required:

```bash
uv run --project config/external/harbor --frozen python \
  -m experiments.post_training.task_curation.datasets.skill2env.prepare \
  --release skill2env/skill2env@sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c \
  --task skill2env/task_api-design-reviewer_jdkbai64@sha256:acfc393449fdad22ae5a2da63d16d27605c02a983059fac0c5dd49384bf6c260 \
  --output /tmp/skill2env-input
```

Repeat `--task` for more tasks; `--all` selects the release's complete population.
Without image builds, conversion retains the native `environment/` files as an
unresolved `DockerBuildContext`. The verifier context adds workspace clearing
so submitted deletions are preserved. Private tests and reference solutions
remain separate from both build contexts. The runtime rejects unresolved
contexts before execution.

Convert the prepared selection with the existing driver:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver --mode quick --run \
  --source skill2env --source-input-root skill2env /tmp/skill2env-input \
  --output-root /tmp/skill2env-converted
```

QUICK writes normalized TaskSpecs for inspection. It does not build images,
run graders or admit training tasks. Retained build recipes do not freeze the
results of package downloads during a later build.

For executable tasks, run preparation on a Docker build host with
`--build-repository ghcr.io/marin-community/iris-task` and a new output directory.
This builds and pins actor and verifier images using the shared curation
publisher's linux/amd64, gzip-layer and disabled-provenance settings. The
verifier derives from the pinned actor image with `/workspace` cleared. Iris
workers pull anonymously, so another repository must also be publicly readable.

For reviewed SAMPLE/FULL execution, use the
[campaign options](../../README.md#reviewed-local-loop-on-real-sources) with that
built input root and an image backend supporting 2 CPUs, 4096 MiB RAM,
4096 MiB storage and disabled network access.

Skill2Env is excluded from default campaigns because it requires prepared input.
Select it explicitly with `--source skill2env`. In mixed campaigns,
`--source-input-root skill2env PATH` leaves other sources' downloads unchanged.
A missing input fails during planning, before workers start. Campaigns do not
download or build this source automatically. A selected export is a
subset, not evidence about all 7,496 tasks.

The grader runs the unchanged `tests/test.sh` in a separate sandbox. Private
tests and reference solutions are excluded from agent resources. It retains
native components in grading details, reports their arithmetic mean, and marks
full pass only when every component is within `1e-9` of one. These last two
rules are curation conventions. The optional written LLM rubric is preserved
but not executed.

Workspace capture preserves file contents, including changes to preexisting
files and deletions. It does not reproduce file modes, symlinks, empty
directories, running processes, or changes outside `/workspace`. Tasks that
depend on these properties need another execution contract. A passing reference
solution alone does not establish grader equivalence.
