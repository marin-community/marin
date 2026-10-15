# Skill2Env conversion

The existing `Harbor Hub:skill2env` catalog entry has a conversion recipe for
network-disabled tasks with a single `/workspace`, native programmatic tests,
and a reference solution. Preparation verifies the release membership and each
task's content digest through the pinned client in `config/external/harbor`.
The Hub release ID and the current SDK's membership hash differ for this
release, so preparation also checks the separately pinned 7,496-task membership
hash. It does not edit upstream tasks.

Prepare a bounded selection on a Docker build host with registry access:

```bash
uv run --project config/external/harbor --frozen python \
  experiments/post_training/task_curation/datasets/skill2env/prepare.py \
  --release skill2env/skill2env@sha256:bef4f739bde8d865af04c04a2cb1c84ecad83b52dd558f4d582dd46c7933666c \
  --task skill2env/task_api-design-reviewer_jdkbai64@sha256:acfc393449fdad22ae5a2da63d16d27605c02a983059fac0c5dd49384bf6c260 \
  --build-repository YOUR_REGISTRY/skill2env --output /tmp/skill2env-input
```

Repeat `--task` for more tasks; `--all` selects the release's complete population.
`--build-repository` builds and pushes two images per task, recording their
immutable digests. The actor uses the original Docker context. The verifier
uses that image with `/workspace` cleared, so copying the submitted workspace
cannot resurrect files the agent deleted. Without `--build-repository`, the
export is available for inspection but conversion defers its unbuilt tasks.

Convert the prepared selection with the existing driver:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver --mode quick --run \
  --source skill2env --input-root /tmp/skill2env-input \
  --output-root /tmp/skill2env-converted
```

QUICK does not run graders or admit training tasks. For reviewed SAMPLE/FULL
execution, use the [campaign options](../../README.md#reviewed-local-loop-on-real-sources)
with the same input root and an image backend supporting 2 CPUs, 4096 MiB RAM,
4096 MiB storage and disabled network access. Campaigns require prepared input;
they do not download or build this source automatically. A selected export is a
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
