# NeMo Gym Agentic-SWE-Pivot-v1 reimport findings

Date: 2026-09-14

## Verdict

The sampled rows cannot be safely imported as real repository repair tasks from
the current NeMo Gym release. They can become such tasks only after joining each
pivot to a pinned source checkout/archive and a private, executable final-state
oracle. The current source is a one-step action-prediction dataset: its verifier
compares the next function-call category, target, and argument similarity. It
does not execute a repository or establish that a repair succeeds.

## Exact sampled rows

Dataset `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`, Hub revision
`4947a3c8ea803413a65f9eca14a96ef521b2ddf5`, config `default`, split `train`:

| offset | trajectory / metadata | source instance | available task material |
| ---: | --- | --- | --- |
| 14364 | 7329 / `codex` | `pandas-dev__pandas-50238` | system and user trajectory prefix; requested workspace `/workspace/pandas-dev__pandas__2.0`; issue text includes the reported-version commit `749d59db6e5935f4b3ddf371e79dacc00fcad1c0`; `ref_patch=null`; no base checkout revision, repository archive, tests, or expected final state |
| 39518 | 10128 / `codeact` | `dask__dask-7656` | user trajectory consists of uploaded workspace path `/workspace/dask__dask__2021.04` plus issue text; `ref_patch=null`; no base checkout revision, repository archive, tests, or expected final state |

Both rows contain `responses_create_params`, tool schemas, a materialized
trajectory prefix, `expected_action`, `ref_message`, and pass-rate metadata.
The source schema does not contain a repository URL/commit, workspace snapshot,
test patch, test command, or final-state oracle. `ref_message` is a prior agent
turn, not a reproducible source checkout. The pandas version commit in the issue
report is the environment in which the bug was observed, not an asserted base
revision for a task workspace.

## Source verifier semantics

At the pinned Gym tree (`NVIDIA-NeMo/Gym@1e668906`),
`resources_servers/swe_pivot/README.md` and `app.py` describe a single-step
verifier. It extracts one model `function_call`, maps tool names into categories,
optionally compares target paths/commands, computes `SequenceMatcher` argument
similarity, and optionally applies diff-size shaping. The configured
`max_steps: 1` confirms that the service stops at a pivot. The dataset config
(`resources_servers/swe_pivot/configs/swe_pivot.yaml`) identifies GitLab artifact
`swe_pivot` version `0.0.1` for train/validation, but does not include repository
instances or repair tests. This contract is insufficient as a final-state
verifier: a matching next action can receive reward without changing or testing
the requested repository.

## Existing TaskCompendium support

`lib/taskcompendium/src/taskcompendium/importers/r2egym.py` is the closest
supported importer. It requires a pinned source image digest, source commit and
repository identity, embedded private tests plus an expected status map, and a
separate verifier runtime. Its Harbor path already supports the requested
private-verifier shape: `ContainerRuntime` can use a common supervisor image
with `ImageOverlay`, while `TaskDockerEnvironment` starts a writable workdir
and uploads declared inputs. `lower_to_harbor` keeps verifier resources private
and validates complete `FinalState` submissions. The current R2E importer,
however, assumes source dependencies/repository content are in the source image
and has R2E-specific status-map parsing and setup; it is not a NeMo SWE importer.

There is no direct SWE-Pivot importer. `nemo_predicted_action.py` is deliberately
an answer-only native-action importer and has no environment or final-state
submission. Reusing it for these rows would discard the repository-repair
semantics.

## Required source acquisition

For every selected pivot, acquire and pin a manifest containing:

1. canonical repository URL and immutable base commit (or a content-addressed
   repository archive), plus archive SHA-256;
2. the exact issue/bug fixture and any source data files needed to reproduce it;
3. the test additions or a pinned upstream test revision, exact test command,
   dependency/runtime requirements, and expected pass/fail interpretation;
4. the source row's trajectory/instance join key and the NeMo/Gym source
   revisions, so a row cannot silently be paired with a different checkout.

The archive must be copied into the writable workdir before the agent starts.
A common immutable runtime image may provide OS/Python dependencies, but it must
be proven compatible with every selected repository; otherwise use one pinned
runtime per source family. Do not infer a checkout from `instance_id`, issue
number, or the path shown in the prompt.

## Minimal importer/Harbor changes after acquisition

Add a `nemo_swe_pivot` importer that rejects rows without the manifest above,
validates the archive digest and repository base revision, and emits:

* an agent-visible repository archive/data resource and a setup command that
  extracts it into the declared workdir;
* a Docker `WorkspaceState` using the pinned common runtime image and explicit
  writable workdir, with filesystem/shell/process capabilities;
* verifier-only test files, test configuration, and expected final-state/test
  policy; and
* a private verifier that runs the pinned tests against the submitted final
  workspace and reports task completion separately from infrastructure failure.

The public instruction should contain only the issue/task context and workspace
location needed by the agent. Keep expected patches, hidden tests, expected
status maps, and any source oracle in verifier/oracle resources. Use a
`FinalState((".",), ...)` rendering so the complete workspace is snapshotted;
exclude only runtime dependency directories when an image overlay preserves
them. Existing `TaskDockerEnvironment` and `lower_to_harbor` then provide the
needed writable upload/snapshot lifecycle. A small generic archive extraction
setup facility may be useful, but no change is justified until a complete
source manifest exists.

## Evidence

* Exact rows: Dataset Viewer `rows` API at offsets 14364 and 39518 for the Hub
  revision above (retrieved 2026-09-14).
* NeMo source verifier: `NVIDIA-NeMo/Gym@1e668906`,
  `resources_servers/swe_pivot/{README.md,app.py,configs/swe_pivot.yaml}`.
* Local source review: `lib/taskcompendium/src/taskcompendium/importers/r2egym.py`,
  `lib/taskcompendium/src/taskcompendium/importers/nemo_predicted_action.py`,
  `lib/taskcompendium/src/taskcompendium/harbor/environments.py`, and
  `lib/taskcompendium/src/taskcompendium/lowering.py`.
* Related prior audit: `.agents/projects/taskcompendium-coverage/nemo-catalog.md`
  records the same family as action-only, with no repository state and no
  proven source join.
