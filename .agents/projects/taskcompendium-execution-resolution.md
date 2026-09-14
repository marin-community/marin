# TaskCompendium Harbor execution resolution

## Goal

Export Harbor-compatible task packages that bind a semantic task to its public
rendering and environment/tool requirements. Let Harbor select the compatible
harness and launch settings when a rollout starts.

## Design

1. Split the current `HarborExecutionConfig` into a task-owned
   `HarborTaskBinding` and a launch-owned `HarborLaunchConfig`.
2. Store only the binding in a canonical lowering package. The binding fixes the
   environment implementation needed to satisfy the rendering and declared
   capabilities, its model-visible tool interface, and required conversation
   retention.
3. Move agent selection, model endpoint, agent environment, timeout, and retry
   policy into `HarborLaunchConfig`. Resolve this config against a binding only
   at launch time.
4. Make `lower_to_harbor` write `binding.json` and omit `execution.json`.
   Provide an explicit reference-execution helper for examples and tests.
5. Require the runner to receive a resolved Harbor execution config. Record the
   binding and launch identities in the rollout metadata.
6. Regenerate example packages and dataset rows. Keep example launch configs as
   reference artifacts, not properties of semantic lowerings.

## Invariants

- `TaskSpecification` and public `Task` contain no Harbor/harness/provider
  selection.
- A workspace task declares filesystem, shell, process, and initial-state
  requirements. Terminus is selected only in `HarborLaunchConfig`.
- Every resolved launch must validate its binding against task requirements and
  its selected agent against the binding's interaction contract.
- Source verifiers and task prompts remain unchanged by launch resolution.
