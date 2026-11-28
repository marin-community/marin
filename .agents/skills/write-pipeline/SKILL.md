---
name: write-pipeline
description: Design or revise a Marin experiment pipeline that binds reusable work into an artifact graph. Use for new workflow entrypoints, stage wiring, and pipeline placement decisions; use a narrower domain skill when it owns the launch.
---

# Write a Marin pipeline

Keep the boundary between work and workflow explicit:

- `lib/**` owns concrete implementations intended for reuse: functions, configurations, and produced data types. New library code must not compose a pipeline, construct an `ArtifactStep` or `StepSpec`, or reference a particular artifact handle, name, version, or output. The execution framework's `Artifact` and `ArtifactStep` definitions are infrastructure, not a model for new library workflow builders.
- `experiments/**` owns artifact graph construction and one-off binding code: choosing inputs, names, versions, dependencies, resources, and which reusable functions to run. Keep a transformation here only when it is specific to that experiment. Move work intended for reuse into a concrete `lib/**` implementation, then bind it in `experiments/**`.
- Put experiment-specific tests beside their experiment, including tests of its graph and one-off transformations; do not put them in root `tests/**` or `lib/**`. Keep tests of reusable library behavior in the library's normal test location. Follow root `TESTING.md` and the relevant module instructions.

SFT data mixes: build training data with `mixture()` over versioned handles (`tokenized`, `pretokenized`, `apply` or `adopt`), never raw cache paths or directory prefixes. Datakit sources expose `StepSpec`s: build their normalization graph first, then adopt the completed output with its normalization-step identity in the config and retain the source receipts. Pass the adopted handle to variants and tokenization; bump fixed versions when recipes or dependency versions change. The mix is `{handle: weight}` plus the shuffle config, so the run's `.artifact.json` record names every component. RL task sources are `RlDataSource` entries with Atlas curation receipts. See [SFT data mixes](../../../docs/references/sft-mixes.md).

Inspect nearby current pipelines before editing. Read [layout and examples](references/layout.md) for placement decisions and [artifact binding](references/artifact-binding.md) when working with `ArtifactStep`, `apply`, paths, or caching. Existing library builders that return steps are exceptions, not precedent for adding more pipeline composition to `lib/**`.

For a change, verify that the experiment builds the intended dependencies and that reusable work can run without importing `experiments/**`. Dry-run the graph where the entrypoint supports it; run focused behavior tests without submitting a live job unless requested. Update user-facing docs when the workflow or API changes. Use the `manage-pr` skill if asked to publish the result.
