# External runtime dependencies

This directory tracks Python tools and artifacts that Marin installs into
isolated runtime environments.

The Evalchemy uv project is excluded from the root
workspace so its dependency graph does not have to resolve with Marin's
training and serving dependencies. Each `pyproject.toml` selects an external
source. Its adjacent `uv.lock` records the exact commit that Marin uses.

Verified Evalchemy launches select revisions from `RUNTIME_COMMITS` in
`lib/marin/src/marin/evaluation/eval_policy.py`. The external dependency updater
advances only the Evalchemy project. Harbor always runs the
source under `vendor/harbor` with its own lock. Historical Harbor commit metadata
validates saved cohort records and does not select executable dependencies.
Harbor is excluded from workspace membership and default installs, including
`uv sync --all-packages`.

`vllm/gpu.toml` records the published CUDA release, Torch backend, and exact
wheel URL and SHA-256 for each architecture. The
[GPU guide](https://github.com/marin-community/marin/blob/main/.agents/skills/refresh-fork/docs/vllm.md#triggers-and-tests)
defines how the fork builds and tests wheels and how Marin adopts them.

`vllm/tpu.toml` records an exact vLLM source from the fork's maintained `main`
lineage and the matching `tpu-inference` SHA. The TPU stack runs from an
isolated uvx env rather than the workspace. It is not a uv project either;
refresh it through
`.agents/skills/refresh-fork/SKILL.md`.

The packaged pin table at
`lib/marin/src/marin/external_dependencies.py` is generated from the locks, the
vLLM GPU release config, and the TPU serving fork descriptor. Runtime code imports
that module instead of reading repository-relative configuration.

Advance one project with:

```bash
uv run config/update-external.py evalchemy
```

Omit the project name to advance all registered Git projects. The command updates
the selected lockfiles and regenerates the packaged requirements.

Update a GPU release from its published manifest with:

```sh
uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json
```

The command updates `vllm/gpu.toml` and regenerates the packaged pins. It
validates and renders both outputs before replacing the descriptor, so a
rendering failure leaves the existing pins in place. Follow the GPU guide's
Snowball test and adoption steps before merging those pins.

Update TPU source pins in `vllm/tpu.toml`; those forks are not part of the
nightly upgrade set.
Verify that all generated state is current without contacting the repositories:

```bash
uv run config/update-external.py --check
```

`Ops - External Dependency Update` runs the all-project command every six hours
and can also be started with `workflow_dispatch`. It opens or refreshes one
`automation/external-dependencies` pull request containing every changed lock
and generated pin. The dedicated updater app merges after the required checks
pass. A failed check or one-hour timeout leaves the pull request open and fails
the scheduled run. The workflow log and pull request body list each resolved
package version and commit, followed by the upstream commit subjects in every
changed range. Generate the same Markdown summary locally with `--summary-file
<path>`; commit metadata is read through the authenticated GitHub CLI.

The six-hour schedule is the update-discovery interval. Under normal GitHub
Actions scheduling, a green update lands in the same run; a blocked update is
visible within the workflow's 90-minute deadline. Launches continue to consume
only commits already landed on Marin's default branch.

The external configurations intentionally model only what Marin needs:

- `evalchemy` resolves the endpoint client core. Benchmark extras are selected
  by each evaluation at runtime.
- `vllm` records the promoted GPU wheels (`gpu.toml`) and a main-line source
  paired with tpu-inference for TPU (`tpu.toml`). The GPU and TPU wheels have
  independent build and promotion paths. Neither is a workspace dependency.

Evaluation provenance stores the generated commit-pinned Evalchemy requirement.
Historical Harbor cohorts identify their exact Git revision. Current Harbor
provenance includes its imported source revision and a digest of the vendored
runtime and its dependency lock.

`migration.toml` describes how to migrate each fork toward upstream — the base to
select, the Marin e2e that validates it, and the constraints to respect. It holds
no pins; the `refresh-fork` skill reads a section to refresh one fork. The weekly
`ops-fork-ferry` workflow runs that skill for each supported pin or group.

To add another external tool, create an isolated project and register its
directory, distribution, and generated constant in `config/update-external.py`,
then add a `migration.toml` section for it. Consumers should use the generated
`ExternalDependency.requirement()` rather than reading a lockfile or copying its
commit.

SkyRL lives in `vendor/skyrl` with its own `pyproject.toml` and `uv.lock`.
Marin invokes its `marinskyrl` executable in an isolated child environment.
GPU workers receive the source, assets, and lock in a hash-verified Iris bundle;
they do not clone the fork. The external updater does not advance this source.
The [SkyRL migration guide](../../docs/dev-guide/skyrl-migration.md) describes
rebuilding the pruned source and reconciling active fork work before archival.
`uv run config/update-external.py --skyrl-marin-packages` advances its published
Marin libraries within the nested project's version constraints. It checks the
isolated import and constructor contract and restores the lock if the update
fails. The runtime smoke also exercises the harness on GPUs.
