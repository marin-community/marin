# External runtime dependencies

This directory tracks Python tools and artifacts that Marin installs into
isolated runtime environments.

The Evalchemy, Harbor, and MarinSkyRL uv projects are excluded from the root
workspace so their dependency graphs do not have to resolve with Marin's
training and serving dependencies. Each `pyproject.toml` follows the external
repository's `main` branch, and its adjacent `uv.lock` records the exact commit
Marin uses.

`vllm/gpu-release.toml` records the promoted CUDA release, Torch backend, and
architecture-specific wheel URLs and SHA-256 digests. It is updated from the
release manifest only after the H100 and GB200 publication gates pass. It is
not a uv project and the nightly update does not advance it.

`vllm/tpu-forks.toml` records the `vllm` and `tpu-inference` fork SHAs for the
TPU serving stack, which runs from an isolated uvx env rather than the workspace.
It is not a uv project either; refresh it through
`.agents/skills/refresh-fork/SKILL.md`.

The packaged pin table at
`lib/marin/src/marin/external_dependencies.py` is generated from the locks, the
vLLM GPU release config, and the TPU serving fork descriptor. Runtime code imports
that module instead of reading repository-relative configuration.

Advance one project with:

```bash
uv run config/update-external.py evalchemy
```

Omit the project name to advance all three Git projects. The command updates
the selected lockfiles and regenerates the packaged requirements. Regenerate
only the promoted vLLM release after editing `vllm/gpu-release.toml` with:

```bash
uv run config/update-external.py vllm
```

The generated module also carries the isolated TPU-vLLM requirements from
`vllm/tpu-forks.toml`; those forks are not part of the nightly upgrade set.
Verify that all generated state is current without contacting the repositories:

```bash
uv run config/update-external.py --check
```

`Ops - External Dependency Update` runs the all-project command at 09:00 UTC
each day and can also be started with `workflow_dispatch`. It opens or refreshes
one `automation/external-dependencies` pull request containing every changed
lock and generated pin, then enables squash auto-merge. The workflow log and
pull request body list each resolved package version and commit, followed by
the upstream commit subjects in every changed range. Generate the same Markdown
summary locally with `--summary-file <path>`; commit metadata is read through
the authenticated GitHub CLI.

The external configurations intentionally model only what Marin needs:

- `evalchemy` resolves the endpoint client core. Benchmark extras are selected
  by each evaluation at runtime.
- `harbor` resolves the Git checkout and pinned Daytona SDK used only by the
  isolated evaluation driver. Harbor is absent from Marin's workspace lock.
- `MarinSkyRL` tracks the repository-root `marinskyrl` distribution. The
  external lock resolves its CPU-safe base for the isolated launcher. The
  launcher synchronizes the selected `fsdp` or `megatron` profile from that
  revision's frozen root lock inside the cluster's standard Iris task image.
- `vllm` records the promoted GPU wheels (`gpu-release.toml`) and the TPU source
  fork pins (`tpu-forks.toml`). Neither is a workspace dependency.

`migration.toml` describes how to migrate each fork toward upstream — the base to
select, the Marin e2e that validates it, and the constraints to respect. It holds
no pins; the `refresh-fork` skill reads a section to refresh one fork. A weekly
coordinator that walks the sections in dependency order is planned but not yet
built.

To add another external tool, create an isolated project and register its
directory, distribution, and generated constant in `config/update-external.py`,
then add a `migration.toml` section for it. Consumers should use the generated
`ExternalDependency.requirement()` rather than reading a lockfile or copying its
commit.
