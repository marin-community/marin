# External Python projects

These uv projects track Python tools that Marin installs into isolated runtime
environments. They are excluded from the root workspace so their dependency
graphs do not have to resolve with Marin's training and serving dependencies.

Each `pyproject.toml` follows the external repository's `main` branch. Its
adjacent `uv.lock` records the exact commit Marin uses. The packaged pin table
at `lib/marin/src/marin/external_dependencies.py` is generated from those
locks; runtime code imports it instead of maintaining another SHA.

Advance one project with:

```bash
uv run config/update-external.py evalchemy
```

Omit the project name to advance all three. The command updates the selected
lockfiles, regenerates the packaged pins, and keeps the root Harbor source and
lock on the same commit. Verify that all generated state is current without
contacting the repositories:

```bash
uv run config/update-external.py --check
```

The projects intentionally model only what Marin needs:

- `evalchemy` resolves the endpoint client core. Benchmark extras are selected
  by each evaluation at runtime.
- `harbor` resolves the Git checkout and Marin's pinned Daytona SDK.
- `MarinSkyRL` tracks the repository-root package. It does not resolve the
  CUDA-heavy `skyrl-train` subproject because Marin has no runtime consumer for
  it yet.

To add another external tool, create an isolated project and register its
directory, distribution, and generated constant in `config/update-external.py`.
Consumers should use the generated `ExternalDependency.requirement()` rather
than reading a lockfile or copying its commit.
