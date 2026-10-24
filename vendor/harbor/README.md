# Harbor evaluation runtime

Marin vendors the [Harbor fork](https://github.com/marin-community/harbor) at the
revision recorded in [UPSTREAM.toml](UPSTREAM.toml). The source is licensed under
[Apache 2.0](LICENSE).

This package contains the evaluation runtime, agents, Shellbox environment adapters,
verification, trajectories, registry downloads, FineStore archives, and telemetry.
It evaluates prebuilt Harbor tasks from local paths, Hugging Face repositories,
or the Harbor registry. The `harbor` distribution provides both the `harbor` and
`harbor_config` Python modules.

The website, results viewer, Hub result sharing, benchmark publishing commands,
standalone config release tooling, and upstream development automation are
omitted. Benchmark adapters, their generation inputs, and parity artifacts are
also omitted; the upstream revision retains their sources. Task collection and
hashing remain part of evaluation job locking. Unused golden agent traces are
omitted.

The full upstream `registry.json` snapshot is also omitted. Named registry datasets
use Harbor's hosted service; explicit registry URLs and paths remain supported.

Run from the Marin repository root in Harbor's isolated project:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime harbor run --help
uv run --isolated --project vendor/harbor --frozen --extra runtime harbor run --config prepared-policy.yaml
```

Harbor is excluded from Marin's uv workspace. Its own lock selects local editable
Marin FineStore and Rigging packages. The normal Marin environment and Iris
`--all-packages` environments do not install Harbor. The `runtime` extra preserves
the deployed Daytona SDK version, 0.200.2.

Harbor executes tasks only through Shellbox. Its environment types are `iris`
(the default), `daytona`, `qemu` and `shellsim`; other provider names and
custom environment import paths fail validation. The adapters depend on
Shellbox's generic machine API; Shellbox does not import Harbor. The `runtime`
extra includes Iris and Daytona; `qemu` and `shellsim` install those optional
backends. The external `harbor.agents.shellbox:BashAgent` also uses this API.

Policies must select prepared task images or a supported trusted image-preparation
path. Iris supports private service groups imported from the validated Compose
subset. Daytona runs one Shellbox machine and rejects Compose; its image builds
run in provider preparation, never in a privileged guest daemon. See
[Harbor execution](../../docs/harbor-integration.md) for preparation requirements.
Existing historical policies are inputs for reproduction, not a promise that a
removed provider or old image recipe can run unchanged.

Edit Harbor and its local Marin dependencies in the same pull request. Update
Harbor's lock when a dependency manifest changes, and run
`uv run --no-project infra/ci/run_harbor_tests.py` for the offline runtime checks.
For an upstream import, export the selected Git tree, reapply the deliberate
removals, and update `UPSTREAM.toml`.
`UPSTREAM.toml` records the upstream import base; it does not select a Git runtime.
All Harbor launches execute this checkout. Reproduce historical evaluation cohorts
from their recorded Marin checkout.
