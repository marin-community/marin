# External runtime dependencies

This directory records promoted vLLM GPU releases and paired vLLM/TPU source
revisions. Runtime code reads the generated
`lib/marin/src/marin/external_dependencies.py` module.

Evalchemy and Harbor live in `vendor/evalchemy` and `vendor/harbor`, each with its
own lock and local FineStore and Rigging sources. Both are excluded from workspace
membership and default installs, including `uv sync --all-packages`. Evaluator
changes land with Marin. Historical commit metadata validates saved cohort
records; reproducing those runs requires their recorded Marin checkout.

`vllm/gpu.toml` records the published CUDA release, Torch backend, wheel URLs and
SHA-256 for each architecture. The
[GPU guide](https://github.com/marin-community/marin/blob/main/.agents/skills/refresh-fork/docs/vllm.md#triggers-and-tests)
defines release and adoption gates. Adopt a promoted manifest with:

```sh
uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json
```

The command validates and renders the descriptor and generated pins before
replacing them. `vllm/tpu.toml` selects the paired TPU source revisions, which
run in an isolated environment. After editing that descriptor, regenerate pins
with `uv run config/update-external.py vllm`. Check generated state without
network access with `uv run config/update-external.py --check`.

No Git-pinned runtime projects remain under `config/external`. The external
update workflow is manual; vendored source updates land directly in Marin.
`migration.toml` describes the remaining fork refreshes and their required e2e
runs. The [refresh-fork skill](../../.agents/skills/refresh-fork/SKILL.md) reads
these descriptors for the weekly fork ferry.

SkyRL lives in `vendor/skyrl` with its own `pyproject.toml` and `uv.lock`.
Marin invokes its `marinskyrl` executable in an isolated child environment.
GPU workers receive source, assets and the lock in a hash-verified Iris bundle.
The [SkyRL migration guide](../../docs/dev-guide/skyrl-migration.md) covers
refreshing the pruned source and reconciling active fork work before archival.
`uv run config/update-external.py --skyrl-marin-packages` advances published
Marin libraries within the nested project's version constraints. It checks the
isolated import and constructor contract and restores the lock on failure.
