# Reproduce the pretraining source pool

Marin publishes the code needed to retrieve and normalize its pretraining
sources. The canonical catalog is
[`lib/marin/src/marin/datakit/sources.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sources.py).
Each entry records a stable Marin source name, an ordered source-processing
recipe, and an approximate token count used for initial mixture weighting.

The catalog describes the active source pool at a particular Marin commit. To
reconstruct the inputs to a specific training run, check out the Marin commit
recorded by that run and use its copy of `sources.py`. The model's training
configuration determines which catalog entries were sampled and at what
weights; the catalog alone does not specify an exact model mixture or token
stream.

## Why Marin publishes source recipes

Marin does not publish the pool as a single mirrored Hugging Face dataset.
Researchers retrieve each dataset from its original provider. This keeps the
provider's attribution, license, and access controls in the retrieval path. It
also means that a provider can change or withdraw the source used by future
reconstructions without relying on Marin to remove a second public copy.

Some providers require an account or acceptance of dataset-specific terms.
Reconstruction does not bypass those requirements. Review each provider's
license and obtain access before running its download step.

For Hugging Face sources, Marin records the dataset repository ID and pins a
specific repository revision. The pin is intended to prevent changes on a
provider's moving branch from changing the reconstructed files. This guarantee
depends on Hugging Face continuing to serve that revision. Sources fetched
from the live web, APIs, or other mutable upstreams do not always provide an
equivalent immutable revision. For those sources, a Marin commit pins the
retrieval and transformation code, but a later run may retrieve different
content.

## Trace a source to its provider

Every catalog entry is a `DatakitSource`. Its `normalize_steps` field contains
an ordered `StepSpec` chain ending in the normalized Parquet artifact consumed
by downstream Datakit stages.

To inspect a source:

1. Find its Marin name in `all_sources()` in
   [`sources.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sources.py).
2. Follow the imported `*_normalize_steps` factory to its module under
   [`marin/datakit/download/`](https://github.com/marin-community/marin/tree/main/lib/marin/src/marin/datakit/download).
3. Read the factory and the dependencies of its first returned step. These
   lead to the raw download and any source-specific transforms.
4. Record the provider dataset ID, pinned revision, selected files, schema,
   and normalization parameters from that module.

For example, the `coderforge` row points to
`coderforge_normalize_steps`. That factory calls `download_hf_step` with the
provider ID `togethercomputer/CoderForge-Preview`, revision `060fca9`, and the
selected Parquet splits. It then renders each trajectory as text and runs the
shared normalization step. The resulting chain preserves both the provider
identity and the exact Marin transform.

Many simple Hugging Face sources use `hf_normalize_steps`, which makes the same
provider ID, revision, file selection, and schema parameters visible in one
call. Family modules such as `nemotron_v2.py` define one shared provider
download and a separate normalization step for each catalog subset.

## Materialize the catalog

Clone Marin and check out the commit associated with the training run:

```bash
git clone https://github.com/marin-community/marin.git
cd marin
git checkout <marin-commit>
uv sync --all-packages --extra=cpu
```

Choose a durable destination for the raw and normalized artifacts. Marin
resolves relative step paths under `MARIN_PREFIX`; the destination can be a
local directory or an object-store URI supported by `fsspec`.

```bash
export MARIN_PREFIX=/path/to/pretraining-pool
export HF_TOKEN=<token-for-sources-you-can-access>
```

List the source names at the checked-out revision:

```bash
uv run python - <<'PY'
from marin.datakit.sources import all_sources

for name in all_sources():
    print(name)
PY
```

Start with one source and materialize its complete chain:

```bash
uv run python -m experiments.datakit.scripts.trigger_sources \
  --sources coderforge
```

Use a comma-separated list to select several sources. Omitting `--sources`
materializes every entry in the catalog:

```bash
uv run python -m experiments.datakit.scripts.trigger_sources
```

This is a large distributed data-processing workload. `trigger_sources` hands
each terminal normalization step to Marin's `StepRunner`, which follows
transitive dependencies, shares common family downloads, and skips artifacts
that already completed under the same prefix. A local run can download simple
sources, but large normalization steps require an Iris/Fray environment or an
equivalent deployment of Marin's execution stack.

Pass `--downloads-only` to stop at each chain's first step. This is useful for
auditing provider retrievals, but a first step may include a source-specific
transform and is not guaranteed to be the untouched provider files. Trace its
dependencies when the raw download itself is required.

## Sources with separate reconstruction steps

Three catalog entries currently begin from pre-staged Parquet because their raw
provider data requires an additional hydration or extraction process. Their
download modules pin the code and configuration that produced those shards:

| Catalog entry | Reconstruction pointer |
| --- | --- |
| `common-crawl-focus-2026-22` | [`common_crawl_focus.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/common_crawl_focus.py) links the exact Focus Crawl extractor and launch configuration. |
| `nemotron_code_v1/content` | [`nemotron_code_v1_content.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/nemotron_code_v1_content.py) links the pinned resolver, Software Heritage graph configuration, ID mapping, and object downloader. |
| `nemotron_code_v2/content` | [`nemotron_code_v2_content.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/nemotron_code_v2_content.py) links the same pinned hydration pipeline for the v2 metadata. |

Run the linked reconstruction process first, place its output at the relative
path named by the module under `MARIN_PREFIX`, and then run `trigger_sources`
for that catalog entry. Each pre-staged step checks that input shards are
present before normalization. Use the shard counts recorded in the module to
verify that staging completed.

Provider availability can change after a Marin revision is published. A
provider removal, license change, or expired credential may prevent a later
byte-for-byte reconstruction even though the original recipe remains pinned.
