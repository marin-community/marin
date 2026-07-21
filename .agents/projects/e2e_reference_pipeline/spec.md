# Spec: Canonical end-to-end reference pipeline

The contract layer for [`design.md`](./design.md). Pins the public surface of the new symbols and
module constants, and states the exact contracts where they meet the existing
`reference_datakit_steps` / Grug launch trainer / `eval_steps` / `eval_report` APIs. This spec
commits to the **two-pass driver + `store_mixture`→`LmDataConfig`** shape; the unified-graph
alternative is out of scope (see Open Questions in `design.md`).

**Resolved by the stress-test:** the `TokenizedCache` + `train_lm(datasets=...)` path does *not*
work for store buckets — they are flat Levanter caches, and `TokenizedCache.as_component()` omits
`flat_cache=True`, so Levanter looks for a `<bucket>/train/` layout the store never writes and
silently skips every component. The trainer therefore uses a directly-built `LmDataConfig` with
`flat_cache=True` per-bucket components (the `launch_datakit_moe_mix.py` shape) and the Grug launch
trainer — **not** `train_lm` and **not** `ArtifactStep.adopt`.

All new code lives in **`experiments/references/reference_training_pipeline.py`** (rewritten,
replacing the current pretrain-only version), except `store_mixture`, which may live in
`experiments/datakit/` (Open Question — it encodes the store→train contract and the `flat_cache`
wiring is a datakit/Levanter concern). No new package.

## New public symbols

### `store_mixture`

```python
def store_mixture(
    store: ClusteredStoreData,
    *,
    validation: Sequence[ArtifactStep[TokenizedCache]] = (),
    weighting: MixtureWeighting = MixtureWeighting.TOKEN_PROPORTIONAL,
    prefix: str | None = None,
) -> LmDataConfig:
    """Build the Levanter training data config from a datakit store's non-empty buckets.

    For each entry in ``store.buckets`` (which already omits buckets that received zero rows),
    emits one ``DatasetComponent`` with ``flat_cache=True`` and
    ``source=UrlDatasetSourceConfig(cache_dir=<bucket>, train_urls=[], ...)``, ``cache_dir=<bucket>``,
    and the component weight. ``store.tokenizer`` is set once as the config's tokenizer (all
    components share it — the store guarantees a single tokenizer). ``validation`` caches are added
    as zero-weight held-out components (ordinary, non-flat ``TokenizedCache`` handles resolved to
    their cache dirs). Returns the ``LmDataConfig`` the Grug launch trainer consumes as
    ``GrugBaseLaunchConfig.data``.

    Bucket path handling: ``bucket.path`` is absolute (``s3://.../datakit/store_<hash>/cluster=C/quality=Q``).
    When ``prefix`` is given it is stripped to a prefix-relative ``cache_dir`` so the training
    fingerprint is region-independent (matching the moe precedent's ``_STORE_PREFIX``); with
    ``prefix=None`` the absolute path is used and the training identity is region-specific.

    Weights:
      - ``TOKEN_PROPORTIONAL``: ``bucket.total_tokens``. Asserts ``total_tokens > 0`` for every
        emitted bucket and raises :class:`ValueError` otherwise — a ``0``-weight component is
        silently dropped by Levanter (``datasets.py`` weight resolution), so a zero here is a bug in
        the store, not a valid mixture. Levanter renormalizes weights internally.
      - ``UNIFORM``: every non-empty bucket weight ``1.0``.

    Raises :class:`ValueError` if ``store.buckets`` is empty (a datakit run that produced no data).
    """
```

`MixtureWeighting` is a new `StrEnum` in the same module:

```python
class MixtureWeighting(StrEnum):
    TOKEN_PROPORTIONAL = "token_proportional"
    UNIFORM = "uniform"
```

### `reference_train_on_store`

```python
def reference_train_on_store(
    store: ClusteredStoreData,
    *,
    model: GrugModelConfig,
    version: str | None = None,
    weighting: MixtureWeighting = MixtureWeighting.TOKEN_PROPORTIONAL,
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble the reference pretrain step over a resolved datakit store.

    Mirrors the current reference: builds an ``ArtifactStep[LevanterCheckpoint]`` named ``REF_NAME``
    whose ``build_config(ctx)`` returns a ``GrugBaseLaunchConfig`` with
    ``data=store_mixture(store, validation=VALIDATION, weighting=weighting, prefix=ctx.prefix)``,
    ``model=model``, ``steps=REFERENCE_STEPS``, ``resources=ctx.runtime_arg("train_resources")``, and
    the standard optimizer/precision/tracker, run via ``run_grug_base_trial``. ``REFERENCE_STEPS`` is
    a fixed step count — **epochs are unsupported** for store buckets (an epoch count would read
    ``<bucket>/train/.stats.json``, which the flat store cache never writes). The store's bucket
    paths and weights are literals in the config, so they bear identity in the fingerprint.
    """
```

### `main`

```python
def main() -> None:
    """Run raw sample → datakit → pretrain → eval → report in one invocation (argparse driver).

    1. datakit pass: ``reference_datakit_steps(sample_sources(SAMPLE_PREFIX), quality_model=QUALITY_MODEL,
       quality_model_version=QUALITY_MODEL_VERSION, scale=SMOKE_SCALE)``;
       ``StepRunner().run(datakit.all_steps, max_concurrent=args.max_concurrent)``;
       ``store = read_artifact(datakit.output_buckets.output_path, ClusteredStoreData)``.
    2. train+eval pass: ``model = reference_train_on_store(store, model=REFERENCE_MODEL, version=version,
       weighting=args.weighting)``; ``report = eval_report(eval_steps(model, core_evals(serve=REFERENCE_SERVE)),
       name=REF_NAME)``; ``run(report, max_concurrent=args.max_concurrent)``.

    Flags: ``--version`` (calendar tag or ``dev``), ``--sample-prefix`` (default ``SAMPLE_PREFIX``),
    ``--weighting`` (``token_proportional`` | ``uniform``), ``--max-concurrent`` (threaded into both
    passes). A single process; both passes resume from their own caches on re-run.
    """


if __name__ == "__main__":
    main()
```

The current `build(*, version) -> ArtifactStep[LevanterCheckpoint]` and `experiment_main(build)()`
entry are **removed** — a dynamic bucket set cannot be authored before datakit runs.

## Module constants (identity-bearing defaults)

| Constant | Value | Meaning |
|---|---|---|
| `REF_NAME` | `"references/reference-pipeline"` | shared `name` for the train + eval + report handles |
| `SAMPLE_PREFIX` | `"s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7"` | raw testbed sample (single-edit point a) |
| `QUALITY_MODEL` | `"s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2"` | datakit quality scorer dir (region-specific; not hashed) |
| `QUALITY_MODEL_VERSION` | `"pooled-junkgate2"` | stable identity tag hashed into the quality step |
| `SMOKE_SCALE` | imported from `experiments.datakit.reference_pipeline` | K=64 / 16-worker datakit sizing |
| `REFERENCE_MODEL` | a nano/150M `GrugModelConfig` (single-edit point b) | tiny pretrain model |
| `REFERENCE_STEPS` | small int (e.g. `100`) | token budget = steps × batch × seq_len |
| `REFERENCE_TRAIN_RESOURCES` | small GPU/TPU `ResourceConfig` (`runtime_args`, off identity) | train dispatch target |
| `REFERENCE_SERVE` | a pinned small `ServeSpec` (concrete slice — see Open Question) | evalchemy serving slice for `core_evals` |
| `VALIDATION` | paloma + uncheatable `TokenizedCache` handles | held-out (zero-weight) eval-loss datasets |

## Contracts at the existing-API seams (unchanged; documented, not modified)

- **`reference_datakit_steps(sources, *, quality_model, quality_model_version, domain_centroids=None,
  centroids_version=None, scale=DEFAULT_SCALE) -> DatakitSteps`** (`reference_pipeline.py:465`).
  Called with `sources=sample_sources(SAMPLE_PREFIX)`, `scale=SMOKE_SCALE`, `domain_centroids=None`
  (train centroids inline — `SMOKE_SCALE.cluster.k_train=64` ≤ sample size). Consumes
  `datakit.output_buckets.output_path` (an absolute, computable-before-run property) and
  `datakit.all_steps`. **No change.**
- **`ClusteredStoreData`** (`datakit_store.py:91`) — read via `read_artifact(store.output_path,
  ClusteredStoreData)`. Fields used: `.buckets[*].path`, `.buckets[*].total_tokens`, `.tokenizer`,
  `.cluster_view`. `.buckets` omits empty buckets. **No change.**
- **`GrugBaseLaunchConfig(data: LmDataConfig, model, output_path, run_id, resources, steps, ...)` +
  `run_grug_base_trial(config)`** (`experiments/grug/base/launch.py:94,186`). `data` is the
  `store_mixture` output. **No change.**
- **`eval_steps(model, groups, *, version=None)` and `eval_report(results, *, name, version=None)`**
  (`evals.py:189,196`). `groups=core_evals(serve=REFERENCE_SERVE)`, `name=REF_NAME`. The
  `EvalReport` is the terminal readout. **No change.**

## Persisted shapes

No new persisted schema. Outputs land at (note the **namespacing caveat**: `model.name =
user_namespaced_name(REF_NAME, version)`, so a `dev` version prefixes the train + eval paths with
`users/{username}/`; the calendar-version case is shown below):

- Datakit intermediates + store: `{MARIN_PREFIX}/datakit/*_<hash>/` (content-addressed StepSpec
  routing); store at `{MARIN_PREFIX}/datakit/store_<hash>/cluster=<C>/quality=<Q>/` with a merged
  `shard_ledger.json` per bucket and `.artifact.json` (= `ClusteredStoreData`) at the store root.
  These are the disposable/versioned intermediates the footprint note documents. **No adoption
  records** (the trainer references buckets by literal path, so no per-bucket provenance is minted).
- Train checkpoint: `{MARIN_PREFIX}/{model.name}/<version>/` — for a calendar version,
  `references/reference-pipeline/<version>/`.
- Eval results: `{MARIN_PREFIX}/evaluation/evalchemy/{model.name}/core/<version>/`.
- Report: `{MARIN_PREFIX}/evaluation/report/{REF_NAME}/<version>/` (the report uses the raw
  `REF_NAME`; the eval steps use the namespaced `model.name` — an intentional split worth noting to
  reviewers).

## Errors

- `store_mixture` raises `ValueError` when `store.buckets` is empty, and (under `TOKEN_PROPORTIONAL`)
  when any emitted bucket has `total_tokens <= 0`.
- `reference_train_on_store` uses a fixed step count; passing an epoch count is unsupported (would
  `FileNotFoundError` on the missing `<bucket>/train/.stats.json`).
- Existing seam-API errors are unchanged (`reference_datakit_steps`' `quality_model_version`
  requirement; the store's cross-source single-tokenizer / co-partitioning asserts).

## Out of scope (reviewers: do not push back on these here)

- The **unified single-`ArtifactStep[EvalReport]` graph** (store-wrapping ArtifactStep + store-aware
  trainer) — the design's lead Open Question, deliberately not committed.
- **CI canary/ferry wiring** — a follow-up once the harness is green.
- **Step-varying (phase) mixtures** — the reference uses one static `store_mixture`; the moe-style
  two-phase schedule and the retired reference's pretrain→midtrain→SFT demonstration are not
  reproduced.
- **Retiring the scaling/HP-sweep references** — only `reference_training_pipeline.py` is replaced.
- **Teaching `TokenizedCache`/`as_component` to carry `flat_cache`** — a real Levanter/marin change;
  the reference works around it with a directly-built `LmDataConfig` instead.

## File summary

| Piece | Location |
|---|---|
| Driver `main()`, `reference_train_on_store`, module constants | `experiments/references/reference_training_pipeline.py` (rewritten) |
| `store_mixture`, `MixtureWeighting` | reference file, or `experiments/datakit/` (Open Question) |
| Runbook (how to use as a change-validation harness) | `experiments/references/README.md` or a docs page (per DoD) |
| Reused, unchanged | `reference_datakit_steps` (`experiments/datakit/reference_pipeline.py`), Grug launch trainer (`experiments/grug/base/launch.py`), `eval_steps`/`eval_report` (`experiments/evals/evals.py`) |
