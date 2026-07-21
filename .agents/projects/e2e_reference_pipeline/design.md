# Canonical end-to-end reference pipeline

_Why are we doing this? What's the benefit?_

There is no single blessed path that runs `raw sample → datakit → pretrain → eval → readout`
in one invocation. Validating a data or model change against the *whole* path today means
stitching stages together by hand each time — datakit, pretraining, and eval are exercised
separately (CI smokes, ad-hoc scripts, individual steps). We want one minimal, documented
reference experiment that answers "does the whole path still work, and what does this change do
to the numbers." An agent (or a human) points at it, makes a small localized edit — a new data
source, a datakit config knob, a model change — and gets back a comparable eval readout with
everything else held fixed. It replaces the current pretrain-only
[`reference_training_pipeline.py`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/references/reference_training_pipeline.py)
so we have *one* reference, not many.

See [`research.md`](./research.md) for the full survey. Closes [#7273](https://github.com/marin-community/marin/issues/7273).

## Background

All three stages already exist as reusable building blocks; the work is composition, not new
stage machinery. Datakit's full DAG is
[`reference_datakit_steps()`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/datakit/reference_pipeline.py#L364)
with a `SMOKE_SCALE` config (K=64, 16 workers) meant for "a true end-to-end run on a testbed
sample", reading pre-built samples like `s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7`
and emitting per-`(cluster, quality)` Levanter caches. Training runs through the Grug launch path
([`GrugBaseLaunchConfig` + `run_grug_base_trial`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/grug/base/launch.py#L94),
which takes a Levanter `LmDataConfig` directly — the same trainer the current reference uses). Eval
landed as reusable steps —
[`eval_steps` / `eval_report`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/evals/evals.py#L164)
over `EvalGroup` menus (`core_evals()`). Train and eval ride the post-Executor lazy `ArtifactStep`
model (#6649) — **identity is `name@version`, path is `{prefix}/{name}/{version}`**, not
content-addressed — while datakit uses the older content-addressed `StepSpec` layer; both execute
under `StepRunner`. That split is the source of the seam below.

## Challenges

_What's hard?_

The real difficulty is the **datakit→train seam**, and it is deeper than reshaping an output.
Two execution primitives coexist. Datakit is built on **`StepSpec`** (content-addressed
`{prefix}/{name}_{hash}/`, run by `StepRunner`);
[`reference_datakit_steps()`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/datakit/reference_pipeline.py#L465)
returns a `DatakitSteps` whose `output_buckets` is the terminal store step. Training and eval are
authored as **`ArtifactStep`** handles (`name@version`) that *lower* to `StepSpec`;
[`train_lm(datasets=...)`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/lib/marin/src/marin/experiment/train.py#L106)
wants `Mapping[ArtifactStep[TokenizedCache], float]`. Bridging them hits two hard walls:

1. **The non-empty bucket set is only known after datakit runs.**
   [`ClusteredStoreData.buckets`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/datakit/store/datakit_store.py#L91)
   omits buckets that received zero rows, so the training mixture's dataset list cannot be
   enumerated at authoring time.
2. **The store's buckets are *flat* Levanter caches, not `TokenizedCache`s.** Each bucket is
   `cluster=C/quality=Q/part-*` with a merged `shard_ledger.json` at its root — no `train/` subdir.
   Routing such a bucket through `train_lm(datasets=...)` → `mixture()` →
   [`TokenizedCache.as_component()`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/lib/marin/src/marin/processing/tokenize/tokenize.py#L118)
   omits `flat_cache` (defaults
   [`False`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/lib/levanter/src/levanter/data/text/datasets.py#L335));
   Levanter then looks for a `<bucket>/train/` cache
   ([`datasets.py:924`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/lib/levanter/src/levanter/data/text/datasets.py#L924)),
   doesn't find one, and **silently skips the component** — training on an empty mixture. The only
   shape that works is a hand-built `flat_cache=True` `DatasetComponent`, exactly what
   [`launch_datakit_moe_mix.py:258`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/grug/moe/launch_datakit_moe_mix.py#L258)
   does. So the `TokenizedCache`/`train_lm` path is a dead end here; the store data config must be
   assembled the moe way.

Together these mean the training data config must be built as per-bucket `flat_cache=True`
components, and because the bucket set is dynamic the reference is a two-pass driver, not a single
`build()`. Today that stitch is manual: the moe launcher hardcodes a store hash
(`datakit/store_8ac06c74`) and a transcribed 167-row bucket-weight table. Automating it is the crux;
train → eval → report downstream already composes.

## Costs / Risks

- **Churn**: retiring `reference_training_pipeline.py` and folding its role in; any docs/CI pointing
  at it must be updated (no back-compat shim, per repo policy).
- **A reference is a maintenance commitment**: to be trustworthy it must stay green, which is why
  the issue wants it canaried. That cost is deferred (CI is a follow-up) but real.
- **Eval adds a serve child**: evalchemy boots the checkpoint via marin-serve (vLLM), so the
  "smoke" run needs a small accelerator, not pure CPU — modest cost, but not free.
- A nano model scores near chance on most tasks; the readout's value is **delta-vs-baseline and
  path-liveness**, not absolute quality. The doc/runbook must say so to avoid misreading.
- **Lost coverage**: the current reference demonstrates pretrain→midtrain→SFT as one run with a
  step-varying mixture. Replacing it with a single static store mixture drops that demonstration; if
  it's worth keeping, it moves out of the blessed reference (or becomes a follow-up phase-schedule
  variant).

## Design

_How are we doing this?_

One driver, `experiments/references/reference_training_pipeline.py` (rewritten), runs the whole
path in one `--run` invocation via **two `StepRunner` passes joined by a dynamic handoff** — the
shape forced by the two challenges above. Because the datakit store is a `StepSpec` graph and the
bucket set is dynamic, the reference is a bespoke `main()` (like datakit's own), *not* the
`experiment_main(build)` single-handle idiom the current file uses.

```python
def main() -> None:
    version = ...  # calendar version for the shared reference; dev for local iteration

    # 1. datakit pass: testbed sample -> (cluster, quality) store, SMOKE_SCALE.
    #    SAMPLE_PREFIX / sample_sources(...) is single-edit point (a): swap the data source.
    datakit = reference_datakit_steps(
        sample_sources(SAMPLE_PREFIX), quality_model=QUALITY_MODEL,
        quality_model_version=QUALITY_MODEL_VERSION, scale=SMOKE_SCALE,
    )
    StepRunner().run(datakit.all_steps)
    store = read_artifact(datakit.output_buckets.output_path, ClusteredStoreData)

    # 3. train + eval pass. REFERENCE_MODEL is single-edit point (b).
    model = reference_train_on_store(store, model=REFERENCE_MODEL, version=version)
    report = eval_report(eval_steps(model, core_evals(serve=REFERENCE_SERVE)), name=REF_NAME)
    run(report)  # lowers + runs the train->eval->report ArtifactStep graph, resuming cache hits
```

`store_mixture(store)` is the one genuinely new helper: from the resolved `ClusteredStoreData` it
builds the Levanter `LmDataConfig` the trainer consumes — one `flat_cache=True` `DatasetComponent`
per **non-empty** bucket (`cache_dir=bucket.path`, weight ∝ `bucket.total_tokens` or uniform,
`tokenizer=store.tokenizer`) plus the validation caches — the shape `launch_datakit_moe_mix.py`
builds by hand. `reference_train_on_store` wraps that config in the same
[`GrugBaseLaunchConfig` + `run_grug_base_trial`](https://github.com/marin-community/marin/blob/6a6699b7ec34344d70ec5fb6599f1af776a2b0bc/experiments/grug/base/launch.py#L94)
the current reference uses, yielding an `ArtifactStep[LevanterCheckpoint]`. **No `ArtifactStep.adopt`,
no `train_lm`**: the store buckets are referenced as literal, content-addressed paths that bear
identity directly in the training fingerprint, so a datakit change re-fingerprints training.

**Resume and bounded footprint fall out of the two caching layers, not new code.** The datakit pass
caches per `StepSpec` content hash; the train+eval pass caches per `name@version` with execution
knobs (TPU type, pool size, region) in `runtime_args` off the fingerprint. Re-running unchanged →
datakit store hash unchanged → same bucket paths in the training config → train fingerprint
unchanged → cache hit, **no new artifact copies** (satisfying the resumable + object-store-footprint
requirements and #5744/#6897). A training-config edit re-fingerprints only train+eval and reuses the
store; a datakit edit changes the store hash and correctly cascades. The design's job is to preserve
those invariants — no content in `name@version` paths, `runtime_args` for execution knobs,
disposable intermediates under the sample prefix — not to build caching.

A fully-unified single `ArtifactStep[EvalReport]` graph (so `build()` + `experiment_main` still
work) is possible but heavier: it needs the datakit DAG wrapped as one
`ArtifactStep[ClusteredStoreData]` (a nested `StepRunner`) plus a store-aware training step that
reads `ctx.resolved(store).buckets` at run time. That trade — clean composition vs new
cross-primitive machinery — is the lead Open Question.

Defaults (from the settled questions): input `sample_0.1b_7d7d8fd7`; datakit `SMOKE_SCALE`; a
nano/150M model on a small GPU/TPU (`REFERENCE_TRAIN_RESOURCES` via `runtime_args`); readout =
`core_evals()` MCQ (12 loglikelihood tasks — cheap, deterministic, no generation load).

## Testing

_Agents make mistakes — how do we catch them?_

The pipeline **is** the test — the DoD's verifications are the acceptance criteria:

- **Path liveness**: one `--run` invocation completes raw→datakit→pretrain→eval→report and
  produces an `EvalReport` with metrics + a W&B/report link.
- **Resume**: run to completion, then bump the train config and re-run; assert the datakit pass is
  a cache hit end-to-end (StepSpec hashes unchanged, no stage recomputes) and only train+eval
  re-execute in the second pass.
- **Footprint**: re-run with unchanged config; assert no new artifact paths are minted (same
  `datakit/*_<hash>` dirs and same train/eval `name@version` dirs, fingerprints stable). A doc note
  records where intermediates land and their retention.
- **Inject points**: exercise both documented single-edit points (swap `SAMPLE_PREFIX`; change
  `REFERENCE_MODEL`) and confirm each yields a comparable readout with everything else fixed.

CI wiring (a ferry/canary like `datakit_ferry.py`) is a **follow-up** once the harness is reliably
green — out of scope here.

## Open Questions

- **Bridge shape (lead question)**: two-pass driver + `store_mixture` (proposed — least new
  machinery, works today, but gives up the single-`build()`/`experiment_main` shape) vs a unified
  lazy graph (wrap datakit as an `ArtifactStep[ClusteredStoreData]` + a store-aware trainer that
  reads `ctx.resolved(store).buckets`). The unified path composes with rjpower's #7267 "everything
  is a reusable step" direction but crosses the StepSpec/ArtifactStep boundary. Which do the area
  owners want as the blessed pattern?
- **`store_mixture` weighting + degenerate buckets**: token-proportional (`bucket.total_tokens`) vs
  uniform — faithful vs stable-as-a-regression-signal when sample composition drifts. Either way the
  helper must assert `total_tokens > 0` per emitted bucket (a `0`-weight component is silently
  dropped by Levanter), and the token-proportional/uniform default is a real call.
- **Serve tier (blocks path-liveness)**: the least-exercised leg is serving the trained checkpoint
  to evalchemy via marin-serve/vLLM, which wants an **HF-exported** checkpoint. Pin the smallest
  accelerator that serves a nano/150M model acceptably, and decide whether to bump to 150M if
  nano-on-vLLM is awkward. Also: `expected_fingerprint`-pin the datakit store for a canary so silent
  upstream drift *fails* rather than warns?
- **`store_mixture` location**: the reference file, or `experiments/datakit/` next to
  `ClusteredStoreData` (datakit owns its own store→train consumption contract, and the `flat_cache`
  wiring is really a datakit/Levanter concern, not reference-specific)?
