# Research — Durable public hosting for analysis HTML pages

Issue: [marin#6802](https://github.com/marin-community/marin/issues/6802). Code refs pin `main` at `65d1cbef5346de50ef5d08e190765c0a84b0403e`.

## Framing

Devs host one-off analysis HTML on personal sites (William Held's `datakit_sidebyside.html` on `oa.williamheld.com`, Ahmad's `delphi-midtraining` dossier on `ahmeda14960.github.io`). These are meant to be permalinks but die when the author leaves or reorganizes, and aren't discoverable/referenceable from code. Goal: a paved path to publish analysis HTML to a durable public GCS location and register it as an Artifact.

## Load-bearing findings

### The `marin-public` bucket already exists and is world-readable

Confirmed live with `gcloud storage`:

- `gs://marin-public` — location `US-WEST2`, project `hai-gcp-models`, **uniform bucket-level access ON**.
- IAM binding `roles/storage.objectViewer → allUsers` — every object is publicly readable over `https://storage.googleapis.com/marin-public/<key>`.
- `publicAccessPrevention` is not enforced.
- Already serving `gs://marin-public/held/datakit_sidebyside.html` (the issue's worked example).

The bucket is single-region US-WEST2 (not part of the region-mirrored data set) and lives in a different project's IAM. **Nothing in the repo references it** — it's provisioned out of band, and not declared in [`infra/configure_buckets.py`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/infra/configure_buckets.py#L68) (which owns TTL rules on the private `marin-<region>` buckets). Counter-signal: [`storage-bucket.md:48`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/docs/tutorials/storage-bucket.md#L48) steers users to "keep the bucket private unless you intentionally publish checkpoints." A public paved path runs against that default and must say so.

### The Artifact model, and the registry that was added then removed (correction)

**An earlier draft of this doc got this wrong** — it called `Artifact.from_id` + `ArtifactRegistry` "proposal-only, not yet built." The real history:

- **#6061** ([`eb21d9505`](https://github.com/marin-community/marin/commit/eb21d9505), 2026-05-30) **added** `Artifact.from_id` + `ArtifactRegistry` — a real `lib/marin/src/marin/execution/artifact_registry.py` (321 lines) with `register`/`lookup`, `ArtifactEntry`, `get_default_registry`, and 277 lines of tests. It shipped on main.
- **#6649** ([`f7f5535c3`](https://github.com/marin-community/marin/commit/f7f5535c3), **2026-06-30**) — "Replace the Executor with lazy ArtifactSteps" — **deleted it on purpose.** Commit message: *"One record … subsumes the old payload sidecar **and the registry record** — `artifact_registry.py` is deleted."* Both the module and its tests were removed; #6649 is on latest main. So the registry was intentionally retired as redundant with `name@version` addressing, not "not yet built."

The current model (post-#6649) is what the design builds on:

- A handle is an inert **`ArtifactStep[T]`** addressed by `name@version` ([`lazy.py:210`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L210)). Constructing one runs nothing.
- The realized output is an **`Artifact`** ([`artifact.py:82`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/artifact.py#L82)) carrying `path` + `record`. Loaded by path with `Artifact.raw_load(source)` ([`artifact.py:114`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/artifact.py#L114)); `raw_load` checks `record.result_type` against the loader.
- A successful step writes **one `ArtifactRecord` to `{output_path}/artifact.json`** (identity, deps, config, typed `result`, provenance). `write_record`/`write_artifact` are the public writers ([`artifact.py:228,261`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/artifact.py#L228)).
- **`ArtifactStep.adopt(name, version, source, *, kind=Artifact, config=None)`** ([`lazy.py:286`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L286)) — "register pre-existing data at `source` as a managed `name@version` handle." This is the shipping registration mechanism.
- **There is no common/global registry.** `adopt` does not write to any shared index. Constructing the handle writes nothing (`.path()` returns `source`); only materializing it (`resolve()` / the `StepRunner`) writes a single `ArtifactRecord` to the canonical **`{marin_prefix()}/{name}/{version}/artifact.json`** — under the *active cluster's region-private* prefix, with `source` recorded (runner at [`lazy.py:377`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L377)). Resolution is by reconstructing the deterministic `name@version` path, not by lookup.
- **Versions are `YYYY.MM.DD[.N]`** (or mutable `dev`/`<label>-dev`), validated at construction (`_CALVER_RE`, [`lazy.py:62`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L62)) — **not** the `YYYY.MM.DD[-suffix]` the deleted registry used.
- `name` is a path segment that **allows internal slashes** (only `..`, URL schemes, and leading/trailing `/` are rejected — `_validate_segment`), so `pages/<user>/<slug>` is a valid `name` directly.

### An 80%-done upload prototype already exists

`experiments/datakit/dedup/ops/render_cluster_report.py` renders a self-contained HTML report and, with `--upload gs://…`, pushes it to a bucket via `rigging.filesystem.url_to_fs` ([:345](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/experiments/datakit/dedup/ops/render_cluster_report.py#L345)):

```python
if args.upload:
    fs, p = url_to_fs(args.upload)
    with fs.open(p, "wb") as fh:
        fh.write(body.encode("utf-8"))
```

A hand-rolled instance of exactly the paved path — build on it, don't greenfield. Note it sets **no** content-type.

### Upload / fsspec building blocks

- `rigging.filesystem`: `url_to_fs(url)`, `open_url(url, mode, **kwargs)`, `marin_prefix()`. `open_url`/`url_to_fs` forward `**kwargs` to the filesystem; **verified** that gcsfs 2026.1.0 threads `content_type` from `_open` → `GCSFile(content_type=…)`, so `open_url(dst, "wb", content_type=...)` sets the object's Content-Type.
- Nearest prior art for "grant `allUsers` read": `lib/levanter/src/levanter/infra/docker.py` grants `allUsers` `roles/artifactregistry.reader` on Docker images (GCP Artifact Registry — unrelated to the marin artifact system).

### Docs surface

- `mkdocs.yml` nav is the source of truth. Reports live under **Experiments** → `reports/index.md` (entries are GitHub-issue + WandB + Data-Browser links, never hosted HTML).
- Natural home for a "how to publish an analysis page" doc: `docs/tutorials/` next to `storage-bucket.md`. Discoverability: link from `docs/reports/index.md`.

## What surprised me / correction

- The registry the issue's phrasing implies (`from_id`) was real and on main, then **deliberately deleted one day before this doc was drafted** (#6649). The first draft mistook the post-deletion crater for an empty lot and called it "proposal-only." The correct current mechanism is `ArtifactStep.adopt` + the single `artifact.json` record.
- `adopt` writes to a *private* per-cluster prefix, not a shared registry — so for a *public* page we write the record into the public dir directly (via `write_record`) to keep it self-describing and independent of `marin_prefix()`.
- The `marin-public` bucket + the exact worked-example object already exist and work; the gap is convention + helper + docs, not new storage infra.
