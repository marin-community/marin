# Durable public hosting for analysis HTML pages

_Why are we doing this? What's the benefit?_

Devs publish one-off analysis pages — William Held's [`datakit_sidebyside.html`](https://oa.williamheld.com/datakit_sidebyside.html), Ahmad's [delphi-midtraining dossier](https://ahmeda14960.github.io/delphi-midtraining/dossier.html) — on personal sites. They're meant to be permalinks, but personal GitHub Pages / domains die when the author leaves or reorganizes, and they can't be discovered or fetched from code. We want a paved path: publish an analysis HTML page to a durable, public location and record it as an Artifact, so the same page is both a stable public URL and a code-resolvable handle.

The substrate already exists. `gs://marin-public` (US-WEST2, project `hai-gcp-models`) grants `roles/storage.objectViewer` to `allUsers`, so every object is readable at `https://storage.googleapis.com/marin-public/<key>`. It already serves `held/datakit_sidebyside.html` — the issue's worked example. The gap is convention, a helper, docs, and a code-fetch contract, not new storage infra.

## Background

Full survey with `file:line` refs: [`research.md`](./research.md). The load-bearing facts: (1) `gs://marin-public` exists and is world-readable today, provisioned out of band. (2) Marin's `Artifact` model addresses artifacts by an explicit `name@version` and writes **one `artifact.json` record** per output ([`artifact.py:82`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/artifact.py#L82)). A prior `Artifact.from_id` + `ArtifactRegistry` (a global `(id,version)→uri` lookup) shipped in #6061 and was **deliberately removed** in the lazy-`ArtifactStep` rewrite (#6649, 2026-06-30) as redundant — so there is **no global registry**; resolution is by deterministic addressing. (3) A hand-rolled `--upload gs://` HTML publisher already lives in [`render_cluster_report.py:345`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/experiments/datakit/dedup/ops/render_cluster_report.py#L345).

## Challenges

_What's hard?_

The mechanics (upload bytes, print a URL) are trivial — the difficulty is the code-fetch contract and two storage quirks.

- **There is no lookup, only addressing — so the path must be deterministic.** Post-#6649 there is no `(id,version)→uri` registry; an artifact is found by reconstructing its address, not by querying an index. That's workable *only* because a public page's location is a pure function of its identity: a fixed bucket (`marin-public`) plus `<user>/<slug>/<version>/`. So `site_uri(user, slug, version)` is pure string construction, and `Artifact.raw_load(site_uri(...))` is the whole fetch path — no registry, no `marin_prefix()` (which varies per cluster and would make a computed artifact's address cluster-relative). A record written into that public directory makes `raw_load` return a typed handle with provenance rather than a bare path.
- **The version is part of the address, not an immutability guarantee.** The path carries the version (`…/<slug>/<version>/index.html`) so a new version publishes alongside old ones without clobbering their links. We don't enforce immutability at v1: re-publishing the same version overwrites (last-writer-wins). Versions are CalVer `YYYY.MM.DD[.N]` — the format the current model validates ([`lazy.py:62`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L62)).
- **GCS doesn't serve `index.html` at a bare directory URL.** `…/marin-public/held/datakit-sidebyside/<version>/index.html` serves, but `…/<version>/` returns not-found from the standard `storage.googleapis.com` endpoint (no `MainPageSuffix` without a bucket Website config). The canonical public URL is the explicit `…/index.html`; pretty directory URLs are a deferred nicety.
- **Browsers need the right `Content-Type`.** The existing `fs.open(p, "wb")` prototype sets none, so a page could land as `application/octet-stream` and download instead of render. `publish_site` sets `content_type` explicitly from a pinned extension table (html/css/js/json/wasm/map/svg + a default), not the platform `mimetypes` DB, whose `.js`/`.wasm` mappings drift across systems.

## Costs / Risks

- Introduces a documented public-write path that runs against [`storage-bucket.md:48`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/docs/tutorials/storage-bucket.md#L48)'s "keep buckets private" default. Docs must be explicit that `marin-public` is world-readable — never publish anything sensitive.
- The public bucket's IAM stays provisioned out of band (documented, not managed in `infra/configure_buckets.py`), so the public-read grant is not reproducible from the repo. Accepted tradeoff to avoid `configure_buckets.py` churn; noted as an Open Question.
- The discovery index (`index.json`) is maintained by a non-atomic read-modify-write, so a rare concurrent double-publish can drop one entry. Acceptable given publish frequency; atomic updates are out of scope.
- No content review, no GC, no per-handle ownership: anyone with bucket write can publish under any handle; stale sites linger. Out of scope at v1.

## Design

_How are we doing this?_

**Path layout — versioned directory per page.** A page, its record, and sibling assets (CSS/JS for a small SPA) live under a version-pinned directory:

```
gs://marin-public/<user>/<slug>/<version>/index.html
gs://marin-public/<user>/<slug>/<version>/artifact.json   # ArtifactRecord, so raw_load resolves
gs://marin-public/<user>/<slug>/<version>/<asset>         # optional js/css/data
```

`<user>` is the author's short handle (`held`, `rav`) and `<slug>` is kebab-case — both coerced to that shape from whatever the caller passes (`"Foo Bar"` → `foo-bar`), `<version>` is CalVer `YYYY.MM.DD[.N]`. Public URL is the explicit index: `https://storage.googleapis.com/marin-public/<user>/<slug>/<version>/index.html`.

**Helper — one library function, thin CLI wrapper.** New module `lib/marin/src/marin/publish/sites.py`:

"site", not "page" (per rjpower): `source` may be a single `.html` *or* a directory (a multi-file SPA — HTML + JS/CSS/data).

```python
def publish_site(
    source: Path,                # a local .html file, or a local dir whose index.html is the entrypoint
    *, user: str, slug: str,
    version: str,                # CalVer YYYY.MM.DD[.N]
    title: str,                  # human label for the discovery index
    summary: str = "",           # optional one-line description
) -> PublishedSite:              # {url, uri, name, version, title}
    ...
```

It uploads each file via `rigging.filesystem.open_url(dst, "wb", content_type=...)` — verified to set the object's Content-Type through gcsfs's `GCSFile(content_type=…)` kwarg (gcsfs 2026.1.0) — then (1) writes the record into the *same public directory* so the site is self-describing, and (2) upserts an entry into the public discovery index:

```python
write_record(ArtifactRecord(
    name=f"sites/{user}/{slug}", version=version,
    output_path=uri, result_type=result_type_name(Artifact),
    config={"user": user, "slug": slug, "version": version, "url": url, "title": title, "summary": summary},
    provenance=Provenance.capture(),
))
```

Writing the record *into the public dir* (rather than letting `adopt` write it into a cluster-private `marin_prefix()`) keeps the artifact self-contained and independent of where it's published from. A `scripts/ops/publish_site.py` CLI wraps the function and prints the public URL:

```
uv run scripts/ops/publish_site.py report/ --user rav --slug dedup-examples --version 2026.07.01 --title "Dedup examples"
# -> https://storage.googleapis.com/marin-public/rav/dedup-examples/2026.07.01/index.html
```

**Discovery index (per rjpower).** `publish_site` also upserts into a public manifest `gs://marin-public/index.json` — a JSON list of `{name, version, url, title, summary}`. It's a read-modify-write (read → upsert `(name, version)` → write); publishes are rare enough that the low conflict chance makes last-writer-wins acceptable (atomic index updates are out of scope). This is the programmatic "list all sites" that the removed registry would otherwise have provided; `docs/reports/index.md` renders / links it for humans.

**Fetching from code — deterministic path + `raw_load`.** Since the address is a pure function of identity, no registry is needed:

```python
from marin.publish.sites import site_uri
from marin.execution.artifact import Artifact
page = Artifact.raw_load(site_uri("held", "datakit-sidebyside", "2026.07.01"))
# page.path == "gs://marin-public/held/datakit-sidebyside/2026.07.01/"; page.record carries provenance + config
```

`site_url(user, slug, version)` and `site_uri(user, slug, version)` are pure string constructors; `raw_load` reads the record we wrote (its `result_type` matches base `Artifact`, so the load's type check passes).

**Deferred — a graph-dependency handle.** A `site_artifact(user, slug, version)` returning `ArtifactStep.adopt(f"sites/{user}/{slug}", version, site_uri(...), kind=Artifact)` ([`lazy.py:286`](https://github.com/marin-community/marin/blob/65d1cbef5346de50ef5d08e190765c0a84b0403e/lib/marin/src/marin/execution/lazy.py#L286)) would let a *step graph* depend on a site (so its `name@version` enters a consumer's fingerprint). Not shipped in v1: it would pull the heavy `lazy`/Fray import onto the publish path and has no consumer. Fetching needs only `site_uri` + `raw_load`; see Open Questions.

**`render_cluster_report.py` — not migrated (per rjpower).** Its report embeds **sampled corpus text** (potentially copyrighted), and the paved path here is world-readable. The conservative call is to leave `render_cluster_report` as-is in v1; sensitive pages belong behind a future `--private` variant that routes to an IAP-gated frontend (see Open Questions). So this design touches only the public helper + Held's worked example, not the dedup report.

**Docs.** A new tutorial `docs/tutorials/publish-analysis-site.md` (added to `mkdocs.yml` nav next to `storage-bucket.md`) covers the layout, the helper, the public-URL shape, the `raw_load` fetch, and a bold "public bucket — nothing sensitive, any handle is unauthenticated" warning. `docs/reports/index.md` gets a "Published analysis sites" section rendered from `index.json`.

**Worked example.** Migrate Held's `datakit_sidebyside.html` to `gs://marin-public/held/datakit-sidebyside/<version>/index.html` via the helper and link it from the docs as the reference. (Smoke-tested live — see the PR thread.)

## Testing

_Agents make mistakes — how do we catch them?_

- Unit: `publish_site` against a local-FS / `memory://` root — asserts files land at `<user>/<slug>/<version>/`, the `artifact.json` record carries `name = sites/<user>/<slug>` and `result_type = Artifact`, a multi-file source dir keeps its relative asset paths, `index.json` gains one entry with the `title`/`summary` (and a re-publish upserts, not duplicates), mixed-case/spaced handles are coerced (`"Foo Bar"` → `foo-bar`), and the returned `url` matches the public template. Rejected before any upload: a handle that coerces to nothing (empty, `!!!`), missing `index.html`, non-CalVer version (`v1`, `2026.7.1`), empty `title`, a symlink in the source dir, and a caller-supplied `artifact.json`.
- Round-trip: after `publish_site`, `Artifact.raw_load(site_uri(...))` returns a handle whose `.path` is the version dir and whose `.record.config` matches; `site_artifact(...).path()` equals the same uri.
- The migration is the end-to-end check: after publishing Held's page, fetch the `…/index.html` URL and confirm 200 **and `Content-Type: text/html`** + expected bytes. Known coverage hole: the `memory://` unit test can validate placement but not the served content-type or the bare-directory serving quirk — those two genuinely risky behaviors are only exercised by this manual ops step, not CI.

## Open Questions

- **A `--private` variant for sensitive pages?** rjpower's steer: don't route corpus-text pages (like the dedup report) to the public bucket; a `--private` mode that publishes behind an IAP-gated frontend would be the right home. Out of scope for v1 — but is it the shape we want, and does the public `publish_site` API need to anticipate it (shared path layout, an `index` split, etc.)?
- **Pretty directory URLs.** Serving `…/<version>/` (no `index.html`) needs a bucket Website config (`MainPageSuffix=index.html`) or a CNAME/load-balancer. Worth the infra, or is the explicit `…/index.html` URL fine?
- **Should the public-read IAM be codified in `infra/configure_buckets.py`** (reproducible grant, at the cost of that script owning a public bucket), rather than documented-only? Leaning documented-only, but reviewers may want it managed.
- **Is the `site_artifact()` adopt handle worth adding?** Deferred in v1 (fetch works via `site_uri` + `raw_load`, and it would drag `lazy`/Fray onto the publish path). Add it when a step graph actually wants a site as a typed `name@version` dependency?
