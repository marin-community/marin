# Spec — Durable public hosting for analysis HTML pages

Contract layer for [`design.md`](./design.md). Pins the public surface: the helper API, the CLI, the path/URL templates, the record shape, content types, and error types. No registry — resolution is by deterministic path (see `design.md` Background).

## Constants

```python
# lib/marin/src/marin/publish/sites.py
PUBLIC_BUCKET = "marin-public"                                   # gs://marin-public, allUsers objectViewer
PUBLIC_URL_BASE = "https://storage.googleapis.com/marin-public"  # <base>/<user>/<slug>/<version>/index.html
PUBLIC_ROOT = f"gs://{PUBLIC_BUCKET}"                            # storage root (a module attr, overridable in tests)
SITE_ENTRYPOINT = "index.html"                                   # the served entrypoint under <slug>/<version>/
SITE_NAME_PREFIX = "sites"                                       # ArtifactRecord.name = sites/<user>/<slug>
PUBLIC_INDEX_KEY = "index.json"                                  # gs://marin-public/index.json — discovery manifest

# <user>/<slug> are coerced to lowercase kebab (non-[a-z0-9] runs -> '-', trimmed); rejected only
# if nothing usable remains. e.g. "Foo Bar" -> "foo-bar", "datakit_sidebyside" -> "datakit-sidebyside".
_NON_KEBAB = r"[^a-z0-9]+"

# Content types are set explicitly, NOT via platform `mimetypes` (whose .js/.wasm mappings drift).
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "text/javascript; charset=utf-8",
    ".mjs":  "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map":  "application/json",
    ".svg":  "image/svg+xml",
    ".wasm": "application/wasm",
    ".png":  "image/png",
    ".woff2": "font/woff2",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"
```

### Identifiers

- **URI (record location + fetch address):** `gs://marin-public/<user>/<slug>/<version>/` — a version-pinned directory. Because `marin-public` is fixed and the path is a pure function of `(user, slug, version)`, this address needs no lookup.
- **Public URL:** `<PUBLIC_URL_BASE>/<user>/<slug>/<version>/index.html`.
- **Artifact name:** `sites/<user>/<slug>` (recorded as `ArtifactRecord.name`; also the `name` of the optional `adopt` handle). Internal slashes are legal in a `name` segment (`_validate_segment` rejects only `..`, URL schemes, and leading/trailing `/`), so no separator encoding is needed.
- **Version:** CalVer `YYYY.MM.DD[.N]`. `publish_site` validates the shape via the shared `marin.execution.artifact.CALVER_RE` (the one canonical CalVer form, also used by `lazy._validate_version`), rejecting `v1`, `2026.7.1`, etc., before any upload. Unlike the execution layer it does not accept mutable `dev` versions — a durable public site should pin a real date.

## Public API — `lib/marin/src/marin/publish/sites.py`

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishedSite:
    """The result of a publish: where the site lives and how to reference it.

    ``url`` is the canonical public link (always the explicit ``index.html``, since GCS does not
    serve a bare directory URL). ``uri`` is the version-pinned ``gs://`` directory holding the site
    and its ``artifact.json`` record. ``name``/``version`` are the artifact identity; ``title`` is
    the human label recorded in the discovery index.
    """
    url: str            # https://storage.googleapis.com/marin-public/<user>/<slug>/<version>/index.html
    uri: str            # gs://marin-public/<user>/<slug>/<version>/
    name: str           # sites/<user>/<slug>
    version: str        # CalVer as supplied
    title: str


def publish_site(
    source: Path,
    *,
    user: str,
    slug: str,
    version: str,
    title: str,
    summary: str = "",
) -> PublishedSite:
    """Upload an analysis site to ``gs://marin-public/<user>/<slug>/<version>/``, record it, and
    add it to the public discovery index.

    ``source`` is either a single local ``.html`` file — uploaded as ``index.html`` — or a local
    directory whose ``index.html`` is the entrypoint (a multi-file site: HTML + JS/CSS/data); every
    file under the directory is uploaded preserving relative paths. Each object's ``Content-Type``
    is set from ``CONTENT_TYPES`` by extension (default ``application/octet-stream``) via
    ``open_url(dst, "wb", content_type=...)``; gcsfs forwards ``content_type`` to ``GCSFile`` and
    sets the object metadata (verified against gcsfs 2026.1.0).

    After the bytes are uploaded:
      1. writes the record into the same public directory via ``write_record(ArtifactRecord(
         name=f"sites/{user}/{slug}", version=version, output_path=uri,
         result_type=result_type_name(Artifact), config={"user","slug","version","url","title",
         "summary"}, provenance=Provenance.capture()))`` — so ``Artifact.raw_load(uri)`` returns a
         typed handle with provenance. The record lives in the public dir (not a cluster-private
         ``marin_prefix()``), keeping the artifact self-contained.
      2. upserts an entry into the discovery index at ``gs://marin-public/index.json`` (read →
         upsert ``(name, version)`` → write). The index is a JSON list of
         ``{name, version, url, title, summary}``. This read-modify-write is **not atomic**;
         concurrent publishes are last-writer-wins (acceptable given how rarely sites are published).

    ``user``/``slug`` are coerced to lowercase kebab (``"Foo Bar"`` → ``"foo-bar"``); the coerced
    form is what appears in the path, record, and index.

    Preconditions, all raised **before any upload**:
      * ``user``/``slug`` must retain at least one usable ``[a-z0-9]`` character after coercion —
        else :class:`InvalidSiteError`.
      * ``version`` must be valid CalVer ``YYYY.MM.DD[.N]`` — else :class:`InvalidSiteError`.
      * ``title`` must be non-empty — else :class:`InvalidSiteError` (every indexed site needs a label).
      * a ``source`` directory must contain ``index.html``, must not contain a file named
        ``artifact.json`` (reserved), and must contain no symlinks (which could escape the tree) —
        else :class:`InvalidSiteError`. A missing ``source`` path raises :class:`FileNotFoundError`.

    Existing objects at the destination are overwritten (last-writer-wins) — no immutability guard
    at v1. Distinct versions live at distinct paths, so a new version never clobbers an old one.

    Returns a :class:`PublishedSite`. Does not delete or GC prior versions.
    """


def site_url(user: str, slug: str, version: str) -> str:
    """Canonical public URL: ``<PUBLIC_URL_BASE>/<user>/<slug>/<version>/index.html``.
    Pure string construction; coerces handles, validates version, does not check existence."""


def site_uri(user: str, slug: str, version: str) -> str:
    """The page's ``gs://`` directory (record location + ``raw_load`` address):
    ``gs://marin-public/<user>/<slug>/<version>/``. Pure; coerces handles, validates version."""


def site_name(user: str, slug: str) -> str:
    """The artifact name ``sites/<user>/<slug>``. Pure; coerces handles."""
```

`site_artifact()` (an `ArtifactStep.adopt(...)` handle for step-graph deps) is **deferred** — it would pull the heavy `marin.execution.lazy` / Fray import onto the publish path, has no consumer, and remains an Open Question in `design.md`. Fetching needs only `site_uri` + `raw_load`.

### Fetching from code

No registry, no `from_id`. The address is deterministic, so:

```python
from marin.publish.sites import site_uri
from marin.execution.artifact import Artifact
site = Artifact.raw_load(site_uri("held", "datakit-sidebyside", "2026.07.01"))
# site.path == "gs://marin-public/held/datakit-sidebyside/2026.07.01/"
# site.record.config == {"user", "slug", "version", "url", "title", "summary"}
```

Discovery: read `gs://marin-public/index.json` for the list of all published sites (`{name, version, url, title, summary}`).

## Errors

```python
class InvalidSiteError(ValueError):
    """Raised before any network/GCS access for: a ``user``/``slug`` that coerces to nothing
    (e.g. ``""`` or ``"!!!"``); a non-CalVer ``version``; an empty ``title``; or a source directory
    that lacks ``index.html``, contains a reserved ``artifact.json``, or contains a symlink."""
```

Propagated unchanged (not wrapped): `FileNotFoundError` (missing `source`) and any `rigging.filesystem` upload error.

## CLI — `scripts/ops/publish_site.py`

Thin wrapper over `publish_site`. Prints the public URL to stdout on success.

```
uv run scripts/ops/publish_site.py <source> --user <user> --slug <slug> --version <YYYY.MM.DD[.N]> --title <title> [--summary <summary>]

  <source>        local .html file or directory with index.html
  --user          author handle (lowercase kebab)
  --slug          site slug (lowercase kebab)
  --version       CalVer version string
  --title         human label for the discovery index (required, non-empty)
  --summary       one-line description for the index (optional)

  stdout on success:   <url>
```

Exit 0 on success; non-zero on `InvalidSiteError` / `FileNotFoundError` (message on stderr).

## `render_cluster_report.py` — not migrated in v1

Deliberately **not** rerouted through `publish_site`. Its report embeds sampled corpus text
(potentially copyrighted), and the paved path here targets the world-readable `marin-public`.
Per review (rjpower), the conservative call is to leave it as-is; a future `--private` variant that
routes sensitive pages to an IAP-gated frontend is the right home for that content (Open Question in
`design.md`). `render_cluster_report`'s existing `--upload gs://…` is untouched.

## Docs

- New `docs/tutorials/publish-analysis-site.md`, added to `mkdocs.yml` `nav:` under **Tutorials**, adjacent to `storage-bucket.md`. Contains the layout, `publish_site`/CLI usage, the public-URL shape, the `raw_load` fetch snippet, and a bold "public bucket — publish nothing sensitive; any handle is unauthenticated" warning.
- `docs/reports/index.md` gains a "Published analysis sites" section — the human-facing view of `gs://marin-public/index.json`, starting with the migrated Held site.

## Discovery index — `gs://marin-public/index.json`

A public JSON manifest, updated by `publish_site` (read → upsert `(name, version)` → write; not atomic, last-writer-wins). Shape:

```json
[
  {"name": "sites/held/datakit-sidebyside", "version": "2026.07.01",
   "url": "https://storage.googleapis.com/marin-public/held/datakit-sidebyside/2026.07.01/index.html",
   "title": "DataKit side-by-side", "summary": "..."}
]
```

## File summary

| Path | What |
| --- | --- |
| `lib/marin/src/marin/publish/__init__.py` | new package |
| `lib/marin/src/marin/publish/sites.py` | `publish_site`, `site_url`, `site_uri`, `site_name`, `PublishedSite`, `InvalidSiteError`, constants (`site_artifact` deferred) |
| `scripts/ops/publish_site.py` | CLI wrapper |
| `docs/tutorials/publish-analysis-site.md` | new tutorial |
| `mkdocs.yml` | nav entry for the tutorial |
| `docs/reports/index.md` | "Published analysis sites" section |
| `tests/publish/test_sites.py` | unit + round-trip tests |
| `gs://marin-public/index.json` | discovery manifest (data, not code) |

## Out of scope (v1)

- Managing `marin-public` IAM / lifecycle in `infra/configure_buckets.py` — the bucket stays provisioned out of band (Open Question in `design.md`).
- Pretty directory URLs (`…/<version>/` without `index.html`) — needs a bucket Website config or CNAME.
- A **private** publishing variant (IAP-gated frontend) for sensitive pages like the dedup report — the conservative future path (Open Question in `design.md`); v1 is public-only.
- Ownership/auth on `<user>` handles, content review, deletion/GC of stale sites, and atomic/concurrent-safe index updates.
- Reintroducing any `(id,version)→uri` registry or `Artifact.from_id` — deliberately removed in #6649; this design does not bring it back.
- Templating/rendering of the HTML itself — callers bring their own bytes.
