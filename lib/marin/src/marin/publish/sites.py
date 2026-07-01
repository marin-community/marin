# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an analysis HTML site to durable public hosting and record it as an Artifact.

A "site" is a single ``.html`` file or a directory whose ``index.html`` is the entrypoint (a
multi-file SPA). ``publish_site`` uploads it to ``gs://marin-public/<user>/<slug>/<version>/``,
writes an :class:`~marin.execution.artifact.ArtifactRecord` next to it so it is self-describing
and code-resolvable via ``Artifact.raw_load``, and upserts a public discovery index at
``gs://marin-public/index.json``. Resolution is by deterministic address — a public page's
location is a pure function of ``(user, slug, version)`` — so no registry is involved.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem import open_url, url_to_fs
from rigging.provenance import Provenance

from marin.execution.artifact import (
    CALVER_RE,
    RECORD_FILENAME,
    Artifact,
    ArtifactRecord,
    result_type_name,
    write_record,
)

logger = logging.getLogger(__name__)

PUBLIC_BUCKET = "marin-public"
# Storage root and public URL base. PUBLIC_ROOT is a module attribute so tests can point it at a
# local directory; the served URL is always the storage.googleapis.com endpoint.
PUBLIC_ROOT = f"gs://{PUBLIC_BUCKET}"
PUBLIC_URL_BASE = f"https://storage.googleapis.com/{PUBLIC_BUCKET}"
SITE_ENTRYPOINT = "index.html"
SITE_NAME_PREFIX = "sites"
PUBLIC_INDEX_KEY = "index.json"

# <user>/<slug> are coerced to lowercase-kebab path segments; this matches non-[a-z0-9] runs.
_NON_KEBAB = re.compile(r"[^a-z0-9]+")

# Content types are pinned by extension rather than deferred to the platform ``mimetypes`` DB,
# whose ``.js``/``.wasm``/``.mjs`` mappings drift across systems and would mis-serve SPA assets.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".woff2": "font/woff2",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"


class InvalidSiteError(ValueError):
    """A malformed ``user``/``slug``/``version``/``title`` or an unpublishable ``source`` tree.

    Always raised before any network/GCS access.
    """


@dataclass(frozen=True)
class PublishedSite:
    """Where a published site lives and how to reference it.

    ``url`` is the canonical public link (the explicit ``index.html``); ``uri`` is the versioned
    ``gs://`` directory holding the site and its ``.artifact.json``; ``name``/``version`` are the
    artifact identity; ``title`` is the label recorded in the discovery index.
    """

    url: str
    uri: str
    name: str
    version: str
    title: str


def _coerce_handle(label: str, value: str) -> str:
    """Normalize a ``user``/``slug`` to a lowercase-kebab path segment.

    Lowercases, collapses non-``[a-z0-9]`` runs to a single ``-``, and trims leading/trailing ``-``
    (so ``"Foo Bar"`` → ``"foo-bar"``, ``"datakit_sidebyside"`` → ``"datakit-sidebyside"``). Raises
    :class:`InvalidSiteError` only when nothing usable remains (e.g. ``""`` or ``"!!!"``).
    """
    coerced = _NON_KEBAB.sub("-", value.lower()).strip("-")
    if not coerced:
        raise InvalidSiteError(f"{label} {value!r} has no usable [a-z0-9] characters")
    return coerced


def _validate_version(version: str) -> None:
    if not CALVER_RE.match(version):
        raise InvalidSiteError(f"version {version!r} must be a calendar version YYYY.MM.DD[.N]")


def site_name(user: str, slug: str) -> str:
    """The artifact name ``sites/<user>/<slug>``. Pure; coerces handles to lowercase kebab."""
    return f"{SITE_NAME_PREFIX}/{_coerce_handle('user', user)}/{_coerce_handle('slug', slug)}"


def site_uri(user: str, slug: str, version: str) -> str:
    """The site's ``gs://`` directory (record location + ``raw_load`` address). Pure; coerces handles."""
    _validate_version(version)
    return f"{PUBLIC_ROOT}/{_coerce_handle('user', user)}/{_coerce_handle('slug', slug)}/{version}/"


def site_url(user: str, slug: str, version: str) -> str:
    """The canonical public URL ``<base>/<user>/<slug>/<version>/index.html``. Pure; coerces handles."""
    _validate_version(version)
    return f"{PUBLIC_URL_BASE}/{_coerce_handle('user', user)}/{_coerce_handle('slug', slug)}/{version}/{SITE_ENTRYPOINT}"


def _content_type(relpath: str) -> str:
    return CONTENT_TYPES.get(Path(relpath).suffix.lower(), DEFAULT_CONTENT_TYPE)


def _collect_files(source: Path) -> list[tuple[str, Path]]:
    """Return ``(relpath, local_path)`` pairs to upload, ``relpath`` using ``/`` separators.

    A single ``.html`` file is uploaded as ``index.html``. A directory is uploaded whole, preserving
    relative paths; it must contain ``index.html``, must not contain a top-level ``.artifact.json``
    (reserved for the record), and must contain no symlinks (which could escape the tree).
    """
    if source.is_file():
        if source.suffix.lower() not in (".html", ".htm"):
            raise InvalidSiteError(f"a single-file source must be .html, got {source.name!r}")
        return [(SITE_ENTRYPOINT, source)]

    files: list[tuple[str, Path]] = []
    has_index = False
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise InvalidSiteError(f"symlink not allowed in source: {path.relative_to(source).as_posix()}")
        if path.is_dir():
            continue
        relpath = path.relative_to(source).as_posix()
        if relpath == RECORD_FILENAME:
            raise InvalidSiteError(f"source may not contain a top-level {RECORD_FILENAME!r} (reserved)")
        has_index = has_index or relpath == SITE_ENTRYPOINT
        files.append((relpath, path))
    if not has_index:
        raise InvalidSiteError(f"source directory must contain {SITE_ENTRYPOINT}")
    return files


def _index_uri() -> str:
    return f"{PUBLIC_ROOT}/{PUBLIC_INDEX_KEY}"


def _read_index() -> list[dict]:
    uri = _index_uri()
    fs = url_to_fs(uri, use_listings_cache=False)[0]
    if not fs.exists(uri):
        return []
    with open_url(uri, "r") as f:
        return json.load(f)


def _upsert_index(entry: dict) -> None:
    """Read the public index, replace any entry with the same ``(name, version)``, append, write.

    Read-modify-write; not atomic. Concurrent publishes are last-writer-wins, acceptable given how
    rarely sites are published.
    """
    entries = [e for e in _read_index() if (e.get("name"), e.get("version")) != (entry["name"], entry["version"])]
    entries.append(entry)
    entries.sort(key=lambda e: (e["name"], e["version"]))
    with open_url(_index_uri(), "w", content_type="application/json") as f:
        json.dump(entries, f, indent=2)


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

    ``source`` is a single local ``.html`` file (uploaded as ``index.html``) or a directory whose
    ``index.html`` is the entrypoint; a directory is uploaded whole, preserving relative paths. Each
    object's ``Content-Type`` is set from :data:`CONTENT_TYPES` by extension (default
    :data:`DEFAULT_CONTENT_TYPE`).

    After upload, writes an :class:`ArtifactRecord` into the same public directory (so
    ``Artifact.raw_load(uri)`` resolves a typed handle with provenance) and upserts an entry into
    ``gs://marin-public/index.json``.

    ``user``/``slug`` are coerced to lowercase kebab (``"Foo Bar"`` → ``"foo-bar"``); the coerced
    form is what appears in the path, record, and index. Other preconditions are checked before any
    upload: ``version`` must be CalVer ``YYYY.MM.DD[.N]``, ``title`` must be non-empty, and a
    directory ``source`` must contain ``index.html``, no top-level ``.artifact.json``, and no
    symlinks. A missing ``source`` raises :class:`FileNotFoundError`; a handle that coerces to
    nothing, a bad version, or an empty title raise :class:`InvalidSiteError`. Existing objects are
    overwritten (last-writer-wins); distinct versions live at distinct paths.
    """
    if not title.strip():
        raise InvalidSiteError("title must be non-empty")
    user = _coerce_handle("user", user)
    slug = _coerce_handle("slug", slug)
    name = site_name(user, slug)
    _validate_version(version)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    files = _collect_files(source)

    uri = site_uri(user, slug, version)
    url = site_url(user, slug, version)
    for relpath, local_path in files:
        with open_url(f"{uri}{relpath}", "wb", content_type=_content_type(relpath)) as dst:
            dst.write(local_path.read_bytes())

    config = {"user": user, "slug": slug, "version": version, "url": url, "title": title, "summary": summary}
    write_record(
        ArtifactRecord(
            name=name,
            version=version,
            output_path=uri,
            result_type=result_type_name(Artifact),
            config=config,
            provenance=Provenance.capture(),
        )
    )
    _upsert_index({"name": name, "version": version, "url": url, "title": title, "summary": summary})
    logger.info("published %s@%s -> %s", name, version, url)
    return PublishedSite(url=url, uri=uri, name=name, version=version, title=title)
