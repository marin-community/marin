# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for marin.publish.sites, against a local filesystem root."""

import json
import os
from typing import ClassVar

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.execution.artifact import RECORD_FILENAME, Artifact
from marin.publish import sites
from marin.publish.sites import InvalidSiteError, publish_site, site_uri


@pytest.fixture
def public_root(tmp_path, monkeypatch):
    """Point the public storage root at a local temp dir; the URL base stays the real endpoint."""
    root = tmp_path / "marin-public"
    root.mkdir()
    monkeypatch.setattr(sites, "PUBLIC_ROOT", str(root))
    return root


def test_publish_single_file_records_indexes_and_round_trips(public_root, tmp_path):
    src = tmp_path / "report.html"
    src.write_text("<h1>hi</h1>")

    site = publish_site(
        src, user="held", slug="datakit-sidebyside", version="2026.07.01", title="DataKit", summary="side by side"
    )

    # uploaded to the versioned path; returned url is the explicit index
    version_dir = public_root / "held" / "datakit-sidebyside" / "2026.07.01"
    assert (version_dir / "index.html").read_text() == "<h1>hi</h1>"
    assert site.url == "https://storage.googleapis.com/marin-public/held/datakit-sidebyside/2026.07.01/index.html"

    # recorded as an Artifact, fetchable from code by deterministic path (raw_load would raise
    # ArtifactTypeMismatchError if the record's result_type were wrong, so returning proves it)
    fetched = Artifact.raw_load(site_uri("held", "datakit-sidebyside", "2026.07.01"))
    assert fetched.record is not None
    assert fetched.record.name == "sites/held/datakit-sidebyside"
    assert fetched.record.config["title"] == "DataKit"
    assert fetched.record.config["summary"] == "side by side"

    # listed in the public discovery index
    index = json.loads((public_root / "index.json").read_text())
    assert index == [
        {
            "name": "sites/held/datakit-sidebyside",
            "version": "2026.07.01",
            "url": site.url,
            "title": "DataKit",
            "summary": "side by side",
        }
    ]


def test_publish_directory_preserves_asset_layout(public_root, tmp_path):
    src = tmp_path / "site"
    (src / "assets").mkdir(parents=True)
    (src / "index.html").write_text("<script src=assets/app.js></script>")
    (src / "assets" / "app.js").write_text("console.log(1)")

    publish_site(src, user="rav", slug="dossier", version="2026.07.01", title="Dossier")

    version_dir = public_root / "rav" / "dossier" / "2026.07.01"
    assert (version_dir / "index.html").exists()
    assert (version_dir / "assets" / "app.js").read_text() == "console.log(1)"


def test_index_upserts_by_name_and_version(public_root, tmp_path):
    src = tmp_path / "report.html"
    src.write_text("<h1>hi</h1>")

    publish_site(src, user="held", slug="ex", version="2026.07.01", title="First")
    publish_site(src, user="held", slug="ex", version="2026.07.01", title="Second")  # upsert in place
    publish_site(src, user="held", slug="ex", version="2026.07.02", title="Third")  # new version, new entry

    index = {(e["name"], e["version"]): e for e in json.loads((public_root / "index.json").read_text())}
    assert len(index) == 2
    assert index[("sites/held/ex", "2026.07.01")]["title"] == "Second"


def test_handles_are_coerced_to_kebab(public_root, tmp_path):
    src = tmp_path / "report.html"
    src.write_text("<h1>hi</h1>")

    site = publish_site(src, user="Held", slug="DataKit_Side By Side", version="2026.07.01", title="t")

    assert site.name == "sites/held/datakit-side-by-side"
    assert (public_root / "held" / "datakit-side-by-side" / "2026.07.01" / "index.html").exists()


def _txt_file(root):
    (root / "report.txt").write_text("x")
    return root / "report.txt"


def _dir_without_index(root):
    (root / "site").mkdir()
    (root / "site" / "page.html").write_text("x")
    return root / "site"


def _dir_with_reserved_record(root):
    (root / "site").mkdir()
    (root / "site" / "index.html").write_text("x")
    (root / "site" / RECORD_FILENAME).write_text("{}")
    return root / "site"


def _dir_with_symlink(root):
    (root / "site").mkdir()
    (root / "site" / "index.html").write_text("x")
    (root / "secret").write_text("s")
    os.symlink(root / "secret", root / "site" / "link")
    return root / "site"


@pytest.mark.parametrize(
    ("build_source", "expected"),
    [
        (_txt_file, InvalidSiteError),  # a single-file source must be .html
        (_dir_without_index, InvalidSiteError),  # a directory must contain index.html
        (_dir_with_reserved_record, InvalidSiteError),  # the record filename is reserved
        (_dir_with_symlink, InvalidSiteError),  # symlinks could escape the tree
        (lambda root: root / "nope.html", FileNotFoundError),  # missing source
    ],
)
def test_unpublishable_source_rejected(public_root, tmp_path, build_source, expected):
    with pytest.raises(expected):
        publish_site(build_source(tmp_path), user="held", slug="ex", version="2026.07.01", title="t")
    assert not any(public_root.rglob("index.html"))  # nothing was uploaded


@pytest.mark.parametrize(
    ("user", "slug", "version", "title"),
    [
        ("held", "", "2026.07.01", "t"),  # slug coerces to nothing
        ("held", "!!!", "2026.07.01", "t"),  # slug has no usable [a-z0-9]
        ("held", "ex", "v1", "t"),  # non-CalVer version
        ("held", "ex", "2026.07.01", ""),  # empty title
    ],
)
def test_invalid_metadata_rejected(public_root, tmp_path, user, slug, version, title):
    src = tmp_path / "report.html"
    src.write_text("<h1>hi</h1>")
    with pytest.raises(InvalidSiteError):
        publish_site(src, user=user, slug=slug, version=version, title=title)


class _ContentTypeRecordingFS(MemoryFileSystem):
    """Memory FS that records the ``content_type`` passed to ``_open``, as gcsfs would honor it."""

    protocol = "ctrec"
    recorded: ClassVar[dict[str, str | None]] = {}

    def _open(self, path, mode="rb", content_type=None, **kwargs):
        if "w" in mode:
            type(self).recorded[path] = content_type
        return super()._open(path, mode, **kwargs)


def test_publish_pins_content_type_on_stored_objects(tmp_path, monkeypatch):
    """The pinned Content-Type must reach fs.open — without it GCS guesses from the extension and
    drops charset=utf-8, so browsers decode UTF-8 pages as Windows-1252 (mojibake)."""
    fsspec.register_implementation("ctrec", _ContentTypeRecordingFS, clobber=True)
    _ContentTypeRecordingFS.recorded.clear()
    monkeypatch.setattr(sites, "PUBLIC_ROOT", "ctrec://marin-public")

    src = tmp_path / "report.html"
    src.write_text("<h1>héllo</h1>")
    publish_site(src, user="held", slug="ex", version="2026.07.01", title="t")

    recorded = _ContentTypeRecordingFS.recorded
    assert recorded["ctrec://marin-public/held/ex/2026.07.01/index.html"] == "text/html; charset=utf-8"
    assert recorded["ctrec://marin-public/index.json"] == "application/json"
