"""Keep the government HTML downloader scoped and archived bytes auditable."""

import hashlib
import json
from pathlib import Path

import pytest

from experiments.post_training.capability_driven_envs.basic_biology_sources.fetch_sources import (
    MEDLINE_BASE,
    scoped_links,
)
from experiments.post_training.capability_driven_envs.basic_biology_sources.publish_snapshot import manifest_files


def test_links_never_cross_scope_or_follow_media() -> None:
    page = b"""<a href="basics/">section</a>
<a href="https://medlineplus.gov/genetics/conditions/anything">other section</a>
<a href="https://example.com/biology">external</a>
<a href="basics/?download=1">query</a>
<a href="basics/#footnote">same section</a>
<img src="figure.png">"""
    assert scoped_links(page, MEDLINE_BASE, MEDLINE_BASE) == [MEDLINE_BASE + "basics/"]


def test_manifest_rejects_mutated_archive(tmp_path: Path) -> None:
    source = tmp_path / "pages/medlineplus/index.html"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    row = {
        "path": "pages/medlineplus/index.html",
        "url": MEDLINE_BASE,
        "bytes": 6,
        "sha256": hashlib.sha256(b"source").hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"status": "raw_official_html_not_cleared_for_model_input", "pages": [row], "rights_policy_snapshots": {}}
        )
    )
    data, rows = manifest_files(tmp_path)
    assert data and rows == [row]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        manifest_files(tmp_path)
