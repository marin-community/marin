# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.data.text import HfDatasetSourceConfig
from marin.evaluation.perplexity_gap import _to_dataset_component, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import (
    GAME_MUSIC_ISSUE,
    PACKAGE_METADATA_ISSUE,
    LongTailPplFamily,
    long_tail_ppl_slices,
    long_tail_raw_validation_sets,
    render_long_tail_ppl_registry_markdown,
)


def test_long_tail_raw_validation_sets_render_deterministic_paths_and_tags():
    datasets = long_tail_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")

    web_key = "long_tail_ppl/web_markup_image_text/common_crawl_warc"
    game_key = "long_tail_ppl/game_music/lichess_pgn"

    assert datasets[web_key].input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/warc.jsonl.gz"
    assert datasets[web_key].tags == ("long_tail_ppl", "epic:5005", "issue:5056", "web_markup_image_text")
    assert datasets[web_key].text_key == "text"

    assert datasets[game_key].input_path == "gs://example-bucket/raw/long_tail/games/lichess/pgn.jsonl.gz"
    assert datasets[game_key].tags == ("long_tail_ppl", "epic:5005", f"issue:{GAME_MUSIC_ISSUE}", "game_music")


def test_long_tail_registry_rendering_mentions_issue_links():
    markdown = render_long_tail_ppl_registry_markdown(family=LongTailPplFamily.GAME_MUSIC)

    assert "long_tail_ppl/game_music/lichess_pgn" in markdown
    assert "database.lichess.org" in markdown
    assert f"#{GAME_MUSIC_ISSUE}" in markdown


def test_hf_backed_raw_dataset_preserves_requested_split():
    dataset = raw_text_dataset(HfDatasetSpec(id="example/dataset"), text_key="body", split="test")

    component = _to_dataset_component(dataset)

    assert component.split == "test"
    assert component.format.text_key == "body"
    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.splits == ["test"]


def test_package_metadata_family_covers_dod_surfaces():
    """Issue #5061 DoD requires slices for registry JSON, dependency graph rows,
    advisory text, release metadata, and lockfile-like records. Verify each surface
    is registered at least once under the PACKAGE_METADATA family."""

    slices = long_tail_ppl_slices(family=LongTailPplFamily.PACKAGE_METADATA)
    surfaces = {slice_.surface_form for slice_ in slices}

    required_surfaces = {
        "registry_json",
        "dependency_rows",
        "dependency_edges",
        "advisory_osv_json",
        "release_metadata",
        "lockfile",
    }
    missing = required_surfaces - surfaces
    assert not missing, f"PACKAGE_METADATA family is missing DoD surfaces: {missing}"

    for slice_ in slices:
        assert slice_.issue_number == PACKAGE_METADATA_ISSUE
        assert slice_.raw_relative_path.startswith("packages/")


def test_file_backed_raw_dataset_rejects_non_validation_split():
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/eval.jsonl", split="test")
