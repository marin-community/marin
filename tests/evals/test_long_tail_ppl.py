# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.long_tail_ppl import (
    DIAGNOSTIC_LOGS_ISSUE,
    GAME_MUSIC_ISSUE,
    PACKAGE_METADATA_ISSUE,
    LongTailPplFamily,
    long_tail_ppl_slices,
    long_tail_raw_validation_sets,
    render_long_tail_ppl_registry_markdown,
)
from experiments.defaults import default_raw_validation_sets
from levanter.data.text import HfDatasetSourceConfig
from levanter.data.text import UrlDatasetSourceConfig
from marin.evaluation.perplexity_gap import _to_dataset_component, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec


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


def test_diagnostic_logs_slices_render_with_issue_5093_tag_and_marin_leakage_note():
    datasets = long_tail_raw_validation_sets(
        raw_root="gs://example-bucket/raw/long_tail",
        family=LongTailPplFamily.DIAGNOSTIC_LOGS,
    )

    ghalogs_key = "long_tail_ppl/diagnostic_logs/ghalogs"
    marin_key = "long_tail_ppl/diagnostic_logs/marin_internal_logs_sanitized"

    assert datasets[ghalogs_key].input_path == "gs://example-bucket/raw/long_tail/diagnostic_logs/ghalogs/runs.jsonl.gz"
    expected_tags = ("long_tail_ppl", "epic:5005", f"issue:{DIAGNOSTIC_LOGS_ISSUE}", "diagnostic_logs")
    assert datasets[ghalogs_key].tags == expected_tags
    assert datasets[marin_key].tags == expected_tags

    slice_names = {s.name for s in long_tail_ppl_slices(family=LongTailPplFamily.DIAGNOSTIC_LOGS)}
    # DoD requires at least two public log sources plus a Marin-internal slice.
    assert {"ghalogs", "logchunks", "marin_internal_logs_sanitized"}.issubset(slice_names)

    # Leakage / contamination handling for Marin-owned logs must be documented in-place.
    marin_notes = next(
        s.notes
        for s in long_tail_ppl_slices(family=LongTailPplFamily.DIAGNOSTIC_LOGS)
        if s.name == "marin_internal_logs_sanitized"
    )
    assert "scrub" in marin_notes.lower()
    assert "never" in marin_notes.lower() and "training" in marin_notes.lower()


def test_diagnostic_logs_slices_excluded_from_unfiltered_default_root():
    # The new family must register through the metadata-only registry, not be wired
    # into default_raw_validation_sets. The unfiltered call uses the default raw_root
    # ("raw/long_tail_ppl") and should still surface the new slices, while a
    # family-filter call must not leak entries from other families.
    all_datasets = long_tail_raw_validation_sets()
    diagnostic_only = long_tail_raw_validation_sets(family=LongTailPplFamily.DIAGNOSTIC_LOGS)

    assert "long_tail_ppl/diagnostic_logs/ghalogs" in all_datasets
    assert all(key.startswith("long_tail_ppl/diagnostic_logs/") for key in diagnostic_only)
    assert len(diagnostic_only) >= 3


def test_diagnostic_logs_are_opt_in_and_not_added_to_default_validation_bundles():
    default_validation_sets = default_raw_validation_sets()
    assert all(not key.startswith("long_tail_ppl/diagnostic_logs/") for key in default_validation_sets)


def test_diagnostic_logs_convert_to_gap_finder_dataset_components():
    datasets = long_tail_raw_validation_sets(
        raw_root="gs://example-bucket/raw/long_tail",
        family=LongTailPplFamily.DIAGNOSTIC_LOGS,
    )

    apache_component = _to_dataset_component(datasets["long_tail_ppl/diagnostic_logs/loghub_apache"])

    assert isinstance(apache_component.source, UrlDatasetSourceConfig)
    assert apache_component.source.validation_urls == [
        "gs://example-bucket/raw/long_tail/diagnostic_logs/loghub/apache.jsonl.gz"
    ]
    assert apache_component.split == "validation"
    assert apache_component.tags == ["long_tail_ppl", "epic:5005", f"issue:{DIAGNOSTIC_LOGS_ISSUE}", "diagnostic_logs"]


def test_diagnostic_logs_registry_markdown_includes_issue_link():
    markdown = render_long_tail_ppl_registry_markdown(family=LongTailPplFamily.DIAGNOSTIC_LOGS)

    assert "long_tail_ppl/diagnostic_logs/loghub_apache" in markdown
    assert "github.com/logpai/loghub" in markdown
    assert f"#{DIAGNOSTIC_LOGS_ISSUE}" in markdown
    # Confirm the GAME_MUSIC family does not bleed into a filtered render.
    assert f"#{GAME_MUSIC_ISSUE}" not in markdown
