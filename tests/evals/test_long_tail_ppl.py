# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.data.text import HfDatasetSourceConfig
from marin.evaluation.perplexity_gap import _to_dataset_component, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import (
    CODE_ECOSYSTEM_ISSUE,
    CODE_ECOSYSTEM_LANGUAGES,
    CODE_ECOSYSTEM_LARGE_TARGET_TOKENS,
    CODE_ECOSYSTEM_SMALL_TARGET_TOKENS,
    GAME_MUSIC_ISSUE,
    CodeEcosystemTier,
    LongTailPplFamily,
    _language_to_slug,
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


def test_file_backed_raw_dataset_rejects_non_validation_split():
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/eval.jsonl", split="test")


@pytest.mark.parametrize(
    "language, expected_slug",
    [
        ("Python", "python"),
        ("C", "c"),
        ("C++", "cpp"),
        ("C#", "c_sharp"),
        ("F#", "f_sharp"),
        ("Objective-C", "objective_c"),
        ("Objective-C++", "objective_cpp"),
        ("Visual Basic .NET", "visual_basic_net"),
        ("Java Server Pages", "java_server_pages"),
        ("Cap'n Proto", "capn_proto"),
        ("Modula-2", "modula_2"),
    ],
)
def test_code_ecosystem_language_slug_is_filesystem_safe(language: str, expected_slug: str):
    assert _language_to_slug(language) == expected_slug


def test_code_ecosystem_slices_cover_every_registered_language_exactly_once():
    slices = long_tail_ppl_slices(family=LongTailPplFamily.CODE_ECOSYSTEM)
    expected_count = sum(len(langs) for langs in CODE_ECOSYSTEM_LANGUAGES.values())

    assert len(slices) == expected_count
    slugs = [slice_.name.removeprefix("stack_v2_") for slice_ in slices]
    assert len(set(slugs)) == len(slugs), "every code-ecosystem slice slug must be unique"
    expected_slugs = {
        _language_to_slug(language) for languages in CODE_ECOSYSTEM_LANGUAGES.values() for language in languages
    }
    assert set(slugs) == expected_slugs


def test_code_ecosystem_explicitly_covers_ecosystems_called_out_in_issue():
    slices = long_tail_ppl_slices(family=LongTailPplFamily.CODE_ECOSYSTEM)
    surface_forms = {slice_.surface_form for slice_ in slices}

    # Issue #5254 explicitly calls out C/C++ systems, Apple/Cocoa, .NET/Xamarin, JVM enterprise.
    for required_language in (
        "C",
        "C++",
        "Objective-C",
        "Objective-C++",
        "Swift",
        "C#",
        "F#",
        "Visual Basic .NET",
        "Java",
        "Kotlin",
        "Scala",
        "Groovy",
    ):
        assert f"source_code:{required_language}" in surface_forms


def test_code_ecosystem_slices_record_target_token_budget_in_notes():
    datasets = long_tail_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")
    java_key = "long_tail_ppl/code_ecosystem/stack_v2_java"
    agda_key = "long_tail_ppl/code_ecosystem/stack_v2_agda"

    assert datasets[java_key].input_path == "gs://example-bucket/raw/long_tail/code/stack_v2/java/heldout.jsonl.gz"
    assert datasets[java_key].tags == ("long_tail_ppl", "epic:5005", f"issue:{CODE_ECOSYSTEM_ISSUE}", "code_ecosystem")
    assert datasets[agda_key].input_path == "gs://example-bucket/raw/long_tail/code/stack_v2/agda/heldout.jsonl.gz"

    java_slice = next(
        slice_
        for slice_ in long_tail_ppl_slices(family=LongTailPplFamily.CODE_ECOSYSTEM)
        if slice_.name == "stack_v2_java"
    )
    agda_slice = next(
        slice_
        for slice_ in long_tail_ppl_slices(family=LongTailPplFamily.CODE_ECOSYSTEM)
        if slice_.name == "stack_v2_agda"
    )

    assert f"{CODE_ECOSYSTEM_LARGE_TARGET_TOKENS:,}" in java_slice.notes
    assert f"{CODE_ECOSYSTEM_SMALL_TARGET_TOKENS:,}" in agda_slice.notes


def test_code_ecosystem_tiers_partition_languages_disjointly():
    large = set(CODE_ECOSYSTEM_LANGUAGES[CodeEcosystemTier.LARGE])
    small = set(CODE_ECOSYSTEM_LANGUAGES[CodeEcosystemTier.SMALL])

    assert large.isdisjoint(small), "a language must be assigned to exactly one tier"


def test_code_ecosystem_markdown_lists_per_language_slices():
    markdown = render_long_tail_ppl_registry_markdown(family=LongTailPplFamily.CODE_ECOSYSTEM)

    assert "long_tail_ppl/code_ecosystem/stack_v2_python" in markdown
    assert "long_tail_ppl/code_ecosystem/stack_v2_objective_cpp" in markdown
    assert f"#{CODE_ECOSYSTEM_ISSUE}" in markdown
