# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.data.text import HfDatasetSourceConfig, SupervisedLmDatasetFormat
from marin.evaluation.perplexity_gap import (
    _cache_key_for_dataset,
    _to_dataset_component,
    raw_text_dataset,
    supervised_text_dataset,
)
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import (
    GAME_MUSIC_ISSUE,
    LongTailPplFamily,
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
    dataset = raw_text_dataset(
        HfDatasetSpec(id="example/dataset", name="raw_config", revision="raw-revision-abc123"),
        text_key="body",
        split="test",
    )

    component = _to_dataset_component(dataset)

    assert component.split == "test"
    assert component.format.text_key == "body"
    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.name == "raw_config"
    assert component.source.revision == "raw-revision-abc123"
    assert component.source.splits == ["test"]


def test_hf_backed_supervised_dataset_uses_supervised_format():
    dataset = supervised_text_dataset(
        HfDatasetSpec(id="example/dataset", name="supervised_config", revision="supervised-revision-def456"),
        input_key="prompt",
        target_key="completion",
        split="test",
    )

    component = _to_dataset_component(dataset)

    assert component.split == "test"
    assert isinstance(component.format, SupervisedLmDatasetFormat)
    assert component.format.input_key == "prompt"
    assert component.format.target_key == "completion"
    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.name == "supervised_config"
    assert component.source.revision == "supervised-revision-def456"
    assert component.source.splits == ["test"]


def test_hf_dataset_revision_changes_per_dataset_score_cache_key():
    old_dataset = supervised_text_dataset(
        HfDatasetSpec(id="example/dataset", name="config", revision="old-commit"),
        input_key="prompt",
        target_key="completion",
    )
    new_dataset = supervised_text_dataset(
        HfDatasetSpec(id="example/dataset", name="config", revision="new-commit"),
        input_key="prompt",
        target_key="completion",
    )

    assert _cache_key_for_dataset(old_dataset)["hf_dataset_revision"] == "old-commit"
    assert _cache_key_for_dataset(new_dataset)["hf_dataset_revision"] == "new-commit"
    assert _cache_key_for_dataset(old_dataset) != _cache_key_for_dataset(new_dataset)


def test_file_backed_raw_dataset_rejects_non_validation_split():
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/eval.jsonl", split="test")
