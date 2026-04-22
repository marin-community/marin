# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.long_tail_ppl import (
    GAME_MUSIC_ISSUE,
    LongTailPplFamily,
    long_tail_raw_validation_sets,
    render_long_tail_ppl_registry_markdown,
)
from experiments.evals.long_tail_ppl_runnable import (
    RUNNABLE_LONG_TAIL_PPL_REGISTRY,
    runnable_long_tail_ppl_slices,
    runnable_long_tail_raw_validation_sets,
)
from levanter.data.text import HfDatasetSourceConfig
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


def test_file_backed_raw_dataset_rejects_non_validation_split():
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/eval.jsonl", split="test")


def test_runnable_game_music_slices_are_registered():
    game_music_slices = runnable_long_tail_ppl_slices(family=LongTailPplFamily.GAME_MUSIC)

    names = {slice_.name for slice_ in game_music_slices}
    assert {"lichess_pgn_2013_06", "irishman_abc", "melodyhub_abc_input"} <= names

    pgn = RUNNABLE_LONG_TAIL_PPL_REGISTRY["long_tail_ppl_runnable/game_music/lichess_pgn_2013_06"]
    assert pgn.hf_dataset == HfDatasetSpec(id="Icannos/lichess_games", name="2013-06")
    assert pgn.text_key == "text"
    # PGN only ships a ``train`` split; we still use it as a diagnostic eval.
    assert pgn.split == "train"
    assert "split:train" in pgn.tags

    irishman = RUNNABLE_LONG_TAIL_PPL_REGISTRY["long_tail_ppl_runnable/game_music/irishman_abc"]
    # IrishMAN's column is literally ``abc notation`` (with the space). Asserting
    # the exact string catches drift if someone "normalizes" it.
    assert irishman.text_key == "abc notation"
    assert irishman.split == "validation"


def test_runnable_game_music_datasets_round_trip_through_dataset_component():
    datasets = runnable_long_tail_raw_validation_sets()

    pgn_key = "long_tail_ppl_runnable/game_music/lichess_pgn_2013_06"
    irishman_key = "long_tail_ppl_runnable/game_music/irishman_abc"

    pgn_component = _to_dataset_component(datasets[pgn_key])
    irishman_component = _to_dataset_component(datasets[irishman_key])

    assert isinstance(pgn_component.source, HfDatasetSourceConfig)
    assert pgn_component.source.id == "Icannos/lichess_games"
    assert pgn_component.source.name == "2013-06"
    assert pgn_component.source.splits == ["train"]
    assert pgn_component.format.text_key == "text"

    assert isinstance(irishman_component.source, HfDatasetSourceConfig)
    assert irishman_component.source.id == "sander-wood/irishman"
    assert irishman_component.source.splits == ["validation"]
    assert irishman_component.format.text_key == "abc notation"
