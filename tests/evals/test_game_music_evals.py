# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.exp5062_game_music_evals import game_music_raw_validation_sets


def test_game_music_raw_validation_sets_render_deterministic_paths_and_tags() -> None:
    datasets = game_music_raw_validation_sets(raw_root="gs://example-bucket/raw/game_music")

    lichess = datasets["game_music/lichess_pgn_2013_01"]
    irishman = datasets["game_music/irishman_abc"]

    assert lichess.input_path == "gs://example-bucket/raw/game_music/lichess/2013-01/data.jsonl.gz"
    assert lichess.tags == (
        "game_music",
        "long_tail_ppl",
        "epic:5005",
        "issue:5062",
        "source:lichess",
        "surface:pgn",
    )
    assert irishman.input_path == "gs://example-bucket/raw/game_music/music/abc/irishman/data.jsonl.gz"
    assert irishman.tags == (
        "game_music",
        "long_tail_ppl",
        "epic:5005",
        "issue:5062",
        "source:irishman",
        "surface:abc_notation",
    )
