# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.sft_shootout_rows.build import check_relay_student, finished_shards

LAYOUT = {
    "swesmith/shard-00/rebuilt.jsonl": True,
    "swesmith/shard-v5/rebuilt.jsonl": True,
    "swesmith/shard-01/rebuilt.jsonl": False,  # still running: no _SUCCESS
    "tmax/shard-00/rebuilt.jsonl": True,
    "tmax/shard-00/trial__abc/rebuilt.jsonl": True,  # inside a trial directory, not a shard file
    "swesmith-relay/shard-00/round-1/rebuilt.jsonl": True,
    "swesmith-relay/shard-00/round-2.failed-7/rebuilt.jsonl": True,  # a failed round moved aside
    "tmax-relay/shard-03/round-2/rebuilt.jsonl": True,
    "shards/shard-00/rebuilt.jsonl": True,  # not a trace family
}


def test_finished_shards_takes_only_the_run_types_finished_shard_files(tmp_path):
    for name, finished in LAYOUT.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        if finished:
            (path.parent / "_SUCCESS").write_text("")
    plain = [p.removeprefix(f"{tmp_path}/") for p in finished_shards(f"{tmp_path}/", relay=False)]
    relay = [p.removeprefix(f"{tmp_path}/") for p in finished_shards(str(tmp_path), relay=True)]
    assert plain == ["swesmith/shard-00/rebuilt.jsonl", "swesmith/shard-v5/rebuilt.jsonl", "tmax/shard-00/rebuilt.jsonl"]
    assert relay == ["swesmith-relay/shard-00/round-1/rebuilt.jsonl", "tmax-relay/shard-03/round-2/rebuilt.jsonl"]


V01 = {
    "location": "s3://marin-us-east-02a/marin/models/open-athena--Snowball-Smartie-RL-2026-10-15-32-step14--6783e974",
    "revision": None,
    "tokenizer": "open-athena/Snowball-Smartie-RL-2026.10.15.32-step14",
    "tokenizer_revision": "6783e9743478bd25a9ce03ab3460e84d6a9814b0",
}
V0 = {
    "location": "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12",
    "revision": "4eeb756b404db25bc04e4daf49dfb9817bceb741",
    "tokenizer": "open-athena/Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026.10.12",
    "tokenizer_revision": "4eeb756b404db25bc04e4daf49dfb9817bceb741",
}


def relay_round(tmp_path, name: str, student: dict | None) -> str:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "driver_config.json").write_text(json.dumps({} if student is None else {"student_model": student}))
    return str(directory / "rebuilt.jsonl")


def test_relay_rounds_must_have_served_the_named_student(tmp_path):
    v01 = relay_round(tmp_path, "v7/round-1", V01)
    v0 = relay_round(tmp_path, "v7-v0/round-1", V0)
    old = relay_round(tmp_path, "v5/round-1", None)
    assert check_relay_student([v01], "v0.1")["files_by_served_student"] == {
        f"{V01['location']} @ {V01['tokenizer_revision']}": 1
    }
    assert sum(check_relay_student([v0, old], "v0")["files_by_served_student"].values()) == 2
    for files, student in (([v0], "v0.1"), ([old], "v0.1"), ([v01], "v0")):
        with pytest.raises(ValueError, match="not the relay student"):
            check_relay_student(files, student)
