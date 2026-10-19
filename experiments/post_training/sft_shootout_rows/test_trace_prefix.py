# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.sft_shootout_rows.build import finished_shards

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
