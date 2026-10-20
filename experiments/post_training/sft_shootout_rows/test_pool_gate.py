# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.sft_shootout_rows.pool_gate import plain_report, relay_report


def touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def finish(directory):
    touch(directory / "_SUCCESS")
    touch(directory / "rebuilt.jsonl")


def test_plain_needs_every_listed_shard_finished(tmp_path):
    for shard in ("00", "01", "02"):
        touch(tmp_path / f"shards/tmax-shard-{shard}.json")
    finish(tmp_path / "tmax/shard-00")
    touch(tmp_path / "tmax/shard-01/_SUCCESS")  # rebuilt.jsonl not written yet
    report, done = plain_report(str(tmp_path), "tmax")
    assert report == {"shards_listed": 3, "finished": 1, "missing": ["01", "02"]}
    assert done == [str(tmp_path / "tmax/shard-00")]


def test_relay_reports_stop_and_unfinished_rounds(tmp_path):
    root = tmp_path / "swesmith-relay"
    finish(root / "shard-00/round-1")
    finish(root / "shard-00/round-2")
    touch(root / "shard-00/round-2-tasks.json")  # a task list beside the rounds, not a round
    touch(root / "shard-01/round-1.failed-1a2b3c4d/requests.jsonl")  # moved aside, ignored
    finish(root / "shard-01/round-1")
    touch(root / "shard-01/round-2/requests.jsonl")  # partly uploaded
    report, done = relay_report(str(tmp_path), "swesmith")
    assert report == {
        "family_stop": False,
        "shards_with_rounds": 2,
        "rounds_finished": 3,
        "without_success": ["shard-01/round-2"],
        "success_without_rebuilt": [],
    }
    assert len(done) == 3
    touch(root / "STOP")
    assert relay_report(str(tmp_path), "swesmith")[0]["family_stop"]
