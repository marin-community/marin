# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import os

from experiments.post_training.sweep_data.prepare_agentic import BATCH, write_prebuilt


def test_prebuilt_files_depend_only_on_their_rows(tmp_path, monkeypatch):
    lines = ['{"input_ids":[1,2],"loss_weights":[0,1]}', '{"input_ids":[3],"loss_weights":[1]}']
    first, second = tmp_path / "a" / "train-epoch-0-prebuilt.jsonl.gz", tmp_path / "b" / "other-name.jsonl.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    assert write_prebuilt(first, lines) == (2, BATCH - 2)
    monkeypatch.setattr("time.time", lambda: 2_000_000_000.0)  # a different clock: gzip would stamp it as mtime
    write_prebuilt(second, lines)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[3:8] == b"\x00\x00\x00\x00\x00"  # no FNAME flag, mtime 0
    assert gzip.decompress(first.read_bytes()).decode().splitlines()[:2] == lines
    assert os.path.getsize(first) > 0
