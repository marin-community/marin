# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.chat_normalize import CHAT_SCHEMA

from experiments.post_training.drafting.mochi_mix import _write_mix


def test_mochi_mix_interleaves_sources_and_keeps_prompt_groups_together(tmp_path):
    sources = []
    for name, quota in (("tools", 96), ("instruction", 48), ("trajectories", 24)):
        source = tmp_path / name
        shards = source / "outputs" / "main"
        shards.mkdir(parents=True)
        rows = [
            {
                "id": f"{name}-{index}",
                "source": name,
                "messages": [{"role": "user", "content": [{"type": "text", "text": f"prompt {index}"}]}],
            }
            for index in range(160)
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=CHAT_SCHEMA), shards / "000.parquet", row_group_size=16)
        sources.append((name, str(source), quota))

    def build(output):
        _write_mix(output=str(output), sources=tuple(sources), seed=42)
        return tuple(pq.read_table(output / split).to_pylist() for split in ("train", "validation"))

    train, validation = build(tmp_path / "mix")
    assert (train, validation) == build(tmp_path / "rebuild")
    for name, _, quota in sources:
        selected = [row for row in train + validation if row["source"] == name]
        assert len({row["id"] for row in selected}) == len(selected) == quota
        assert any(int(row["id"].split("-")[-1]) >= quota for row in selected)
    for window in (train[:24], train[len(train) // 2 : len(train) // 2 + 24], train[-24:]):
        assert {row["source"] for row in window} == {name for name, _, _ in sources}

    prompts = [{row["messages"][0]["content"][0]["text"] for row in split} for split in (train, validation)]
    assert sum(map(len, prompts)) < len(train) + len(validation)
    assert prompts[0].isdisjoint(prompts[1])
