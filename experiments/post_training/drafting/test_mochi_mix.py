# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import CHAT_SCHEMA

from experiments.post_training.drafting.mochi_mix import _write_mix


@pytest.fixture
def mix_sources(tmp_path):
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
    return tuple(sources)


def test_mochi_mix_keeps_prompt_groups_together(tmp_path, mix_sources):
    output = tmp_path / "mix"
    _write_mix(output=str(output), sources=mix_sources, seed=42)
    train, validation = (pq.read_table(output / split).to_pylist() for split in ("train", "validation"))
    prompts = [{row["messages"][0]["content"][0]["text"] for row in split} for split in (train, validation)]
    assert sum(map(len, prompts)) < len(train) + len(validation)
    assert prompts[0].isdisjoint(prompts[1])
