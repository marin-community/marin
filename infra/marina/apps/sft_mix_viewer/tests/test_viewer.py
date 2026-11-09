# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from marina.applets import AppletServices
from marina.client import AppletQuery
from marina.testing import EngineServices, engine_query, migrated_schema
from rigging.filesystem.storage_path import StoragePath
from sqlalchemy import text
from sqlalchemy.engine import Engine

from infra.marina.apps.sft_mix_viewer.load import load_snapshot
from infra.marina.apps.sft_mix_viewer.sample import (
    HEAD_TOKENS,
    MIX_PATH,
    REPLAY_PATH,
    TAIL_TOKENS,
    clip_tokens,
    replay_sources,
    sft_sources,
)
from infra.marina.apps.sft_mix_viewer.server.app import create_api, migrate
from infra.marina.apps.sft_mix_viewer.snapshot import (
    MixSource,
    Sample,
    SampledSource,
    Snapshot,
    SourceKind,
    read_snapshot,
    write_snapshot,
)


@pytest.fixture
def viewer_database(database_url: str) -> Iterator[Engine]:
    with migrated_schema(database_url, "sft_mix_test", migrate) as engine:
        yield engine


@pytest.fixture
def viewer_query(viewer_database: Engine) -> AppletQuery:
    return engine_query(viewer_database)


VOCAB = ["<|start_think|>", "\n", "five", "two", "nine", "zero", "one"]


def sampled_source(name: str, kind: SourceKind, share: float, documents: dict[int, tuple[int, ...]]) -> SampledSource:
    """A source whose documents are given as token ids into ``VOCAB``."""
    source = MixSource(name, kind, name.split("/")[0], share, f"gs://bucket/{name}")
    samples = []
    for index, ids in documents.items():
        text_value = "".join(VOCAB[i] for i in ids)
        samples.append(Sample(index, len(ids), len(text_value), text_value, len(ids), ids))
    return SampledSource(source, 100, tuple(samples))


def test_loaded_snapshot_replaces_the_served_one(viewer_database: Engine, viewer_query: AppletQuery, tmp_path) -> None:
    written = Snapshot(
        [
            sampled_source("chat/a", SourceKind.SFT, 0.8, {2: (3,), 5: (0, 1, 2)}),
            sampled_source("replay/c01/q0/gt_64k", SourceKind.REPLAY, 0.2, {9: (4,)}),
        ],
        VOCAB,
    )
    write_snapshot(StoragePath.parse(str(tmp_path)), 0, written)
    load_snapshot(read_snapshot(StoragePath.parse(str(tmp_path))), viewer_query)

    with TestClient(create_api(cast(AppletServices, EngineServices(viewer_database)))) as client:
        sources = client.get("/sources").json()
        assert [(source["name"], source["share"]) for source in sources] == [
            ("chat/a", 0.8),
            ("replay/c01/q0/gt_64k", 0.2),
        ]
        assert sources[0]["samples"] == [
            {"doc_index": 2, "tokens": 1, "chars": 3},
            {"doc_index": 5, "tokens": 3, "chars": 20},
        ]
        document = {"source": "chat/a", "doc_index": 5}
        assert client.get("/sample", params=document).json()["text"] == "<|start_think|>\nfive"
        assert client.get("/tokens", params=document).json() == {
            "head_tokens": 3,
            "token_ids": [0, 1, 2],
            "pieces": ["<|start_think|>", "\n", "five"],
        }

        load_snapshot(Snapshot([sampled_source("chat/b", SourceKind.SFT, 1.0, {1: (6,)})], VOCAB), viewer_query)
        assert [source["name"] for source in client.get("/sources").json()] == ["chat/b"]
        assert client.get("/sample", params=document).status_code == 404
        assert client.get("/tokens", params=document).status_code == 404
    with viewer_database.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM mix_samples")).scalar_one() == 1
        assert connection.execute(text("SELECT count(*) FROM mix_vocab")).scalar_one() == len(VOCAB)


def test_load_rejects_a_source_without_samples(viewer_query: AppletQuery) -> None:
    snapshot = Snapshot(
        [
            sampled_source("chat/a", SourceKind.SFT, 0.5, {0: (5,)}),
            sampled_source("chat/empty", SourceKind.SFT, 0.5, {}),
        ],
        VOCAB,
    )
    with pytest.raises(ValueError, match="chat/empty"):
        load_snapshot(snapshot, viewer_query)


def test_pinned_mix_shares_cover_all_training_positions() -> None:
    mix = json.loads(MIX_PATH.read_text())
    stores = {name: {"cache_path": f"gs://bucket/{name}"} for name in mix["allocations_tokens"]}
    sources = sft_sources(mix, stores) + replay_sources(mix, json.loads(REPLAY_PATH.read_text()))

    assert len({source.name for source in sources}) == len(sources)
    assert sum(source.share for source in sources) == pytest.approx(1.0)
    long_replay = next(source for source in sources if source.name == "replay/c01/q0/gt_64k")
    short_replay = next(source for source in sources if source.name == "replay/c01/q0/lte_64k")
    assert long_replay.share == pytest.approx(8 * short_replay.share)


def test_long_documents_keep_both_ends() -> None:
    ids = np.arange(HEAD_TOKENS + 1234 + TAIL_TOKENS)

    kept, head_tokens = clip_tokens(ids)

    assert head_tokens == HEAD_TOKENS
    assert kept[:head_tokens].tolist() == ids[:HEAD_TOKENS].tolist()
    assert kept[head_tokens:].tolist() == ids[-TAIL_TOKENS:].tolist()
    whole, whole_head = clip_tokens(np.arange(10))
    assert (len(whole), whole_head) == (10, 10)
