# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from marina.applets import AppletServices, validate_query
from marina.client import AppletQuery
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from infra.marina.apps.sft_mix_viewer.load import load_snapshot
from infra.marina.apps.sft_mix_viewer.sample import (
    HEAD_CHARS,
    MIX_PATH,
    REPLAY_PATH,
    TAIL_CHARS,
    clip_text,
    replay_sources,
    sft_sources,
)
from infra.marina.apps.sft_mix_viewer.server.app import create_api, migrate


@dataclass(frozen=True)
class ViewerServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


@pytest.fixture
def viewer_database(database_url: str) -> Iterator[Engine]:
    admin = create_engine(database_url)
    schema = "sft_mix_test_" + uuid.uuid4().hex
    with admin.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def select_schema(connection, _record):
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {schema}, public")

    try:
        with engine.begin() as connection:
            migrate(connection)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


@pytest.fixture
def viewer_query(viewer_database: Engine) -> AppletQuery:
    def query(sql: str, parameters: dict[str, Any]) -> dict[str, Any]:
        validate_query(sql)
        with viewer_database.begin() as connection:
            result = connection.execute(text(sql), parameters)
            return {
                "rows": [dict(row) for row in result.mappings()] if result.returns_rows else [],
                "row_count": result.rowcount,
            }

    return query


def source_row(name: str, kind: str, share: float) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "family": name.split("/")[0],
        "share": share,
        "documents": 100,
        "cache_path": f"gs://bucket/{name}",
    }


def sample_row(source: str, doc_index: int, text_value: str) -> dict[str, Any]:
    return {"source": source, "doc_index": doc_index, "tokens": 7, "chars": len(text_value), "text": text_value}


def test_loaded_snapshot_replaces_the_served_one(viewer_database: Engine, viewer_query: AppletQuery) -> None:
    load_snapshot(
        [source_row("chat/a", "sft", 0.8), source_row("replay/c01/q0/gt_64k", "replay", 0.2)],
        [
            sample_row("chat/a", 5, "<|start_think|>\nfive"),
            sample_row("chat/a", 2, "two"),
            sample_row("replay/c01/q0/gt_64k", 9, "nine"),
        ],
        viewer_query,
    )
    with TestClient(create_api(cast(AppletServices, ViewerServices(viewer_database)))) as client:
        sources = client.get("/sources").json()
        assert [(source["name"], source["share"]) for source in sources] == [
            ("chat/a", 0.8),
            ("replay/c01/q0/gt_64k", 0.2),
        ]
        assert sources[0]["samples"] == [
            {"doc_index": 2, "tokens": 7, "chars": 3},
            {"doc_index": 5, "tokens": 7, "chars": 20},
        ]
        sample = client.get("/sample", params={"source": "chat/a", "doc_index": 5}).json()
        assert sample["text"] == "<|start_think|>\nfive"

        load_snapshot([source_row("chat/b", "sft", 1.0)], [sample_row("chat/b", 1, "one")], viewer_query)
        assert [source["name"] for source in client.get("/sources").json()] == ["chat/b"]
        assert client.get("/sample", params={"source": "chat/a", "doc_index": 5}).status_code == 404
    with viewer_database.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM mix_samples")).scalar_one() == 1


def test_load_rejects_a_source_without_samples(viewer_query: AppletQuery) -> None:
    with pytest.raises(ValueError, match="chat/empty"):
        load_snapshot(
            [source_row("chat/a", "sft", 0.5), source_row("chat/empty", "sft", 0.5)],
            [sample_row("chat/a", 0, "zero")],
            viewer_query,
        )


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
    text_value = "H" * HEAD_CHARS + "m" * 1234 + "T" * TAIL_CHARS

    clipped = clip_text(text_value)

    assert clipped.startswith("H" * HEAD_CHARS + "\n\n[… 1,234 characters omitted …]\n\n")
    assert clipped.endswith("T" * TAIL_CHARS)
    assert clip_text("short\x00text") == "short␀text"
