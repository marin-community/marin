# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the cluster-text verification stage and its marker artifact."""

import json
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.copartitioned import SOURCE_MANIFEST_FILENAME
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.cluster_dedup import ClusterDedupParams
from marin.processing.classification.deduplication.cluster_text import (
    CLUSTER_TEXT_SUCCESS_FILENAME,
    ClusterTextManifest,
    ClusterTextShard,
    write_cluster_text_manifest,
    write_cluster_text_success,
)
from marin.processing.classification.deduplication.cluster_verify import (
    ClusterVerifiedFuzzyDupsAttrData,
    verify_cluster_text,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VERIFIED_FUZZY_DUPLICATE_SCHEMA,
    VerifiedFuzzyDupsArtifact,
    VerifiedFuzzyDupsPerSource,
)
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

ORIGINAL = (
    "the reference implementation walks every cluster member from the longest "
    "document and keeps the first representative that already holds the member "
    "content so a shorter copy never survives a longer original"
)
"""31 words, thus 29 distinct 3-grams."""

NEAR_COPY = ORIGINAL.replace("already holds", "now holds")
"""The original with one word replaced: shorter, and containment 26/29."""

UNRELATED = (
    "a calibration table maps each humidity reading to the pressure coefficient "
    "that the sensor firmware applies before it reports a value to the flight recorder bus"
)
"""Shares no 3-gram with the original."""

SOURCES = {"datakit/normalize/left": "source_000", "datakit/normalize/right": "source_001"}


@pytest.fixture
def local_client():
    client = LocalClient()
    try:
        with set_current_client(client):
            yield client
    finally:
        client.shutdown()


def _write_cluster_text(root: Path, rows: list[dict]) -> str:
    """Materialize one grouped text file and the manifest that names the shards.

    The cluster-text stage writes ``file_idx`` 0 for the left source's only
    shard and 1 for the right source's, and sorts the text by ``cluster_key``.
    """
    text_dir = root / "text"
    text_dir.mkdir(parents=True)
    write_parquet_file(
        sorted(rows, key=lambda row: (row["cluster_key"], row["id"])), str(text_dir / "part-000000.parquet")
    )
    manifest = ClusterTextManifest(
        candidates="candidates",
        max_cluster_size=100_000,
        output_shards=1,
        groups_per_shard=1,
        split_ngram_size=5,
        oversized_clusters={},
        oversized_cluster_members=0,
        shards=[
            ClusterTextShard(
                file_idx=file_idx,
                source_key=source_key,
                source_tag=source_tag,
                basename="shard-000.parquet",
            )
            for file_idx, (source_key, source_tag) in enumerate(SOURCES.items())
        ],
    )
    write_cluster_text_manifest(str(root), manifest)
    write_cluster_text_success(str(root))
    return str(root)


def _member(*, cluster: str, doc_id: str, text: str, file_idx: int) -> dict:
    return {
        "cluster_key": cluster,
        "dup_cluster_id": cluster,
        "id": doc_id,
        "text": text,
        "text_truncated": False,
        "file_idx": file_idx,
    }


def test_marker_lands_beside_the_normalized_shard_that_holds_the_duplicate(tmp_path, local_client):
    """A cross-source duplicate is marked in the *member's* tree, not its representative's."""
    cluster_text = _write_cluster_text(
        tmp_path / "cluster_text",
        [
            _member(cluster="c1", doc_id="left-original", text=ORIGINAL, file_idx=0),
            _member(cluster="c1", doc_id="right-copy", text=NEAR_COPY, file_idx=1),
            _member(cluster="c2", doc_id="left-only", text=UNRELATED, file_idx=0),
        ],
    )
    output_path = str(tmp_path / "verified")

    result = verify_cluster_text(
        cluster_text=cluster_text,
        output_path=output_path,
        params=ClusterDedupParams(),
    )

    markers = list(load_parquet(str(Path(output_path) / "outputs/source_001/shard-000.parquet")))
    assert set(markers[0]) == set(VERIFIED_FUZZY_DUPLICATE_SCHEMA.names)
    assert [(row["id"], row["dup_doc"], row["dup_cluster_id"], row["dup_representative_id"]) for row in markers] == [
        ("right-copy", True, "c1", "left-original")
    ]
    assert markers[0]["dup_representative_source_key"] == "datakit/normalize/left"
    assert markers[0]["dup_representative_kind"] == "cluster_longest"
    assert markers[0]["dup_member_containment"] == pytest.approx(26 / 29, rel=1e-3)
    # The left source holds only a representative and a singleton, so it gets no
    # file at all: the store reads a missing shard as "no duplicates here".
    assert not list((Path(output_path) / "outputs/source_000").glob("*.parquet"))
    assert result.counters["fuzzy/cluster_verify/markers"] == 1


def test_every_source_is_resolvable_even_when_it_has_no_markers(tmp_path, local_client):
    """The store resolves one attribute directory per source key, marked or not."""
    cluster_text = _write_cluster_text(
        tmp_path / "cluster_text",
        [
            _member(cluster="c1", doc_id="left-original", text=ORIGINAL, file_idx=0),
            _member(cluster="c1", doc_id="left-copy", text=NEAR_COPY, file_idx=0),
        ],
    )
    output_path = str(tmp_path / "verified")

    result = verify_cluster_text(
        cluster_text=cluster_text,
        output_path=output_path,
        params=ClusterDedupParams(),
    )

    assert {source_key: entry.attr_dir for source_key, entry in result.sources.items()} == {
        source_key: f"{output_path}/outputs/{source_tag}" for source_key, source_tag in SOURCES.items()
    }
    manifest = json.loads((Path(output_path) / SOURCE_MANIFEST_FILENAME).read_text())
    assert {entry["source_key"]: entry["attribute_dir"] for entry in manifest["sources"]} == {
        source_key: f"outputs/{source_tag}" for source_key, source_tag in SOURCES.items()
    }


def test_a_rule_the_cluster_cannot_satisfy_marks_nothing(tmp_path, local_client):
    """The threshold is a parameter: raised above the pair's containment, nothing is marked."""
    cluster_text = _write_cluster_text(
        tmp_path / "cluster_text",
        [
            _member(cluster="c1", doc_id="left-original", text=ORIGINAL, file_idx=0),
            _member(cluster="c1", doc_id="right-copy", text=NEAR_COPY, file_idx=1),
        ],
    )
    output_path = str(tmp_path / "verified")

    result = verify_cluster_text(
        cluster_text=cluster_text,
        output_path=output_path,
        params=ClusterDedupParams(minimum_containment=0.95),
    )

    assert result.counters["fuzzy/cluster_verify/markers"] == 0
    assert result.rule.minimum_containment == 0.95


def test_result_round_trips_through_the_artifact_record(tmp_path, monkeypatch):
    """A written record reproduces the rule and the absolute attribute directories.

    The store reads the stage's output through :func:`read_artifact`, and the
    paths inside it are stored relative to ``MARIN_PREFIX``.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    output_path = str(tmp_path / "verified")
    params = ClusterDedupParams(minimum_containment=0.75, ngram_size=4)
    written = ClusterVerifiedFuzzyDupsAttrData(
        rule=params,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(attr_dir=f"{output_path}/outputs/{source_tag}", source_tag=source_tag)
            for source_key, source_tag in SOURCES.items()
        },
        counters={"fuzzy/cluster_verify/markers": 3},
    )

    write_artifact(written, output_path)
    loaded = read_artifact(output_path, ClusterVerifiedFuzzyDupsAttrData)
    common = read_artifact(output_path, VerifiedFuzzyDupsArtifact)

    assert loaded.rule == params
    assert common.producer == "cluster" and common.version == "v1"
    assert loaded.attr_dir_for_source("datakit/normalize/right") == f"{output_path}/outputs/source_001"


def test_attr_dir_for_an_unknown_source_names_the_missing_key(tmp_path, monkeypatch):
    """The store drops or rebuilds a source on this error, so it must name the key."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    result = ClusterVerifiedFuzzyDupsAttrData(rule=ClusterDedupParams(), sources={}, counters={})

    with pytest.raises(KeyError, match="datakit/normalize/left"):
        result.attr_dir_for_source("datakit/normalize/left")


def test_incomplete_cluster_text_is_rejected(tmp_path, local_client):
    cluster_text = _write_cluster_text(
        tmp_path / "cluster_text",
        [_member(cluster="c1", doc_id="left-original", text=ORIGINAL, file_idx=0)],
    )
    (Path(cluster_text) / CLUSTER_TEXT_SUCCESS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        verify_cluster_text(
            cluster_text=cluster_text,
            output_path=str(tmp_path / "verified"),
            params=ClusterDedupParams(),
        )


def test_truncated_document_is_not_removed_or_used_as_a_representative(tmp_path, local_client):
    original = _member(cluster="c1", doc_id="left-original", text=ORIGINAL, file_idx=0)
    original["text_truncated"] = True
    cluster_text = _write_cluster_text(
        tmp_path / "cluster_text",
        [original, _member(cluster="c1", doc_id="right-copy", text=NEAR_COPY, file_idx=1)],
    )

    result = verify_cluster_text(
        cluster_text=cluster_text,
        output_path=str(tmp_path / "verified"),
        params=ClusterDedupParams(),
    )

    assert result.counters["fuzzy/cluster_verify/markers"] == 0
    assert result.counters["fuzzy/cluster_verify/truncated_documents_skipped"] == 1
