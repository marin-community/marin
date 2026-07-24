# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
)
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams, NgramKind

from experiments.datakit.scripts.dedup_ab_continue import snapshot_dedup_outputs


def test_snapshot_dedup_outputs_copies_every_marker_and_rewrites_paths(tmp_path) -> None:
    dedup_path = tmp_path / "dedup"
    source_dir = dedup_path / "outputs" / "source_000"
    source_dir.mkdir(parents=True)
    (source_dir / "part-00000.parquet").write_bytes(b"first")
    (source_dir / "part-00001.parquet").write_bytes(b"second")
    source_main_dir = str(tmp_path / "normalized")
    write_artifact(
        FuzzyDupsAttrData(
            params=MinHashParams(
                num_perms=10,
                num_bands=2,
                ngram_size=5,
                ngram_kind=NgramKind.CHAR,
                seed=42,
            ),
            sources={source_main_dir: FuzzyDupsPerSource(attr_dir=str(source_dir))},
            counters={"dedup/fuzzy/document/cluster_members": 2},
        ),
        str(dedup_path),
    )
    snapshot_path = tmp_path / "dedup-cap50"

    snapshot_dedup_outputs(
        dedup_path=str(dedup_path),
        snapshot_path=str(snapshot_path),
        copy_workers=2,
    )

    snapshot = read_artifact(str(snapshot_path), FuzzyDupsAttrData)
    snapshot_source = snapshot.sources[source_main_dir].attr_dir
    assert snapshot_source == str(snapshot_path / "outputs" / "source_000")
    assert (snapshot_path / "outputs" / "source_000" / "part-00000.parquet").read_bytes() == b"first"
    assert (snapshot_path / "outputs" / "source_000" / "part-00001.parquet").read_bytes() == b"second"
    assert snapshot.counters == {"dedup/fuzzy/document/cluster_members": 2}
