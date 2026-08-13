# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.execution import ZephyrContext

from experiments.datakit.cluster.domain.v1 import materialize_10pct_harrier_tokens
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM


def test_co_partitions_aligns_renumbered_sample_shards_by_source_order(tmp_path):
    directories = {name: tmp_path / name for name in ("sample", "main", "nemotron", "marin", "quality")}
    for directory in directories.values():
        directory.mkdir()
    sample_paths = [directories["sample"] / f"part-{index:05d}-of-00002.parquet" for index in range(2)]
    original_paths = {
        name: [directories[name] / f"part-{index:05d}-of-00010.parquet" for index in range(3)]
        for name in ("main", "nemotron", "quality")
    }
    marin_paths = [directories["marin"] / path.name for path in sample_paths]
    for path in [*sample_paths, *marin_paths, *(path for paths in original_paths.values() for path in paths)]:
        path.touch()
    plan = materialize_10pct_harrier_tokens.InputPlan(
        sample_files=tuple(materialize_10pct_harrier_tokens.SampleFile("example", str(path)) for path in sample_paths),
        main_dirs={"example": str(directories["main"])},
        fuzzy_dirs={},
        nemotron_dirs={"example": str(directories["nemotron"])},
        marin_dirs={"example": str(directories["marin"])},
        quality_dirs={"example": str(directories["quality"])},
        schema_paths={"example": str(sample_paths[0])},
    )

    partitions = materialize_10pct_harrier_tokens.co_partitions(plan)

    assert [partition.main_path for partition in partitions] == [str(path) for path in original_paths["main"][:2]]
    assert [partition.marin_path for partition in partitions] == [str(path) for path in marin_paths]


def test_materialize_dataset_joins_co_partitions_and_concatenates_token_chunks(tmp_path, monkeypatch):
    basename = "part.parquet"
    sample_path = tmp_path / "sample" / basename
    main_path = tmp_path / "main" / basename
    fuzzy_path = tmp_path / "fuzzy" / basename
    nemotron_path = tmp_path / "nemotron" / basename
    marin_path = tmp_path / "marin" / basename
    quality_path = tmp_path / "quality" / basename
    output_root = tmp_path / "output"
    for path in (sample_path, main_path, fuzzy_path, nemotron_path, marin_path, quality_path):
        path.parent.mkdir()

    pq.write_table(
        pa.table({"id": ["a", "b", "c"], "text": ["alpha", "beta", "gamma"]}),
        sample_path,
    )
    embedding_type = pa.list_(pa.int8(), HARRIER_DIM)
    pq.write_table(
        pa.table({"id": ["a"], "embedding": pa.array([[1] * HARRIER_DIM], type=embedding_type)}),
        main_path,
    )
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b"],
                "embedding": pa.array([[9] * HARRIER_DIM, [2] * HARRIER_DIM], type=embedding_type),
            }
        ),
        fuzzy_path,
    )
    token_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("input_ids", pa.list_(pa.int32())),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "a", "chunk_index": 0, "input_ids": [10, 11]},
                {"id": "a", "chunk_index": 1, "input_ids": [12]},
                {"id": "b", "chunk_index": 0, "input_ids": [13]},
            ],
            schema=token_schema,
        ),
        nemotron_path,
    )
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "b"],
                "input_ids": pa.array([[20], [21], [22, 23]], type=pa.list_(pa.int32())),
            }
        ),
        marin_path,
    )
    pq.write_table(pa.table({"id": ["b"], "score": [0.75]}), quality_path)

    class Index:
        def search(self, embeddings, _neighbors):
            clusters = (embeddings[:, :1] > 0.003).astype(np.int64)
            return np.zeros_like(clusters, dtype=np.float32), clusters

    monkeypatch.setattr(
        materialize_10pct_harrier_tokens,
        "_get_index",
        lambda _centroids, _lookups: {"index": Index(), "lookups": {40: np.asarray([7, 8], dtype=np.int32)}},
    )
    partition = materialize_10pct_harrier_tokens.CoPartition(
        source="example",
        basename=basename,
        sample_path=str(sample_path),
        schema_path=str(sample_path),
        main_path=str(main_path),
        fuzzy_path=str(fuzzy_path),
        nemotron_path=str(nemotron_path),
        marin_path=str(marin_path),
        quality_path=str(quality_path),
    )
    dataset = materialize_10pct_harrier_tokens._materialize_dataset(
        (partition,),
        str(output_root),
        "cluster",
    )
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=1,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="test-materialize-copartitioned",
    )
    try:
        context.execute(dataset)
    finally:
        context.shutdown()
        client.shutdown(wait=True)

    result = pq.read_table(output_root / "example" / basename)
    assert result["id"].to_pylist() == ["a", "b"]
    assert result["embedding"].to_pylist() == [[1] * HARRIER_DIM, [2] * HARRIER_DIM]
    assert result["cluster_5000"].to_pylist() == [0, 1]
    assert result["domain_id"].to_pylist() == [7, 8]
    assert result["quality_score_pooled_junkgate2"].to_pylist() == [None, 0.75]
    assert result["nemotron_input_ids"].to_pylist() == [[10, 11, 12], [13]]
    assert result["marin_input_ids"].to_pylist() == [[20], [21, 22, 23]]
