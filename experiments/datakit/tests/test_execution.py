# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from zephyr.context import ZephyrContext

from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.embeddings.luxical.pipeline import LUXICAL_DIM, EmbeddingAttrData
from experiments.datakit.execution import run_steps_in_pool


@pytest.fixture
def shared_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    groups = []
    create_group = client.create_actor_group

    def record_group(actor_class, *args, **kwargs):
        groups.append((actor_class, kwargs["count"]))
        return create_group(actor_class, *args, **kwargs)

    monkeypatch.setattr(client, "create_actor_group", record_group)
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=1,
                resources=ResourceConfig(cpu=2, ram="64g"),
                chunk_storage_prefix=str(tmp_path / "chunks"),
            ) as pool,
        ):
            yield pool, groups
    finally:
        client.shutdown()


def test_shared_pool_downloads_and_normalizes_source(tmp_path, shared_pool):
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "first", "text": "Shared pools retain the original document text."},
                {"id": "second", "text": "Each source uses the same worker pool."},
            ]
        ),
        source / "data.parquet",
    )
    download = StepSpec(
        name="download",
        fn=remote(
            lambda output_path: download_hf(
                DownloadConfig(
                    hf_dataset_id="local-test",
                    revision="test",
                    source_url_override=str(source),
                    gcs_output_path=output_path,
                )
            )
        ),
    )
    normalized = normalize_step(name="normalize", download=download, file_extensions=(".parquet",))
    pool, groups = shared_pool
    initial_groups = list(groups)
    run_steps_in_pool([normalized], pool=pool, max_concurrent=2)
    assert groups == initial_groups
    artifact = read_artifact(normalized.output_path, NormalizedData)
    result = pq.read_table(artifact.main_output_dir).to_pylist()
    assert sorted(row["text"] for row in result) == [
        "Each source uses the same worker pool.",
        "Shared pools retain the original document text.",
    ]


def test_centroid_sampling_threads_use_shared_pool(tmp_path, shared_pool):
    embeddings = {}
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        pq.write_table(
            pa.table({"embedding": [[1] * LUXICAL_DIM, [2] * LUXICAL_DIM]}),
            directory / "part.parquet",
        )
        embeddings[name] = EmbeddingAttrData(
            output_dir=str(directory),
            source_key=name,
            model_name="test",
            embedding_dim=LUXICAL_DIM,
            quantization_scale=1.0,
            quantization_range=1.0,
            batch_size=2,
        )
    pool, groups = shared_pool
    initial_groups = list(groups)
    output = tmp_path / "sample"
    with pool.execution_scope():
        sample_centroid_inputs(str(output), embeddings, n_per_source=1, parallel_sources=2)
    assert groups == initial_groups
    rows = pq.read_table(output).to_pylist()
    assert sorted(row["source"] for row in rows) == ["first", "second"]
    assert all(len(row["embedding"]) == LUXICAL_DIM for row in rows)
