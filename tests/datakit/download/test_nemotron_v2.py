# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray import LocalClient, set_current_client
from marin.datakit.download.nemotron_v2 import normalize_nemotron_v2_step
from marin.datakit.normalize import generate_id
from marin.datakit.sources import all_sources
from marin.execution.step_spec import StepSpec

NEMOTRON_SPECIALIZED_V1_2_FAMILY = "nemotron_pretraining_specialized_v1_2"
NEMOTRON_SPECIALIZED_V1_2_REGISTRY_PREFIX = "nemotron_specialized_v1_2"
NEMOTRON_SPECIALIZED_V1_2_DATASET_ID = "nvidia/Nemotron-Pretraining-Specialized-v1.2"
NEMOTRON_SPECIALIZED_V1_2_REVISION = "807afc1fa65c441d46ebc7d9b95295a35499a527"

EXPECTED_SPECIALIZED_V1_2_SUBSETS = {
    "fact_seeking": ("Nemotron-Pretraining-Fact-Seeking", 35.03),
    "generative": ("Nemotron-Pretraining-Generative", 0.69),
    "moral_scenarios": ("Nemotron-Pretraining-Moral-Scenarios", 0.02),
    "multiple_choice": ("Nemotron-Pretraining-Multiple-Choice", 6.10),
}


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with set_current_client(LocalClient()):
        yield


def _read_main_parquet(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((output_dir / "outputs" / "main").glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_nemotron_specialized_v1_2_sources_build_download_and_normalize_chain() -> None:
    sources = {
        name: source
        for name, source in all_sources().items()
        if name.startswith(f"{NEMOTRON_SPECIALIZED_V1_2_REGISTRY_PREFIX}/")
    }

    assert set(sources) == {
        f"{NEMOTRON_SPECIALIZED_V1_2_REGISTRY_PREFIX}/{subset}" for subset in EXPECTED_SPECIALIZED_V1_2_SUBSETS
    }

    first_download = next(iter(sources.values())).normalize_steps[0]
    assert first_download.name == f"raw/{NEMOTRON_SPECIALIZED_V1_2_FAMILY}"

    for registry_name, source in sources.items():
        subset = registry_name.removeprefix(f"{NEMOTRON_SPECIALIZED_V1_2_REGISTRY_PREFIX}/")
        upstream_directory, token_count = EXPECTED_SPECIALIZED_V1_2_SUBSETS[subset]
        download, normalize = source.normalize_steps

        assert download is first_download
        assert download.hash_attrs["hf_dataset_id"] == NEMOTRON_SPECIALIZED_V1_2_DATASET_ID
        assert download.hash_attrs["revision"] == NEMOTRON_SPECIALIZED_V1_2_REVISION
        assert download.override_output_path is None
        assert source.rough_token_count_b == token_count
        assert normalize.name == f"normalized/{NEMOTRON_SPECIALIZED_V1_2_FAMILY}/{subset}"
        assert normalize.deps == [download]
        assert normalize.hash_attrs["id_field"] == "uuid"
        assert normalize.hash_attrs["text_field"] == "text"
        assert normalize.hash_attrs["relative_input_path"] == upstream_directory
        assert normalize.hash_attrs["file_extensions"] == (".parquet",)


def test_nemotron_specialized_v1_2_normalize_reads_subset_parquet_and_preserves_uuid(tmp_path: Path) -> None:
    fact_seeking_dir = tmp_path / "raw" / "Nemotron-Pretraining-Fact-Seeking"
    fact_seeking_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "uuid": "fact-seeking-0001",
                    "text": "Answer the fact-seeking question.",
                    "license": "CC-BY-4.0",
                }
            ]
        ),
        fact_seeking_dir / "part-00000.parquet",
    )

    ignored_dir = tmp_path / "raw" / "Nemotron-Pretraining-Generative"
    ignored_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "uuid": "generative-0001",
                    "text": "This sibling subset must not be normalized.",
                    "license": "CC-BY-2.0",
                }
            ]
        ),
        ignored_dir / "part-00000.parquet",
    )

    download = StepSpec(
        name=f"raw/{NEMOTRON_SPECIALIZED_V1_2_FAMILY}",
        override_output_path=str(tmp_path / "raw"),
    )
    normalize = normalize_nemotron_v2_step(download, family=NEMOTRON_SPECIALIZED_V1_2_FAMILY, subset="fact_seeking")

    assert normalize.fn is not None
    normalize.fn(str(tmp_path / "normalized"))

    rows = _read_main_parquet(tmp_path / "normalized")
    assert rows == [
        {
            "text": "Answer the fact-seeking question.",
            "license": "CC-BY-4.0",
            "id": generate_id("Answer the fact-seeking question."),
            "source_id": "fact-seeking-0001",
        }
    ]
