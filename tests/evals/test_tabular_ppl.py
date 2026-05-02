# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from marin.execution.executor import InputName

from experiments.evals import tabular_ppl


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]


def _read_records(output_dir: Path) -> list[dict]:
    with gzip.open(output_dir / "data.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_materialize_fivethirtyeight_marvel_wikia_downloads_and_stages(tmp_path, monkeypatch):
    payload = (
        b"page_id,name,align,appearances\n"
        b"1678,Spider-Man (Peter Parker),Good Characters,4043\n"
        b'2114,"Madame Web, Julia Carpenter",,\n'
        b"2151,Venom (Eddie Brock),Bad Characters,337\n"
    )

    monkeypatch.setattr(
        tabular_ppl.requests,
        "get",
        lambda url, *, stream, timeout: _FakeResponse(payload),
    )

    result = tabular_ppl.materialize_fivethirtyeight_marvel_wikia(tmp_path.as_posix())

    assert result["record_count"] == 1
    records = _read_records(tmp_path)
    assert records[0]["text"] == payload.decode("utf-8")
    assert '"Madame Web, Julia Carpenter",,\n' in records[0]["text"]
    assert "4043" in records[0]["text"]

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["source_license"] == "CC BY 4.0"
    assert metadata["source_manifest"]["epic_issue"] == 5005
    assert metadata["source_manifest"]["issue_numbers"] == [5059]
    assert metadata["source_manifest"]["source_urls"] == [
        tabular_ppl.FIVETHIRTYEIGHT_COMIC_CHARACTERS_DATASET_URL,
        tabular_ppl.FIVETHIRTYEIGHT_MARVEL_WIKIA_CSV_URL,
    ]


def test_tabular_raw_validation_sets_point_at_materialized_jsonl():
    datasets = tabular_ppl.tabular_raw_validation_sets()

    dataset = datasets[tabular_ppl.FIVETHIRTYEIGHT_MARVEL_SLICE_KEY]
    assert isinstance(dataset.input_path, InputName)
    assert dataset.input_path.name == "data.jsonl.gz"
    assert "tabular" in dataset.tags
    assert dataset.tags[-1] == tabular_ppl.FIVETHIRTYEIGHT_MARVEL_SLICE_KEY
