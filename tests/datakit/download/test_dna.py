# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow.parquet as pq
from marin.datakit.download.dna import DnaDatasetSpec, dna_document_prefix, dna_document_text, write_balanced_dna


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


def _test_dataset(
    name: str,
    text_field: str,
    region_type: str,
    id_fields: tuple[str, ...],
) -> DnaDatasetSpec:
    return DnaDatasetSpec(
        name=name,
        hf_dataset_id=f"test/{name}",
        revision="revision",
        text_field=text_field,
        region_type=region_type,
        id_fields=id_fields,
        shard_globs=("*.jsonl",),
        num_download_shards=1,
    )


def test_write_balanced_dna_caps_each_source_by_utf8_text_bytes(tmp_path: Path):
    genomes = _test_dataset("genomes", "seq", "CLASS_A", ("id",))
    zoonomia = _test_dataset("zoonomia", "sequence", "CLASS_B", ("query_name", "species"))

    genomes_path = tmp_path / "genomes.jsonl"
    zoonomia_path = tmp_path / "zoonomia.jsonl"
    _write_jsonl(
        genomes_path,
        [
            {"id": "g1", "seq": "ACGT"},
            {"id": "g2", "seq": "TGCA"},
            {"id": "g3", "seq": "AAAA"},
        ],
    )
    _write_jsonl(
        zoonomia_path,
        [
            {"query_name": "q1", "species": "sp1", "sequence": "éé"},
            {"query_name": "q2", "species": "sp2", "sequence": "éé"},
            {"query_name": "q3", "species": "sp3", "sequence": "éé"},
        ],
    )

    output_path = tmp_path / "balanced"
    bytes_per_document = len(dna_document_text("ACGT", "CLASS_A").encode("utf-8"))
    write_balanced_dna(
        source_files={
            genomes: [str(genomes_path)],
            zoonomia: [str(zoonomia_path)],
        },
        output_path=str(output_path),
        target_text_bytes_per_dataset=2 * bytes_per_document,
    )

    rows = [row for path in output_path.glob("*.parquet") for row in pq.read_table(path).to_pylist()]
    bytes_by_source = {
        source: sum(len(row["text"].encode("utf-8")) for row in rows if row["source"] == source)
        for source in (genomes.hf_dataset_id, zoonomia.hf_dataset_id)
    }

    assert bytes_by_source == {
        genomes.hf_dataset_id: 2 * bytes_per_document,
        zoonomia.hf_dataset_id: 2 * bytes_per_document,
    }
    assert {row["text"] for row in rows if row["source"] == genomes.hf_dataset_id} == {
        dna_document_text("ACGT", "CLASS_A"),
        dna_document_text("TGCA", "CLASS_A"),
    }
    assert {row["region_type"] for row in rows} == {"CLASS_A", "CLASS_B"}
    assert dna_document_prefix("CLASS_A") == "[DNA]\n[REGION=CLASS_A]\n"
    assert {row["source_id"] for row in rows} == {
        "test/genomes:g1",
        "test/genomes:g2",
        "test/zoonomia:q1:sp1",
        "test/zoonomia:q2:sp2",
    }
