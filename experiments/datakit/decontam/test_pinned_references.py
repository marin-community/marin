# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark examples exercise extraction, the stage-1 key scan, and the sealed-manifest gate."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.decon import ITEM_SCHEMA, build_keys
from marin.datakit.decon_keys import KeyTable, record_pairs

from experiments.datakit.decontam import pinned_references
from experiments.datakit.decontam.eval_manifests import validate_required_manifest
from experiments.datakit.decontam.pinned_references import (
    ReferenceFile,
    prepare_references,
    reference_catalog,
    reference_records,
    reference_sources_sha256,
)

SAMPLES = json.loads(Path(__file__).with_name("fixtures").joinpath("reference_samples.json").read_text())
SOURCES = reference_catalog()


@pytest.fixture(scope="module")
def stage_one_candidates(tmp_path_factory):
    """Each sample's id mapped to the items whose key its exact copy holds, with keys built from the samples as
    free-text items (as the item table holds the pinned references)."""
    root = tmp_path_factory.mktemp("reference-behavior")
    items = root / "items"
    items.mkdir()
    rows = [
        {
            "id": name,
            "eval": name,
            "suite": "coverage",
            "question": sample["text"],
            "choices": [],
            "answer": "",
            "prompt": "",
            "passage": "",
            "choices_mode": "none",
            "reason": "",
        }
        for name, sample in SAMPLES.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=ITEM_SCHEMA), items / "references.parquet")
    reference = root / "reference.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "r", "text": "an unrelated document about gardening"}]), reference)
    with set_current_client(LocalClient()):
        keys = build_keys(
            items_dir=str(items),
            reference_shards=[str(reference)],
            output_path=str(root / "keys"),
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
            max_workers=1,
        )
    locators = pq.read_table(keys.locators_path)
    table = KeyTable(locators.column("key1_hash").to_numpy(), np.arange(locators.num_rows))
    return {
        name: {pair.item for pair in record_pairs(f"{name}:copy", sample["text"], table, locators)}
        for name, sample in SAMPLES.items()
    }


@pytest.mark.parametrize("name", SAMPLES)
def test_benchmark_problem_copy_reaches_stage_two(name, stage_one_candidates):
    source = next(source for source in SOURCES if source.name == name)
    sample = SAMPLES[name]
    [record] = reference_records(iter([sample["row"]]), source)
    assert record["id"] == sample["record_id"]
    assert record["text"] == sample["text"]
    assert name in stage_one_candidates[name]


def test_empty_configured_family_does_not_publish_manifest(tmp_path, monkeypatch):
    source = next(source for source in SOURCES if source.name == "micro210-gsm8k-0shot")
    raw = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([SAMPLES[source.name]["row"]]), raw)
    source = replace(
        source,
        benchmark="absent-family",
        file=ReferenceFile(path="input.parquet", sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
    )
    monkeypatch.setattr(pinned_references, "hf_hub_download", lambda *args, **kwargs: str(raw))
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="expected 10 records, extracted 0"):
        prepare_references(str(output), (source,))
    assert not (output / "_manifest.json").exists()


def test_changed_reference_bytes_are_rejected_before_the_item_table_is_built(tmp_path, monkeypatch):
    source = next(source for source in SOURCES if source.name == "micro210-gsm8k-0shot")
    raw = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([SAMPLES[source.name]["row"]]), raw)
    source = replace(
        source,
        file=ReferenceFile(path="input.parquet", sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
        expected_records=1,
    )
    monkeypatch.setattr(pinned_references, "hf_hub_download", lambda *args, **kwargs: str(raw))
    output = tmp_path / "release"
    manifest = prepare_references(str(output), (source,))
    artifact = output / manifest["benchmarks"][0]["artifact"]
    pq.write_table(
        pa.Table.from_pylist([{"id": "changed", "text": "A different benchmark problem with changed bytes"}]), artifact
    )
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        validate_required_manifest(str(output / "_manifest.json"), manifest["corpus_version"], (source.name,))


def test_source_pin_update_cannot_reuse_sealed_reference_release(tmp_path, monkeypatch):
    source = next(source for source in SOURCES if source.name == "micro210-gsm8k-0shot")
    raw = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([SAMPLES[source.name]["row"]]), raw)
    source = replace(
        source,
        file=ReferenceFile(path="input.parquet", sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
        expected_records=1,
    )
    monkeypatch.setattr(pinned_references, "hf_hub_download", lambda *args, **kwargs: str(raw))
    output = tmp_path / "release"
    manifest = prepare_references(str(output), (source,))
    updated = replace(source, revision="a-new-dataset-commit")
    with pytest.raises(ValueError, match="source catalog differs"):
        validate_required_manifest(
            str(output / "_manifest.json"),
            manifest["corpus_version"],
            (source.name,),
            expected_source_catalog_sha256=reference_sources_sha256((updated,)),
        )
