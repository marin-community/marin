# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark examples exercise extraction and the production lexical matcher."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.decon import NGramConfig, RequiredEvalManifest, build_eval_bloom, decon_to_parquet
from marin.datakit.normalize import NormalizedData

from experiments.datakit.decontam import pinned_references
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
def benchmark_decisions(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference-behavior")
    evals = root / "evals.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": name, "text": sample["text"]} for name, sample in SAMPLES.items()]), evals
    )
    documents = []
    for name, sample in SAMPLES.items():
        tokens = sample["text"].split()
        near_miss = " ".join(f"replacement{i}" if i % 7 == 0 else token for i, token in enumerate(tokens))
        documents.extend(
            [{"id": f"{name}:copy", "text": sample["text"]}, {"id": f"{name}:near-miss", "text": near_miss}]
        )
    inputs = root / "inputs"
    inputs.mkdir()
    pq.write_table(pa.Table.from_pylist(documents), inputs / "part.parquet")
    ngram = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
    with set_current_client(LocalClient()):
        build_eval_bloom(eval_data_sources=str(evals), output_path=str(root / "bloom"), ngram=ngram)
        result = decon_to_parquet(
            normalized_data=NormalizedData(main_output_dir=str(inputs), dup_output_dir=str(root / "dups"), counters={}),
            output_path=str(root / "marked"),
            prebuilt_bloom_dir=str(root / "bloom"),
            ngram=ngram,
        )
    rows = pq.read_table(result.main_output_dir).to_pylist()
    return {row["id"]: row["contaminated"] for row in rows}


@pytest.mark.parametrize("name", SAMPLES)
def test_benchmark_problem_is_detected_and_near_miss_is_retained(name, benchmark_decisions):
    source = next(source for source in SOURCES if source.name == name)
    sample = SAMPLES[name]
    [record] = reference_records(iter([sample["row"]]), source)
    assert record["id"] == sample["record_id"]
    assert record["text"] == sample["text"]
    assert benchmark_decisions[f"{source.name}:copy"] is True
    assert benchmark_decisions[f"{source.name}:near-miss"] is False


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


def test_changed_reference_bytes_are_rejected_before_bloom_build(tmp_path, monkeypatch):
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
        build_eval_bloom(
            eval_data_sources=str(output),
            output_path=str(tmp_path / "bloom"),
            additional_required_eval_manifests=(
                RequiredEvalManifest(str(output / "_manifest.json"), manifest["corpus_version"], (source.name,)),
            ),
        )
    assert not (tmp_path / "bloom" / "_bloom" / "filter.bin").exists()


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
        build_eval_bloom(
            eval_data_sources=str(output),
            output_path=str(tmp_path / "bloom"),
            additional_required_eval_manifests=(
                RequiredEvalManifest(
                    str(output / "_manifest.json"),
                    manifest["corpus_version"],
                    (source.name,),
                    reference_sources_sha256((updated,)),
                ),
            ),
        )
    assert not (tmp_path / "bloom" / "_bloom" / "filter.bin").exists()
