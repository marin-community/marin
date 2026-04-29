# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest

from experiments import exp5056_raw_web_markup_ppl as raw_web_markup
from levanter.analysis.perplexity_gap import iter_raw_text_documents
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text import HfDatasetSourceConfig
from marin.evaluation.perplexity_gap import (
    RawTextEvaluationDataset,
    _to_dataset_component,
    raw_text_dataset,
)


class _SingleShardRawSource(ShardedDataSource[dict]):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    @property
    def shard_names(self) -> list[str]:
        return ["records"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        assert shard_name == "records"
        return iter(self.records[row:])


def test_prefixed_raw_web_markup_validation_sets_prefixes_each_slice() -> None:
    warc = RawTextEvaluationDataset(input_path="raw/common_crawl/warc.jsonl.gz")
    wat = RawTextEvaluationDataset(input_path="raw/common_crawl/wat.jsonl.gz")

    prefixed = raw_web_markup.prefixed_raw_web_markup_validation_sets(
        {
            "cc_warc_html": warc,
            "cc_wat_json": wat,
        }
    )

    assert prefixed == {
        f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/cc_warc_html": warc,
        f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/cc_wat_json": wat,
    }


def test_raw_web_markup_raw_validation_sets_reads_active_registry(monkeypatch) -> None:
    svg = RawTextEvaluationDataset(input_path="raw/svg_stack/svg.xml.jsonl.gz")

    monkeypatch.setattr(raw_web_markup, "ACTIVE_RAW_WEB_MARKUP_DATASETS", {"svg_xml": svg})

    assert raw_web_markup.raw_web_markup_raw_validation_sets() == {
        f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/svg_xml": svg
    }


def test_raw_web_markup_raw_validation_sets_registers_svg_stack_hf_slices() -> None:
    datasets = raw_web_markup.raw_web_markup_raw_validation_sets()

    val_key = f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/svg_stack/svg_xml_val"
    test_key = f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/svg_stack/svg_xml_test"

    assert set(datasets) == {val_key, test_key}

    val_dataset = datasets[val_key]
    assert val_dataset.hf_dataset_id == "starvector/svg-stack"
    assert val_dataset.text_key == "Svg"
    assert val_dataset.split == "val"
    assert val_dataset.tags == (
        "raw_web_markup",
        "issue:5056",
        "source:svg_stack",
        "surface:svg_xml",
        "split:val",
    )


def test_svg_stack_slice_materializes_as_hf_dataset_component() -> None:
    datasets = raw_web_markup.raw_web_markup_raw_validation_sets()
    component = _to_dataset_component(datasets[f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/svg_stack/svg_xml_test"])

    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.id == "starvector/svg-stack"
    assert component.source.splits == ["test"]
    assert component.split == "test"
    assert component.format.text_key == "Svg"
    assert component.tags == ["raw_web_markup", "issue:5056", "source:svg_stack", "surface:svg_xml", "split:test"]


def test_svg_stack_slice_runtime_uses_configured_hf_split(monkeypatch) -> None:
    requested_splits: list[str] = []

    def get_shard_source(self: HfDatasetSourceConfig, split: str) -> ShardedDataSource[dict] | None:
        requested_splits.append(split)
        if split != "test":
            return None
        return _SingleShardRawSource([{"Svg": "<svg></svg>"}])

    monkeypatch.setattr(HfDatasetSourceConfig, "get_shard_source", get_shard_source)

    datasets = raw_web_markup.raw_web_markup_raw_validation_sets()
    component = _to_dataset_component(datasets[f"{raw_web_markup.RAW_WEB_MARKUP_PREFIX}/svg_stack/svg_xml_test"])

    documents = list(
        iter_raw_text_documents(
            {"svg": component},
            max_docs_per_dataset=None,
            max_doc_bytes=None,
        )
    )

    assert requested_splits == ["test"]
    assert len(documents) == 1
    assert documents[0].text == "<svg></svg>"
    assert documents[0].tags == (
        "raw_web_markup",
        "issue:5056",
        "source:svg_stack",
        "surface:svg_xml",
        "split:test",
        "svg",
    )


def test_file_backed_raw_web_markup_dataset_rejects_non_validation_split() -> None:
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/raw_web_markup.jsonl.gz", split="test")
