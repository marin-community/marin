# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from experiments import exp5056_raw_web_markup_ppl as raw_web_markup
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset


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
        os.path.join(raw_web_markup.RAW_WEB_MARKUP_PREFIX, "cc_warc_html"): warc,
        os.path.join(raw_web_markup.RAW_WEB_MARKUP_PREFIX, "cc_wat_json"): wat,
    }


def test_raw_web_markup_raw_validation_sets_reads_active_registry(monkeypatch) -> None:
    svg = RawTextEvaluationDataset(input_path="raw/svg_stack/svg.xml.jsonl.gz")

    monkeypatch.setattr(raw_web_markup, "ACTIVE_RAW_WEB_MARKUP_DATASETS", {"svg_xml": svg})

    assert raw_web_markup.raw_web_markup_raw_validation_sets() == {
        os.path.join(raw_web_markup.RAW_WEB_MARKUP_PREFIX, "svg_xml"): svg
    }
