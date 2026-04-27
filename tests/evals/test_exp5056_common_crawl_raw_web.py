# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.exp5056_common_crawl_raw_web import common_crawl_raw_validation_sets


def test_common_crawl_raw_validation_sets_render_paths_and_tags() -> None:
    datasets = common_crawl_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")

    warc = datasets["raw_web_markup/common_crawl_warc"]
    wat = datasets["raw_web_markup/common_crawl_wat"]

    assert warc.input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/warc.jsonl.gz"
    assert wat.input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/wat.jsonl.gz"
    assert warc.tags == (
        "raw_web_markup",
        "issue:5056",
        "source:common_crawl",
        "surface:warc",
        "crawl:CC-MAIN-2026-12",
    )
    assert wat.tags == (
        "raw_web_markup",
        "issue:5056",
        "source:common_crawl",
        "surface:wat",
        "crawl:CC-MAIN-2026-12",
    )
