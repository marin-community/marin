# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.download.uncheatable_eval import (
    UncheatableEvalTransformConfig,
    transform_uncheatable_eval,
)


def test_transform_uncheatable_eval_splits_categories(tmp_path) -> None:
    input_path = tmp_path / "input" / "data"
    input_path.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "content": "BBC text",
                    "category": "bbc_news",
                    "url": "https://example.com/news",
                },
                {
                    "content": "C++ text",
                    "category": "github_cpp",
                    "url": "https://example.com/code",
                },
                {
                    "content": "Excluded text",
                    "category": "ao3_nonenglish",
                    "url": "https://example.com/excluded",
                },
            ]
        ),
        input_path / "test.parquet",
    )

    output_path = tmp_path / "output"
    result = transform_uncheatable_eval(
        UncheatableEvalTransformConfig(
            input_path=str(tmp_path / "input"),
            output_path=str(output_path),
            categories=("bbc_news", "github_cpp"),
        )
    )

    assert result == {"categories": ["bbc_news", "github_cpp"]}
    with gzip.open(output_path / "bbc_news.jsonl.gz", "rt") as f:
        assert json.load(f) == {
            "id": "https://example.com/news",
            "text": "BBC text",
            "source": "uncheatable_eval/bbc_news",
        }
    with gzip.open(output_path / "github_cpp.jsonl.gz", "rt") as f:
        assert json.load(f) == {
            "id": "https://example.com/code",
            "text": "C++ text",
            "source": "uncheatable_eval/github_cpp",
        }
    assert not (output_path / "ao3_nonenglish.jsonl.gz").exists()
