# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from marin.transform.huggingface.raw_text import (
    HfRawTextMaterializationConfig,
    HfRawTextRenderMode,
    HfRawTextSurfaceConfig,
    materialize_hf_raw_text,
    render_hf_raw_text,
)


class RowsHandler(BaseHTTPRequestHandler):
    rows: ClassVar[list[dict]] = []

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = [{"row_idx": offset + index, "row": row} for index, row in enumerate(self.rows[offset : offset + length])]
        payload = json.dumps({"rows": rows}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_: str, *args: object) -> None:
        return


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_render_hf_raw_text_modes() -> None:
    row = {
        "Svg": "<svg><text>Hello</text></svg>",
        "ocr_tokens": ["Hello", "world"],
        "metadata": {"title": "Book", "answers": ["A", "B"]},
    }

    assert (
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="svg",
                dataset_id="test/svg",
                config_name="default",
                split="test",
                output_filename="svg.jsonl.gz",
                render_mode=HfRawTextRenderMode.STRING_FIELD,
                field="Svg",
            ),
        )
        == "<svg><text>Hello</text></svg>"
    )
    assert (
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="ocr",
                dataset_id="test/ocr",
                config_name="default",
                split="test",
                output_filename="ocr.jsonl.gz",
                render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
                field="ocr_tokens",
            ),
        )
        == "Hello\nworld"
    )
    assert json.loads(
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="json",
                dataset_id="test/json",
                config_name="default",
                split="test",
                output_filename="json.jsonl.gz",
                render_mode=HfRawTextRenderMode.JSON_FIELDS,
                fields=("metadata.title", "metadata.answers"),
            ),
        )
    ) == {"metadata.answers": ["A", "B"], "metadata.title": "Book"}


def test_materialize_hf_raw_text_reads_dataset_viewer_rows(tmp_path: Path) -> None:
    RowsHandler.rows = [
        {"texts": ["Alpha", "Beta"], "bboxes": [[0.0, 1.0, 2.0, 3.0]], "num_text_regions": 2},
        {"texts": ["Gamma"], "bboxes": [[4.0, 5.0, 6.0, 7.0]], "num_text_regions": 1},
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), RowsHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        cfg = HfRawTextMaterializationConfig(
            output_path=str(tmp_path),
            datasets_server_url=f"http://127.0.0.1:{server.server_port}",
            surfaces=(
                HfRawTextSurfaceConfig(
                    name="textocr_ocr_strings",
                    dataset_id="test/textocr",
                    config_name="default",
                    split="TextOCR",
                    output_filename="textocr/ocr_strings.jsonl.gz",
                    render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
                    field="texts",
                    max_rows=2,
                    page_length=1,
                    license_note="apache-2.0",
                ),
            ),
        )

        result = materialize_hf_raw_text(cfg)
    finally:
        server.shutdown()
        thread.join()

    records = _read_jsonl_gz(tmp_path / "textocr" / "ocr_strings.jsonl.gz")
    assert [record["text"] for record in records] == ["Alpha\nBeta", "Gamma"]
    assert records[0]["metadata"] == {
        "config": "default",
        "split": "TextOCR",
        "row_idx": 0,
        "surface": "textocr_ocr_strings",
    }

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata[0]["license"] == "apache-2.0"
    assert metadata[0]["sampling_plan"] == "First 2 non-empty rendered rows from TextOCR."
    assert result["metadata"] == str(tmp_path / "metadata.json")
