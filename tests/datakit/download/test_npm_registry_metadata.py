# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from marin.datakit.download.npm_registry_metadata import (
    DEFAULT_OUTPUT_FILENAME,
    DownloadNpmRegistryMetadataConfig,
    NpmRegistryMetadataSource,
    download_npm_registry_metadata,
)


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture()
def local_npm_server() -> Iterator[str]:
    payloads = {
        "/express": {
            "_id": "express",
            "name": "express",
            "dist-tags": {"latest": "5.1.0"},
            "time": {
                "created": "2010-01-01T00:00:00.000Z",
                "modified": "2025-04-01T00:00:00.000Z",
                "5.1.0": "2025-03-31T00:00:00.000Z",
                "5.0.1": "2025-03-01T00:00:00.000Z",
            },
            "repository": {"type": "git", "url": "git+https://github.com/expressjs/express.git"},
            "homepage": "https://expressjs.com/",
            "bugs": {"url": "https://github.com/expressjs/express/issues"},
            "versions": {
                "5.1.0": {
                    "name": "express",
                    "version": "5.1.0",
                    "dependencies": {"accepts": "^2.0.0", "type-is": "^2.0.1"},
                    "dist": {
                        "tarball": "https://registry.npmjs.org/express/-/express-5.1.0.tgz",
                        "integrity": "sha512-express510",
                        "shasum": "express510",
                    },
                    "author": "Alice <alice@example.com>",
                },
                "5.0.1": {
                    "name": "express",
                    "version": "5.0.1",
                    "dependencies": {"debug": "^4.3.0"},
                    "dist": {
                        "tarball": "https://registry.npmjs.org/express/-/express-5.0.1.tgz",
                        "integrity": "sha512-express501",
                        "shasum": "express501",
                    },
                },
            },
            "readme": "# express docs",
        },
        "/%40babel%2Fcore": {
            "_id": "@babel/core",
            "name": "@babel/core",
            "dist-tags": {"latest": "7.27.1"},
            "time": {
                "created": "2014-01-01T00:00:00.000Z",
                "modified": "2025-04-15T00:00:00.000Z",
                "7.27.1": "2025-04-14T00:00:00.000Z",
                "7.26.0": "2025-02-20T00:00:00.000Z",
            },
            "repository": {"type": "git", "url": "https://github.com/babel/babel.git"},
            "versions": {
                "7.27.1": {
                    "name": "@babel/core",
                    "version": "7.27.1",
                    "peerDependencies": {"supports-color": "^8.0.0"},
                    "dependencies": {"@babel/types": "^7.27.1"},
                    "dist": {
                        "tarball": "https://registry.npmjs.org/@babel/core/-/core-7.27.1.tgz",
                        "integrity": "sha512-babel7271",
                        "shasum": "babel7271",
                    },
                },
                "7.26.0": {
                    "name": "@babel/core",
                    "version": "7.26.0",
                    "dependencies": {"@babel/parser": "^7.26.0"},
                    "dist": {
                        "tarball": "https://registry.npmjs.org/@babel/core/-/core-7.26.0.tgz",
                        "integrity": "sha512-babel7260",
                        "shasum": "babel7260",
                    },
                },
            },
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # stdlib signature
            payload = payloads.get(self.path)
            if payload is None:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002  # stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}/"
    finally:
        httpd.shutdown()
        thread.join()


def test_download_npm_registry_metadata_writes_version_records_and_sidecar(
    tmp_path: Path, local_npm_server: str
) -> None:
    output_dir = tmp_path / "output"
    result = download_npm_registry_metadata(
        DownloadNpmRegistryMetadataConfig(
            source=NpmRegistryMetadataSource(
                registry_base_url=local_npm_server,
                package_names=("express", "@babel/core"),
                max_versions_per_package=2,
            ),
            output_path=str(output_dir),
        )
    )

    records = _read_jsonl_gz(output_dir / DEFAULT_OUTPUT_FILENAME)
    assert len(records) == 4
    assert records[0]["id"] == "package_metadata/npm_registry_metadata#express@5.1.0"
    assert records[1]["id"] == "package_metadata/npm_registry_metadata#express@5.0.1"
    assert records[2]["id"] == "package_metadata/npm_registry_metadata#%40babel%2Fcore@7.27.1"
    assert '"dependencies":{"accepts":"^2.0.0","type-is":"^2.0.1"}' in records[0]["text"]
    assert '"integrity":"sha512-express510"' in records[0]["text"]
    assert '"tarball":"https://registry.npmjs.org/@babel/core/-/core-7.27.1.tgz"' in records[2]["text"]
    assert '"peerDependencies":{"supports-color":"^8.0.0"}' in records[2]["text"]
    assert "alice@example.com" not in records[0]["text"]
    assert "# express docs" not in records[0]["text"]

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert result["metadata_path"] == str(output_dir / "metadata.json")
    assert metadata["source_manifest"]["slice_key"] == "package_metadata/npm_registry_metadata"
    assert metadata["source_manifest"]["policy"]["eval_only"] is True
    assert metadata["materialized_output"]["record_count"] == 4
    assert metadata["materialized_output"]["metadata"]["packages"][0]["selected_versions"] == ["5.1.0", "5.0.1"]
