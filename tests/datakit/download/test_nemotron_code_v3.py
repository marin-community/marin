# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from marin.datakit.download.nemotron_code_v3 import (
    FAILURES_DIR,
    HF_DATASET_ID,
    SUCCESS_DIR,
    GitHubBlobRef,
    NemotronCodeV3MaterializeConfig,
    TransientFetchError,
    blob_ref_from_row,
    github_raw_url,
    materialize_blob,
    materialize_nemotron_code_v3,
    materialize_nemotron_code_v3_step,
)
from marin.execution.step_spec import StepSpec


def _config(**overrides) -> NemotronCodeV3MaterializeConfig:
    values = {
        "view_name": "test",
        "metadata_relative_glob": "Nemotron-Code-Metadata/part_00000.parquet",
        "allowed_languages": ("Python",),
        "max_rows": 10,
        "max_file_bytes": 100,
        "request_timeout_seconds": 5.0,
        "retry_attempts": 2,
        "raw_base_url": "http://example.test",
        "batch_rows": 1,
        "record_transient_failures": False,
    }
    values.update(overrides)
    return NemotronCodeV3MaterializeConfig(**values)


def _metadata_row(**overrides) -> dict:
    row = {
        "repo": "owner/repo",
        "rel_path": "src/main.py",
        "language": "Python",
        "commit_id": "abc1234",
    }
    row.update(overrides)
    return row


def _write_metadata(root: Path, rows: list[dict]) -> None:
    metadata_dir = root / "Nemotron-Code-Metadata"
    metadata_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), metadata_dir / "part_00000.parquet")


def _read_rows(root: Path, subdir: str) -> list[dict]:
    directory = root / subdir
    if not directory.exists():
        return []
    return [row for path in sorted(directory.glob("*.parquet")) for row in pq.read_table(path).to_pylist()]


@pytest.fixture()
def local_raw_server() -> Iterator[tuple[str, dict]]:
    state = {
        "payloads": {},
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # stdlib signature
            state["requests"].append(self.path)
            status, body, headers = state["payloads"].get(self.path, (404, b"", {}))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002  # stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host = httpd.server_address[0]
        port = httpd.server_address[1]
        yield f"http://{host}:{port}", state
    finally:
        httpd.shutdown()
        thread.join()


def test_github_raw_url_encodes_path_segments():
    ref = GitHubBlobRef(
        repo="owner/repo",
        rel_path="src/file name+#.py",
        language="Python",
        commit_id="abc/123",
    )

    assert github_raw_url(ref, raw_base_url="https://raw.example.test/") == (
        "https://raw.example.test/owner/repo/abc%2F123/src/file%20name%2B%23.py"
    )


@pytest.mark.parametrize(
    "row",
    [
        _metadata_row(repo=""),
        _metadata_row(repo="owner/repo/extra"),
        _metadata_row(rel_path="/src/main.py"),
        _metadata_row(rel_path="../src/main.py"),
        _metadata_row(language=None),
        _metadata_row(commit_id=""),
    ],
    ids=["empty-repo", "invalid-repo", "absolute-path", "parent-path", "non-string-language", "empty-commit"],
)
def test_blob_ref_from_row_rejects_malformed_metadata(row):
    with pytest.raises(ValueError):
        blob_ref_from_row(row)


def test_materializer_writes_success_rows_and_provenance(tmp_path: Path, local_raw_server):
    base_url, state = local_raw_server
    state["payloads"]["/owner/repo/abc1234/src/main.py"] = (200, b"print('hello')\n", {})
    _write_metadata(tmp_path / "input", [_metadata_row()])

    output_dir = tmp_path / "output"
    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(output_dir),
        config=_config(raw_base_url=base_url),
    )

    rows = _read_rows(output_dir, SUCCESS_DIR)
    assert counts == {"metadata_rows": 1, "fetch_attempts": 1, "successes": 1, "failures": 0}
    assert rows == [
        {
            "id": hashlib.sha256(b"print('hello')\n").hexdigest(),
            "text": "print('hello')\n",
            "source": HF_DATASET_ID,
            "repo": "owner/repo",
            "rel_path": "src/main.py",
            "commit_id": "abc1234",
            "language": "Python",
            "source_url": f"{base_url}/owner/repo/abc1234/src/main.py",
        }
    ]
    assert _read_rows(output_dir, FAILURES_DIR) == []

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["source"] == HF_DATASET_ID
    assert metadata["counts"] == counts


def test_materializer_records_permanent_fetch_failures(tmp_path: Path, local_raw_server):
    base_url, _state = local_raw_server
    _write_metadata(tmp_path / "input", [_metadata_row()])

    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        config=_config(raw_base_url=base_url),
    )

    assert counts == {"metadata_rows": 1, "fetch_attempts": 1, "successes": 0, "failures": 1}
    assert _read_rows(tmp_path / "output", SUCCESS_DIR) == []
    [failure] = _read_rows(tmp_path / "output", FAILURES_DIR)
    assert failure["status"] == "not_found"
    assert failure["http_status"] == 404
    assert failure["source_url"] == f"{base_url}/owner/repo/abc1234/src/main.py"


def test_materializer_records_invalid_metadata_without_fetching(tmp_path: Path, local_raw_server):
    base_url, state = local_raw_server
    _write_metadata(tmp_path / "input", [_metadata_row(repo="owner/repo/extra")])

    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        config=_config(raw_base_url=base_url),
    )

    assert counts == {"metadata_rows": 1, "fetch_attempts": 0, "successes": 0, "failures": 1}
    assert state["requests"] == []
    [failure] = _read_rows(tmp_path / "output", FAILURES_DIR)
    assert failure["status"] == "invalid_metadata"
    assert failure["source_url"] is None


def test_materializer_records_disallowed_language_without_fetching(tmp_path: Path, local_raw_server):
    base_url, state = local_raw_server
    _write_metadata(tmp_path / "input", [_metadata_row(language="CSS")])

    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        config=_config(raw_base_url=base_url),
    )

    assert counts == {"metadata_rows": 1, "fetch_attempts": 0, "successes": 0, "failures": 1}
    assert state["requests"] == []
    [failure] = _read_rows(tmp_path / "output", FAILURES_DIR)
    assert failure["status"] == "language"
    assert failure["source_url"] == f"{base_url}/owner/repo/abc1234/src/main.py"


def test_materializer_raises_on_transient_fetch_failures(local_raw_server):
    base_url, state = local_raw_server
    state["payloads"]["/owner/repo/abc1234/src/main.py"] = (500, b"unavailable", {})
    ref = blob_ref_from_row(_metadata_row())

    with requests.Session() as session:
        with pytest.raises(TransientFetchError):
            materialize_blob(ref, config=_config(raw_base_url=base_url), session=session)

    assert state["requests"] == [
        "/owner/repo/abc1234/src/main.py",
        "/owner/repo/abc1234/src/main.py",
    ]


def test_diagnostics_mode_records_transient_http_failures(local_raw_server):
    base_url, state = local_raw_server
    state["payloads"]["/owner/repo/abc1234/src/main.py"] = (500, b"unavailable", {})
    ref = blob_ref_from_row(_metadata_row())

    with requests.Session() as session:
        result = materialize_blob(
            ref,
            config=_config(raw_base_url=base_url, record_transient_failures=True),
            session=session,
        )

    assert result.status == "http_error"
    assert result.http_status == 500


def test_materializer_respects_max_file_bytes(tmp_path: Path, local_raw_server):
    base_url, state = local_raw_server
    state["payloads"]["/owner/repo/abc1234/src/main.py"] = (200, b"this is too long", {})
    _write_metadata(tmp_path / "input", [_metadata_row()])

    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        config=_config(raw_base_url=base_url, max_file_bytes=4),
    )

    assert counts == {"metadata_rows": 1, "fetch_attempts": 1, "successes": 0, "failures": 1}
    [failure] = _read_rows(tmp_path / "output", FAILURES_DIR)
    assert failure["status"] == "too_large"


def test_max_rows_caps_fetch_attempts_deterministically(tmp_path: Path, local_raw_server):
    base_url, state = local_raw_server
    rows = [
        _metadata_row(rel_path="src/one.py"),
        _metadata_row(rel_path="src/two.py"),
        _metadata_row(rel_path="src/three.py"),
    ]
    for row in rows:
        state["payloads"][f"/owner/repo/abc1234/{row['rel_path']}"] = (200, b"print('x')\n", {})
    _write_metadata(tmp_path / "input", rows)

    counts = materialize_nemotron_code_v3(
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        config=_config(raw_base_url=base_url, max_rows=2),
    )

    assert counts["fetch_attempts"] == 2
    assert len(state["requests"]) == 2
    assert [row["rel_path"] for row in _read_rows(tmp_path / "output", SUCCESS_DIR)] == ["src/one.py", "src/two.py"]


def test_materialization_cache_key_tracks_fetch_policy():
    metadata = StepSpec(name="raw/test-metadata")
    fast_retries = _config(request_timeout_seconds=5.0, retry_attempts=2)
    slow_retries = _config(request_timeout_seconds=20.0, retry_attempts=4)

    fast_step = materialize_nemotron_code_v3_step(metadata, config=fast_retries)
    slow_step = materialize_nemotron_code_v3_step(metadata, config=slow_retries)

    assert fast_step.hash_attrs["request_timeout_seconds"] == 5.0
    assert fast_step.hash_attrs["retry_attempts"] == 2
    assert slow_step.hash_attrs["request_timeout_seconds"] == 20.0
    assert slow_step.hash_attrs["retry_attempts"] == 4
    assert fast_step.hash_attrs != slow_step.hash_attrs
