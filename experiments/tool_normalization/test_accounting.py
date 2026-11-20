# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sqlite3

from experiments.tool_normalization.accounting import summarize


def test_accounting_includes_rule_converted_verified_exports(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"id": "row", "source": "corpus", "tokens": 12}))
    record = tmp_path / "records/row"
    export = record / "export"
    export.mkdir(parents=True)
    payload = b"verified export"
    (export / "tokens.bin").write_bytes(payload)
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "tokens": 10,
                "supervised_tokens": 7,
                "files": {"tokens.bin": hashlib.sha256(payload).hexdigest()},
            }
        )
    )
    (record / "source-reference.json").write_text(
        json.dumps({"sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    result = {"status": "rule_converted_adapter_checked", "verification": {"tokens": 10}}
    (record / "result.json").write_text(json.dumps(result))
    with sqlite3.connect(tmp_path / "queue.sqlite3") as db:
        db.execute("CREATE TABLE rows (id TEXT, source_file TEXT, status TEXT, detail TEXT)")
        db.execute("INSERT INTO rows VALUES (?,?,?,?)", ("row", str(source), result["status"], json.dumps(result)))
    report = summarize(tmp_path)
    assert report["producer_verified_exports"] == 1
    assert report["by_source"]["corpus"]["candidate_tokens"] == 10
    assert report["by_source"]["corpus"]["supervised_tokens"] == 7
    index = json.loads((tmp_path / "candidate-index.jsonl").read_text())
    assert index["id"] == "row"
    assert index["status"] == result["status"]
