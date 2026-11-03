# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.store.davinci_provider_rows import DavinciCtxReadAudit, read_davinci_context
from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    normalized_provider_documents,
    reconstruction_code_digests,
)
from experiments.datakit.store.reconstruct_selected_source import reconstruct_selected_source


def _original_with_encrypted_content(tmp_path, edited_title="Fix6"):
    encrypted = b"\x00GITCRYPT\x00" + b"\x00" * 5 + b"\x9craw-encrypted-bytes"
    values = [f"text{i}".encode() for i in range(33)]
    values[5] = encrypted
    # Preserve an original provider defect: physical bytes are not UTF8, but
    # the Parquet leaf is declared as a string inside a nested file struct.
    content = pa.array(values, type=pa.binary()).cast(pa.string(), safe=False)
    files = pa.ListArray.from_arrays(
        pa.array(range(34), type=pa.int32()),
        pa.StructArray.from_arrays([pa.array(["a.py"] * 33), content], names=["path", "content"]),
    )
    raw = tmp_path / "original.parquet"
    pq.write_table(
        pa.table(
            {
                "repo_name": [f"repo{i}" for i in range(33)],
                "title": [edited_title if i == 6 else f"Fix{i}" for i in range(33)],
                "relevant_files": files,
            }
        ),
        raw,
        row_group_size=7,
    )
    return raw, encrypted


def test_historical_filter_keeps_exact16row_boundaries_and_audits_encrypted_original_bytes(tmp_path):
    raw, encrypted = _original_with_encrypted_content(tmp_path)
    audit = DavinciCtxReadAudit()
    recovered = list(read_davinci_context(str(raw), audit))
    assert [row["repo_name"] for row in recovered] == [f"repo{i}" for i in range(16, 33)]
    report = audit.report()
    assert (report["total_rows"], report["decoded_rows"], report["excluded_rows"]) == (33, 17, 16)
    assert report["policy"]["batch_size"] == 16
    rejected = report["excluded_batches"]
    assert len(rejected) == 1
    assert (rejected[0]["batch_index"], rejected[0]["raw_row_start"], rejected[0]["raw_row_count"]) == (0, 0, 16)
    fault = rejected[0]["faults"][0]
    assert (fault["raw_row"], fault["field"], fault["byte_start"]) == (5, "relevant_files[0].content", 15)
    assert fault["original_bytes_sha256"] == hashlib.sha256(encrypted).hexdigest()
    assert report["complete"] and report["expected_rows"] == 33
    original_batch_hash = rejected[0]["original_batch_arrow_ipc_sha256"]
    changed, _ = _original_with_encrypted_content(tmp_path, edited_title="Different excluded valid row")
    changed_audit = DavinciCtxReadAudit()
    assert list(read_davinci_context(str(changed), changed_audit)) == recovered
    changed_batch = changed_audit.report()["excluded_batches"][0]
    assert changed_batch["faults"][0]["original_bytes_sha256"] == fault["original_bytes_sha256"]
    assert changed_batch["original_batch_arrow_ipc_sha256"] != original_batch_hash


def test_historical_raw_exclusion_never_silently_removes_a_selected_document(tmp_path):
    raw, _ = _original_with_encrypted_content(tmp_path)
    n, t = reconstruction_code_digests(ProviderTransform.DAVINCI_CTX)
    recipe = ReconstructionRecipe(ProviderTransform.DAVINCI_CTX, "text", "id", 128, n, t)

    def document(i):
        return next(
            normalized_provider_documents(
                recipe,
                [
                    {
                        "repo_name": f"repo{i}",
                        "title": f"Fix{i}",
                        "relevant_files": [{"path": "a.py", "content": f"text{i}"}],
                    }
                ],
            )
        )

    expected = [document(16), document(32)]
    selected = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "ctx", "id": d["id"]} for d in expected]), selected)
    good = tmp_path / "good"
    report = reconstruct_selected_source([str(selected)], [str(raw)], str(good), marin_source="ctx", recipe=recipe)
    recovered = pq.ParquetFile(good / "text.parquet").read().to_pylist()
    assert sorted(row["text"] for row in recovered) == sorted(d["text"] for d in expected)
    assert report["historical_provider_filter"][0]["excluded_rows"] == 16
    excluded_selected = tmp_path / "excluded-selected.parquet"
    pq.write_table(pa.Table.from_pylist([{"marin_source": "ctx", "id": document(6)["id"]}]), excluded_selected)
    incomplete = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="1 selected occurrences unresolved"):
        reconstruct_selected_source(
            [str(excluded_selected)], [str(raw)], str(incomplete), marin_source="ctx", recipe=recipe
        )
    assert not (incomplete / "_complete.json").exists()
    assert json.loads((incomplete / "report.json").read_text())["missing_occurrences"] == 1
