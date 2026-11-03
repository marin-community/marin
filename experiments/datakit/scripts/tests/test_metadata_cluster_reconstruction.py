# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rigging.timing
import xxhash
from fsspec.exceptions import FSTimeoutError

import experiments.datakit.scripts.metadata_cluster_reconstruction as reconstruction
from experiments.datakit.scripts.metadata_cluster_reconstruction import (
    aggregate_reconstruction,
    build_selected_index,
    migrate_unversioned_source_roots,
    prepare_reconstruction_plan,
    reconcile_source,
    scan_provider_file,
)
from experiments.datakit.scripts.metadata_cluster_selection import _digest
from experiments.datakit.scripts.metadata_reconstruction_migration import (
    checkpoint_inventory,
    migrate_source_checkpoints,
)
from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    reconstruction_code_digests,
)
from experiments.datakit.store.software_heritage_empty_rows import SCHEMA as SOFTWARE_HERITAGE_SCHEMA


def _inputs(tmp_path, second_text="second"):
    first_id = xxhash.xxh3_128_hexdigest(b"first")
    second_id = xxhash.xxh3_128_hexdigest(b"second")
    ledger = tmp_path / "metadata.parquet"
    rows = [
        {"id": doc_id, "normalized_row": i, "normalized_shard": "normalized", "cluster_id": 22, "marin_source": "books"}
        for i, doc_id in enumerate([first_id, first_id, second_id])
    ]
    pq.write_table(pa.Table.from_pylist(rows), ledger)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["id"], row["normalized_row"]], separators=(",", ":")).encode() + b"\n")
    marker_path = tmp_path / "selected-complete.json"
    marker_path.write_text(
        json.dumps(
            {
                "manifest": {"marin_source": "books", "normalized": "normalized"},
                "target_cluster": 22,
                "selected_occurrences": 3,
                "ordered_occurrences_sha256": digest.hexdigest(),
                "output": str(ledger),
            }
        )
    )
    descriptor = {
        "output": str(ledger),
        "source": "books",
        "selected_occurrences": 3,
        "completion_marker": str(marker_path),
        "ordered_occurrences_sha256": digest.hexdigest(),
        "status": "selected",
    }
    reports = [
        {
            "config": {"task_index": i, "task_count": 2, "inventory_sha256": "frozen", "target_cluster": 22},
            "complete": True,
            "sampled": False,
            "completed": [descriptor] if i == 0 else [],
        }
        for i in range(2)
    ]
    selection = {
        "complete": True,
        "complete_cluster": True,
        "target_cluster": 22,
        "task_count": 2,
        "inventory_sha256": "frozen",
        "selected_occurrences": 3,
        "sources": {"books": {"selected_occurrences": 3, "ledgers": [str(ledger)]}},
    }
    raw = tmp_path / "original"
    raw.mkdir()
    paths = [raw / "part-0.parquet", raw / "part-1.parquet"]
    pq.write_table(pa.Table.from_pylist([{"text": "first"}, {"text": "not selected"}]), paths[0])
    pq.write_table(pa.Table.from_pylist([{"text": "first"}, {"text": second_text}]), paths[1])
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.DIRECT)
    providers = {
        "cluster_id": 22,
        "sources": [
            {
                "marin_source": "books",
                "provider_paths": list(map(str, paths)),
                "provider_references": [
                    {"dataset": "example/books", "revision": "frozen-provider", "artifact": str(raw)}
                ],
                "recipe": asdict(ReconstructionRecipe(ProviderTransform.DIRECT, "text", "id", 128, normalizer, adapter)),
            }
        ],
    }
    plan = prepare_reconstruction_plan(selection, reports, providers, str(tmp_path / "reconstruction"))
    index_cache = tmp_path / "index-cache"
    index_cache.mkdir()
    return plan, str(index_cache), [first_id, first_id, second_id]


def test_parallel_provider_files_require_full_coverage_and_preserve_selected_multiplicity(tmp_path):
    plan, cache, ids = _inputs(tmp_path)
    index = build_selected_index(plan, "books", batch_size=2)
    assert (index["selected_occurrences"], index["distinct_ids"]) == (3, 2)
    first = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    # An unrelated source's provider recovery changes the enclosing plan, but
    # must not invalidate this source's pinned selected index and completed scan.
    plan["provider_manifest_sha256"] = "another-source-recovered"
    plan["manifest_sha256"] = _digest({key: value for key, value in plan.items() if key != "manifest_sha256"})
    assert build_selected_index(plan, "books") == index
    assert scan_provider_file(plan, "books", 0, cache) == first
    with pytest.raises(FileNotFoundError):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    assert not (Path(plan["output_prefix"]) / "_complete.json").exists()
    scan_provider_file(plan, "books", 1, cache, batch_size=1)
    source = reconcile_source(plan, "books", cache, sqlite_cache_mib=1, text_shard_bytes=5)
    rows = [row for output in source["outputs"] for row in pq.read_table(output["path"]).to_pylist()]
    assert sorted(row["id"] for row in rows) == sorted(ids)
    assert [row["text"] for row in rows].count("first") == 2
    assert source["duplicate_candidate_ids"] == 1
    assert aggregate_reconstruction(plan)["reconstructed_occurrences"] == 3


def test_complete_file_scans_with_missing_selected_text_never_certify_source_or_cluster(tmp_path):
    plan, cache, _ = _inputs(tmp_path, second_text="different provider document")
    build_selected_index(plan, "books")
    for file_index in range(2):
        scan_provider_file(plan, "books", file_index, cache)
    with pytest.raises(ValueError, match="1 selected occurrences unresolved"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    with pytest.raises(FileNotFoundError):
        aggregate_reconstruction(plan)
    assert not (Path(plan["output_prefix"]) / "_complete.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EBUSY, "Reduce your rate of simultaneous reads on the same object."),
        FSTimeoutError("HTTP read timeout"),
        OSError(errno.EIO, "An error occurred (429) when calling the HeadObject operation: Too Many Requests"),
    ],
    ids=["R2-throttle", "HTTP-timeout", "HeadObject429"],
)
def test_transient_provider_read_restarts_file_without_certifying_partial_or_duplicating_rows(
    tmp_path, monkeypatch, failure
):
    plan, cache, ids = _inputs(tmp_path)
    build_selected_index(plan, "books")
    original_rows = reconstruction.provider_rows
    calls = 0
    failed_attempt_outputs = []
    root = Path(plan["source_roots"]["books"])

    def interrupted_rows(path, columns, batch_size):
        nonlocal calls
        calls += 1
        for row in original_rows(path, columns, batch_size):
            yield row
            if calls == 1:
                failed_attempt_outputs.extend((root / "candidates/file=000000").glob("attempt=*/candidates.parquet"))
                assert not (root / "candidates/file=000000/_complete.json").exists()
                raise failure

    monkeypatch.setattr(reconstruction, "provider_rows", interrupted_rows)
    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _: None)
    marker = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    assert calls == 2
    assert len(failed_attempt_outputs) == 1
    assert Path(marker["candidates"]) not in failed_attempt_outputs
    assert len(list((root / "candidates/file=000000").glob("attempt=*/candidates.parquet"))) == 2
    assert marker["raw_rows"] == 2
    assert marker["candidate_rows"] == 1
    assert pq.read_table(marker["candidates"]).column("text").to_pylist() == ["first"]
    scan_provider_file(plan, "books", 1, cache, batch_size=1)
    completed = reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    rows = [row for output in completed["outputs"] for row in pq.read_table(output["path"]).to_pylist()]
    assert sorted(row["id"] for row in rows) == sorted(ids)


def test_provider_snapshot_head429_retries_before_reading_original_rows(tmp_path, monkeypatch):
    plan, cache, _ = _inputs(tmp_path)
    build_selected_index(plan, "books")
    raw_path = plan["sources"][0]["provider_paths"][0]
    original_filesystem_for = reconstruction.filesystem_for
    raw_fs, raw_key = original_filesystem_for(raw_path)
    calls = 0

    def info(key):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, "An error occurred (429) when calling the HeadObject operation: Too Many Requests")
        return raw_fs.info(key)

    def routed_filesystem(path):
        if path == raw_path:
            return SimpleNamespace(info=info), raw_key
        return original_filesystem_for(path)

    monkeypatch.setattr(reconstruction, "filesystem_for", routed_filesystem)
    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _: None)
    marker = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    assert calls == 3  # Failed HEAD, successful initial HEAD, successful final HEAD.
    assert (marker["raw_rows"], marker["candidate_rows"]) == (2, 1)
    assert pq.read_table(marker["candidates"]).column("text").to_pylist() == ["first"]
    assert len(list((Path(plan["source_roots"]["books"]) / "candidates/file=000000").glob("attempt=*"))) == 1


@pytest.mark.parametrize(
    ("failure", "expected_calls"),
    [
        (OSError(errno.EBUSY, "Reduce your rate of simultaneous reads on the same object."), 4),
        (FSTimeoutError("HTTP read timeout"), 4),
        (OSError(errno.EIO, "An error occurred (429) when calling the HeadObject operation: Too Many Requests"), 4),
        (OSError(errno.EIO, "provider data read failed"), 1),
        (OSError(errno.EIO, "An error occurred (403) when calling the HeadObject operation: Forbidden"), 1),
        (FileNotFoundError("HTTP404: provider file not found"), 1),
        (ValueError("Provider code digest or canonical content differs"), 1),
    ],
)
def test_failed_provider_reads_exhaust_bounded_retry_without_completion(tmp_path, monkeypatch, failure, expected_calls):
    plan, cache, _ = _inputs(tmp_path)
    build_selected_index(plan, "books")
    calls = 0

    def failed_rows(path, columns, batch_size):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(reconstruction, "provider_rows", failed_rows)
    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _: None)
    with pytest.raises(type(failure)) as raised:
        scan_provider_file(plan, "books", 0, cache)
    assert calls == expected_calls
    assert "source=books file_index=0 raw_path=" in raised.value.__notes__[0]
    root = Path(plan["source_roots"]["books"])
    assert not (root / "candidates/file=000000/_complete.json").exists()
    assert not (root / "reconstructed/_complete.json").exists()


def test_provider_candidates_are_checked_against_selected_hashes_after_scan(tmp_path):
    plan, cache, _ = _inputs(tmp_path)
    build_selected_index(plan, "books")
    first = scan_provider_file(plan, "books", 0, cache)
    scan_provider_file(plan, "books", 1, cache)
    table = pq.ParquetFile(first["candidates"]).read()
    table = table.set_column(1, "text", pa.array(["changed after scan"]))
    pq.write_table(table, first["candidates"])
    with pytest.raises(ValueError, match="does not hash"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    assert not (Path(plan["output_prefix"]) / "_complete.json").exists()


def test_provider_plan_update_reuses_legacy_source_root_only_when_source_is_unchanged(tmp_path):
    plan, _, _ = _inputs(tmp_path)
    # Model the already submitted first plan, whose workers use name-only roots.
    plan.pop("source_roots")
    plan["manifest_sha256"] = _digest({key: value for key, value in plan.items() if key != "manifest_sha256"})
    source = plan["sources"][0]
    ledgers = source["ledgers"]
    selection = {
        "complete": True,
        "complete_cluster": True,
        "target_cluster": 22,
        "task_count": 1,
        "inventory_sha256": plan["selection_inventory_sha256"],
        "selected_occurrences": 3,
        "sources": {"books": {"selected_occurrences": 3, "ledgers": [item["output"] for item in ledgers]}},
    }
    reports = [
        {
            "config": {
                "task_index": 0,
                "task_count": 1,
                "inventory_sha256": selection["inventory_sha256"],
                "target_cluster": 22,
            },
            "complete": True,
            "sampled": False,
            "completed": ledgers,
        }
    ]
    provider = {key: source[key] for key in ("marin_source", "provider_paths", "provider_references", "recipe")}
    providers = {"cluster_id": 22, "sources": [provider]}
    same = prepare_reconstruction_plan(selection, reports, providers, plan["output_prefix"], plan)
    assert same["source_roots"] == migrate_unversioned_source_roots(plan)
    previous_index = build_selected_index(same, "books")
    provider["provider_paths"] = provider["provider_paths"][:1]
    changed = prepare_reconstruction_plan(selection, reports, providers, plan["output_prefix"], plan)
    assert changed["source_roots"]["books"] != same["source_roots"]["books"]
    assert "/version=" in changed["source_roots"]["books"]
    assert build_selected_index(changed, "books")["index_path"] != previous_index["index_path"]


@pytest.mark.parametrize("corruption", ["wrong_index", "wrong_source", "missing_occurrence", "repeated_output"])
def test_cluster_completion_rejects_source_marker_with_false_coverage(tmp_path, corruption):
    plan, cache, _ = _inputs(tmp_path)
    build_selected_index(plan, "books")
    for file_index in range(2):
        scan_provider_file(plan, "books", file_index, cache)
    marker = reconcile_source(plan, "books", cache, sqlite_cache_mib=1, text_shard_bytes=5)
    if corruption == "wrong_index":
        marker["binding"]["index_sha256"] = "different selected index"
    elif corruption == "wrong_source":
        marker["marin_source"] = "another source"
    elif corruption == "missing_occurrence":
        marker["missing_occurrences"] = 1
    else:
        # Three references to the same one-row output have the expected total
        # and valid Parquet footers, but do not prove three selected occurrences.
        marker["outputs"] = [marker["outputs"][0]] * 3
    marker_path = Path(plan["source_roots"]["books"]) / "reconstructed/_complete.json"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        aggregate_reconstruction(plan)
    assert not (Path(plan["output_prefix"]) / "_complete.json").exists()


@pytest.mark.parametrize("corrupt_candidate", [False, True])
def test_checkpoint_migration_copies_only_verified_complete_candidates(tmp_path, corrupt_candidate):
    old, cache, _ = _inputs(tmp_path)
    build_selected_index(old, "books")
    candidate = scan_provider_file(old, "books", 0, cache)
    new = json.loads(json.dumps(old))
    new["output_prefix"] = str(tmp_path / "destination")
    new["source_roots"] = {"books": str(tmp_path / "destination/books")}
    new["manifest_sha256"] = _digest({key: value for key, value in new.items() if key != "manifest_sha256"})
    root = Path(old["source_roots"]["books"])
    paths = {str(root / "index/_complete.json"), str(root / "candidates/file=000000/_complete.json")}
    inventory = checkpoint_inventory(old, new, paths, workers=1)
    assert (inventory["indices"], inventory["candidate_files"]) == (1, 1)
    if corrupt_candidate:
        table = pq.ParquetFile(candidate["candidates"]).read().set_column(1, "text", pa.array(["tampered text"]))
        pq.write_table(table, candidate["candidates"])
        with pytest.raises(ValueError, match="canonical selected ID"):
            migrate_source_checkpoints(inventory["sources"][0])
        assert not (Path(new["source_roots"]["books"]) / "candidates/file=000000/_complete.json").exists()
        return
    migrate_source_checkpoints(inventory["sources"][0])
    copied_index = build_selected_index(new, "books")
    assert copied_index["index_path"].startswith(new["source_roots"]["books"])
    copied = scan_provider_file(new, "books", 0, cache)
    assert copied["candidate_digest"] == candidate["candidate_digest"]
    assert copied["candidates"].startswith(new["source_roots"]["books"])
    assert not (Path(new["source_roots"]["books"]) / "candidates/file=000001/_complete.json").exists()


def _ctx_inputs(tmp_path, selected_first_row=16):
    plan, cache, _ = _inputs(tmp_path)
    source = plan["sources"][0]
    raw = Path(source["provider_paths"][0])
    values = [b"valid"] * 33
    values[5] = b"\x00GITCRYPT\x00\xff"
    pq.write_table(
        pa.table(
            {
                "repo_name": [f"repo{i}" for i in range(33)],
                "title": [f"Fix{i}" for i in range(33)],
                "ignored_text": pa.array(values, type=pa.binary()).cast(pa.string(), safe=False),
            }
        ),
        raw,
        row_group_size=7,
    )

    def expected_id(i):
        text = f"# Repository Context\nName: repo{i}\nDescription: \n\n# Pull Request\n## Fix{i}"
        return xxhash.xxh3_128_hexdigest(text.encode())

    ids = [expected_id(selected_first_row), expected_id(selected_first_row), expected_id(32)]
    ledger = source["ledgers"][0]
    rows = pq.ParquetFile(ledger["output"]).read().to_pylist()
    for row, doc_id in zip(rows, ids, strict=True):
        row["id"] = doc_id
    pq.write_table(pa.Table.from_pylist(rows), ledger["output"])
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["id"], row["normalized_row"]], separators=(",", ":")).encode() + b"\n")
    ledger["ordered_occurrences_sha256"] = digest.hexdigest()
    marker = json.loads(Path(ledger["completion_marker"]).read_text())
    marker["ordered_occurrences_sha256"] = digest.hexdigest()
    Path(ledger["completion_marker"]).write_text(json.dumps(marker))
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.DAVINCI_CTX)
    source["recipe"] = asdict(
        ReconstructionRecipe(ProviderTransform.DAVINCI_CTX, "text", "id", 128, normalizer, adapter)
    )
    source["provider_paths"] = [str(raw)]
    source["manifest_sha256"] = _digest({k: v for k, v in source.items() if k != "manifest_sha256"})
    plan["source_roots"]["books"] = str(tmp_path / "ctx-reconstructed")
    plan["manifest_sha256"] = _digest({k: v for k, v in plan.items() if k != "manifest_sha256"})
    return plan, cache, ids


def test_parallel_ctx_reader_audit_is_required_for_cache_source_and_cluster_completion(tmp_path):
    plan, cache, ids = _ctx_inputs(tmp_path)
    build_selected_index(plan, "books")
    marker = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    audit = marker["reader_audit"]
    assert (marker["raw_rows"], audit["total_rows"], audit["excluded_rows"]) == (17, 33, 16)
    assert audit["excluded_batches"][0]["faults"][0]["field"] == "ignored_text"
    marker_path = Path(plan["source_roots"]["books"]) / "candidates/file=000000/_complete.json"
    without_audit = {k: v for k, v in marker.items() if k != "reader_audit"}
    marker_path.write_text(json.dumps(without_audit))
    with pytest.raises(ValueError, match="historical reader audit"):
        scan_provider_file(plan, "books", 0, cache)
    with pytest.raises(ValueError, match="historical reader audit"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    marker_path.write_text(json.dumps(marker))
    report = reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    assert report["reader_audit"]["excluded_rows"] == 16
    assert len(report["reader_audit"]["file_audits"]) == 1
    actual = [row["id"] for output in report["outputs"] for row in pq.ParquetFile(output["path"]).read().to_pylist()]
    assert sorted(actual) == sorted(ids)
    complete = Path(plan["source_roots"]["books"]) / "reconstructed/_complete.json"
    complete.write_text(json.dumps({k: v for k, v in report.items() if k != "reader_audit"}))
    with pytest.raises(ValueError, match="historical exclusion audit"):
        aggregate_reconstruction(plan)
    with pytest.raises(ValueError, match="per-file reader audits"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    complete.write_text(json.dumps(report))
    assert aggregate_reconstruction(plan)["reconstructed_occurrences"] == 3


def test_parallel_ctx_historical_exclusion_cannot_drop_a_selected_occurrence(tmp_path):
    plan, cache, _ = _ctx_inputs(tmp_path, selected_first_row=6)
    build_selected_index(plan, "books")
    marker = scan_provider_file(plan, "books", 0, cache)
    assert marker["reader_audit"]["excluded_rows"] == 16
    assert marker["candidate_rows"] == 1
    with pytest.raises(ValueError, match="2 selected occurrences unresolved"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    root = Path(plan["source_roots"]["books"])
    assert not (root / "reconstructed/_complete.json").exists()
    assert not (Path(plan["output_prefix"]) / "_complete.json").exists()


def _swh_row(payload, *, present=True):
    return {
        "sha1_git": hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(),
        "sha1": hashlib.sha1(payload).hexdigest(),
        "content": payload,
        "present": present,
    }


def _swh_empty_inputs(tmp_path, *, include_second=True, corrupt=None):
    plan, cache, ids = _inputs(tmp_path)
    source = plan["sources"][0]
    rows = [
        _swh_row(b"first"),
        {**_swh_row(b""), "sha1": "a" * 40, "sha1_git": "b" * 40},
        {**_swh_row(b"", present=False), "sha1": "c" * 40, "sha1_git": "d" * 40},
        _swh_row(b" \n"),
    ]
    if include_second:
        rows.append(_swh_row(b"second"))
    if corrupt is not None:
        rows.append({**_swh_row(corrupt), "sha1": "f" * 40})
    raw = tmp_path / "swh-original.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SOFTWARE_HERITAGE_SCHEMA), raw, row_group_size=2)
    normalizer, adapter = reconstruction_code_digests(ProviderTransform.SOFTWARE_HERITAGE_EMPTY)
    source["recipe"] = asdict(
        ReconstructionRecipe(
            ProviderTransform.SOFTWARE_HERITAGE_EMPTY, "content", "sha1_git", 128, normalizer, adapter, bare=True
        )
    )
    source["provider_paths"] = [str(raw)]
    source["manifest_sha256"] = _digest({k: v for k, v in source.items() if k != "manifest_sha256"})
    plan["source_roots"]["books"] = str(tmp_path / "swh-reconstructed")
    plan["manifest_sha256"] = _digest({k: v for k, v in plan.items() if k != "manifest_sha256"})
    return plan, cache, ids


def test_swh_empty_faults_are_audited_without_hiding_selected_multiplicity(tmp_path):
    plan, cache, ids = _swh_empty_inputs(tmp_path)
    build_selected_index(plan, "books")
    marker = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    audit = marker["reader_audit"]
    assert (
        audit["total_rows"],
        marker["raw_rows"],
        audit["verified_nonempty_rows"],
        audit["verified_nonempty_blank_rows"],
    ) == (5, 4, 3, 1)
    assert len(audit["excluded_present_empty"]) == len(audit["existing_absent_empty"]) == 1
    excluded = audit["excluded_present_empty"][0]
    assert (excluded["raw_row"], excluded["row_group"], excluded["row_within_group"]) == (1, 0, 1)
    assert excluded["plain_identity_matches"] is excluded["git_identity_matches"] is False
    assert excluded["advertised_identity_membership"] == "unknown"
    report = reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    actual = [r["id"] for o in report["outputs"] for r in pq.read_table(o["path"]).to_pylist()]
    assert sorted(actual) == sorted(ids)
    assert report["reader_audit"]["identity_repair"] is False
    assert aggregate_reconstruction(plan)["reconstructed_occurrences"] == 3


@pytest.mark.parametrize("payload", [b"unselected corruption", b" \t"])
def test_swh_nonempty_identity_fault_remains_fatal_even_unselected_or_blank(tmp_path, payload):
    plan, cache, _ = _swh_empty_inputs(tmp_path, corrupt=payload)
    build_selected_index(plan, "books")
    with pytest.raises(ValueError, match="plain SHA1 identity"):
        scan_provider_file(plan, "books", 0, cache)
    assert not (Path(plan["source_roots"]["books"]) / "candidates/file=000000/_complete.json").exists()


def test_swh_empty_exclusion_cannot_remove_a_required_selected_id(tmp_path):
    plan, cache, _ = _swh_empty_inputs(tmp_path, include_second=False)
    build_selected_index(plan, "books")
    scan_provider_file(plan, "books", 0, cache)
    with pytest.raises(ValueError, match="1 selected occurrences unresolved"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    assert not (Path(plan["source_roots"]["books"]) / "reconstructed/_complete.json").exists()


def test_swh_cached_and_source_audits_cannot_be_missing_or_tampered(tmp_path):
    plan, cache, _ = _swh_empty_inputs(tmp_path)
    build_selected_index(plan, "books")
    marker = scan_provider_file(plan, "books", 0, cache)
    root = Path(plan["source_roots"]["books"])
    path = root / "candidates/file=000000/_complete.json"
    path.write_text(json.dumps({k: v for k, v in marker.items() if k != "reader_audit"}))
    with pytest.raises(ValueError, match="absent or different policy"):
        scan_provider_file(plan, "books", 0, cache)
    with pytest.raises(ValueError, match="absent or different policy"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    invalid = json.loads(json.dumps(marker))
    invalid["reader_audit"]["excluded_present_empty"][0]["plain_identity_matches"] = True
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="byte/hash faults"):
        scan_provider_file(plan, "books", 0, cache)
    with pytest.raises(ValueError, match="byte/hash faults"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    path.write_text(json.dumps(marker))
    report = reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    complete = root / "reconstructed/_complete.json"
    complete.write_text(json.dumps({k: v for k, v in report.items() if k != "reader_audit"}))
    with pytest.raises(ValueError, match="historical-compatible empty audit"):
        aggregate_reconstruction(plan)
    with pytest.raises(ValueError, match="per-file reader audits"):
        reconcile_source(plan, "books", cache, sqlite_cache_mib=1)


def test_swh_retry_discards_partial_empty_audit_state(tmp_path, monkeypatch):
    plan, cache, ids = _swh_empty_inputs(tmp_path)
    build_selected_index(plan, "books")
    original = reconstruction.read_software_heritage
    calls = 0

    def interrupted(path, audit, batch_size):
        nonlocal calls
        calls += 1
        for i, row in enumerate(original(path, audit, batch_size)):
            yield row
            if calls == 1 and i == 1:
                assert len(audit.excluded_present_empty) == 1
                raise FSTimeoutError("partial SWH stream")

    monkeypatch.setattr(reconstruction, "read_software_heritage", interrupted)
    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _: None)
    marker = scan_provider_file(plan, "books", 0, cache, batch_size=1)
    assert calls == 2
    assert marker["reader_audit"]["total_rows"] == 5
    assert len(marker["reader_audit"]["excluded_present_empty"]) == 1
    report = reconcile_source(plan, "books", cache, sqlite_cache_mib=1)
    assert report["reconstructed_occurrences"] == len(ids)


@pytest.mark.parametrize(("field", "value"), [("present", None), ("content", None), ("present", False)])
def test_swh_empty_policy_rejects_invalid_original_present_content_shapes(tmp_path, field, value):
    plan, cache, _ = _swh_empty_inputs(tmp_path)
    raw = plan["sources"][0]["provider_paths"][0]
    rows = pq.read_table(raw).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=SOFTWARE_HERITAGE_SCHEMA), raw)
    build_selected_index(plan, "books")
    with pytest.raises(ValueError, match=r"shape is invalid|absent content"):
        scan_provider_file(plan, "books", 0, cache)
    assert not (Path(plan["source_roots"]["books"]) / "candidates/file=000000/_complete.json").exists()


def test_swh_historical_empty_policy_cannot_classify_an_unrelated_source_as_ready(tmp_path):
    plan, _, _ = _swh_empty_inputs(tmp_path)
    kind, blocked = reconstruction._provider_kind(plan["sources"][0])
    assert kind == "blocked"
    assert blocked is not None
    assert "restricted to its frozen v1 source" in blocked
