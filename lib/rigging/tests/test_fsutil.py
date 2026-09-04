# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fsutil's copy semantics and file rendering, over local paths — which
``filesystem_for`` routes exactly as it routes an object store."""

import bz2
import gzip
import io
import json
import lzma
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
import rigging.filesystem.bulk_deletion as bulk_deletion
import rigging.fsutil.cli as cli_module
import rigging.fsutil.transfer as transfer_module
import rigging.fsutil.verified_copy as verified_copy_module
import rigging.timing as timing
from botocore.exceptions import EndpointConnectionError
from click.testing import CliRunner
from rigging.filesystem.cross_region import CrossRegionGuardedFS
from rigging.filesystem.paged_listing import with_listing
from rigging.fsutil import deletion, listing
from rigging.fsutil.cli import cli
from rigging.fsutil.listing import MAX_PREVIEW_BYTES, read_decompressed_preview
from rigging.fsutil.render import file_lines
from rigging.fsutil.usage import (
    PrefixGroup,
    UsageStats,
    parse_byte_size,
    ranked_groups,
    scan_usage,
    threshold_prefix_groups,
)
from rigging.fsutil.verified_copy import COMPLETION_MANIFEST, VerifiedCopyError, verified_copy_prefix


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("nested")
    return tmp_path


def test_cp_handles_the_awkward_destination_shapes(tree, tmp_path, monkeypatch):
    """A prefix copy mirrors the tree; the shapes around it must not silently misfire.

    A directory without -r is an error rather than a partial copy, a missing source is
    an error rather than "0 objects", and file destinations are not silently turned
    into directories.
    """
    run = CliRunner().invoke

    result = run(cli, ["cp", "-r", str(tree / "sub"), str(tmp_path / "tree-out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tree-out" / "c.txt").read_text() == "nested"

    result = run(cli, ["cp", str(tree / "sub"), str(tmp_path / "no-flag")])
    assert result.exit_code != 0 and "-r" in result.output

    result = run(cli, ["cp", "-r", str(tree / "absent"), str(tmp_path / "missing")])
    assert result.exit_code != 0 and "does not exist" in result.output

    result = run(cli, ["cp", "-r", str(tree / "b.txt"), str(tmp_path / "file-out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "file-out").read_text() == "hello"

    monkeypatch.chdir(tmp_path)
    result = run(cli, ["cp", str(tree / "b.txt"), "bare.txt"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "bare.txt").read_text() == "hello"


def test_cp_file_to_existing_directory_preserves_source_name(tree, tmp_path):
    destination = tmp_path / "downloads"
    destination.mkdir()

    result = CliRunner().invoke(cli, ["cp", str(tree / "b.txt"), str(destination)])

    assert result.exit_code == 0, result.output
    assert (destination / "b.txt").read_text() == "hello"


def test_cp_multiple_files_places_each_beneath_destination(tree, tmp_path):
    destination = tmp_path / "downloads"

    result = CliRunner().invoke(
        cli,
        ["cp", str(tree / "b.txt"), str(tree / "sub" / "c.txt"), str(destination)],
    )

    assert result.exit_code == 0, result.output
    assert (destination / "b.txt").read_text() == "hello"
    assert (destination / "c.txt").read_text() == "nested"


def test_cp_expands_quoted_globs(tree, tmp_path):
    destination = tmp_path / "downloads"

    result = CliRunner().invoke(cli, ["cp", str(tree / "*.txt"), str(destination)])

    assert result.exit_code == 0, result.output
    assert (destination / "b.txt").read_text() == "hello"


def test_cp_no_clobber_preserves_existing_destination(tree, tmp_path):
    destination = tmp_path / "downloads"
    destination.mkdir()
    (destination / "b.txt").write_text("keep")

    result = CliRunner().invoke(cli, ["cp", "--no-clobber", str(tree / "b.txt"), str(destination)])

    assert result.exit_code == 0, result.output
    assert (destination / "b.txt").read_text() == "keep"


def test_verified_copy_publishes_content_manifest_after_verification(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "config.json").write_bytes(b"config")
    (source / "nested" / "weights.bin").write_bytes(b"weights")

    result = verified_copy_prefix(str(source), str(destination), workers=2)

    manifest = json.loads((destination / COMPLETION_MANIFEST).read_text())
    assert result.total_files == 2
    assert result.copied_files == 2
    assert result.resumed_files == 0
    assert {entry["path"]: entry["size"] for entry in manifest["files"]} == {
        "config.json": 6,
        "nested/weights.bin": 7,
    }
    assert (destination / "config.json").read_bytes() == b"config"
    assert (destination / "nested" / "weights.bin").read_bytes() == b"weights"


def test_verified_copy_retry_uses_verified_objects_without_source_reads(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    verified_copy_prefix(str(source), str(destination), workers=1)
    (destination / COMPLETION_MANIFEST).unlink()

    source_filesystem, source_path = verified_copy_module.filesystem_for(str(source))
    destination_filesystem, destination_path = verified_copy_module.filesystem_for(str(destination))
    status_filesystem, status_path = verified_copy_module.filesystem_for(f"{destination}.verified-copy-status")

    class UnreadableSource:
        def __getattr__(self, name):
            return getattr(source_filesystem, name)

        def open(self, _path, _mode):
            raise AssertionError("a verified resumed object must not reread its source")

    def routed_filesystem(url, *, fixed_upload_size=False):
        del fixed_upload_size
        if url == str(source):
            return UnreadableSource(), source_path
        if url == str(destination):
            return destination_filesystem, destination_path
        return status_filesystem, status_path

    monkeypatch.setattr(verified_copy_module, "filesystem_for", routed_filesystem)

    result = verified_copy_prefix(str(source), str(destination), workers=1)

    assert result.copied_files == 0
    assert result.resumed_files == 1
    assert (destination / "weights.bin").read_bytes() == b"weights"
    assert (destination / COMPLETION_MANIFEST).exists()


def test_verified_copy_interruption_leaves_no_completion_and_retry_resumes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "a.bin").write_bytes(b"first")
    (source / "b.bin").write_bytes(b"second")

    source_filesystem, source_path = verified_copy_module.filesystem_for(str(source))
    original_filesystem_for = verified_copy_module.filesystem_for

    class FailingSource:
        def __getattr__(self, name):
            return getattr(source_filesystem, name)

        def open(self, path, mode):
            if path.endswith("b.bin") and mode == "rb":
                raise OSError("injected source failure")
            return source_filesystem.open(path, mode)

    def failing_route(url, *, fixed_upload_size=False):
        if url == str(source):
            return FailingSource(), source_path
        return original_filesystem_for(url, fixed_upload_size=fixed_upload_size)

    monkeypatch.setattr(verified_copy_module, "filesystem_for", failing_route)
    with pytest.raises(OSError, match="injected source failure"):
        verified_copy_prefix(str(source), str(destination), workers=1)
    assert not (destination / COMPLETION_MANIFEST).exists()
    assert (destination / "a.bin").read_bytes() == b"first"

    monkeypatch.setattr(verified_copy_module, "filesystem_for", original_filesystem_for)
    result = verified_copy_prefix(str(source), str(destination), workers=1)

    assert result.copied_files == 1
    assert result.resumed_files == 1
    assert (destination / "b.bin").read_bytes() == b"second"
    assert (destination / COMPLETION_MANIFEST).exists()


def test_verified_copy_does_not_reuse_status_from_another_source(tmp_path):
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "second-source"
    destination = tmp_path / "destination"
    first_source.mkdir()
    second_source.mkdir()
    (first_source / "weights.bin").write_bytes(b"first!!")
    (second_source / "weights.bin").write_bytes(b"second!")
    verified_copy_prefix(str(first_source), str(destination), workers=1)
    (destination / COMPLETION_MANIFEST).unlink()

    result = verified_copy_prefix(str(second_source), str(destination), workers=1)

    assert result.copied_files == 1
    assert result.resumed_files == 0
    assert (destination / "weights.bin").read_bytes() == b"second!"


def test_verified_copy_same_size_source_overwrite_invalidates_resume_marker(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    source_file = source / "weights.bin"
    source_file.write_bytes(b"first!!")
    verified_copy_prefix(str(source), str(destination), workers=1)
    (destination / COMPLETION_MANIFEST).unlink()
    original_mtime = source_file.stat().st_mtime_ns
    source_file.write_bytes(b"second!")
    os.utime(source_file, ns=(original_mtime + 1_000_000_000, original_mtime + 1_000_000_000))

    result = verified_copy_prefix(str(source), str(destination), workers=1)

    assert result.copied_files == 1
    assert result.resumed_files == 0
    assert (destination / "weights.bin").read_bytes() == b"second!"


def test_verified_copy_same_size_source_overwrite_invalidates_completion(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    source_file = source / "weights.bin"
    source_file.write_bytes(b"first!!")
    verified_copy_prefix(str(source), str(destination), workers=1)
    original_mtime = source_file.stat().st_mtime_ns
    source_file.write_bytes(b"second!")
    os.utime(source_file, ns=(original_mtime + 1_000_000_000, original_mtime + 1_000_000_000))

    with pytest.raises(VerifiedCopyError, match="source identity changed"):
        verified_copy_prefix(str(source), str(destination), workers=1)

    assert (destination / "weights.bin").read_bytes() == b"first!!"


def test_verified_copy_identityless_source_rereads_content_on_completion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    source_file = source / "weights.bin"
    source_file.write_bytes(b"first!!")

    source_filesystem, source_path = verified_copy_module.filesystem_for(str(source))
    original_filesystem_for = verified_copy_module.filesystem_for

    class IdentitylessSource:
        def __getattr__(self, name):
            return getattr(source_filesystem, name)

        def find(self, path, *, detail):
            found = source_filesystem.find(path, detail=detail)
            return {
                name: {"name": name, "size": info["size"], "type": info.get("type", "file")}
                for name, info in found.items()
            }

        def info(self, path):
            info = source_filesystem.info(path)
            return {"name": path, "size": info["size"], "type": info.get("type", "file")}

    def identityless_route(url, *, fixed_upload_size=False):
        if url == str(source):
            return IdentitylessSource(), source_path
        return original_filesystem_for(url, fixed_upload_size=fixed_upload_size)

    monkeypatch.setattr(verified_copy_module, "filesystem_for", identityless_route)
    verified_copy_prefix(str(source), str(destination), workers=1)
    source_file.write_bytes(b"second!")

    with pytest.raises(VerifiedCopyError, match="source content changed"):
        verified_copy_prefix(str(source), str(destination), workers=1)


def test_verified_copy_hash_mismatch_removes_object_and_withholds_completion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")

    destination_filesystem, destination_path = verified_copy_module.filesystem_for(str(destination))
    original_filesystem_for = verified_copy_module.filesystem_for

    class CorruptingWriter:
        def __init__(self, file, path):
            self.file = file
            self.path = path

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            result = self.file.__exit__(exc_type, exc_value, traceback)
            if exc_type is None:
                path = Path(self.path)
                path.write_bytes(b"x" * path.stat().st_size)
            return result

        def write(self, data):
            return self.file.write(data)

    class CorruptingDestination:
        def __getattr__(self, name):
            return getattr(destination_filesystem, name)

        def open(self, path, mode):
            file = destination_filesystem.open(path, mode)
            if mode == "wb" and path.endswith("weights.bin"):
                return CorruptingWriter(file, path)
            return file

    def corrupting_route(url, *, fixed_upload_size=False):
        if url == str(destination):
            return CorruptingDestination(), destination_path
        return original_filesystem_for(url, fixed_upload_size=fixed_upload_size)

    monkeypatch.setattr(verified_copy_module, "filesystem_for", corrupting_route)

    with pytest.raises(VerifiedCopyError, match="destination hash mismatch"):
        verified_copy_prefix(str(source), str(destination), workers=1)

    assert not (destination / "weights.bin").exists()
    assert not (destination / COMPLETION_MANIFEST).exists()


@pytest.mark.parametrize(("command", "destination"), [(["cp", "-r"], "copy"), (["rsync"], "sync")])
def test_recursive_transfers_accept_relative_local_directories(tmp_path, monkeypatch, command, destination):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "value.txt").write_text("value")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, [*command, "source", destination])

    assert result.exit_code == 0, result.output
    assert (tmp_path / destination / "nested" / "value.txt").read_text() == "value"


def test_cp_recursive_preserves_repeated_object_key_separators(monkeypatch):
    class ObjectFileSystem:
        protocol = "s3"

        def __init__(self):
            self.files = {
                "bucket/source/a/b": b"single",
                "bucket/source/a//b": b"double",
            }

        def exists(self, path):
            prefix = f"{path.rstrip('/')}/"
            return path in self.files or any(name.startswith(prefix) for name in self.files)

        def isdir(self, path):
            prefix = f"{path.rstrip('/')}/"
            return path not in self.files and any(name.startswith(prefix) for name in self.files)

        def find(self, path, *, detail, withdirs):
            assert detail is True
            assert withdirs is True
            prefix = f"{path.rstrip('/')}/"
            return {
                name: {"name": name, "size": len(data), "type": "file"}
                for name, data in self.files.items()
                if name.startswith(prefix)
            }

        def makedirs(self, _path, *, exist_ok):
            assert exist_ok is True

        def open(self, path, mode):
            if mode == "rb":
                return io.BytesIO(self.files[path])

            filesystem = self

            class WriteBuffer(io.BytesIO):
                def close(self):
                    filesystem.files[path] = self.getvalue()
                    super().close()

            return WriteBuffer()

    filesystem = ObjectFileSystem()
    monkeypatch.setattr(
        transfer_module,
        "filesystem_for",
        lambda url: (filesystem, url.split("://", 1)[1]),
    )

    result = CliRunner().invoke(cli, ["cp", "-r", "s3://bucket/source", "s3://bucket/destination"])

    assert result.exit_code == 0, result.output
    assert filesystem.files["bucket/destination/a/b"] == b"single"
    assert filesystem.files["bucket/destination/a//b"] == b"double"


def test_mv_file_to_directory_removes_source(tree, tmp_path):
    destination = tmp_path / "archive"
    destination.mkdir()

    result = CliRunner().invoke(cli, ["mv", str(tree / "b.txt"), str(destination)])

    assert result.exit_code == 0, result.output
    assert not (tree / "b.txt").exists()
    assert (destination / "b.txt").read_text() == "hello"


def test_mv_recursive_preserves_empty_local_directories(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "empty" / "nested").mkdir(parents=True)
    (source / "value.txt").write_text("value")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["mv", "-r", "source", "moved"])

    assert result.exit_code == 0, result.output
    assert not source.exists()
    assert (tmp_path / "moved" / "empty" / "nested").is_dir()
    assert (tmp_path / "moved" / "value.txt").read_text() == "value"


def test_mv_rejects_gcs_aliases_for_the_same_object(monkeypatch):
    removed = []

    class AliasFileSystem:
        protocol = ("gs", "gcs")

        def exists(self, _path):
            return True

        def isdir(self, _path):
            return False

        def rm(self, path, *, recursive):
            removed.append((path, recursive))

    monkeypatch.setattr(
        transfer_module,
        "filesystem_for",
        lambda _url: (AliasFileSystem(), "bucket/key"),
    )

    result = CliRunner().invoke(cli, ["mv", "gs://bucket/key", "gcs://bucket/key"])

    assert result.exit_code != 0
    assert removed == []


def test_rsync_copies_changed_files_and_deletes_only_when_requested(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same.txt").write_text("same")
    (source / "changed.txt").write_text("new-value")
    (source / "new.txt").write_text("new")
    (destination / "same.txt").write_text("same")
    (destination / "changed.txt").write_text("old")
    (destination / "extra.txt").write_text("keep unless --delete is passed")

    result = CliRunner().invoke(cli, ["rsync", str(source), str(destination)])

    assert result.exit_code == 0, result.output
    assert (destination / "changed.txt").read_text() == "new-value"
    assert (destination / "new.txt").read_text() == "new"
    assert (destination / "extra.txt").exists()

    dry_run = CliRunner().invoke(cli, ["rsync", "--delete", "--dry-run", str(source), str(destination)])

    assert dry_run.exit_code == 0, dry_run.output
    assert (destination / "extra.txt").read_text() == "keep unless --delete is passed"

    result = CliRunner().invoke(cli, ["rsync", "--delete", str(source), str(destination)])

    assert result.exit_code == 0, result.output
    assert not (destination / "extra.txt").exists()
    assert {path.name: path.read_text() for path in destination.iterdir()} == {
        "same.txt": "same",
        "changed.txt": "new-value",
        "new.txt": "new",
    }


def test_rsync_checksum_detects_equal_sized_changes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "value.txt").write_text("new")
    (destination / "value.txt").write_text("old")
    timestamp = 1_700_000_000
    os.utime(source / "value.txt", (timestamp, timestamp))
    os.utime(destination / "value.txt", (timestamp, timestamp))

    result = CliRunner().invoke(cli, ["rsync", str(source), str(destination)])
    assert result.exit_code == 0, result.output
    assert (destination / "value.txt").read_text() == "old"

    result = CliRunner().invoke(cli, ["rsync", "--checksum", str(source), str(destination)])
    assert result.exit_code == 0, result.output
    assert (destination / "value.txt").read_text() == "new"


def test_rsync_detects_equal_sized_files_with_different_mtimes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "value.txt").write_text("new")
    (destination / "value.txt").write_text("old")
    os.utime(source / "value.txt", (1_700_000_001, 1_700_000_001))
    os.utime(destination / "value.txt", (1_700_000_000, 1_700_000_000))

    result = CliRunner().invoke(cli, ["rsync", str(source), str(destination)])

    assert result.exit_code == 0, result.output
    assert (destination / "value.txt").read_text() == "new"


def test_rsync_rejects_overlapping_directories(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("value")

    result = CliRunner().invoke(cli, ["rsync", str(source), str(source / "nested")])

    assert result.exit_code != 0
    assert not (source / "nested").exists()


def test_hash_reports_md5_in_base64(tmp_path):
    path = tmp_path / "digits.txt"
    path.write_bytes(b"123456789")

    result = CliRunner().invoke(cli, ["hash", str(path)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["url", "md5"]
    assert lines[2].split() == [str(path), "JfnnlDI7RTiF9RgfG2JNCw=="]


def test_rm_requires_recursive_for_directories(tree):
    run = CliRunner().invoke

    result = run(cli, ["rm", str(tree / "sub")])
    assert result.exit_code != 0
    assert (tree / "sub" / "c.txt").exists()

    result = run(cli, ["rm", "-R", str(tree / "sub")])
    assert result.exit_code == 0, result.output
    assert not (tree / "sub").exists()

    result = run(cli, ["rm", str(tree / "b.txt")])
    assert result.exit_code == 0, result.output
    assert not (tree / "b.txt").exists()


def test_rm_removes_multiple_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")

    result = CliRunner().invoke(cli, ["rm", str(first), str(second)])

    assert result.exit_code == 0, result.output
    assert not first.exists()
    assert not second.exists()


def test_rm_recursive_unlinks_local_directory_symlink_without_deleting_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    result = CliRunner().invoke(cli, ["rm", "-R", str(link)])

    assert result.exit_code == 0, result.output
    assert not link.is_symlink()
    assert (target / "keep.txt").read_text() == "keep"


def _install_object_store_filesystem(monkeypatch, fs):
    fs = bulk_deletion.with_bulk_deletion(with_listing(fs))
    for module in (deletion, listing, cli_module):
        monkeypatch.setattr(module, "filesystem_for", lambda _url, fs=fs: (fs, "bucket/prefix"))
    return fs


def test_rm_uses_s3_bulk_delete_batches(monkeypatch):
    class RecordingS3FileSystem:
        protocol = "s3"

        def __init__(self):
            self.requests = []
            self.config_kwargs = {}

        def isdir(self, _path):
            return True

        def split_path(self, path):
            bucket, key = path.split("/", 1)
            return bucket, key, None

        def call_s3(self, method, **kwargs):
            if method == "list_objects_v2":
                assert kwargs["Prefix"] == "prefix/"
                contents = [{"Key": f"prefix/{index}", "Size": 1} for index in range(1001)]
                return {"Contents": contents}
            assert method == "delete_objects"
            self.requests.append(kwargs)
            return {}

        def invalidate_cache(self):
            pass

    fs = _install_object_store_filesystem(monkeypatch, RecordingS3FileSystem())

    result = CliRunner().invoke(cli, ["rm", "-R", "s3://bucket/prefix"])

    assert result.exit_code == 0, result.output
    batches = [request["Delete"]["Objects"] for request in fs.requests]
    assert sorted(map(len, batches)) == [1, 1000]
    assert {item["Key"] for batch in batches for item in batch} == {f"prefix/{index}" for index in range(1001)}


class _BulkDeleteS3FileSystem:
    """An S3 stub whose bulk delete reports per-key failures in the response body."""

    protocol = "s3"

    def __init__(self, failures):
        self.failures = list(failures)
        self.attempts = 0
        self.config_kwargs = {}

    def isdir(self, _path):
        return True

    def split_path(self, path):
        bucket, key = path.split("/", 1)
        return bucket, key, None

    def call_s3(self, method, **kwargs):
        if method == "list_objects_v2":
            return {"Contents": [{"Key": "prefix/0", "Size": 1}]}
        self.attempts += 1
        errors = self.failures.pop(0) if self.failures else []
        return {"Errors": errors}

    def invalidate_cache(self):
        pass


def test_rm_retries_a_throttled_bulk_delete(monkeypatch):
    """S3 answers 200 and reports throttling per key, so it never raises on its own.

    A large removal provokes exactly this, and treating it as fatal aborts the run after
    a partial delete.
    """
    fs = _install_object_store_filesystem(
        monkeypatch,
        _BulkDeleteS3FileSystem([[{"Key": "prefix/0", "Code": "SlowDown"}], []]),
    )
    monkeypatch.setattr(bulk_deletion, "_DELETE_BACKOFF", timing.ExponentialBackoff(initial=0.001, maximum=0.001))

    result = CliRunner().invoke(cli, ["rm", "-R", "s3://bucket/prefix"])

    assert result.exit_code == 0, result.output
    assert fs.attempts == 2


def test_rm_does_not_retry_a_batch_holding_a_permanent_error(monkeypatch):
    failures = [
        [
            {"Key": "prefix/0", "Code": "SlowDown"},
            {"Key": "prefix/1", "Code": "AccessDenied"},
        ]
    ]
    fs = _install_object_store_filesystem(
        monkeypatch,
        _BulkDeleteS3FileSystem(failures),
    )

    result = CliRunner().invoke(cli, ["rm", "-R", "s3://bucket/prefix"])

    assert result.exit_code != 0
    assert fs.attempts == 1


def test_rm_pages_and_batches_through_a_filesystem_wrapper(monkeypatch):
    class RecordingGCSFileSystem:
        protocol = ("gs", "gcs")

        def __init__(self):
            self.batch_sizes = []

        def isdir(self, _path):
            return True

        def split_path(self, path):
            bucket, key = path.split("/", 1)
            return bucket, key, None

        def call(self, _method, _template, _bucket, *, pageToken=None, **_kwargs):
            start = int(pageToken or 0)
            end = min(start + 100, 250)
            page = {"items": [{"name": f"prefix/{index}", "size": "1"} for index in range(start, end)]}
            if end < 250:
                page["nextPageToken"] = str(end)
            return page

        def _process_object(self, bucket, item):
            return {"name": f"{bucket}/{item['name']}", "size": int(item["size"]), "type": "file"}

        def rm(self, paths, **kwargs):
            self.batch_sizes.append((len(paths), kwargs["batchsize"]))

        def invalidate_cache(self):
            pass

    inner = RecordingGCSFileSystem()
    guarded = CrossRegionGuardedFS(inner, cross_region_checker=lambda _bucket: False)
    _install_object_store_filesystem(monkeypatch, guarded)

    result = CliRunner().invoke(cli, ["rm", "-R", "gs://bucket/prefix"])

    assert result.exit_code == 0, result.output
    assert sorted(inner.batch_sizes) == [(50, 100), (100, 100), (100, 100)]


def test_rm_deletes_while_the_listing_still_streams(monkeypatch):
    """A prefix wider than one page must not wait for the whole scan before deleting.

    The first delete has to land while later pages are still outstanding; otherwise a
    prefix with tens of millions of objects buys a full scan, and the memory to hold it,
    before it removes anything.
    """

    class BlockingS3FileSystem:
        protocol = "s3"
        pages = 3

        def __init__(self):
            self.first_delete = threading.Event()
            self.streamed = False
            self.deleted_keys = set()
            self.config_kwargs = {}

        def isdir(self, _path):
            return True

        def split_path(self, path):
            bucket, key = path.split("/", 1)
            return bucket, key, None

        def call_s3(self, method, **kwargs):
            if method == "delete_objects":
                self.deleted_keys.update(item["Key"] for item in kwargs["Delete"]["Objects"])
                self.first_delete.set()
                return {}
            page = int((kwargs.get("ContinuationToken") or "token-0").removeprefix("token-"))
            start = page * 1000
            contents = [{"Key": f"prefix/{index}", "Size": 1} for index in range(start, start + 1000)]
            if page < self.pages - 1:
                return {"Contents": contents, "NextContinuationToken": f"token-{page + 1}"}
            # The final page is served only once an earlier batch has already been
            # deleted, so a scan-then-delete implementation deadlocks here instead of
            # quietly passing.
            self.streamed = self.first_delete.wait(timeout=30)
            return {"Contents": contents}

        def invalidate_cache(self):
            pass

    fs = _install_object_store_filesystem(monkeypatch, BlockingS3FileSystem())

    result = CliRunner().invoke(cli, ["rm", "-R", "s3://bucket/prefix", "--workers", "1"])

    assert result.exit_code == 0, result.output
    assert fs.streamed, "the last listing page was reached before any object was deleted"
    assert fs.deleted_keys == {f"prefix/{index}" for index in range(3000)}


def test_json_previews_render_as_tables_and_degrade_safely():
    """JSONL records become a column per key; a truncated record and binary data still
    produce something readable instead of an error."""
    table = file_lines("metrics.jsonl", b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 2.0}\n')
    assert table[0].split() == ["step", "loss"]
    assert table[2].split() == ["1", "3.5"]

    assert file_lines("metrics.jsonl", b'{"step": 1}\n{"step": 2, trunc') == ['{"step": 1}', '{"step": 2, trunc']
    assert file_lines("nested.json", json.dumps({"a": {"b": 1}}).encode())[2].split()[0] == "a"
    assert file_lines("model.bin", b"\x00\xff\xfe") == ["[binary file, 3 bytes]"]


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
        (".lzma", lambda data: lzma.compress(data, format=lzma.FORMAT_ALONE)),
    ],
)
def test_compressed_json_preview_decompresses_and_renders_as_a_table(tmp_path, suffix, compress):
    payload = b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 2.0}\n'
    path = tmp_path / f"metrics.jsonl{suffix}"
    path.write_bytes(compress(payload))

    preview = read_decompressed_preview(str(path))

    assert preview.truncated is False
    assert file_lines(path.name, preview.data)[2].split() == ["1", "3.5"]


def test_compressed_preview_applies_limit_to_decompressed_data(tmp_path):
    path = tmp_path / "large.txt.gz"
    path.write_bytes(gzip.compress(b"x" * (MAX_PREVIEW_BYTES + 1)))

    preview = read_decompressed_preview(str(path))

    assert preview.truncated is True
    assert preview.data == b"x" * MAX_PREVIEW_BYTES


def test_cat_raw_keeps_compressed_bytes(tmp_path):
    compressed = gzip.compress(b'{"step": 1}\n')
    path = tmp_path / "metrics.jsonl.gz"
    path.write_bytes(compressed)

    result = CliRunner().invoke(cli, ["cat", "--raw", str(path)])

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == compressed


@pytest.mark.parametrize("command", [["cat"], ["head", "-n", "3"]])
def test_formatted_cli_commands_decompress_json(tmp_path, command):
    path = tmp_path / "metrics.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"step": 1, "loss": 3.5}\n'))

    result = CliRunner().invoke(cli, [*command, str(path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[2].split() == ["1", "3.5"]


def test_cat_reports_full_size_when_uncompressed_preview_is_truncated(tmp_path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_text("abcdef")
    monkeypatch.setattr(listing, "MAX_PREVIEW_BYTES", 4)

    result = CliRunner().invoke(cli, ["cat", str(path)])

    assert result.exit_code == 0, result.output
    assert result.stderr == "[truncated: read 4 B of 6 B]\n"


def test_bare_url_opens_browser_for_directories_and_viewer_for_files(tree, monkeypatch):
    """A URL in command position needs no command name: directories open in the
    browser and files in the viewer."""
    opened = []
    monkeypatch.setattr(cli_module, "run_browser", lambda url: opened.append(("browse", url)))
    monkeypatch.setattr(cli_module, "show_viewer", lambda url: opened.append(("show", url)))

    assert CliRunner().invoke(cli, [str(tree)]).exit_code == 0
    assert CliRunner().invoke(cli, [str(tree / "b.txt")]).exit_code == 0
    assert opened == [("browse", str(tree)), ("show", str(tree / "b.txt"))]


def test_command_typos_still_fail_as_commands(tmp_path, monkeypatch):
    """Only path-shaped tokens take the URL fallback, so a mistyped command name is
    reported rather than opened as a missing file."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["catt"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_show_rejects_directories(tree):
    """The viewer holds one file; a directory belongs to the browser, and saying so
    beats opening curses on an unreadable path."""
    result = CliRunner().invoke(cli, ["show", str(tree)])

    assert result.exit_code != 0
    assert "use `fsutil browse" in result.output


def test_ls_long_renders_local_directory(tree):
    result = CliRunner().invoke(cli, ["ls", "-l", str(tree)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["size", "modified", "name"]
    assert any(line.endswith("b.txt") for line in lines)
    assert any(line.endswith("sub/") for line in lines)


def test_ls_recursive_renders_paths_relative_to_a_relative_directory(tree, monkeypatch):
    monkeypatch.chdir(tree.parent)

    result = CliRunner().invoke(cli, ["ls", "-R", tree.name])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["sub/", "b.txt", "sub/c.txt"]


def test_ls_glob_renders_matches_with_listing_metadata(monkeypatch):
    class GlobFileSystem:
        def glob(self, path, *, detail):
            assert path == "bucket/*/foo"
            assert detail is True
            return {
                "bucket/first/foo": {"name": "bucket/first/foo", "size": 3, "type": "file"},
                "bucket/second/foo": {"name": "bucket/second/foo", "size": 5, "type": "file"},
            }

    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (GlobFileSystem(), "bucket/*/foo"))

    result = CliRunner().invoke(cli, ["ls", "s3://bucket/*/foo"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["first/foo", "second/foo"]


def test_du_scans_directories_in_parallel_using_listing_metadata(monkeypatch):
    class ParallelListingFileSystem:
        protocol = "gcs"

        def __init__(self):
            self.child_listings_started = threading.Barrier(2)

        def info(self, path):
            assert path == "bucket/root"
            return {"name": path, "size": 0, "type": "directory"}

        def ls(self, path, *, detail):
            assert detail is True
            if path == "bucket/root":
                return [
                    {"name": "bucket/root/recon", "size": 0, "type": "directory"},
                    {"name": "bucket/root/top", "size": 5, "type": "file"},
                ]
            if path in ("bucket/root/recon/a", "bucket/root/recon/b"):
                self.child_listings_started.wait(timeout=2)
            return {
                "bucket/root/recon": [
                    {"name": "bucket/root/recon/a", "size": 0, "type": "directory"},
                    {"name": "bucket/root/recon/b", "size": 0, "type": "directory"},
                ],
                "bucket/root/recon/a": [
                    {"name": "bucket/root/recon/a/nested", "size": 0, "type": "directory"},
                    {"name": "bucket/root/recon/a/one", "size": 7, "type": "file"},
                ],
                "bucket/root/recon/b": [{"name": "bucket/root/recon/b/two", "size": 11, "type": "file"}],
                "bucket/root/recon/a/nested": [{"name": "bucket/root/recon/a/nested/three", "size": 13, "type": "file"}],
            }[path]

    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (ParallelListingFileSystem(), "bucket/root"))

    assert listing.total_size("gs://bucket/root") == (36, 4)


def test_usage_scan_descends_to_exact_prefixes_below_threshold_and_orders_by_size(monkeypatch):
    old = datetime(2020, 1, 1, tzinfo=UTC)
    recent = datetime(2026, 1, 1, tzinfo=UTC)

    def file(name, size, modified):
        return {"name": name, "size": size, "type": "file", "LastModified": modified}

    def directory(name):
        return {"name": name, "size": 0, "type": "directory"}

    listings = {
        "bucket": [directory("bucket/scratch"), directory("bucket/users"), file("bucket/root.bin", 10, old)],
        "bucket/scratch": [file("bucket/scratch/cache", 50, old)],
        "bucket/users": [directory("bucket/users/iris")],
        "bucket/users/iris": [
            directory("bucket/users/iris/a"),
            directory("bucket/users/iris/b"),
            directory("bucket/users/iris/c"),
            directory("bucket/users/iris/d"),
        ],
        "bucket/users/iris/a": [file("bucket/users/iris/a/data", 60, old)],
        "bucket/users/iris/b": [file("bucket/users/iris/b/data", 50, old)],
        "bucket/users/iris/c": [file("bucket/users/iris/c/data", 70, recent)],
        "bucket/users/iris/d": [file("bucket/users/iris/d/data", 30, recent)],
    }

    class UsageFileSystem:
        def ls(self, path, *, detail):
            assert detail is True
            return listings[path]

    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (UsageFileSystem(), "bucket"))

    scan = scan_usage("s3://bucket", workers=4)
    groups = threshold_prefix_groups(scan, threshold_bytes=100)

    assert scan.root.total == UsageStats(size_bytes=270, object_count=6, last_modified=recent)
    assert [(group.label, group.stats.size_bytes) for group in groups] == [
        ("users/iris/c/", 70),
        ("users/iris/a/", 60),
        ("scratch/", 50),
        ("users/iris/b/", 50),
        ("users/iris/d/", 30),
        ("[root objects]", 10),
    ]
    assert sum(group.stats.size_bytes for group in groups) == scan.root.total.size_bytes


def test_usage_scan_splits_large_s3_prefixes_to_bounded_depth(monkeypatch):
    modified = datetime(2025, 1, 1, tzinfo=UTC)

    class PagedS3FileSystem:
        protocol = "s3"

        def __init__(self):
            self.config_kwargs = {}
            self.requested_prefixes = []

        def split_path(self, path):
            bucket, _, key = path.partition("/")
            return bucket, key, None

        def info(self, path):
            assert path == "bucket"
            return {"name": path, "size": 0, "type": "directory"}

        def call_s3(self, method, **kwargs):
            assert method == "list_objects_v2"
            prefix = kwargs["Prefix"]
            token = kwargs.get("ContinuationToken")
            delimiter = kwargs["Delimiter"]
            self.requested_prefixes.append((prefix, delimiter, token))
            if prefix == "" and delimiter == "":
                return {
                    "Contents": [{"Key": "ignored-probe", "Size": 999, "LastModified": modified}],
                    "NextContinuationToken": "split-root",
                }
            if prefix == "" and token is None:
                return {
                    "CommonPrefixes": [{"Prefix": "users/"}],
                    "Contents": [{"Key": "root-a", "Size": 10, "LastModified": modified}],
                    "NextContinuationToken": "root-page-2",
                }
            if prefix == "" and token == "root-page-2":
                return {
                    "CommonPrefixes": [{"Prefix": "scratch//"}],
                    "Contents": [{"Key": "root-b", "Size": 20, "LastModified": modified}],
                }
            if prefix == "users/" and delimiter == "":
                return {
                    "Contents": [{"Key": "users/ignored-probe", "Size": 999, "LastModified": modified}],
                    "NextContinuationToken": "split-users",
                }
            if prefix == "users/":
                return {"CommonPrefixes": [{"Prefix": "users/iris/"}]}
            if prefix == "users/iris/" and delimiter == "":
                return {
                    "Contents": [{"Key": "users/iris/ignored-probe", "Size": 999, "LastModified": modified}],
                    "NextContinuationToken": "split-iris",
                }
            if prefix == "users/iris/":
                return {"CommonPrefixes": [{"Prefix": "users/iris/archive/"}]}
            if prefix == "users/iris/archive/" and token is None:
                return {
                    "Contents": [{"Key": "users/iris/archive/a", "Size": 10, "LastModified": modified}],
                    "NextContinuationToken": "archive-page-2",
                }
            if prefix == "users/iris/archive/":
                return {"Contents": [{"Key": "users/iris/archive/b", "Size": 20, "LastModified": modified}]}
            assert prefix == "scratch//"
            return {"Contents": [{"Key": "scratch//data", "Size": 40, "LastModified": modified}]}

    fs = with_listing(PagedS3FileSystem())
    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (fs, "bucket"))

    progress = []
    scan = scan_usage("s3://bucket", workers=3, progress=progress.append)

    assert scan.root.total == UsageStats(size_bytes=100, object_count=5, last_modified=modified)
    assert {child.prefix: child.total.size_bytes for child in scan.root.children} == {
        "scratch/": 40,
        "users/": 30,
    }
    assert {prefix for prefix, _, _ in fs.requested_prefixes} == {
        "",
        "scratch//",
        "users/",
        "users/iris/",
        "users/iris/archive/",
    }
    assert all(delimiter == "" for prefix, delimiter, _ in fs.requested_prefixes if prefix == "users/iris/archive/")
    assert progress[-1].listing_pages == 10
    assert progress[-1].prefixes_completed == progress[-1].prefixes_discovered == 5
    assert listing.total_size("s3://bucket") == (100, 5)


def test_usage_scan_retries_transient_s3_page_failures(monkeypatch):
    class ResettingS3FileSystem:
        protocol = "s3"

        def __init__(self):
            self.config_kwargs = {}
            self.requests = 0

        def split_path(self, path):
            bucket, _, key = path.partition("/")
            return bucket, key, None

        def call_s3(self, method, **_kwargs):
            assert method == "list_objects_v2"
            self.requests += 1
            if self.requests == 1:
                raise EndpointConnectionError(endpoint_url="https://bucket.example.com")
            return {"Contents": [{"Key": "data", "Size": 10}]}

    fs = with_listing(ResettingS3FileSystem())
    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (fs, "bucket"))
    monkeypatch.setattr(timing.time, "sleep", lambda _delay: None)

    scan = scan_usage("s3://bucket", workers=1)

    assert scan.root.total == UsageStats(size_bytes=10, object_count=1, last_modified=None)


def test_usage_ranking_combines_reclaimable_size_with_inactivity():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recent_huge = PrefixGroup(
        label="recent/",
        stats=UsageStats(size_bytes=2 * 1024**4, object_count=1, last_modified=datetime(2025, 12, 1, tzinfo=UTC)),
    )
    old_large = PrefixGroup(
        label="old/",
        stats=UsageStats(size_bytes=1024**4, object_count=1, last_modified=datetime(2023, 1, 1, tzinfo=UTC)),
    )

    assert ranked_groups([recent_huge, old_large], now) == [old_large, recent_huge]


def test_usage_size_parser_accepts_decimal_and_binary_units():
    assert parse_byte_size("1TB") == 1000**4
    assert parse_byte_size("1TiB") == 1024**4
    with pytest.raises(ValueError):
        parse_byte_size("1iB")
