# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from fsspec.asyn import sync
from rigging.filesystem.atomic import atomic_rename, fetch_file_atomic, unique_temp_path
from s3fs import S3FileSystem


@pytest.fixture
def s3_objects(monkeypatch):
    objects = {}
    missing_copies = set()
    filesystem = S3FileSystem(key="test", secret="test", skip_instance_cache=True)
    client = sync(filesystem.loop, filesystem.set_session)

    async def request(operation, params):
        key = params.get("Key")
        if operation == "HeadObject":
            if key not in objects:
                raise ClientError({"Error": {"Code": "404", "Message": "Not found"}}, operation)
            return {"ContentLength": len(objects[key])}
        if operation == "ListObjectsV2":
            prefix = params.get("Prefix", "")
            delimiter = params.get("Delimiter", "")
            contents = []
            prefixes = set()
            for name, data in sorted(objects.items()):
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix) :]
                if delimiter and delimiter in suffix:
                    prefixes.add(prefix + suffix.split(delimiter, 1)[0] + delimiter)
                else:
                    contents.append({"Key": name, "Size": len(data)})
            return {
                "Contents": contents,
                "CommonPrefixes": [{"Prefix": name} for name in sorted(prefixes)],
                "IsTruncated": False,
            }
        if operation == "CopyObject":
            source = params["CopySource"]["Key"]
            if source not in objects or source in missing_copies:
                raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, operation)
            objects[key] = objects[source]
            return {}
        if operation == "DeleteObjects":
            deleted = params["Delete"]["Objects"]
            for item in deleted:
                objects.pop(item["Key"], None)
            return {"Deleted": deleted}
        raise AssertionError(f"Unexpected S3 operation: {operation}")

    # Fake the remote API, retaining s3fs/fsspec's real listing and move logic.
    monkeypatch.setattr(client, "_make_api_call", request)
    yield filesystem, objects, missing_copies
    sync(filesystem.loop, client.close)


@pytest.mark.parametrize("output", ["s3://bucket/cache", "bucket/cache"])
def test_atomic_rename_moves_s3_directory_contents(s3_objects, output):
    filesystem, objects, _ = s3_objects
    with atomic_rename(output, filesystem) as temp_path:
        prefix = temp_path.removeprefix("s3://bucket/").removeprefix("bucket/")
        objects[f"{prefix}/ledger.json"] = b"ledger"
        objects[f"{prefix}/data/0"] = b"chunk"

    assert objects == {"cache/ledger.json": b"ledger", "cache/data/0": b"chunk"}


def test_atomic_rename_propagates_missing_s3_object(s3_objects):
    filesystem, objects, missing_copies = s3_objects
    with pytest.raises(FileNotFoundError):
        with atomic_rename("s3://bucket/cache", filesystem) as temp_path:
            prefix = temp_path.removeprefix("s3://bucket/")
            objects[f"{prefix}/ledger.json"] = b"ledger"
            missing_copies.add(f"{prefix}/ledger.json")

    assert objects == {f"{prefix}/ledger.json": b"ledger"}


def test_unique_temp_path_produces_distinct_paths():
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    assert all(path.startswith("/some/output.txt.tmp.") for path in paths)


def test_atomic_rename_replaces_output(tmp_path):
    output = tmp_path / "out.txt"

    with atomic_rename(str(output)) as temp_path:
        Path(temp_path).write_text("data")

    assert output.read_text() == "data"
    assert not Path(temp_path).exists()


def test_atomic_rename_cleans_up_on_error(tmp_path):
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()


def test_fetch_file_atomic_copies_source(tmp_path):
    source = tmp_path / "remote" / "tokenizer.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"version": 1}')
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(source), str(destination)) is True
    assert destination.read_bytes() == b'{"version": 1}'


def test_fetch_file_atomic_missing_source_returns_false(tmp_path):
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)

    assert fetch_file_atomic(str(tmp_path / "remote" / "absent.json"), str(destination)) is False
    assert not destination.exists()


def test_fetch_file_atomic_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    source = tmp_path / "remote" / "tokenizer.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"version": 2}')
    destination = tmp_path / "cache" / "tokenizer.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'{"version": 1}')

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated failure finalizing the fetch")

    monkeypatch.setattr("os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated failure"):
        fetch_file_atomic(str(source), str(destination))

    assert destination.read_bytes() == b'{"version": 1}'
    assert [path.name for path in destination.parent.iterdir()] == ["tokenizer.json"]
