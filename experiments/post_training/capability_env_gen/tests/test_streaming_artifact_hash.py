import hashlib

from capability_pipeline.runtime import sha256, tree_sha256


def test_streaming_hash_preserves_file_and_tree_identity(tmp_path, monkeypatch):
    data = b"retained trace\n" * 100000
    path = tmp_path / "trace.jsonl"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("artifact hashing must not read the whole file at once")

    monkeypatch.setattr(type(path), "read_bytes", forbidden)
    assert sha256(path) == expected
    assert tree_sha256(tmp_path) == hashlib.sha256(
        b"trace.jsonl\0" + expected.encode() + b"\n"
    ).hexdigest()
