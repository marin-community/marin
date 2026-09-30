import gzip
import io
import json

import pytest

from capability_pipeline.oci_artifact import (
    ImageArtifactError,
    build_oci_metadata,
    digest,
    validate_layer,
    verify_blob,
)


def check(data, **overrides):
    kwargs = {
        "expected_digest": digest(data),
        "expected_bytes": len(data),
        "max_uncompressed_bytes": 8 << 20,
    }
    kwargs.update(overrides)
    return validate_layer(io.BytesIO(data), **kwargs)


def test_streaming_integrity_includes_full_uncompressed_content():
    payload = b"repeated data\x00" * (300_000)
    compressed = gzip.compress(payload, mtime=0)
    layer = check(compressed)
    assert layer.compressed_digest == digest(compressed)
    assert layer.compressed_bytes == len(compressed)
    assert layer.diff_id == digest(payload)
    assert layer.uncompressed_bytes == len(payload)


@pytest.mark.parametrize("removed", [1, 8, 12])
def test_truncation_rejected_even_with_matching_truncated_input_hash(removed):
    compressed = gzip.compress(b"fixture" * 1000, mtime=0)
    with pytest.raises(ImageArtifactError, match="truncated"):
        check(compressed[:-removed])


def test_corrupt_crc_is_not_suppressed():
    compressed = bytearray(gzip.compress(b"fixture", mtime=0))
    compressed[-8] ^= 1
    with pytest.raises(ImageArtifactError, match="invalid gzip"):
        check(bytes(compressed))


@pytest.mark.parametrize("suffix", [b"junk", gzip.compress(b"second", mtime=0)])
def test_trailing_or_second_member_rejected(suffix):
    with pytest.raises(ImageArtifactError, match="trailing"):
        check(gzip.compress(b"first", mtime=0) + suffix)


@pytest.mark.parametrize("delta", [-1, 1])
def test_wrong_compressed_size_rejected(delta):
    compressed = gzip.compress(b"fixture", mtime=0)
    with pytest.raises(ImageArtifactError, match="size"):
        check(compressed, expected_bytes=len(compressed) + delta)


def test_wrong_compressed_hash_rejected():
    with pytest.raises(ImageArtifactError, match="digest mismatch"):
        check(gzip.compress(b"fixture", mtime=0), expected_digest="sha256:" + "0" * 64)


def test_expansion_bound_enforced():
    with pytest.raises(ImageArtifactError, match="uncompressed layer exceeds"):
        check(gzip.compress(b"a" * (3 << 20), mtime=0), max_uncompressed_bytes=1 << 20)


def test_exact_source_config_and_deterministic_manifest():
    layer = check(gzip.compress(b"fixture", mtime=0))
    source = {
        "Env": ["PATH=/opt/bin:/usr/bin", "LANG=C"],
        "WorkingDir": "/task",
        "User": "1001:1001",
        "Entrypoint": ["/opt/bin/start"],
        "Cmd": ["--ready"],
        "Labels": {"fixture": "reviewed"},
    }
    args = {
        "image_config": source,
        "architecture": "amd64",
        "operating_system": "linux",
    }
    config, manifest = build_oci_metadata(layer, **args)
    assert (config, manifest) == build_oci_metadata(layer, **args)
    assert json.loads(config)["config"] == source
    parsed = json.loads(manifest)
    assert parsed["config"]["digest"] == digest(config)
    assert parsed["config"]["size"] == len(config)
    assert parsed["layers"][0]["digest"] == layer.compressed_digest
    assert json.loads(config)["rootfs"]["diff_ids"] == [layer.diff_id]
    verify_blob(
        config, expected_digest=parsed["config"]["digest"], expected_bytes=len(config)
    )
    with pytest.raises(ImageArtifactError, match="readback mismatch"):
        verify_blob(
            config + b"x", expected_digest=digest(config), expected_bytes=len(config)
        )


def test_unknown_source_config_is_not_replaced_with_guessed_defaults():
    layer = check(gzip.compress(b"fixture", mtime=0))
    with pytest.raises(ImageArtifactError, match="provenance fields"):
        build_oci_metadata(
            layer,
            image_config={"Env": []},
            architecture="amd64",
            operating_system="linux",
        )
