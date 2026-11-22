# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io

import pytest

from experiments.post_training.task_curation.environment import Download
from experiments.post_training.task_curation.images import build


@pytest.mark.parametrize("valid_digest", [True, False])
def test_verbatim_download_is_published_only_after_checksum_validation(tmp_path, monkeypatch, valid_digest):
    content = b"model weights" * 100_000
    digest = hashlib.sha256(content).hexdigest() if valid_digest else "0" * 64
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url: io.BytesIO(content))
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous")
    download = Download("https://example.invalid/model.bin", digest)
    if valid_digest:
        build.materialize(download, target)
        assert target.read_bytes() == content
    else:
        with pytest.raises(ValueError, match="SHA-256"):
            build.materialize(download, target)
        assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
