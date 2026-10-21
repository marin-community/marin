# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI behavior for Iris transfers and private Hugging Face destinations."""

import json
import os
import sys

import pytest
import rigging.fsutil.huggingface as huggingface_module
import rigging.fsutil.remote as remote_module
from click.testing import CliRunner
from httpx import Request, Response
from huggingface_hub.errors import BucketNotFoundError
from rigging.fsutil.cli import cli


@pytest.mark.parametrize("operation", ["cp", "mv"])
def test_remote_transfer_forwards_saved_hf_token_without_putting_it_in_command(tmp_path, monkeypatch, operation):
    capture = tmp_path / "submission.json"
    iris = tmp_path / "iris"
    iris.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FSUTIL_CAPTURE_PATH'], 'w') as output:\n"
        "    json.dump({'args': sys.argv[1:], 'token': os.environ.get('HF_TOKEN')}, output)\n"
    )
    iris.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("FSUTIL_CAPTURE_PATH", str(capture))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(remote_module, "get_token", lambda: "saved-token")

    result = CliRunner().invoke(
        cli,
        [operation, "--remote", "s3://marin-us-east-02a/example", "hf://buckets/marin/test/example"],
    )

    assert result.exit_code == 0, result.output
    submission = json.loads(capture.read_text())
    assert submission["token"] == "saved-token"
    assert "saved-token" not in submission["args"]
    assert submission["args"] == [
        "--cluster=cw-us-east-02a",
        "job",
        "run",
        "--cpu",
        "1",
        "--memory",
        "2GB",
        "--disk",
        "5GB",
        "--sync-package",
        "marin-rigging",
        "--",
        "fsutil",
        operation,
        "s3://marin-us-east-02a/example",
        "hf://buckets/marin/test/example",
    ]


def test_new_hf_bucket_is_private(monkeypatch):
    calls = []

    def missing_bucket(_bucket_id):
        raise BucketNotFoundError(
            "missing", response=Response(404, request=Request("GET", "https://huggingface.co/api/buckets"))
        )

    monkeypatch.setattr(huggingface_module, "bucket_info", missing_bucket)
    monkeypatch.setattr(huggingface_module, "create_bucket", lambda *args, **kwargs: calls.append((args, kwargs)))

    huggingface_module.prepare_hf_destination("hf://buckets/marin/test/example")

    assert calls == [(("marin/test",), {"private": True, "exist_ok": True})]


def test_existing_hf_bucket_keeps_visibility(monkeypatch):
    monkeypatch.setattr(huggingface_module, "bucket_info", lambda _bucket_id: object())
    monkeypatch.setattr(
        huggingface_module, "create_bucket", lambda *args, **kwargs: pytest.fail("existing bucket recreated")
    )

    huggingface_module.prepare_hf_destination("hf://buckets/marin/existing/file")
