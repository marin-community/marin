import hashlib
from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.vllm_source import ATTENTION_HELPER_PATH, verify_vllm_attention_source


def test_installed_helper_must_match_qualified_bytes_without_rewriting(tmp_path, monkeypatch):
    helper = tmp_path / ATTENTION_HELPER_PATH
    helper.parent.mkdir(parents=True)
    qualified_source = b"qualified attention helper\n"
    helper.write_bytes(qualified_source)
    expected = hashlib.sha256(qualified_source).hexdigest()
    monkeypatch.setattr(
        "skyrl_train.inference_engines.vllm_source.metadata.distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: tmp_path / path),
    )
    verify_vllm_attention_source(expected)
    unqualified_source = b"old attention helper\n"
    helper.write_bytes(unqualified_source)
    actual = hashlib.sha256(unqualified_source).hexdigest()
    with pytest.raises(RuntimeError, match=f"expected {expected}, actual {actual}"):
        verify_vllm_attention_source(expected)
    assert helper.read_bytes() == unqualified_source
