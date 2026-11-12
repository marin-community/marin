"""Tests for content-sensitive, stable FineStore resume fingerprints."""

import hashlib
from pathlib import Path

import pytest

from eval.resume import RunFingerprint
from eval.resume.fingerprint import (
    MATERIAL_FIELDS,
    digest_file,
    digest_files,
    normalize_decoding,
    resolve_model_revision,
)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _base_kwargs(tmp_path: Path):
    """A fully-resolved, realistic from_run_inputs kwargs set (MATH500-like)."""
    data = tmp_path / "math500.jsonl"
    data.write_text('{"problem": "1+1", "answer": "2"}\n')
    grader = tmp_path / "grader.py"
    grader.write_text("def is_equiv(a, b): return a == b\n")
    return dict(
        model_repo="org/model",
        model_revision="a" * 40,
        task_name="MATH500",
        task_data_path=data,
        grader_source_path=grader,
        grader_version="hendrycks_math.v1",
        rendered_config={"task": "MATH500", "limit": None},
        apply_chat_template=True,
        chat_template="{{messages}}",
        decoding={"temperature": 0.7, "top_p": 1.0, "do_sample": False,
                  "max_new_tokens": 32768, "n": 1, "num_samples": 1},
        seed_set=[0, 1234, 1234, 1234],
        max_model_len=40960,
        num_fewshot=0,
        passk_batch_size=64,
    )


def _fp(tmp_path, **overrides):
    kw = _base_kwargs(tmp_path)
    kw.update(overrides)
    return RunFingerprint.from_run_inputs(**kw)


# --------------------------------------------------------------------------------------------
# from_run_inputs resolves the curated MATERIAL set
# --------------------------------------------------------------------------------------------
def test_from_run_inputs_resolves_material_set(tmp_path: Path):
    fp = _fp(tmp_path)
    payload = fp._canonical_payload()
    # every resolved key is in the MATERIAL partition (or a *_digest derived from it)
    for k in payload:
        assert k in MATERIAL_FIELDS, f"{k} leaked outside MATERIAL partition"
    # files were content-hashed into digests
    assert payload["task_data_digest"].startswith("sha256:")
    assert payload["grader_source_digest"].startswith("sha256:")
    assert payload["template_digest"].startswith("sha256:")
    # alias normalized: max_new_tokens collapsed to max_tokens
    assert "max_tokens" in payload
    assert "max_new_tokens" not in payload
    assert payload["max_tokens"] == 32768
    assert fp.value().startswith("sha256:")


def test_from_run_inputs_stable_across_key_order(tmp_path: Path):
    # decoding dict order / kwarg order must not change the hash
    fp1 = _fp(tmp_path, decoding={"n": 1, "temperature": 0.7, "top_p": 1.0,
                                  "do_sample": False, "max_new_tokens": 32768, "num_samples": 1})
    fp2 = _fp(tmp_path, decoding={"max_new_tokens": 32768, "num_samples": 1, "do_sample": False,
                                  "top_p": 1.0, "temperature": 0.7, "n": 1})
    assert fp1.value() == fp2.value()


# --------------------------------------------------------------------------------------------
# alias normalization
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alias", ["max_tokens", "max_new_tokens", "max_gen_toks"])
def test_alias_normalization_same_hash(tmp_path: Path, alias):
    ref = _fp(tmp_path, decoding={"temperature": 0.7, "max_tokens": 4096})
    other = _fp(tmp_path, decoding={"temperature": 0.7, alias: 4096})
    assert ref.value() == other.value(), f"{alias} should normalize to max_tokens"


def test_alias_normalization_unit():
    assert normalize_decoding({"max_gen_toks": 100}) == {"max_tokens": 100}
    assert normalize_decoding({"max_new_tokens": 100}) == {"max_tokens": 100}
    assert normalize_decoding({"max_tokens": 100}) == {"max_tokens": 100}
    # None values dropped; non-alias passes through
    assert normalize_decoding({"temperature": 0.7, "top_p": None}) == {"temperature": 0.7}


def test_alias_conflict_raises():
    with pytest.raises(ValueError):
        normalize_decoding({"max_gen_toks": 64, "max_tokens": 128})


def test_alias_agreeing_values_ok():
    # same value via two aliases is not a conflict
    assert normalize_decoding({"max_gen_toks": 64, "max_tokens": 64}) == {"max_tokens": 64}


# --------------------------------------------------------------------------------------------
# content-hash sensitivity: editing a controlling FILE changes the fingerprint
# --------------------------------------------------------------------------------------------
def test_data_file_content_change_changes_fingerprint(tmp_path: Path):
    # build run-1 kwargs ONCE (writes the files), fingerprint, then mutate the file in place and
    # re-fingerprint from the SAME kwargs (do NOT go through _fp, which rewrites the files).
    kw = _base_kwargs(tmp_path)
    fp1 = RunFingerprint.from_run_inputs(**kw)
    # mutate the data file content (e.g. a debug [:2] slice -> different rows)
    Path(kw["task_data_path"]).write_text('{"problem": "2+2", "answer": "4"}\n')
    fp2 = RunFingerprint.from_run_inputs(**kw)
    assert fp1.value() != fp2.value()
    assert "task_data_digest" in fp1.diff_material(fp2)


def test_grader_source_change_changes_fingerprint(tmp_path: Path):
    kw = _base_kwargs(tmp_path)
    fp1 = RunFingerprint.from_run_inputs(**kw)
    Path(kw["grader_source_path"]).write_text("def is_equiv(a, b): return str(a) == str(b)\n")
    fp2 = RunFingerprint.from_run_inputs(**kw)
    assert fp1.value() != fp2.value()
    assert "grader_source_digest" in fp1.diff_material(fp2)


def test_template_string_change_changes_fingerprint(tmp_path: Path):
    fp1 = _fp(tmp_path)
    fp2 = _fp(tmp_path, chat_template="{{messages}}<|im_end|>")
    assert fp1.value() != fp2.value()
    assert "template_digest" in fp1.diff_material(fp2)


def test_digest_file_helpers(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert digest_file(f) == "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert digest_file(None) is None
    with pytest.raises(FileNotFoundError):
        digest_file(tmp_path / "missing.txt")
    # multi-file grader: order-independent, content-sensitive
    g1 = tmp_path / "a.py"; g1.write_text("A")
    g2 = tmp_path / "b.py"; g2.write_text("B")
    d_ab = digest_files([g1, g2])
    d_ba = digest_files([g2, g1])
    assert d_ab == d_ba and d_ab.startswith("sha256:")
    g2.write_text("B2")
    assert digest_files([g1, g2]) != d_ab


def test_precomputed_digest_for_in_memory_dataset(tmp_path: Path):
    # HF dataset with no single file: caller hashes serialized rows, passes the digest directly
    fp = _fp(tmp_path, task_data_path=None, task_data_digest="sha256:deadbeef")
    assert fp._canonical_payload()["task_data_digest"] == "sha256:deadbeef"


# --------------------------------------------------------------------------------------------
# model revision resolution
# --------------------------------------------------------------------------------------------
def test_resolve_model_revision_pinned_passthrough():
    sha = "b" * 40
    assert resolve_model_revision("org/m", sha) == sha  # already a commit -> unchanged
    # offline / no-network: branch name passes through unchanged (no HF_HUB_OFFLINE network hit)
    assert resolve_model_revision("org/m", "main", allow_network=False) == "main"
    assert resolve_model_revision("org/m", None, allow_network=False) is None


def test_model_revision_is_material(tmp_path: Path):
    fp1 = _fp(tmp_path, model_revision="a" * 40)
    fp2 = _fp(tmp_path, model_revision="c" * 40)
    assert fp1.value() != fp2.value()
    assert "model_revision" in fp1.diff_material(fp2)
