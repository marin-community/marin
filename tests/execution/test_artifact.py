# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The artifact record: full-record provenance, advisory drift, dev mutability, pins.

Drives the real lower -> StepRunner pipeline against a tmp prefix so the drift check is
exercised exactly as it fires in production (before a cached SUCCESS is served), plus the
manual ``read_artifact``/``write_artifact`` payload API.
"""

import dataclasses
import json
import logging

import pytest
from marin.execution.artifact import (
    Artifact,
    FingerprintMismatchError,
    read_artifact,
    read_record,
    write_artifact,
)
from marin.execution.lazy import ArtifactStep, run
from pydantic import BaseModel

# A frozen calendar version for the toy artifacts these tests build.
V = "2026.06.28"


class Toy(Artifact):
    payload: str


def _toy(version: str, payload: str) -> ArtifactStep[Toy]:
    """A toy value artifact whose recipe (and thus fingerprint) is keyed by ``payload``."""
    return ArtifactStep(
        name="datasets/toy",
        version=version,
        artifact_type=Toy,
        run=lambda config: Toy(payload=config["payload"]),
        build_config=lambda ctx: {"out": ctx.output_path, "payload": payload},
    )


# --- the full record -----------------------------------------------------------


def test_records_full_provenance_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    art = _toy(V, "a")
    run(art)

    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert record is not None
    assert (record.name, record.version, record.fingerprint) == ("datasets/toy", V, art.fingerprint())
    # The unified record carries the full provenance, not just the payload.
    assert record.result_type.endswith(".Toy")
    assert record.provenance is not None
    assert record.provenance.command_line  # sys.argv of the launching process
    # result is the artifact's declared value fields only (``path`` is not a value field).
    assert record.result == {"payload": "a"}


def test_record_carries_fingerprint_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    art = _toy(V, "a")
    run(art)

    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert record.fingerprint_payload == art.fingerprint_payload()


def test_same_recipe_rerun_is_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))
    # Identical recipe + version: a cache hit, no error.
    run(_toy(V, "a"))


# --- advisory drift (no more ImmutableArtifactError) ---------------------------


def test_changed_recipe_warns_and_serves_cached(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))

    # Same name@version, different recipe: a warning, not an error, and the cached output stands.
    with caplog.at_level(logging.WARNING):
        run(_toy(V, "b"))

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "drift" in messages
    assert "payload: 'a' -> 'b'" in messages  # the field-level diff names the changed value

    record = read_record(f"{tmp_path}/datasets/toy/{V}")
    assert record.result == {"payload": "a"}  # original output served, not rebuilt


def test_expected_fingerprint_pin_raises_at_lower(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    art = _toy(V, "a")

    # A matching pin lowers cleanly; a stale pin fails even before the first build.
    dataclasses.replace(art, expected_fingerprint=art.fingerprint()).lower()
    with pytest.raises(FingerprintMismatchError):
        dataclasses.replace(art, expected_fingerprint="deadbeef").lower()


def test_expected_fingerprint_pin_hard_fails_drift(tmp_path, monkeypatch):
    """A pinned artifact rebuilt from a changed recipe is a hard error on the cache hit too."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    run(_toy(V, "a"))

    # Same address, different recipe, but now pinned to the *changed* recipe's fingerprint.
    changed = _toy(V, "b")
    pinned = dataclasses.replace(changed, expected_fingerprint=changed.fingerprint())
    with pytest.raises(FingerprintMismatchError):
        run(pinned)


def test_fixed_version_rejects_dev_dependency(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    dep = _toy("dev", "a")
    parent = ArtifactStep(
        name="checkpoints/p",
        version=V,
        artifact_type=Toy,
        run=lambda config: Toy(payload="p"),
        build_config=lambda ctx: {"out": ctx.output_path, "data": ctx.artifact_path(dep)},
        deps=(dep,),
    )
    with pytest.raises(ValueError, match="mutable"):
        parent.lower()


def test_dev_version_is_mutable(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    run(_toy("dev", "a"))
    # A changed recipe under a dev version rebuilds in place instead of warning.
    run(_toy("dev", "b"))

    record = read_record(f"{tmp_path}/datasets/toy/dev")
    assert record.result == {"payload": "b"}


# --- the manual typed-payload API ----------------------------------------------


class _Doc(BaseModel):
    title: str
    tokens: int


def test_write_then_read_artifact_round_trips_a_payload(tmp_path):
    out = (tmp_path / "step").as_posix()
    write_artifact(_Doc(title="x", tokens=7), out)

    loaded = read_artifact(out, _Doc)
    assert loaded == _Doc(title="x", tokens=7)


def test_read_artifact_requires_a_pydantic_schema(tmp_path):
    out = (tmp_path / "step").as_posix()
    write_artifact(_Doc(title="x", tokens=7), out)
    with pytest.raises(TypeError):
        read_artifact(out, dict)


def test_read_artifact_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_artifact((tmp_path / "nothing").as_posix(), _Doc)


# --- the legacy Ray executor .executor_info fallback ---------------------------

# A trimmed per-step ``.executor_info`` in the old ``ExecutorStepInfo`` schema, as written for the
# pinned llama3 Nemotron-CC caches: the tokenizer lives only under ``config``.
_EXECUTOR_INFO = json.dumps(
    {
        "name": "tokenized/nemotron_cc/hq_actual",
        "fn_name": "tokenize",
        "config": {"tokenizer": "meta-llama/Meta-Llama-3.1-8B", "tags": ["nemotron"]},
        "description": None,
        "override_output_path": None,
        "version": {},
        "dependencies": ["gs://bucket/raw/nemotron"],
        "output_path": "gs://bucket/tokenized/nemotron_cc/hq_actual-5af4cc",
    }
)


def test_read_record_recovers_config_from_executor_info(tmp_path):
    """A cache with only a legacy ``.executor_info`` resolves its config, not ``None``."""
    (tmp_path / ".executor_info").write_text(_EXECUTOR_INFO)

    record = read_record(tmp_path.as_posix())
    assert record is not None
    assert record.config == {"tokenizer": "meta-llama/Meta-Llama-3.1-8B", "tags": ["nemotron"]}
    assert record.name == "tokenized/nemotron_cc/hq_actual"
    assert record.deps == ["gs://bucket/raw/nemotron"]


def test_read_record_prefers_modern_record_over_executor_info(tmp_path):
    """``.executor_info`` is a last resort: a modern ``.artifact.json`` still wins."""
    (tmp_path / ".executor_info").write_text(_EXECUTOR_INFO)
    (tmp_path / ".artifact.json").write_text('{"name": "modern", "config": {"tokenizer": "gpt2"}}')

    record = read_record(tmp_path.as_posix())
    assert record.name == "modern"
    assert record.config == {"tokenizer": "gpt2"}


# --- the pre-#6649 bare-payload .artifact.json ---------------------------------

# Before #6649 the value model was serialized straight into ``.artifact.json`` with no record
# wrapper: every payload field sits at top level and there is no name/fingerprint/result_type/result.
_LEGACY_BARE_PAYLOAD = json.dumps(
    {"version": "v1", "output_dir": "gs://bucket/datakit/embed/cp/peps_52c03790", "counters": {"rows": 128}}
)


class _EmbedPayload(BaseModel):
    version: str
    output_dir: str
    counters: dict


def test_read_artifact_adopts_pre_6649_bare_payload(tmp_path):
    """A pre-#6649 bare ``.artifact.json`` loads: the whole document becomes the result payload."""
    (tmp_path / ".artifact.json").write_text(_LEGACY_BARE_PAYLOAD)

    loaded = read_artifact(tmp_path.as_posix(), _EmbedPayload)
    assert loaded == _EmbedPayload(
        version="v1", output_dir="gs://bucket/datakit/embed/cp/peps_52c03790", counters={"rows": 128}
    )


def test_read_record_leaves_a_genuine_data_record_untouched(tmp_path):
    """A modern record with identity but no ``result`` is not mistaken for a bare payload."""
    (tmp_path / ".artifact.json").write_text(
        '{"name": "datasets/x", "version": "v1", "fingerprint": "abc", "result": null}'
    )

    record = read_record(tmp_path.as_posix())
    assert record.name == "datasets/x"
    assert record.result is None
