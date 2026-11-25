# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adoption of pre-existing data as ``name@version`` handles.

Consumers resolve an adopted artifact to its existing source. Registered adoption records the
alias at its canonical address. Read-only adoption verifies the source.
"""

import logging

import pytest
from marin.execution.artifact import Artifact, read_record
from marin.execution.lazy import ArtifactStep, materialized_config, run


def test_adopt_resolves_to_source_not_canonical():
    art = ArtifactStep.adopt("datasets/external", "2026.06.28", source="gs://elsewhere/data")
    # A consumer resolves to the pre-existing location, not {prefix}/{name}/{version}.
    assert art.path("gs://prefix") == "gs://elsewhere/data"
    assert art.artifact_type is Artifact


@pytest.mark.parametrize("read_only", [False, True], ids=["register", "read-only"])
def test_adopt_writes_a_pointer_only_when_registered(tmp_path, monkeypatch, read_only):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source = tmp_path / "pre_existing"
    source.mkdir()
    run(ArtifactStep.adopt("datasets/external", "2026.06.28", source="pre_existing", read_only=read_only))

    record = read_record(f"{tmp_path}/datasets/external/2026.06.28")
    if read_only:
        assert record is None
        assert not (tmp_path / "datasets/external/2026.06.28").exists()
        assert not any(source.iterdir())
        return
    assert record is not None
    assert record.source == f"{tmp_path}/pre_existing"
    assert record.output_path == f"{tmp_path}/datasets/external/2026.06.28"


def test_readopt_same_source_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "src").mkdir()
    run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="src"))
    run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="src"))


@pytest.mark.parametrize("read_only_first", [False, True])
def test_read_only_input_cannot_share_a_scheduled_output(tmp_path, monkeypatch, read_only_first):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "external").mkdir()
    output = ArtifactStep.adopt("datasets/x", "2026.06.28", source="external")
    read_only = ArtifactStep.adopt("datasets/alias", "2026.06.28", source="datasets/x/2026.06.28", read_only=True)
    ordered = (read_only, output) if read_only_first else (output, read_only)
    with pytest.raises(ValueError, match="both a read-only input and a step output"):
        run(*ordered)


def test_readopt_different_source_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="a"))

    # A different source re-fingerprints: advisory warning, not an error.
    with caplog.at_level(logging.WARNING):
        run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="b"))
    assert "drift" in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("read_only", [False, True], ids=["register", "read-only"])
def test_adopting_missing_data_fails(tmp_path, monkeypatch, read_only):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # The adopt step runs in a runner worker thread, so the missing-data error surfaces as
    # the runner's wrapping RuntimeError with the FileNotFoundError as its cause.
    with pytest.raises(RuntimeError) as exc_info:
        run(ArtifactStep.adopt("datasets/x", "2026.06.28", source="does_not_exist", read_only=read_only))
    causes = []
    err: BaseException | None = exc_info.value
    while err is not None:
        causes.append(err)
        err = err.__cause__
    assert any(isinstance(err, FileNotFoundError) for err in causes)


def test_adopted_artifact_is_a_usable_dependency(tmp_path):
    (tmp_path / "tokens").mkdir()
    tokens = ArtifactStep.adopt("datasets/tokens", "2026.06.28", source="tokens")
    consumer = ArtifactStep(
        name="checkpoints/model",
        version="2026.06.28",
        artifact_type=Artifact,
        run=lambda config: config,
        build_config=lambda ctx: {"data": ctx.artifact_path(tokens)},
        deps=(tokens,),
    )
    config = materialized_config(consumer, str(tmp_path))
    assert config["data"] == f"{tmp_path}/tokens"


def test_adopt_and_pin_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ArtifactStep(
            name="datasets/x",
            version="2026.06.28",
            artifact_type=Artifact,
            run=lambda config: config,
            build_config=lambda ctx: None,
            override_path="some/pin",
            adopt_source="some/source",
        )


def test_lower_adopted_handle_has_no_deps_guard(tmp_path, monkeypatch):
    """An adopted handle lowers cleanly (its recipe is the no-op adopt recipe)."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    spec = ArtifactStep.adopt("datasets/x", "2026.06.28", source="src").lower()
    assert spec.override_output_path == "datasets/x/2026.06.28"
