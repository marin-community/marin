# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adoption: registering pre-existing data as a managed ``name@version``.

An adopted artifact points at data already on disk — consumers resolve to that
source and nothing is recomputed — while a provenance record at the canonical
address puts the alias under the build-once guard.
"""

import pytest
from marin.execution.lazy import Dataset, Recipe, adopt, lower, materialized_config
from marin.execution.registry import ImmutableArtifactError, read_record
from marin.execution.step_runner import StepRunner


def _run(artifact) -> None:
    StepRunner().run([lower(artifact)])


def test_adopt_resolves_to_source_not_canonical():
    art = adopt("datasets/external", "v1", source="gs://elsewhere/data")
    # A consumer resolves to the pre-existing location, not {prefix}/{name}/{version}.
    assert art.path("gs://prefix") == "gs://elsewhere/data"


def test_adopt_records_pointer_at_canonical_address(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "pre_existing").mkdir()
    _run(adopt("datasets/external", "v1", source="pre_existing"))

    record = read_record(f"{tmp_path}/datasets/external/v1")
    assert record is not None
    assert record.source == f"{tmp_path}/pre_existing"
    assert record.output_path == f"{tmp_path}/datasets/external/v1"


def test_readopt_same_source_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "src").mkdir()
    _run(adopt("datasets/x", "v1", source="src"))
    _run(adopt("datasets/x", "v1", source="src"))


def test_readopt_different_source_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _run(adopt("datasets/x", "v1", source="a"))
    with pytest.raises(ImmutableArtifactError):
        _run(adopt("datasets/x", "v1", source="b"))


def test_adopting_missing_data_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # The adopt step runs in a runner worker thread, so the missing-data error surfaces
    # as the runner's wrapping RuntimeError with the FileNotFoundError as its cause.
    with pytest.raises(RuntimeError) as exc_info:
        _run(adopt("datasets/x", "v1", source="does_not_exist"))
    causes = []
    err: BaseException | None = exc_info.value
    while err is not None:
        causes.append(err)
        err = err.__cause__
    assert any(isinstance(err, FileNotFoundError) for err in causes)


def test_adopted_artifact_is_a_usable_dependency(tmp_path):
    (tmp_path / "tokens").mkdir()
    tokens = adopt("datasets/tokens", "v1", source="tokens")
    consumer = Dataset(
        name="checkpoints/model",
        version="v1",
        recipe=Recipe(
            fn=lambda config: config,
            build_config=lambda ctx: {"data": ctx.path(tokens)},
            deps=(tokens,),
        ),
    )
    config = materialized_config(consumer, str(tmp_path))
    assert config["data"] == f"{tmp_path}/tokens"


def test_adopt_and_pin_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Dataset(
            name="datasets/x",
            version="v1",
            recipe=Recipe(fn=lambda config: config, build_config=lambda ctx: None),
            override_path="some/pin",
            adopt_source="some/source",
        )
