# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adoption: registering pre-existing data as a managed ``name@version``.

An adopted artifact points at data already on disk — consumers resolve to that source and
nothing is recomputed — while a provenance record at the canonical address records the alias.
"""

import logging

import pytest
from marin.execution.artifact import Dataset, read_record
from marin.execution.lazy import Lazy, Recipe, adopt, lower, materialized_config, run


def test_adopt_resolves_to_source_not_canonical():
    art = adopt("datasets/external", "v1", source="gs://elsewhere/data")
    # A consumer resolves to the pre-existing location, not {prefix}/{name}/{version}.
    assert art.path("gs://prefix") == "gs://elsewhere/data"
    assert art.result_type is Dataset


def test_adopt_records_pointer_at_canonical_address(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "pre_existing").mkdir()
    run(adopt("datasets/external", "v1", source="pre_existing"))

    record = read_record(f"{tmp_path}/datasets/external/v1")
    assert record is not None
    assert record.source == f"{tmp_path}/pre_existing"
    assert record.output_path == f"{tmp_path}/datasets/external/v1"


def test_readopt_same_source_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "src").mkdir()
    run(adopt("datasets/x", "v1", source="src"))
    run(adopt("datasets/x", "v1", source="src"))


def test_readopt_different_source_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run(adopt("datasets/x", "v1", source="a"))

    # A different source re-fingerprints: advisory warning, not an error.
    with caplog.at_level(logging.WARNING):
        run(adopt("datasets/x", "v1", source="b"))
    assert "drift" in "\n".join(r.getMessage() for r in caplog.records)


def test_adopting_missing_data_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # The adopt step runs in a runner worker thread, so the missing-data error surfaces as
    # the runner's wrapping RuntimeError with the FileNotFoundError as its cause.
    with pytest.raises(RuntimeError) as exc_info:
        run(adopt("datasets/x", "v1", source="does_not_exist"))
    causes = []
    err: BaseException | None = exc_info.value
    while err is not None:
        causes.append(err)
        err = err.__cause__
    assert any(isinstance(err, FileNotFoundError) for err in causes)


def test_adopted_artifact_is_a_usable_dependency(tmp_path):
    (tmp_path / "tokens").mkdir()
    tokens = adopt("datasets/tokens", "v1", source="tokens")
    consumer = Lazy(
        name="checkpoints/model",
        version="v1",
        result_type=Dataset,
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
        Lazy(
            name="datasets/x",
            version="v1",
            result_type=Dataset,
            recipe=Recipe(fn=lambda config: config, build_config=lambda ctx: None),
            override_path="some/pin",
            adopt_source="some/source",
        )


def test_lower_adopted_handle_has_no_deps_guard(tmp_path, monkeypatch):
    """An adopted handle lowers cleanly (its recipe is the no-op adopt recipe)."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    spec = lower(adopt("datasets/x", "v1", source="src"))
    assert spec.override_output_path == "datasets/x/v1"
