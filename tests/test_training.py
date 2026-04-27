# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from fray import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.main import train_lm
from levanter.trainer import TrainerConfig

from marin.training.training import (
    TrainLmOnPodConfig,
    _doublecheck_paths,
    _enforce_run_id,
)


@pytest.fixture
def trainer_config():
    """Create a basic trainer config for tests."""
    return TrainerConfig(
        id="test-run",
        checkpointer=CheckpointerConfig(),
    )


@dataclasses.dataclass
class MockDataConfig:
    """Mock data config for testing."""

    cache_dir: str


@dataclasses.dataclass
class MockNestedDataConfig:
    """Mock nested data config for testing."""

    cache_dir: str
    subdir: dict


@dataclasses.dataclass
class MockNestedConfig:
    """Mock nested config for testing."""

    path: str


def test_lm_config_with_train_urls_allowed_out_of_region(trainer_config):
    """train/validation source URLs are exempt from region checks."""
    with (
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data={"train_urls": ["gs://bucket/path"]},  # type: ignore[arg-type]
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        _doublecheck_paths(config)


def test_recursive_path_checking(trainer_config):
    """Paths are checked recursively in nested structures."""
    with (
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
    ):
        nested_data = MockNestedDataConfig(
            cache_dir="gs://bucket/path", subdir={"file": "gs://bucket/other/path", "list": ["gs://bucket/another/path"]}
        )
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=nested_data,
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)


def test_dataclass_recursive_checking(trainer_config):
    """Paths are checked recursively in dataclass objects."""
    with (
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=MockNestedConfig(path="gs://bucket/path")),  # type: ignore
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)


def test_pathlib_path_handling(trainer_config):
    """pathlib.Path objects that represent GCS URIs are handled correctly."""
    with (
        patch("rigging.filesystem.marin_region", return_value="us-central1"),
        patch("rigging.filesystem.get_bucket_location", return_value="us-east1"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir=Path("gs://bucket/path")),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
        )
        with pytest.raises(ValueError, match="not in the same region"):
            _doublecheck_paths(config)


# ---------------------------------------------------------------------------
# Cross-region temp checkpoint search-path wiring (mirrortmp://)
# ---------------------------------------------------------------------------


def _make_train_config(*, output_path: str | None, run_id: str | None, impute: bool) -> TrainLmOnPodConfig:
    return TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            data={"train_urls": ["gs://bucket/path"]},  # type: ignore[arg-type]
            trainer=TrainerConfig(id=run_id, checkpointer=CheckpointerConfig()),
        ),
        resources=ResourceConfig.with_tpu("v4-8"),
        output_path=output_path,
        impute_run_id_from_output_path=impute,
    )


def test_enforce_run_id_imputed_includes_run_id_literal_in_mirror_search_path():
    """impute_run_id_from_output_path=True ⇒ search path contains the imputed run-id literal,
    not the bare ``checkpoints-temp/`` root (which would glob across all runs)."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    enforced = _enforce_run_id(config)
    search_paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/foo-abc123" in search_paths
    # Bare prefix without run-id must never appear.
    assert "mirrortmp://ttl=14d/checkpoints-temp" not in search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/" not in search_paths


def test_enforce_run_id_explicit_includes_run_id_literal_in_mirror_search_path():
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/whatever",
        run_id="my-explicit-run",
        impute=False,
    )
    enforced = _enforce_run_id(config)
    search_paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/my-explicit-run" in search_paths


def test_enforce_run_id_appends_to_existing_search_paths():
    """Pre-existing entries on the user-supplied config are preserved; mirror entry is appended."""
    base_config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    base_config = dataclasses.replace(
        base_config,
        train_config=dataclasses.replace(
            base_config.train_config,
            trainer=dataclasses.replace(
                base_config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    base_config.train_config.trainer.checkpointer,
                    temporary_search_paths=["custom://prior/entry"],
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(base_config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths == [
        "custom://prior/entry",
        "mirrortmp://ttl=14d/checkpoints-temp/foo-abc123",
    ]


def test_enforce_run_id_dedupes_existing_mirror_entry():
    """Idempotent: re-running _enforce_run_id (or a user pre-supplying the entry) doesn't duplicate."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    config = dataclasses.replace(
        config,
        train_config=dataclasses.replace(
            config.train_config,
            trainer=dataclasses.replace(
                config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    config.train_config.trainer.checkpointer,
                    temporary_search_paths=["mirrortmp://ttl=14d/checkpoints-temp/foo-abc123"],
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths.count("mirrortmp://ttl=14d/checkpoints-temp/foo-abc123") == 1


def _with_temp_base_path(config: TrainLmOnPodConfig, temp_base: str) -> TrainLmOnPodConfig:
    """Inject a ``temporary_base_path`` (simulating ``_update_config_to_use_out_path``)."""
    return dataclasses.replace(
        config,
        train_config=dataclasses.replace(
            config.train_config,
            trainer=dataclasses.replace(
                config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    config.train_config.trainer.checkpointer,
                    temporary_base_path=temp_base,
                ),
            ),
        ),
    )


def test_enforce_run_id_inlines_run_id_into_temp_base_path_in_midtraining():
    """Midtraining (impute=True ⇒ append_run_id_to_base_path=False) ⇒ Levanter's
    ``expanded_temporary_path`` would NOT append run_id, so writes would land at
    ``.../checkpoints-temp/step-N`` and the mirror search path with run_id would never
    match.  Marin must inline run_id into ``temporary_base_path`` so writes and search
    agree on the same directory."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    config = _with_temp_base_path(config, "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp")

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    # 1. Run-id is inlined.
    assert cp.temporary_base_path == "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/foo-abc123"
    # 2. expanded_temporary_path does NOT double-append (append_run_id_to_base_path=False).
    assert cp.expanded_temporary_path("foo-abc123") == "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/foo-abc123"
    # 3. Aligned with the mirror search path — write destination and search root agree.
    assert "mirrortmp://ttl=14d/checkpoints-temp/foo-abc123" in cp.temporary_search_paths


def test_enforce_run_id_does_not_double_append_when_levanter_will_append():
    """Non-midtraining (impute=False ⇒ append_run_id_to_base_path=True) ⇒ Levanter
    *will* append run_id in ``expanded_temporary_path``.  Marin must NOT pre-inline,
    or we'd end up with ``.../checkpoints-temp/run-X/run-X``."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/whatever",
        run_id="my-explicit-run",
        impute=False,
    )
    config = _with_temp_base_path(config, "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp")

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    # Marin did NOT pre-inline; Levanter appends on its own.
    assert cp.temporary_base_path == "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp"
    assert cp.append_run_id_to_base_path is True
    assert (
        cp.expanded_temporary_path("my-explicit-run")
        == "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/my-explicit-run"
    )


def test_enforce_run_id_temp_base_path_inline_is_idempotent():
    """Pre-supplied ``temporary_base_path`` already ending in ``/{run_id}`` is left alone."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    config = _with_temp_base_path(config, "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/foo-abc123")
    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer
    assert cp.temporary_base_path == "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/foo-abc123"


def test_enforce_run_id_leaves_temp_base_path_alone_if_none():
    """No ``temporary_base_path`` set (e.g. ``_update_config_to_use_out_path`` short-circuited
    on ``output_path is None``) ⇒ no inline; Marin doesn't fabricate a path."""
    config = _make_train_config(output_path=None, run_id="my-run", impute=False)
    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer
    assert cp.temporary_base_path is None


def test_temp_write_path_matches_mirror_search_path_for_resume():
    """Alignment invariant: the directory the temp writes target must equal the
    directory the mirrortmp:// search path resolves to.  If they diverge, a cross-region
    resume can't find the temp checkpoint written before preemption."""
    config = _make_train_config(
        output_path="gs://marin-us-central1/runs/foo-abc123",
        run_id=None,
        impute=True,
    )
    config = _with_temp_base_path(config, "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp")

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    write_dir = cp.expanded_temporary_path("foo-abc123")
    search_url = next(p for p in cp.temporary_search_paths if p.startswith("mirrortmp://"))

    # Strip the bucket-family prefix from each side and compare the *path* portion.
    assert write_dir.endswith("/ttl=14d/checkpoints-temp/foo-abc123")
    assert search_url.endswith("/ttl=14d/checkpoints-temp/foo-abc123")
