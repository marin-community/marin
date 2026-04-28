# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
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
    _update_config_to_use_out_path,
    temporary_checkpoint_base_path,
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


def test_temporary_checkpoint_base_path_follows_output_path_region():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        assert temporary_checkpoint_base_path("gs://marin-us-east5/experiments/grug/base-trial") == (
            "gs://marin-tmp-us-east5/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
        )


def test_update_config_to_use_out_path_sets_run_specific_temp_checkpoints(trainer_config):
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v4-8"),
            output_path="gs://marin-us-east5/experiments/grug/base-trial",
        )

        updated = _update_config_to_use_out_path(config)

        checkpointer = updated.train_config.trainer.checkpointer
        assert checkpointer.base_path == "gs://marin-us-east5/experiments/grug/base-trial/checkpoints"
        assert checkpointer.temporary_base_path == (
            "gs://marin-tmp-us-east5/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
        )


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


# ``_update_config_to_use_out_path`` builds the temp base path via
# ``temporary_checkpoint_base_path``, which embeds the output-path component into the
# prefix.  The resulting structure is:
#   gs://marin-tmp-{region}/ttl=14d/checkpoints-temp/{output_component}/checkpoints[/{run_id}]
# where the run-id is appended only when ``append_run_id_to_base_path`` is True
# (non-imputed runs).  ``_enforce_run_id`` derives the mirror search URL from this
# write path so the two always agree.

_MIDTRAIN_OUTPUT = "gs://marin-us-central1/runs/foo-abc123"
_MIDTRAIN_TEMP_BASE = "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/marin-us-central1/runs/foo-abc123/checkpoints"
_MIDTRAIN_MIRROR_SEARCH = "mirrortmp://ttl=14d/checkpoints-temp/marin-us-central1/runs/foo-abc123/checkpoints"

_PRETRAIN_OUTPUT = "gs://marin-us-central1/runs/whatever"
_PRETRAIN_TEMP_BASE = "gs://marin-tmp-us-central1/ttl=14d/checkpoints-temp/marin-us-central1/runs/whatever/checkpoints"
_PRETRAIN_RUN_ID = "my-explicit-run"
# When append_run_id_to_base_path=True, Levanter appends the run_id, so the search path
# tracks the same suffix.
_PRETRAIN_MIRROR_SEARCH = (
    "mirrortmp://ttl=14d/checkpoints-temp/marin-us-central1/runs/whatever/checkpoints/" + _PRETRAIN_RUN_ID
)


def test_enforce_run_id_mirror_search_matches_imputed_temp_write_path():
    """Imputed run-id (midtraining): search URL equals temp write directory under
    ``mirrortmp://`` so cross-region resume finds the checkpoint."""
    config = _make_train_config(output_path=_MIDTRAIN_OUTPUT, run_id=None, impute=True)
    config = _with_temp_base_path(config, _MIDTRAIN_TEMP_BASE)

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    assert _MIDTRAIN_MIRROR_SEARCH in cp.temporary_search_paths
    # The bare prefix without the run-discriminating output_component must never appear.
    assert "mirrortmp://ttl=14d/checkpoints-temp" not in cp.temporary_search_paths
    assert "mirrortmp://ttl=14d/checkpoints-temp/" not in cp.temporary_search_paths


def test_enforce_run_id_mirror_search_matches_explicit_run_id_temp_write_path():
    """Explicit run-id (non-imputed): Levanter appends run_id; the search URL appends it too."""
    config = _make_train_config(output_path=_PRETRAIN_OUTPUT, run_id=_PRETRAIN_RUN_ID, impute=False)
    config = _with_temp_base_path(config, _PRETRAIN_TEMP_BASE)

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    assert _PRETRAIN_MIRROR_SEARCH in cp.temporary_search_paths


def test_enforce_run_id_appends_to_existing_search_paths():
    """Pre-existing entries on the user-supplied config are preserved; mirror entry is appended."""
    base_config = _make_train_config(output_path=_MIDTRAIN_OUTPUT, run_id=None, impute=True)
    base_config = _with_temp_base_path(base_config, _MIDTRAIN_TEMP_BASE)
    base_config = dataclasses.replace(
        base_config,
        train_config=dataclasses.replace(
            base_config.train_config,
            trainer=dataclasses.replace(
                base_config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    base_config.train_config.trainer.checkpointer,
                    temporary_search_paths=["custom://prior/entry"],
                    temporary_base_path=_MIDTRAIN_TEMP_BASE,
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(base_config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths == ["custom://prior/entry", _MIDTRAIN_MIRROR_SEARCH]


def test_enforce_run_id_dedupes_existing_mirror_entry():
    """Idempotent: re-running _enforce_run_id (or a user pre-supplying the entry) doesn't duplicate."""
    config = _make_train_config(output_path=_MIDTRAIN_OUTPUT, run_id=None, impute=True)
    config = _with_temp_base_path(config, _MIDTRAIN_TEMP_BASE)
    config = dataclasses.replace(
        config,
        train_config=dataclasses.replace(
            config.train_config,
            trainer=dataclasses.replace(
                config.train_config.trainer,
                checkpointer=dataclasses.replace(
                    config.train_config.trainer.checkpointer,
                    temporary_search_paths=[_MIDTRAIN_MIRROR_SEARCH],
                    temporary_base_path=_MIDTRAIN_TEMP_BASE,
                ),
            ),
        ),
    )
    enforced = _enforce_run_id(config)
    paths = enforced.train_config.trainer.checkpointer.temporary_search_paths
    assert paths.count(_MIDTRAIN_MIRROR_SEARCH) == 1


def test_enforce_run_id_skips_mirror_search_when_no_temp_base_path():
    """No ``temporary_base_path`` (e.g. ``_update_config_to_use_out_path`` short-circuited on
    ``output_path=None``) ⇒ no mirror search URL fabricated."""
    config = _make_train_config(output_path=None, run_id="my-run", impute=False)
    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer
    assert cp.temporary_base_path is None
    assert not any(p.startswith("mirrortmp://") for p in cp.temporary_search_paths)


def test_enforce_run_id_skips_mirror_search_for_non_marin_tmp_path():
    """If a user supplies a custom ``temporary_base_path`` that isn't under marin-tmp-*,
    we don't fabricate a mirrortmp:// URL — there are no cross-region siblings to scan."""
    config = _make_train_config(output_path=_MIDTRAIN_OUTPUT, run_id=None, impute=True)
    config = _with_temp_base_path(config, "file:///tmp/marin/tmp/checkpoints-temp/foo")
    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer
    assert not any(p.startswith("mirrortmp://") for p in cp.temporary_search_paths)


def test_temp_write_path_matches_mirror_search_path_for_resume():
    """Alignment invariant: the directory the temp writes target equals the directory
    the mirrortmp:// search resolves to (modulo bucket prefix).  If they diverge, a
    cross-region resume can't find the temp checkpoint written before preemption."""
    config = _make_train_config(output_path=_MIDTRAIN_OUTPUT, run_id=None, impute=True)
    config = _with_temp_base_path(config, _MIDTRAIN_TEMP_BASE)

    enforced = _enforce_run_id(config)
    cp = enforced.train_config.trainer.checkpointer

    write_dir = cp.expanded_temporary_path("foo-abc123")
    search_url = next(p for p in cp.temporary_search_paths if p.startswith("mirrortmp://"))

    # Both paths end in the same key: ``ttl=14d/checkpoints-temp/{output_component}/checkpoints``.
    common_suffix = "/ttl=14d/checkpoints-temp/marin-us-central1/runs/foo-abc123/checkpoints"
    assert write_dir.endswith(common_suffix)
    assert search_url.endswith(common_suffix)
