# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from fray.v2 import JobStatus, ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.elastic import ElasticTrainingConfig
from levanter.distributed import RayConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.training.elastic_budget_compare import (
    _cached_data_config,
    _experiment_model_config,
    _num_train_steps_for_target_flops,
)
from marin.training.elastic_fault_benchmark import _shared_data_config, _trainer_config
from marin.training.training import (
    MARIN_FAULT_INJECTION_STEPS_ENV,
    TrainLmOnPodConfig,
    _compilation_cache_dir_for_resources,
    _elastic_worker_count,
    _fault_steps_for_worker,
    _wait_for_elastic_jobs,
    _with_elastic_worker_assignment,
    _doublecheck_paths,
)


@pytest.fixture
def trainer_config():
    """Create a basic trainer config for tests."""
    return TrainerConfig(
        id="test-run",
        checkpointer=CheckpointerConfig(),
        ray=RayConfig(),
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
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
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
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
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


def test_path_checking_uses_target_tpu_region_when_explicit(trainer_config):
    with (
        patch("iris.marin_fs.marin_region", return_value="us-west4"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east5"),
    ):
        config = TrainLmOnPodConfig(
            train_config=train_lm.TrainLmConfig(
                data=MockDataConfig(cache_dir="gs://bucket/path"),
                trainer=trainer_config,
            ),
            resources=ResourceConfig.with_tpu("v5p-8", regions=["us-east5"]),
        )
        _doublecheck_paths(config)


def test_dataclass_recursive_checking(trainer_config):
    """Paths are checked recursively in dataclass objects."""
    with (
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
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
        patch("iris.marin_fs.marin_region", return_value="us-central1"),
        patch("iris.marin_fs.get_bucket_location", return_value="us-east1"),
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


def test_elastic_worker_assignment_splits_multislice_request(trainer_config):
    trainer_config = dataclasses.replace(
        trainer_config,
        elastic=ElasticTrainingConfig(enabled=True, worker_count=4),
        tracker=WandbConfig(project="marin", entity="marin-community", name="elastic-bench"),
    )
    config = TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            trainer=trainer_config,
            data_seed=17,
        ),
        resources=ResourceConfig.with_tpu("v5p-8", slice_count=4),
        output_path="/tmp/out",
    )

    assert _elastic_worker_count(config) == 4

    worker_config, worker_env = _with_elastic_worker_assignment(
        config,
        worker_index=2,
        worker_count=4,
    )

    assert worker_config.train_config.trainer.id == "test-run-w002"
    assert worker_config.train_config.data_seed == 17 + 2 * 10_000
    assert worker_config.resources.replicas == 1
    assert worker_config.train_config.trainer.elastic.group_id == "test-run"
    assert worker_config.train_config.trainer.elastic.worker_id == "w002"
    assert worker_config.train_config.trainer.elastic.state_path.endswith("/_elastic/test-run")
    assert worker_config.train_config.trainer.tracker.group == "test-run"
    assert worker_config.train_config.trainer.tracker.name == "elastic-bench-w002"
    assert worker_env["RUN_ID"] == "test-run-w002"
    assert worker_env["MARIN_ELASTIC_WORKER_COUNT"] == "4"


def test_elastic_worker_assignment_injects_worker_specific_fault_steps(trainer_config):
    trainer_config = dataclasses.replace(
        trainer_config,
        elastic=ElasticTrainingConfig(enabled=True, worker_count=2),
    )
    config = TrainLmOnPodConfig(
        train_config=train_lm.TrainLmConfig(
            trainer=trainer_config,
            data_seed=17,
        ),
        resources=ResourceConfig.with_tpu("v5p-8", slice_count=2),
        output_path="/tmp/out",
    )

    _, worker_env = _with_elastic_worker_assignment(
        config,
        worker_index=1,
        worker_count=2,
        fault_injection_by_worker='{"w001": [400, 1200]}',
    )

    assert worker_env[MARIN_FAULT_INJECTION_STEPS_ENV] == "[400, 1200]"


def test_fault_steps_for_worker_ignores_unconfigured_workers():
    assert _fault_steps_for_worker('{"w001": [400, 1200]}', "w000") is None


def test_compilation_cache_dir_uses_target_region_bucket():
    resources = ResourceConfig.with_tpu("v5p-8", regions=["us-east5"])
    assert _compilation_cache_dir_for_resources(resources) == "gs://marin-tmp-us-east5/ttl=30d/compilation-cache"


def test_elastic_fault_benchmark_uses_base_validation_components(tmp_path):
    base_data = LmDataConfig(
        tokenizer="gpt2",
        cache_dir="gs://levanter-data/tokenized",
        auto_build_caches=True,
        components={
            "openwebtext": DatasetComponent(
                source=UrlDatasetSourceConfig(
                    train_urls=["gs://pubmed-mosaic/openwebtext-train.jsonl.gz"],
                    validation_urls=["gs://pubmed-mosaic/openwebtext-val.jsonl.gz"],
                ),
                cache_dir="gs://levanter-data/tokenized/openwebtext/",
            )
        },
        train_weights={"openwebtext": 1.0},
    )

    data = _shared_data_config(base_data, str(tmp_path), train_docs=8, validation_docs=4)

    assert data.train_weights == {"synthetic": 1.0}
    assert set(data.components) == {"synthetic", "openwebtext"}

    synthetic = data.components["synthetic"]
    assert isinstance(synthetic, DatasetComponent)
    assert isinstance(synthetic.source, UrlDatasetSourceConfig)
    assert synthetic.source.train_urls == [f"{tmp_path}/synthetic-data/train.jsonl"]
    assert synthetic.source.validation_urls == []

    openwebtext = data.components["openwebtext"]
    assert isinstance(openwebtext, DatasetComponent)
    assert openwebtext.cache_dir is None
    assert isinstance(openwebtext.source, UrlDatasetSourceConfig)
    assert openwebtext.source.validation_urls == ["gs://pubmed-mosaic/openwebtext-val.jsonl.gz"]


def test_elastic_fault_benchmark_uses_positive_eval_budget(trainer_config):
    base = train_lm.TrainLmConfig(data=MockDataConfig(cache_dir="/tmp/data"), trainer=trainer_config)

    benchmark_trainer = _trainer_config(
        base,
        run_id="run",
        run_name="run",
        num_steps=200,
        train_batch_size=128,
        checkpoint_every=100,
        steps_per_eval=500,
        max_eval_batches=1,
        elastic=ElasticTrainingConfig(enabled=False),
        tags=[],
    )

    assert benchmark_trainer.steps_per_eval == 200
    assert benchmark_trainer.max_eval_batches == 1


def test_elastic_budget_compare_uses_cache_only_train_data():
    base_data = LmDataConfig(
        tokenizer="gpt2",
        cache_dir="gs://old/cache",
        auto_build_caches=True,
        components={
            "old": DatasetComponent(
                source=UrlDatasetSourceConfig(
                    train_urls=["gs://pubmed-mosaic/train.jsonl.gz"],
                    validation_urls=["gs://pubmed-mosaic/val.jsonl.gz"],
                )
            )
        },
        train_weights={"old": 1.0},
    )

    data = _cached_data_config(
        base_data,
        dataset_cache_dir="gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-6fbcbb",
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
    )

    assert data.tokenizer == "meta-llama/Meta-Llama-3.1-8B"
    assert data.cache_dir is None
    assert data.auto_build_caches is False
    assert data.train_weights == {"fineweb-edu-10b": 1.0}
    assert set(data.components) == {"fineweb-edu-10b"}

    component = data.components["fineweb-edu-10b"]
    assert isinstance(component, DatasetComponent)
    assert isinstance(component.source, UrlDatasetSourceConfig)
    assert component.source.train_urls == []
    assert component.source.validation_urls == []
    assert component.cache_dir == "gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-6fbcbb"


def test_elastic_budget_compare_matches_known_1e19_300m_step_count():
    model = _experiment_model_config()

    steps = _num_train_steps_for_target_flops(
        model_config=model,
        vocab_size=128_256,
        train_batch_size=128,
        train_seq_len=4096,
        target_flops=1e19,
    )

    assert steps == 11_456


class _StaticJobHandle:
    def __init__(self, job_id: str, status: JobStatus):
        self._job_id = job_id
        self._status = status

    @property
    def job_id(self) -> str:
        return self._job_id

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        return self._status

    def status(self) -> JobStatus:
        return self._status

    def terminate(self) -> None:
        return None


def test_wait_for_elastic_jobs_accepts_failed_workers_after_group_completion(tmp_path):
    completion_path = tmp_path / "completed.json"
    completion_path.write_text(
        '{"completed_step": 100, "run_id": "run", "updated_at": "2026-03-10T12:00:00+00:00", "worker_id": "w000"}'
    )

    jobs = [
        _StaticJobHandle("worker-0", JobStatus.SUCCEEDED),
        _StaticJobHandle("worker-1", JobStatus.FAILED),
    ]

    _wait_for_elastic_jobs(jobs, completion_path=str(completion_path))
