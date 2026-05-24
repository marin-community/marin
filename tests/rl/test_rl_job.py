# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from levanter.trainer import TrainerConfig
from marin.rl.curriculum import CurriculumConfig
from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType


def test_rl_job_seed_overrides_trainer_and_vllm_config(monkeypatch):
    monkeypatch.setattr("marin.rl.rl_job.make_tokenizer", lambda _tokenizer: SimpleNamespace(vocab_size=128))

    inference_config = vLLMInferenceContextConfig(
        model_name="test-model",
        max_model_len=16,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        sampling_params=VLLMSamplingConfig(),
        seed=0,
    )
    job_config = RLJobConfig(
        model=SimpleNamespace(),
        trainer=TrainerConfig(seed=0),
        train_params=TrainParams(
            optimizer=SimpleNamespace(),
            rl_loss=SimpleNamespace(),
        ),
        curriculum=CurriculumConfig(
            lessons={},
            max_seq_len=16,
        ),
        tokenizer="test-tokenizer",
        inference_type="vllm",
        inference_config=inference_config,
        rollout_storage=RolloutStorageConfig(storage_type=StorageType.IN_MEMORY, queue_name="seed-test"),
        seed=77,
    )

    train_config, rollout_config = RLJob(job_config).to_worker_configs()

    assert train_config.seed == 77
    assert train_config.trainer.seed == 77
    assert rollout_config.seed == 1077
    assert rollout_config.trainer.seed == 77
    assert rollout_config.inference_config.seed == 1077
