# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from marin.rl.rollout_worker import RolloutWorker, VLLMRolloutWorkerConfig, RolloutStorageConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig, EnvConfig
from transformers import AutoTokenizer
import uuid
from marin.rl.weight_transfer.base import WeightTransferConfig, WeightTransferMode
from levanter.trainer import TrainerConfig
from levanter.checkpoint import CheckpointerConfig
from pathlib import Path
from levanter.tracker.json_logger import JsonLoggerConfig
import jmp
from levanter.distributed import RayConfig


def test_initialize_vllm_rollout(tmpdir):
    # weight_transfer_config = WeightTransferConfig(
    #         mode=WeightTransferMode.ARROW_FLIGHT,
    #         sync_interval_steps=1,
    #         checkpoint_dir=tempfile.mkdtemp(),
    #         coordinator_name=f"test_coordinator_{uuid.uuid4().hex[:8]}",
    #     )

    # server = create_weight_transfer_server(
    #     config=weight_transfer_config,
    #     mesh=mesh,
    #     axis_mapping=axis_mapping,
    # )

    # client = create_weight_transfer_client(
    #     config=weight_transfer_config,
    #     mesh=mesh,
    #     axis_mapping=axis_mapping,
    # )

    rollout_worker = RolloutWorker(
        config=VLLMRolloutWorkerConfig(
            inference_type="vllm",
            vllm_model_name="meta-llama/Llama-3.2-1B-Instruct",
            vllm_max_model_len=1024,
            vllm_tensor_parallel_size=4,
            tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            run_id=f"test_rollout_worker_vllm_{uuid.uuid4().hex[:8]}",
            log_freq=10,
            max_rollouts=10,
            trainer=TrainerConfig(
                # model=LlamaConfig(
                #     seq_len=4096,
                #     hidden_dim=2048,
                #     intermediate_dim=8192,
                #     num_heads=32,
                #     num_kv_heads=8,
                #     num_layers=16,
                #     rope=Llama3RotaryEmbeddingsConfig(),
                #     tie_word_embeddings=True,
                # ),
                tracker=JsonLoggerConfig(),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=16,
                num_train_steps=1000,
                steps_per_eval=1,
                checkpointer=CheckpointerConfig(
                    base_path=Path(tmpdir) / "checkpoints",
                    save_interval=datetime.timedelta(seconds=10),
                ),
                tensor_parallel_axes=["mlp", "kv_heads"],
                fsdp_axis="embed",
                batch_axis="batch",
                ray=RayConfig(auto_start_cluster=False),
            ),
            rollout_storage=RolloutStorageConfig(
                storage_type="memory",
                queue_name=f"test_rollout_worker_vllm_{uuid.uuid4().hex[:8]}",
            ),
            weight_transfer=WeightTransferConfig(
                mode=WeightTransferMode.ARROW_FLIGHT,
                sync_interval_steps=4,
                max_weight_transfer_wait_time=1,
                # not really that often since just want to test out rollout worker capability
            ),
            curriculum_config=CurriculumConfig(
                lessons=[
                    LessonConfig(
                        lesson_id="test_lesson",
                        env_config=EnvConfig(
                            env_class="marin.rl.environments.mock_env.MockEnv",
                            env_args={
                                "num_examples": 1,
                                "num_generations": 5,
                                "max_tokens": 256,
                            },
                        ),
                    )
                ],
            ),
        )
    )
    rollout_worker.run()
