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

import tempfile
import uuid

import haliax as hax
import jax
import jmp
import numpy as np
import ray
from jax.sharding import Mesh
from levanter.models.llama import LlamaConfig
from transformers import AutoConfig, AutoTokenizer
from levanter.compat.hf_checkpoints import RepoRef, HFCheckpointConverter

from marin.rl.weight_transfer import (
    WeightTransferConfig,
    WeightTransferMode,
    create_weight_transfer_client,
    create_weight_transfer_server,
)
from marin.rl.weight_utils import levanter_to_nnx_state, MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
import datetime
from marin.rl.rollout_worker import RolloutWorker, VLLMRolloutWorkerConfig, RolloutStorageConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig, EnvConfig
from levanter.trainer import TrainerConfig
from levanter.checkpoint import CheckpointerConfig
from pathlib import Path
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.distributed import RayConfig
from vllm import SamplingParams
from marin.rl.environments.inference_ctx.vllm import vLLMInferenceContext


@ray.remote(resources={"TPU-v6e-8-head": 1}, runtime_env={"env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
def run_vllm():
    from vllm import LLM

    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", max_model_len=1024)
    outputs = llm.generate(["Hello, how are you?"])
    return outputs


@ray.remote(resources={"TPU-v4-8-head": 1}, runtime_env={"env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
def test_arrow_flight_transfer_to_vllm():
    """Test Arrow Flight weight transfer to vLLM."""
    from vllm import LLM

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    devices = jax.devices("tpu")
    num_devices = len(devices)
    print(f"Found {num_devices} TPU devices")

    # Only use device one for now so we don't have to deal with sharded arrays.
    # This is fine since the model will fit on single TPU.
    mesh = Mesh(
        np.array(devices)[:1].reshape(
            1,
        ),
        axis_names=("data",),
    )
    print(f"Mesh created with shape {mesh.shape}: {mesh.devices}")

    # Create weight transfer config
    weight_transfer_config = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        checkpoint_dir=tempfile.mkdtemp(),
        coordinator_name=f"test_coordinator_{uuid.uuid4().hex[:8]}",
    )

    server = create_weight_transfer_server(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping={},
    )

    client = create_weight_transfer_client(
        config=weight_transfer_config,
        mesh=mesh,
        axis_mapping={},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_checkpoint = RepoRef.from_string(MODEL_NAME)
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = LlamaConfig.from_hf_config(hf_config)
    converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)

    # Load pretrained weights from HuggingFace
    with hax.partitioning.set_mesh(mesh):
        model = converter.load_pretrained(
            model_config.model_type,
            ref=hf_checkpoint,
            config=model_config,
            dtype=jmp.get_policy("p=float32,c=bfloat16").compute_dtype,
        )

    print(f"Model built with vocab size: {hf_config.vocab_size}")

    llm = LLM(MODEL_NAME, gpu_memory_utilization=0.50)
    try:
        # Test weight transfer
        print("Starting weight transfer test to vllm...")
        print("Transfer iteration 0")
        server.serve_weights(0, model)
        update = client.receive_weights(model)

        assert update is not None, "Weight update failed"
        assert update.weight_id == 0, f"Expected weight_id 0, got {update.weight_id}"
        model = update.model
        model_nnx_state = levanter_to_nnx_state(model)
        llm.llm_engine.model_executor.driver_worker.sync_weights(
            model_nnx_state,
            mappings=MODEL_MAPPINGS[MODEL_NAME],
            transpose_keys=MODEL_TRANSPOSE_KEYS[MODEL_NAME],
            reshard_fn=None,
        )

        output = llm.generate(["Hello, how are you?"])

        key_phrase = "I'm excited to be here"
        generated_text = output[0].outputs[0].text
        assert key_phrase in generated_text, f"Key phrase: {key_phrase} not found in output: {generated_text}"

        print(f"Client metrics: {client.get_metrics()}")

        print("All transfers completed successfully")

    finally:
        server.cleanup()
        client.cleanup()


@ray.remote(resources={"TPU-v6e-8-head": 1}, runtime_env={"env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
def test_vllm_inference():
    """Test vLLM inference."""

    inference_ctx = vLLMInferenceContext(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=1024,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.60,
        sampling_params=SamplingParams(temperature=0.8, n=1, max_tokens=16, stop=None, logprobs=1),
    )
    outputs = inference_ctx.batch_completions(prompts=["Hello, how are you?"], temperature=0.8, n=1)
    response_tokens = inference_ctx.response_tokens_from_choice(outputs[0].outputs[0])
    logprobs = inference_ctx.logprobs_from_choice(outputs[0].outputs[0])
    print(f"Response tokens: {response_tokens}")
    print(logprobs)


@ray.remote(resources={"TPU-v4-8-head": 1}, runtime_env={"env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}})
def test_initialize_vllm_rollout():
    rollout_worker = RolloutWorker(
        config=VLLMRolloutWorkerConfig(
            model=LlamaConfig(
                seq_len=4096,
                hidden_dim=2048,
                intermediate_dim=8192,
                num_heads=32,
                num_kv_heads=8,
                num_layers=16,
            ),
            inference_type="vllm",
            vllm_model_name="meta-llama/Llama-3.2-1B-Instruct",
            vllm_max_model_len=1024,
            vllm_tensor_parallel_size=1,
            gpu_memory_utilization=0.60,
            sampling_params=SamplingParams(
                temperature=1.0,
                n=4,
                max_tokens=16,
                stop=None,
                logprobs=1,
            ),
            tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
            run_id=f"test_rollout_worker_vllm_{uuid.uuid4().hex[:8]}",
            log_freq=10,
            max_rollouts=10,
            trainer=TrainerConfig(
                tracker=JsonLoggerConfig(),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=16,
                num_train_steps=1000,
                steps_per_eval=1,
                checkpointer=CheckpointerConfig(
                    base_path=Path("/tmp") / "checkpoints",
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
                lessons={
                    "test": LessonConfig(
                        lesson_id="test",
                        env_config=EnvConfig(
                            env_class="marin.rl.environments.mock_env.MockEnv",
                            env_args={"seed": 42, "task_type": "number_comparison"},
                        ),
                        dependencies=[],
                    ),
                }
            ),
        )
    )
    rollout_worker.run()


if __name__ == "__main__":
    ray.get(test_vllm_inference.remote())
    # print(outputs)
