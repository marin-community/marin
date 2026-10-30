import asyncio
import os
import ray
import torch
import torch.multiprocessing as mp
import time
import requests
import importlib
from loguru import logger
from ray.util.placement_group import placement_group
from omegaconf import DictConfig
import hydra
from typing import List, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from functools import lru_cache
import subprocess

from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.dataset import PromptDataset
from skyrl_train.training_batch import TensorBatch, TrainingInputBatch, TrainingOutputBatch
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.trajectory_runners.base import ConversationType
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata
from skyrl_train.trajectory_runners.types import (
    TaskRequestBatch,
)
from skyrl_train.dataset.tasks import source_row_task
from skyrl_train.utils.utils import peer_access_supported, print_mem, initialize_ray, validate_cfg
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines

TEST_DATA_PATH = os.path.expanduser("~/data/gsm8k/validation.parquet")

# Rank diagnostics file the DPO equivalence runner cats at the end of the job stream.
# run_equivalence.sh references the same filename; keep them in sync.
DPO_DIAGNOSTICS_LOG = "dpo-diagnostics.log"


def run_distributed_with_deadline(target, args, world_size: int, timeout_seconds: int = 240):
    """Spawn NCCL rank workers and guarantee no process outlives the deadline."""
    context = mp.spawn(target, args=args, nprocs=world_size, join=False)
    deadline = time.monotonic() + timeout_seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"distributed workers exceeded their {timeout_seconds}-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.trainer.logger = "console"
        validate_cfg(cfg)

        return cfg


def get_rank_0_memory(actor_group, message: str):
    mem = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
    print_mem(message, mem)
    return mem["allocated"]


def make_dummy_tensorbatch(seq_len=10, num_actions=4) -> TensorBatch:
    B, T = 2, seq_len
    data = TensorBatch(
        sequences=torch.ones(B, T, dtype=int, device="cpu"),
        attention_mask=torch.ones(B, T, dtype=int, device="cpu"),
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_training_batch(batch_size=2, seq_len=10, num_actions=4) -> TrainingInputBatch:
    """Create a dummy TrainingInputBatch"""

    torch.manual_seed(42)

    # Add all the required fields for training
    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len), device="cpu"),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=int, device="cpu"),
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
            "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "response_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_experience(seq_len=10, num_actions=4) -> Experience:
    torch.manual_seed(42)
    B, T = 2, seq_len
    num_actions = num_actions

    return Experience(
        sequences=torch.randint(0, 100, (B, T), device="cpu"),
        action_log_probs=0.4 * torch.ones((B, num_actions), device="cpu"),
        base_action_log_probs=0.3 * torch.ones((B, num_actions), device="cpu"),
        rollout_logprobs=0.2 * torch.ones((B, num_actions), device="cpu"),
        values=0.5 * torch.ones((B, num_actions), device="cpu"),
        returns=0.5 * torch.ones((B, num_actions), device="cpu"),
        advantages=0.6 * torch.ones((B, num_actions), device="cpu"),
        attention_mask=torch.ones((B, T), dtype=int, device="cpu"),
        loss_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        action_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        num_actions=num_actions,
        info={},
    )


def import_worker(strategy: str, worker_type: str):
    if strategy != "megatron":
        raise ValueError(f"Unknown strategy type for {worker_type}: {strategy}")
    module_path = "skyrl_train.workers.megatron.megatron_worker"

    module = importlib.import_module(module_path)
    return getattr(module, f"{worker_type.capitalize()}Worker")


def init_worker_with_type(
    worker_type: str,
    shared_pg=None,
    colocate_all=False,
    num_gpus_per_node=1,
    num_nodes=1,
    cfg=None,
    num_training_steps: int | None = None,
) -> PPORayActorGroup:
    if cfg is None:
        cfg = get_test_actor_config()

    if shared_pg is not None:
        pg = shared_pg
        num_gpus_per_actor = 0.2
    else:
        bundles = [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node} for _ in range(num_nodes)]
        pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        num_gpus_per_actor = 0.75

    worker_cls = import_worker(cfg.trainer.strategy, worker_type)
    model = PPORayActorGroup(
        cfg,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        colocate_all=colocate_all,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    # we use policy model path for all tests (regardless of actor type)
    init_kwargs = {} if num_training_steps is None else {"num_training_steps": num_training_steps}
    ray.get(model.async_init_model(cfg.trainer.policy.model.path, **init_kwargs))
    return model


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"{self.message}, time cost: {time.time() - self.start_time:.2f}s")


def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or all available GPUs"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES (can be comma-separated list)
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
        return gpu_ids
    else:
        # If not set, warn user but proceed with all GPUs
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
                print(f"CUDA_VISIBLE_DEVICES not set. Using all {gpu_count} GPUs: {gpu_ids}")
                print("This might conflict with other processes. Consider setting CUDA_VISIBLE_DEVICES explicitly.")
                return gpu_ids
            else:
                return []
        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []


def wait_for_server(url: str, health_path: str, timeout: int = 60, interval: float = 1.0):
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"http://{url}/{health_path}")
            if response.ok:
                return
        except requests.exceptions.ConnectionError:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Server at {url} did not come online within {timeout} seconds")
            time.sleep(interval)


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    for i in range(m + 1):
        dp[i][0] = i  # Deletion from s1 to empty string
    for j in range(n + 1):
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[m][n]


def are_responses_similar(responses_a: List[str], responses_b: List[str], tolerance: float = 0.01) -> float:
    if len(responses_a) != len(responses_b):
        return False

    total_length = 0
    total_diff = 0

    for s1, s2 in zip(responses_a, responses_b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        diff = levenshtein(s1, s2)
        total_diff += diff

    difference = float(total_diff / total_length)
    return difference <= tolerance


def get_test_prompts(model: str, num_samples: int = 20) -> List[ConversationType]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PromptDataset(
        datasets=[TEST_DATA_PATH],
        tokenizer=tokenizer,
        max_prompt_length=512,
    )

    # Extract the actual prompts from the dataset
    prompts = []
    for i in range(min(num_samples, len(dataset))):
        prompt_data, _, _, _ = dataset[i]  # dataset returns (messages, env_class, extra, uid)
        prompts.append(prompt_data)

    return prompts


def get_test_trajectory_request(
    model: str,
    *,
    environment_configs: dict,
    num_prompts: int = 20,
    n_samples_per_prompt: int = 1,
    max_prompt_length: int = 512,
    data_path: str = TEST_DATA_PATH,
    env_class: str = "gsm8k",
) -> TaskRequestBatch:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = PromptDataset(
        datasets=[data_path],
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    rows = []
    for index in range(min(num_prompts, len(dataset))):
        prompt, _, extras, uid = dataset[index]
        lowered = source_row_task(
            {"prompt": prompt, "env_class": env_class, **extras},
            index,
            source_name="gpu-test",
            environment_configs=environment_configs,
        )
        row = TaskInput(
            lowered=lowered,
            metadata=TrajectoryMetadata(uid=uid, metric_class=env_class, data_source=lowered.task.source.dataset),
        )
        rows.extend([row] * n_samples_per_prompt)
    return TaskRequestBatch(rows=tuple(rows))


def get_model_logits_from_actor(actor_group: PPORayActorGroup, input_sequences, attention_mask):
    """Helper function to get model logits for comparison"""

    seq_len = input_sequences.shape[1]
    num_actions_val = seq_len - 5  # Leave some tokens for response

    data = TrainingInputBatch(
        {
            "sequences": input_sequences,
            "attention_mask": attention_mask,
        }
    )
    data.metadata = {"response_length": num_actions_val}

    results_refs = actor_group.async_run_ray_method("mesh", "forward", data)
    results = ray.get(results_refs)
    ret_databatch: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, results)
    logits = ret_databatch["output"]

    return logits


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def ray_init_for_tests():
    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=4):
        log_once("Disabling NCCL P2P for test environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    # TODO (erictang000): refactor this to use the same prepare_runtime_environment function as in utils.py for tests
    # to remove duplicate code
    env_vars["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["NVTE_FUSED_ATTN"] = "0"
    env_vars["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    ray.init(runtime_env={"env_vars": env_vars})


async def run_inference(client, prompts, sampling_params):
    engine_input = InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)
    return await client.generate(engine_input)


# TODO: this is kind of messy. All these information are inside cfg but we are passing them in
# again. Make a global get_test_config function that is parametrized.
def init_inference_engines(
    cfg,
    model,
    use_local,
    tp_size,
    colocate_all,
    backend,
    gpu_memory_utilization=0.6,
    num_inference_engines=1,
    sleep_level=2,  # use level 1 in unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_num_seqs=1024,
    engine_init_kwargs=None,
):
    assert use_local, "This test does not yet support remote engines."
    assert backend in ["vllm", "sglang"]
    if not ray.is_initialized():
        initialize_ray(cfg)
    if colocate_all:
        pg = placement_group([{"GPU": 1, "CPU": 1}] * tp_size * num_inference_engines, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        sleep = True
    else:
        pg, sleep = None, False

    tokenizer = AutoTokenizer.from_pretrained(model)
    eps = create_ray_wrapped_inference_engines(
        num_inference_engines=num_inference_engines,
        tensor_parallel_size=tp_size,
        model_dtype="bfloat16",
        pretrain=model,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        shared_pg=pg,
        engine_init_timeout_seconds=cfg.generator.engine_init_timeout_seconds,
        gpu_memory_utilization=gpu_memory_utilization,
        inference_engine_enable_sleep=sleep,
        max_num_batched_tokens=8192,
        max_num_seqs=max_num_seqs,
        tokenizer=tokenizer,
        backend=backend,
        sleep_level=sleep_level,
        enable_lora=enable_lora,
        engine_init_kwargs=engine_init_kwargs or {},
        weight_sync_pause_policy=resolve_weight_sync_pause_policy(cfg.generator),
    )
    client = InferenceEngineClient(eps, tokenizer, cfg)
    if sleep:
        asyncio.run(client.wake_up())
    return client, pg


def init_remote_inference_servers(
    tp_size: int,
    backend: str,
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    model: str,
) -> Tuple[InferenceEngineClient, subprocess.Popen]:
    available_gpus = get_available_gpus()
    assert len(available_gpus) >= tp_size, (
        f"Not enough GPUs available. Need {tp_size}, but only {len(available_gpus)} available: {available_gpus}"
    )

    selected_gpus = available_gpus[:tp_size]
    gpu_ids_str = ",".join(map(str, selected_gpus))
    print(f"Using GPUs {gpu_ids_str} for vLLM server (tensor_parallel_size={tp_size})")

    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    engine_port = get_free_port()

    # Launch vLLM server using subprocess
    if backend == "vllm":
        remote_server_command = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "vllm",
            "-m",
            "skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            model,
            "--gpu-memory-utilization",
            "0.8",
            "--tensor-parallel-size",
            str(tp_size),
            # NOTE (sumanthrh): Currently, there's an issue with distributed executor backend ray for vllm 0.9.2.
            # For standalone server, we use mp for now.
            "--distributed-executor-backend",
            "mp",
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--worker-extension-cls",
            "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
        ]
    elif backend == "sglang":
        remote_server_command = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "sglang",
            "-m",
            "skyrl_train.inference_engines.sglang.sglang_server",
            "--model-path",
            model,
            "--tp-size",
            str(tp_size),
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--mm-attention-backend",
            "fa3",
            "--attention-backend",
            "fa3",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Set CUDA_VISIBLE_DEVICES environment variable for the subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    # Start the vLLM server process
    server_process = subprocess.Popen(remote_server_command, env=env)

    wait_for_server(url=f"localhost:{engine_port}", health_path="health", timeout=120)
    print(f"Server at localhost:{engine_port} is online")

    engines = create_remote_inference_engines(
        urls=[f"localhost:{engine_port}"],
        model_name=model,
        tokenizer=tokenizer,
        engine_backend=backend,
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        expert_parallel_size=1,
    )

    client = InferenceEngineClient(engines, tokenizer, config)
    return client, server_process
