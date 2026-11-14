import asyncio
import copy
from hashlib import sha256

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group
from safetensors import safe_open
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.batch_source import WorkerBatchStep
from skyrl_train.utils import initialize_ray, validate_cfg
from transformers import AutoConfig, AutoTokenizer

from tests.cpu.tiny_training.tiny_model import write_gsm8k_dataset
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.utils import import_worker, get_test_actor_config


def random_unique_routes(shape: tuple[int, ...], num_experts: int, *, generator: torch.Generator) -> torch.Tensor:
    """Return random top-K indices with no duplicate expert in a token row."""
    scores = torch.rand((*shape[:-1], num_experts), generator=generator)
    return scores.argsort(dim=-1)[..., : shape[-1]]


class _RecordedRollouts:
    def __init__(self, groups):
        self.groups = {group.uid: group.trajectory_batch for group in groups}
        self.order = list(self.groups)
        self.ready = asyncio.Condition()

    def set_trajectory_sink(self, sink):
        sink.bind_runner(type(self).__name__)

    async def run_task(self, task, writer):
        batch = copy.deepcopy(self.groups[task.prompt["uid"]])
        async with self.ready:
            await self.ready.wait_for(lambda: self.order[0] == task.prompt["uid"])
            await writer.write_rollout(
                task.lease, RolloutGroup(batch, task.prompt["uid"], task.lease.policy_step, task.prompt)
            )
            self.order.pop(0)
            self.ready.notify_all()
        return sum(map(len, batch["response_ids"]))


@pytest.mark.slow
def test_megatron_worker_batches_match_driver_update(tmp_path):
    world_size = 4
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, num_experts_per_tok=4, shape={**TOY_SHAPE, "num_key_value_heads": 2})
    model_config = AutoConfig.from_pretrained(model_path)
    num_layers, num_experts, top_k = (
        model_config.num_hidden_layers,
        model_config.num_local_experts,
        model_config.num_experts_per_tok,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        parameter_names = list(checkpoint.keys())
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = world_size
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.policy.optimizer_config.lr = 2.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.0
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.use_sample_packing = True
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.group_advantage_min_size = 2
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.generator.n_samples_per_prompt = 4
    validate_cfg(cfg)
    cfg.trainer.training_metrics = True
    cfg.trainer.rollout_buffer.max_staleness_steps = 0
    cfg.trainer.rollout_buffer.max_in_flight = 3
    cfg.generator.trajectory_retention.enabled = False
    cfg.data.shuffle = False
    dataset = SourceTaskDataset(
        datasets=[str(write_gsm8k_dataset(tmp_path / "data/train.jsonl", 3, max_turns=1))],
        environment_configs=OmegaConf.to_container(cfg.environment.task_sessions, resolve=True),
        tokenizer=tokenizer,
        max_prompt_length=cfg.trainer.max_prompt_length,
        num_workers=1,
    )
    generator = torch.Generator().manual_seed(888)
    groups = []
    for index in range(3):
        lengths = [17, 22, 25, 32 if index == 2 else 27]
        batch = {
            "prompt_token_ids": [[10 + index] * (11 + index)] * 4,
            "response_ids": [[20 + row] * length for row, length in enumerate(lengths)],
            "loss_masks": [[1] * length for length in lengths],
            "rewards": [0.0, 1.0, 3.0, 2.0],
            "rollout_logprobs": [np.full(length, -3.0, dtype=np.float32) for length in lengths],
            "rollout_routed_experts": [
                random_unique_routes((length, num_layers, top_k), num_experts, generator=generator)
                .numpy()
                .astype(np.uint8)
                for length in lengths
            ],
            "is_last_step": [True] * 4,
            "rollout_metrics": {},
        }
        if index == 2:
            batch["rewards"] = [
                [0.1, 0.2] + [0.0] * (length - 3) + [reward]
                for length, reward in zip(lengths, batch["rewards"], strict=True)
            ]
        prompt = dataset.collate_fn([dataset[index]])[0]
        groups.append(RolloutGroup(batch, dataset.uid(index), 1, prompt))
    initialize_ray(cfg)
    shared_pg = placement_group([{"GPU": 1, "CPU": 1}] * world_size, strategy="PACK")

    def parameter_hashes(policy):
        return {
            name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
            for name, value in rank0_validation_snapshot(policy, parameter_names).items()
        }

    results = {}
    trainer = None
    try:
        ray.get(shared_pg.ready(), timeout=120)
        for builder in ("driver", "worker"):
            current = copy.deepcopy(cfg)
            current.trainer.batch_builder = builder
            validate_cfg(current)
            runner = _RecordedRollouts(groups)
            trainer = RayPPOTrainer(
                cfg=current,
                tracker=None,
                tokenizer=tokenizer,
                train_dataset=dataset,
                inference_engine_client=None,
                trajectory_runner=runner,
                context=TrainingContext.from_config(current, dataset, runner),
                callbacks=[],
            )
            trainer.build_models(import_worker("megatron", "policy"), None, None, policy_pg=shared_pg)
            trainer.global_step = 1
            policy = trainer.policy_model
            initial = parameter_hashes(policy)
            if builder == "worker":
                assert initial == results["driver"]["initial"]

            async def run_admitted_batch():
                context = trainer.context
                context.start()
                try:
                    await context.publish(1)
                    if builder == "driver":
                        admitted, metadata = await context.next_batch(
                            stall_timeout=30, on_admitted=lambda _: asyncio.sleep(0)
                        )
                        reference = trainer.convert_rollout_groups_to_training_input(
                            admitted, facts=tuple(group.row_facts for group in metadata.groups)
                        )
                        status = await trainer._run_training(reference)
                    else:
                        source = WorkerBatchStep(
                            context=context,
                            policy=policy,
                            cfg=current,
                            pad_token_id=tokenizer.pad_token_id,
                            num_experts=num_experts,
                        )
                        trainer.all_metrics.update(await source.admit(stall_timeout=30))
                        metrics, _ = await source.prepare(global_step=1)
                        trainer.all_metrics.update(metrics)
                        await trainer._drain_policy_event_loops()
                        trainer.all_metrics.update(await source.prepare_training())
                        inputs = ray.get(
                            [
                                actor.__ray_call__.remote(
                                    lambda worker, batch_id=source.admitted_plan.global_step: copy.deepcopy(
                                        worker._loaded_batches[batch_id].batch
                                    ).to("cpu")
                                )
                                for actor in policy._actor_handlers
                            ]
                        )
                        reference = results["driver"]["batch"]
                        chunks = reference.chunk(reference.batch_size // policy.actor_infos[0].rank.dp_size)
                        for actor, batch in zip(policy.actor_infos, inputs, strict=True):
                            expected = chunks[actor.rank.dp]
                            assert expected.get("values") is None
                            assert batch.keys() == expected.keys() - {"values"}
                            for key, value in batch.items():
                                if value is None:
                                    assert expected[key] is None, key
                                else:
                                    assert value.dtype == expected[key].dtype, key
                                    assert torch.equal(value.view(torch.uint8), expected[key].view(torch.uint8)), key
                            assert all(
                                np.array_equal(a, b)
                                for a, b in zip(
                                    batch.routed_expert_rows.rows, expected.routed_expert_rows.rows, strict=True
                                )
                            )
                        async with trainer._policy_training_window(None):
                            status, cache_refs = await asyncio.to_thread(
                                trainer._train_policy,
                                lambda: policy.async_run_ray_method("pass_through", "train_loaded", 1),
                            )
                            await context.wait(asyncio.gather(*cache_refs))
                        trainer._finish_policy_update(status)
                        await source.unload()
                    assert status["router_replay/hit_fraction"] == 1.0
                    parameters = parameter_hashes(policy)
                    assert parameters.keys() == initial.keys() == set(parameter_names)
                    assert parameters != initial
                    return {
                        "initial": initial,
                        "parameters": parameters,
                        "status": status,
                        "metrics": dict(trainer.all_metrics),
                        **({"batch": copy.deepcopy(reference)} if builder == "driver" else {}),
                    }
                finally:
                    await context.close()

            result = asyncio.run(run_admitted_batch())
            trainer.cleanup_ray_actors()
            trainer = None
            results[builder] = result
        for key in ("policy_loss", "policy_entropy", "raw_grad_norm"):
            assert results["worker"]["status"][key] == results["driver"]["status"][key], key
        assert results["worker"]["parameters"] == results["driver"]["parameters"]
        for key in results["worker"]["metrics"].keys() & results["driver"]["metrics"].keys():
            if not key.startswith(("timing/", "perf/", "async/performance/", "system/")):
                assert results["worker"]["metrics"][key] == pytest.approx(
                    results["driver"]["metrics"][key], rel=1e-6, abs=1e-7, nan_ok=True
                ), key
    finally:
        if trainer is not None:
            trainer.cleanup_ray_actors()
        ray.util.remove_placement_group(shared_pg)
        ray.shutdown()
