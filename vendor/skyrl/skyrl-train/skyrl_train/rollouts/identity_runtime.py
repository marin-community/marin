"""Frozen repair-teacher fleet and worker-local identity grading configuration."""

import asyncio
import json
from pathlib import Path

from omegaconf import OmegaConf
from marinskyrl.identity_repair import IdentityRuntimeSettings

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.identity_repair.facts import load_fact_manifest
from skyrl_gym.identity_repair.grading import ProductionIdentityJudge
from skyrl_gym.identity_repair.reward import IdentityFact, RewardWeights
from skyrl_train.inference_engines.configuration import InferenceEngineRoleConfig, inference_engine_kwargs_from_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.vllm_teacher_oracle import close_owned_inference_engine, tokenizer_vocabulary_fingerprint
from skyrl_train.rollouts.identity_groups import IdentityRepairGroups
from skyrl_train.rollouts.repair_branching import ModelClientBranchChat, TeacherRepairSpec
from skyrl_train.tokenizer import create_tokenizer
from skyrl_train.trajectory_runners.model_clients import DirectModelClient



def identity_settings(cfg) -> IdentityRuntimeSettings | None:
    raw = cfg.get("identity_repair")
    if raw is None:
        return None
    settings = IdentityRuntimeSettings.model_validate(OmegaConf.to_container(raw, resolve=True))
    if cfg.trainer.algorithm.policy_loss_type != "mixed_policy_grpo" or cfg.trainer.step_wise_training:
        raise ValueError("Identity repair requires whole-trajectory mixed_policy_grpo")
    if cfg.generator.n_samples_per_prompt != 4:
        raise ValueError("Identity repair requires four continuations per prompt")
    if cfg.trainer.algorithm.advantage_estimator != "grpo":
        raise ValueError("Identity repair requires GRPO group advantages")
    params = settings.teacher_sampling_params
    if params.get("temperature") != 1.0 or params.get("top_p", 1.0) != 1.0 or params.get("top_k", -1) != -1:
        raise ValueError("Repair teacher requires untruncated temperature-one behavior probabilities")
    return settings


def start_repair_engines(cfg, tokenizer):
    settings = identity_settings(cfg)
    if settings is None:
        return []
    teacher = settings.teacher
    teacher_tokenizer = create_tokenizer(teacher.model_path, revision=teacher.revision,
                                         disable_fast_tokenizer=bool(cfg.trainer.disable_fast_tokenizer))
    if tokenizer_vocabulary_fingerprint(teacher_tokenizer) != tokenizer_vocabulary_fingerprint(tokenizer):
        raise ValueError("Repair teacher and student must have identical token-ID semantics")
    role = InferenceEngineRoleConfig(
        pretrain=teacher.model_path, backend="vllm", num_inference_engines=teacher.engines,
        tensor_parallel_size=teacher.tensor_parallel_size, pipeline_parallel_size=1,
        data_parallel_size=teacher.data_parallel_size, expert_parallel_size=teacher.expert_parallel_size,
        decode_context_parallel_size=1, shared_pg=None, inference_engine_enable_sleep=False, max_logprobs=1,
    )
    engine_init = dict(OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True))
    engine_init.update(revision=teacher.revision, served_model_name=teacher.model_path, max_model_len=teacher.max_model_len)
    engine_init.pop("openai_sampling_params", None)
    kwargs = inference_engine_kwargs_from_config(cfg, tokenizer=teacher_tokenizer, role=role,
                                                 engine_init_kwargs=engine_init)
    kwargs["gpu_memory_utilization"] = teacher.gpu_memory_utilization
    return create_ray_wrapped_inference_engines(**kwargs)


async def close_repair_engines(engines):
    results = await asyncio.gather(*(close_owned_inference_engine(engine) for engine in engines), return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise BaseExceptionGroup("Repair teacher cleanup failed", errors)


def build_identity_groups(cfg, tokenizer, student_client, engines) -> IdentityRepairGroups | None:
    settings = identity_settings(cfg)
    if settings is None:
        return None
    if not engines:
        raise ValueError("Identity repair configured without a frozen teacher fleet")
    client_config = OmegaConf.merge(cfg, {"generator": {"enable_http_endpoint": False, "engine_init_kwargs": {
        "served_model_name": settings.teacher.model_path,
    }}})
    teacher = DirectModelClient(InferenceEngineClient(list(engines), tokenizer, client_config))
    data_root = Path(cfg.data.train_data[0]).parent
    requested = json.loads((data_root / settings.requested_facts_manifest).read_text())
    requested = {task: tuple(tuple(IdentityFact(fact) for fact in turn) for turn in turns)
                 for task, turns in requested.items()}
    criteria = {}
    if settings.task_criteria_manifest is not None:
        criteria = json.loads((data_root / settings.task_criteria_manifest).read_text())
        if not isinstance(criteria, dict) or set(criteria) != set(requested):
            raise ValueError("Task criteria must cover exactly the requested-fact tasks")
        for task, turns in criteria.items():
            if (not isinstance(turns, list) or len(turns) != len(requested[task])
                    or not all(isinstance(criterion, str) and criterion.strip() for criterion in turns)):
                raise ValueError(f"Task criteria must provide nonempty text for every turn: {task}")
        criteria = {task: tuple(turns) for task, turns in criteria.items()}
    judge = ProductionIdentityJudge(OpenAIJudge(settings.judge_base_url, settings.judge_model,
                                               api_key_env=settings.judge_api_key_env, strict_completion=True),
                                    settings.judge_revision, settings.judge_max_tokens)
    return IdentityRepairGroups(
        tokenizer, load_fact_manifest(data_root / settings.fact_manifest), judge,
        ModelClientBranchChat(teacher), ModelClientBranchChat(student_client),
        TeacherRepairSpec(settings.teacher_sampling_params,
                         dict(OmegaConf.to_container(cfg.generator.sampling_params, resolve=True)),
                         settings.teacher.max_model_len, chat_template_kwargs=dict(cfg.generator.get("chat_template_kwargs", {})),
                         max_resume_turns=settings.max_resume_turns),
        requested, RewardWeights(**settings.reward_weights), criteria,
    )
