"""Prepare an independent Pi teacher before scheduling any policy or rollout actors."""

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import httpx
from omegaconf import DictConfig, OmegaConf
from rigging.secrets import resolve_secret_spec
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.rollouts.agent_runtime import InstalledAgentSettings
from skyrl_train.rollouts.pi_teacher_binding import PiTeacherBinding
from skyrl_train.tokenizer import create_tokenizer


@dataclass(frozen=True)
class PreparedPiTeacher:
    """Pinned tokenizer and independently selected teacher source."""

    binding: PiTeacherBinding
    tokenizer: PreTrainedTokenizerBase


def prepare_pi_teacher(cfg: DictConfig) -> PreparedPiTeacher | None:
    """Load the teacher tokenizer and authenticate/inspect an external endpoint before actor allocation."""
    if OmegaConf.select(cfg, "scenario.preset") not in ("teacher_demo_pi_base", "teacher_demo_pi_verified_base"):
        return None
    settings = InstalledAgentSettings.model_validate(
        OmegaConf.to_container(cfg.terminal_bench_config.installed_agent, resolve=True)
    )
    teacher = settings.teacher
    assert teacher is not None
    tokenizer = create_tokenizer(
        teacher.model_path,
        revision=teacher.revision,
        disable_fast_tokenizer=bool(cfg.trainer.disable_fast_tokenizer),
    )
    if teacher.source == "external_vllm":
        assert teacher.endpoint is not None and teacher.auth is not None
        secret = resolve_secret_spec(teacher.auth).value
        with httpx.Client(timeout=10, headers={"Authorization": f"Bearer {secret}"}) as client:
            response = client.get(f"{teacher.endpoint.rstrip('/')}/models")
            response.raise_for_status()
            models = response.json()
        if not isinstance(models, dict) or not isinstance(models.get("data"), list):
            raise ValueError("External Pi teacher /models did not return a model list")
        matched = [model for model in models["data"] if isinstance(model, dict) and model.get("id") == teacher.model]
        if len(matched) != 1 or matched[0].get("max_model_len") != teacher.max_context_tokens:
            raise ValueError("External Pi teacher alias or context limit differs from its binding")
    return PreparedPiTeacher(teacher, tokenizer)


def start_pi_managed_teacher(
    cfg: DictConfig,
    prepared: PreparedPiTeacher | None,
    *,
    engine_factory: Callable[..., list[InferenceEngineInterface]] | None = None,
) -> list[InferenceEngineInterface] | None:
    """Allocate a disjoint teacher vLLM role; its weights are never synchronized with the student."""
    if prepared is None or prepared.binding.source != "skyrl_managed":
        return None
    from skyrl_train.inference_engines.configuration import (
        InferenceEngineRoleConfig,
        inference_engine_kwargs_from_config,
    )

    if engine_factory is None:
        from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

        engine_factory = create_ray_wrapped_inference_engines

    teacher = prepared.binding
    resources = teacher.resources
    assert resources is not None
    engine_init_kwargs = dict(OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True))
    engine_init_kwargs.pop("openai_sampling_params", None)
    engine_init_kwargs.update(
        {
            "revision": teacher.revision,
            "max_model_len": teacher.max_context_tokens,
            "served_model_name": teacher.model,
        }
    )
    role = InferenceEngineRoleConfig(
        pretrain=teacher.model_path,
        backend=resources.backend,
        num_inference_engines=resources.num_engines,
        tensor_parallel_size=resources.tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        decode_context_parallel_size=1,
        shared_pg=None,
        inference_engine_enable_sleep=False,
    )
    kwargs = inference_engine_kwargs_from_config(
        cfg, tokenizer=prepared.tokenizer, role=role, engine_init_kwargs=engine_init_kwargs
    )
    kwargs["gpu_memory_utilization"] = resources.gpu_memory_utilization
    return engine_factory(**kwargs)


async def close_pi_managed_teacher(engines: Sequence[InferenceEngineInterface] | None) -> None:
    """Release independent teacher actors after the rollout worker stops using them."""
    if engines is None:
        return
    from skyrl_train.inference_engines.vllm_teacher_oracle import close_owned_inference_engine

    failures = []
    for engine in reversed(engines):
        try:
            await close_owned_inference_engine(engine)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("Pi teacher fleet cleanup failed", failures)


def student_eval_harbor_config(harbor: DictConfig, student_model: str) -> DictConfig:
    """Construct a student-only literal Pi evaluator; never route validation through a teacher."""
    settings = InstalledAgentSettings.model_validate(OmegaConf.to_container(harbor.installed_agent, resolve=True))
    if settings.teacher is None:
        raise ValueError("Pi teacher evaluation needs a bound teacher training configuration")
    evaluation = OmegaConf.create(OmegaConf.to_container(harbor, resolve=True))
    OmegaConf.update(evaluation, "installed_agent.agent.pi.mode", "literal")
    OmegaConf.update(evaluation, "installed_agent.agent.model_name", f"hosted_vllm/{student_model}")
    del evaluation.installed_agent.teacher
    InstalledAgentSettings.model_validate(OmegaConf.to_container(evaluation.installed_agent, resolve=True))
    return evaluation
