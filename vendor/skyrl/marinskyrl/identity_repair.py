"""Typed configuration shared by identity-repair launch planning and execution."""

from pydantic import BaseModel, ConfigDict, Field


class RepairTeacherSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    revision: str
    engines: int = Field(gt=0)
    tensor_parallel_size: int = Field(gt=0)
    data_parallel_size: int = Field(gt=0)
    expert_parallel_size: int = Field(gt=0)
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, lt=1)


class IdentityRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_manifest: str
    task_criteria_manifest: str | None = None
    requested_facts_manifest: str
    teacher: RepairTeacherSettings
    judge_base_url: str
    judge_model: str
    judge_revision: str
    judge_api_key_env: str
    judge_max_tokens: int = Field(gt=0)
    judge_chat_template_kwargs: dict = Field(default_factory=dict)
    teacher_sampling_params: dict
    max_resume_turns: int = Field(ge=0)
    reward_weights: dict[str, float]

