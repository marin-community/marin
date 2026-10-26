"""Configuration-only Pi teacher source binding; safe to validate before starting Iris."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator


class PiManagedResources(BaseModel):
    """Disjoint GPU capacity for an independently allocated vLLM teacher."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["vllm"] = "vllm"
    num_engines: int
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.85

    @model_validator(mode="after")
    def validate_capacity(self) -> "PiManagedResources":
        if self.num_engines <= 0 or self.tensor_parallel_size <= 0:
            raise ValueError("Pi managed teacher needs positive engine and GPU counts")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("Pi managed teacher GPU memory utilization must be between 0 and 1")
        return self


class PiTeacherBinding(BaseModel):
    """Independent completion model selected before a teacher-trace Pi trial."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["skyrl_managed", "external_vllm"]
    model: str
    model_path: str
    revision: str
    student_model: str
    max_context_tokens: int
    endpoint: str | None = None
    auth: str | None = None
    resources: PiManagedResources | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "PiTeacherBinding":
        if (
            not self.model
            or not self.model_path
            or not self.revision
            or not self.student_model
            or self.model_path == self.student_model
            or self.model == self.student_model
        ):
            raise ValueError("Pi teacher needs a pinned model distinct from the student")
        if type(self.max_context_tokens) is not int or self.max_context_tokens <= 0:
            raise ValueError("Pi teacher context limit must be a positive integer")
        if self.source == "skyrl_managed" and (self.endpoint is not None or self.auth is not None):
            raise ValueError("SkyRL-managed Pi teacher cannot use an external endpoint")
        if self.source == "external_vllm" and self.resources is not None:
            raise ValueError("External Pi teacher cannot allocate managed GPU resources")
        if self.source == "external_vllm" and (not self.endpoint or not self.auth or not self.auth.startswith("env:")):
            raise ValueError("External Pi teacher requires a vLLM endpoint and env-backed authentication")
        if self.source == "external_vllm":
            parsed = urlsplit(self.endpoint)
            if not parsed.hostname or parsed.query or parsed.fragment:
                raise ValueError("External Pi teacher endpoint requires a host and no URL query or fragment")
            if parsed.scheme != "https" and not (
                parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
            ):
                raise ValueError("External Pi teacher requires HTTPS (or a loopback HTTP test fixture)")
        return self
