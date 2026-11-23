# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM local inference backend."""

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT, VLLM_GPU_RELEASE
from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec
from marin.inference.config import (
    DEFAULT_CUDA_VLLM_VERSION,
    InferenceModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)
from marin.inference.model_preparation import read_tool_chat_template
from marin.inference.vllm_router import available_port
from marin.inference.vllm_server import (
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    PreinstalledVllm,
    VllmEnvironment,
    VllmLauncher,
    VllmLauncherWithEnvironment,
    VllmType,
    _is_object_store_path,
)

_VLLM_ENGINE_READY_TIMEOUT_ENV = "VLLM_ENGINE_READY_TIMEOUT_S"


def vllm_launcher(config: VllmEngineConfig) -> VllmLauncher:
    if config.launcher is VllmLauncherType.PREINSTALLED:
        return PreinstalledVllm()
    if config.launcher is VllmLauncherType.TPU:
        return IsolatedTpuVllm(
            vllm_ref=VLLM_FORK_REQUIREMENT,
            tpu_inference_ref=TPU_INFERENCE_FORK_REQUIREMENT,
        )
    source = VllmType.MARIN_FORK if config.source is VllmSource.MARIN_FORK else VllmType.UPSTREAM
    version = config.version if source is VllmType.UPSTREAM else None
    if source is VllmType.UPSTREAM and version is None:
        version = DEFAULT_CUDA_VLLM_VERSION
    return IsolatedCudaVllm(
        source=source,
        version=version,
        extra_requirements=config.extra_requirements,
        release=config.release or VLLM_GPU_RELEASE,
    )


def _with_subprocess_env(
    launcher: VllmLauncher,
    subprocess_env: Mapping[str, str] | None,
    startup_timeout_seconds: int,
) -> VllmLauncher:
    environment = {_VLLM_ENGINE_READY_TIMEOUT_ENV: str(startup_timeout_seconds)}
    if subprocess_env:
        environment.update(subprocess_env)
    return VllmLauncherWithEnvironment(launcher, environment)


RUNAI_STREAMER_LOADER_FLAG = "--model-loader-extra-config"
DISTRIBUTED_STREAMER_BOUNDS = {"concurrency": 4, "memory_limit": 16 * (1 << 30)}
BOUNDED_DISTRIBUTED_STREAMER_CONFIG = json.dumps({"distributed": True, **DISTRIBUTED_STREAMER_BOUNDS})


def _fill_streamer_bounds(value: str) -> str:
    """Return the loader-config JSON with missing streamer bounds filled in."""
    try:
        config = json.loads(value)
    except ValueError as error:
        raise ValueError(f"{RUNAI_STREAMER_LOADER_FLAG} is not valid JSON: {value!r}") from error
    if not isinstance(config, dict) or config.get("distributed") is not True:
        return value
    missing = {key: value for key, value in DISTRIBUTED_STREAMER_BOUNDS.items() if key not in config}
    if not missing:
        return value
    return json.dumps(dict(config, **missing))


def bound_distributed_streamer_buffers(extra_args: Sequence[str]) -> tuple[str, ...]:
    """Fill the runai streamer's host RAM buffer bounds into any distributed loader config.

    The streamer holds streamed checkpoint shards in host RAM before they reach device memory,
    all charged to the serve child's cgroup; a distributed config without bounds buffers
    unbounded and OOMKills the container mid-load. Both accepted forms are rewritten in place:
    the flag and its JSON as separate tokens, or the glued ``--model-loader-extra-config=<json>``
    form. Keys the caller set explicitly pass through untouched, and a config that does not
    select distributed loading is returned unchanged. Raises ``ValueError`` for a config that
    is not valid JSON.
    """
    args = list(extra_args)
    for position, token in enumerate(args):
        flag, separator, inline_value = token.partition("=")
        if flag != RUNAI_STREAMER_LOADER_FLAG:
            continue
        if separator:
            args[position] = f"{flag}={_fill_streamer_bounds(inline_value)}"
        elif position + 1 < len(args):
            args[position + 1] = _fill_streamer_bounds(args[position + 1])
    return tuple(args)


@contextlib.contextmanager
def _chat_template_argument(content: str | None) -> Iterator[tuple[str, ...]]:
    if content is None:
        yield ()
        return
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", prefix="marin_serve_chat_", delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        yield ("--chat-template", path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def _resolved_chat_template(spec: ModelSpec) -> str | None:
    if spec.chat_template_content is not None:
        return spec.chat_template_content
    return read_tool_chat_template(spec.tokenizer_source, spec.tokenizer_revision)


@dataclass(frozen=True)
class VllmServedModel:
    base_url: str
    model_id: str
    chat_template_content: str | None
    environment: VllmEnvironment

    @property
    def metrics_url(self) -> str:
        return self.environment.metrics_url

    def check_alive(self) -> None:
        self.environment.check_alive()


@dataclass(frozen=True)
class VllmBackend:
    config: VllmEngineConfig
    host: str = "127.0.0.1"
    port: int | None = None
    name: str = "vllm"

    @contextlib.contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[VllmServedModel]:
        chat_template_content = _resolved_chat_template(spec)
        resolved_spec = replace(spec, chat_template_content=chat_template_content)
        with self.start(resolved_spec) as environment:
            environment.wait_until_ready()
            yield VllmServedModel(
                base_url=environment.server_url.removesuffix(OPENAI_API_SUFFIX),
                model_id=spec.api_model,
                chat_template_content=chat_template_content,
                environment=environment,
            )

    @contextlib.contextmanager
    def start(
        self,
        spec: ModelSpec,
        *,
        extra_args: Sequence[str] = (),
        subprocess_env: Mapping[str, str] | None = None,
    ) -> Iterator[VllmEnvironment]:
        """Yield a VllmEnvironment; call ``wait_until_ready()`` before using its endpoint."""
        chat_template_content = _resolved_chat_template(spec)
        resolved_port = available_port(self.host) if self.port is None or self.config.router is not None else self.port
        model = self._model_config(spec)
        launcher = _with_subprocess_env(
            vllm_launcher(self.config),
            subprocess_env,
            self.config.startup_timeout_seconds,
        )
        with _chat_template_argument(chat_template_content) as chat_template_args:
            with VllmEnvironment(
                model=model,
                host=self.host,
                port=resolved_port,
                timeout_seconds=self.config.startup_timeout_seconds,
                extra_args=self._serve_args(spec, chat_template_args, extra_args),
                launcher=launcher,
                compilation_cache_mode=self.config.compilation_cache,
                extra_metric_families=self.config.extra_metric_families,
                router=self.config.router,
                router_port=self.port,
                wait_for_ready=False,
            ) as environment:
                yield environment

    def _serve_args(
        self,
        spec: ModelSpec,
        chat_template_args: Sequence[str],
        extra_args: Sequence[str],
    ) -> list[str]:
        loader_args = (*self.config.extra_args, *extra_args)
        if (
            self.config.launcher is not VllmLauncherType.TPU
            and _is_object_store_path(spec.weights)
            and not any(arg.partition("=")[0] in {"--load-format", RUNAI_STREAMER_LOADER_FLAG} for arg in loader_args)
        ):
            loader_args = (RUNAI_STREAMER_LOADER_FLAG, BOUNDED_DISTRIBUTED_STREAMER_CONFIG, *loader_args)
        loader_args = bound_distributed_streamer_buffers(loader_args)
        return [
            *(
                ("--tensor-parallel-size", str(spec.tensor_parallel_size))
                if spec.tensor_parallel_size is not None
                else ()
            ),
            "--served-model-name",
            spec.api_model,
            *(("--revision", spec.revision) if spec.revision is not None else ()),
            *(("--tokenizer", spec.tokenizer) if spec.tokenizer is not None else ()),
            *(("--tokenizer-revision", spec.tokenizer_revision) if spec.tokenizer_revision is not None else ()),
            *chat_template_args,
            *(
                ("--speculative-config", self.config.speculative.vllm_argument())
                if self.config.speculative is not None
                else ()
            ),
            *loader_args,
        ]

    def _model_config(self, spec: ModelSpec) -> InferenceModelConfig:
        engine_kwargs: dict[str, object] = {"dtype": spec.dtype}
        if spec.max_model_len is not None:
            engine_kwargs["max_model_len"] = spec.max_model_len
        if self.config.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self.config.max_num_batched_tokens
        if self.config.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self.config.max_num_seqs
        return InferenceModelConfig(name=spec.weights, path=spec.weights, engine_kwargs=engine_kwargs)
