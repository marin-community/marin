# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM local inference backend."""

import contextlib
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
    normalize_vllm_option,
)
from marin.inference.lmcache_server import LMCACHE_KV_TRANSFER_FLAG, LMCACHE_REQUIREMENT, lmcache_server
from marin.inference.model_preparation import read_tool_chat_template
from marin.inference.vllm_router import available_ports
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
        extra_requirements=(
            *config.extra_requirements,
            *((LMCACHE_REQUIREMENT,) if config.lmcache_ram_gib is not None else ()),
        ),
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
        configured_ports = () if self.port is None else (self.port,)
        resolved_port = (
            available_ports(self.host, 1, exclude=configured_ports)[0]
            if self.port is None or self.config.router is not None
            else self.port
        )
        model = self._model_config(spec)
        runtime_launcher = vllm_launcher(self.config)
        launcher = _with_subprocess_env(
            runtime_launcher,
            subprocess_env,
            self.config.startup_timeout_seconds,
        )
        with contextlib.ExitStack() as stack:
            chat_template_args = stack.enter_context(_chat_template_argument(chat_template_content))
            serve_args = self._serve_args(spec, chat_template_args, extra_args)
            cache = None
            if self.config.lmcache_ram_gib is not None:
                for arg in serve_args:
                    option = normalize_vllm_option(arg.partition("=")[0])
                    if option in {LMCACHE_KV_TRANSFER_FLAG, "--kv-offloading-size", "--kv-offloading-backend"}:
                        raise ValueError(f"lmcache_ram_gib owns {option}; remove it from extra args")
                cache_command = (
                    runtime_launcher.command(executable="python")
                    if isinstance(runtime_launcher, IsolatedCudaVllm)
                    else PreinstalledVllm().command(executable="python")
                )
                cache = stack.enter_context(
                    lmcache_server(
                        cache_command,
                        self.config.lmcache_ram_gib,
                        environment={**os.environ, **launcher.env()},
                        startup_timeout_seconds=self.config.startup_timeout_seconds,
                        excluded_ports=(resolved_port, *configured_ports),
                    )
                )
                serve_args.extend(cache.vllm_args)
            environment = stack.enter_context(
                VllmEnvironment(
                    model=model,
                    host=self.host,
                    port=resolved_port,
                    timeout_seconds=self.config.startup_timeout_seconds,
                    extra_args=serve_args,
                    launcher=launcher,
                    compilation_cache_mode=self.config.compilation_cache,
                    extra_metric_families=self.config.extra_metric_families,
                    router=self.config.router,
                    router_port=self.port,
                    wait_for_ready=False,
                    check_dependencies=cache.check_alive if cache is not None else lambda: None,
                )
            )
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
            and not any(arg.partition("=")[0] in {"--load-format", "--model-loader-extra-config"} for arg in loader_args)
        ):
            loader_args = ("--model-loader-extra-config", '{"distributed":true}', *loader_args)
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
