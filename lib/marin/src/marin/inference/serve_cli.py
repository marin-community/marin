# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entrypoint for local and Iris inference serving."""

import contextlib
import logging
import time
from pathlib import Path

import click

from marin.inference.config import (
    DEFAULT_CUDA_VLLM_VERSION,
    LevanterEngineConfig,
    ServedModelConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)
from marin.inference.iris_cli import main as iris
from marin.inference.iris_cli import reject_backend_options
from marin.inference.serve import local_inference
from marin.inference.tpu_vllm_pins import vllm_fork_ref

_LOCAL_VLLM_OPTIONS = {
    "launcher": "--launcher",
    "vllm_source": "--vllm-source",
    "vllm_version": "--vllm-version",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "vllm_args": "--vllm-arg",
    "startup_timeout": "--startup-timeout",
}
_LOCAL_LEVANTER_OPTIONS = {
    "max_seqs": "--max-seqs",
    "page_size": "--page-size",
    "hbm_utilization": "--hbm-utilization",
}


@click.group(context_settings={"show_default": True})
def main() -> None:
    """Start inference locally or on Iris."""


@main.command("local")
@click.argument("model")
@click.option("--backend", type=click.Choice(["vllm", "levanter"]), default="vllm")
@click.option("--launcher", type=click.Choice([item.value for item in VllmLauncherType]), default="workspace")
@click.option("--vllm-source", type=click.Choice(["upstream", "marin-fork"]), default="upstream")
@click.option("--vllm-version", default=DEFAULT_CUDA_VLLM_VERSION)
@click.option("--tokenizer", default=None)
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=None)
@click.option("--dtype", default="bfloat16")
@click.option("--max-model-len", type=int, default=None)
@click.option("--tensor-parallel-size", type=int, default=None)
@click.option("--max-num-batched-tokens", type=int, default=None)
@click.option("--chat-template", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--vllm-arg", "vllm_args", multiple=True)
@click.option("--startup-timeout", type=int, default=1800)
@click.option("--max-seqs", type=int, default=16)
@click.option("--page-size", type=int, default=128)
@click.option("--hbm-utilization", type=float, default=0.8)
def local(
    model: str,
    backend: str,
    launcher: str,
    vllm_source: str,
    vllm_version: str,
    tokenizer: str | None,
    host: str,
    port: int | None,
    dtype: str,
    max_model_len: int | None,
    tensor_parallel_size: int | None,
    max_num_batched_tokens: int | None,
    chat_template: str | None,
    vllm_args: tuple[str, ...],
    startup_timeout: int,
    max_seqs: int,
    page_size: int,
    hbm_utilization: float,
) -> None:
    """Start one inference server on the current host."""

    logging.basicConfig(level=logging.INFO, format="[marin-serve] %(message)s")
    ctx = click.get_current_context()
    if backend == "levanter":
        reject_backend_options(backend, _LOCAL_VLLM_OPTIONS)
    else:
        reject_backend_options(backend, _LOCAL_LEVANTER_OPTIONS)

    launcher_type = VllmLauncherType(launcher)
    source = VllmSource.MARIN_FORK if vllm_source == "marin-fork" else VllmSource.UPSTREAM
    if launcher_type is not VllmLauncherType.CUDA:
        if ctx.get_parameter_source("vllm_source") is click.core.ParameterSource.COMMANDLINE:
            raise click.ClickException("--vllm-source requires --launcher cuda")
        if ctx.get_parameter_source("vllm_version") is click.core.ParameterSource.COMMANDLINE:
            raise click.ClickException("--vllm-version requires --launcher cuda")
    if source is VllmSource.MARIN_FORK and launcher_type is not VllmLauncherType.CUDA:
        raise click.ClickException("--vllm-source marin-fork requires --launcher cuda")

    model_config = ServedModelConfig(
        model=model,
        tokenizer=tokenizer,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        chat_template_content=Path(chat_template).read_text() if chat_template else None,
    )
    if backend == "vllm":
        engine = VllmEngineConfig(
            launcher=launcher_type,
            source=source,
            version=vllm_version,
            startup_timeout_seconds=startup_timeout,
            max_num_batched_tokens=max_num_batched_tokens,
            extra_args=vllm_args,
        )
    else:
        engine = LevanterEngineConfig(
            max_seqs=max_seqs,
            page_size=page_size,
            hbm_utilization=hbm_utilization,
        )
    if backend == "vllm" and launcher_type is VllmLauncherType.TPU:
        click.echo(f"Using pinned TPU vLLM {vllm_fork_ref()}")
    with local_inference(model_config, engine, host=host, port=port) as session:
        click.echo(f"OpenAI endpoint: {session.model.endpoint.base_url}")
        click.echo("Press Ctrl-C to stop.")
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                session.check_alive()
                time.sleep(30)


main.add_command(iris, name="iris")
