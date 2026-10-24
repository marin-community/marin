# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the frozen Snowball/EAGLE3 pair, or smoke all modes on one Iris GPU task."""

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path

import click
import requests
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.inference_metrics import InferenceMetricWindow
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import (
    RejectionSampleMethod,
    RemoteInferenceConfig,
    ResolvedModelLocator,
    SpeculativeMethod,
    SpeculativeServingConfig,
    VllmEngineConfig,
)
from marin.inference.iris import (
    IrisServiceConfig,
    RemoteInferenceSession,
    _prepared_local_inference,
    _register_dashboard,
    run_iris_service,
)
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.provenance import Provenance
from rigging.timing import Deadline

TARGET = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/"
DRAFT = "laion/snowball-64k-eagle3-draft-r2egym@4bdb47c08e5b5190bea3c7a93c3e14470230e469"
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
MODEL_NAME = "snowball-67b-a2b-step1903"
DEPTH = 3
CONTEXT = 65536
SMOKE_OUTPUT_TOKENS = 64
CODING_PROMPTS = (
    "Write a Python function that merges two sorted lists. Include a short example.",
    "Write a Python function that checks balanced parentheses. Explain its time complexity.",
    "Implement binary search in Python and show a test for a missing value.",
)


class Mode(StrEnum):
    BASELINE = "baseline"
    EAGLE3 = "eagle3"
    SYNTHETIC = "synthetic"


def snowball_config(mode: Mode, acceptance_length: float | None = None) -> RemoteInferenceConfig:
    """Build the fixed pair using the evaluation serving configuration path."""
    if mode is not Mode.SYNTHETIC and acceptance_length is not None:
        raise ValueError("acceptance_length requires synthetic mode")
    speculative = None
    if mode is not Mode.BASELINE:
        speculative = SpeculativeServingConfig(
            method=SpeculativeMethod.EAGLE3,
            model=ResolvedModelLocator(uri=DRAFT, identity=DRAFT),
            num_speculative_tokens=DEPTH,
            rejection_sample_method=(
                RejectionSampleMethod.SYNTHETIC if mode is Mode.SYNTHETIC else RejectionSampleMethod.STANDARD
            ),
            synthetic_acceptance_length=acceptance_length,
        )
    model = ModelConfig(
        name=MODEL_NAME,
        location=TARGET,
        identity=TARGET,
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
        resource_hint=ResourceHint(gpu={"H100": 8}, cpu=16, memory="512g", disk="200g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=CONTEXT,
            max_num_batched_tokens=7168,
            max_num_seqs=32,
            tool_call_parser="hermes",
            auto_overrides=False,
            vllm_extra_args=("--enable-expert-parallel", "--model-loader-extra-config", '{"distributed":true}'),
            speculative=speculative,
        ),
    )
    config = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        api_model=MODEL_NAME,
    )
    # A direct endpoint preserves the engine's ordinary streaming responses.
    return replace(config, broker=None)


def _request(endpoint: OpenAIEndpoint, prompt: str, *, streaming: bool) -> dict:
    payload: dict[str, object] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 17,
        "max_tokens": SMOKE_OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": streaming,
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    with requests.post(endpoint.url("chat/completions"), json=payload, stream=streaming, timeout=180) as response:
        response.raise_for_status()
        if not streaming:
            body = response.json()
            assert body["choices"][0]["message"]["content"]
            return {"request": payload, "response": body}
        events = [line.decode() for line in response.iter_lines() if line]
        assert events[-1] == "data: [DONE]"
        chunks = [json.loads(line.removeprefix("data: ")) for line in events[:-1]]
        assert any(choice["delta"].get("content") for chunk in chunks for choice in chunk["choices"])
        return {"request": payload, "response": chunks}


def smoke(service: IrisServiceConfig, output: Path) -> None:
    """Exercise the same worker and dashboard lifecycle as a persistent Iris serve."""
    job = get_job_info()
    assert job is not None
    assert isinstance(service.engine, VllmEngineConfig)
    output.mkdir(parents=True, exist_ok=True)
    with _prepared_local_inference(service.model, service.engine, service.iris) as local:
        engine_root = local.model.endpoint.server_url("")
        response = requests.get(engine_root + "server_info?config_format=json", timeout=180)
        response.raise_for_status()
        effective = response.json()["vllm_config"]
        version = requests.get(engine_root + "version", timeout=30)
        version.raise_for_status()
        assert version.json()["version"] == VLLM_GPU_RELEASE.version
        expected = service.engine.speculative
        actual = effective["speculative_config"]
        assert effective["model_config"]["max_model_len"] == CONTEXT
        if expected is None:
            assert actual is None
        else:
            assert actual["method"] == "eagle3" and actual["num_speculative_tokens"] == DEPTH
            assert actual["rejection_sample_method"] == expected.rejection_sample_method.value
            assert actual["draft_model_config"]["max_model_len"] == CONTEXT
            if expected.synthetic_acceptance_length is not None:
                assert abs(1 + sum(actual["synthetic_acceptance_rates"]) - expected.synthetic_acceptance_length) < 1e-6
        with _register_dashboard(
            service,
            local.model,
            tensor_parallel_size=local.tensor_parallel_size,
            backend_name=local.backend_name,
            streaming=True,
            chat_template_content=local.chat_template_content,
        ) as root:
            endpoint = OpenAIEndpoint(base_url=root + "/v1", model=MODEL_NAME)
            models = requests.get(endpoint.url("models"), timeout=30)
            models.raise_for_status()
            assert models.json()["data"][0]["id"] == MODEL_NAME
            session = RemoteInferenceSession(
                model=RunningModel(endpoint=endpoint, tokenizer=TOKENIZER),
                jobs=(),
                endpoint_name=service.endpoint_name,
                endpoint_health_timeout_seconds=180,
                streaming=True,
                tensor_parallel_size=local.tensor_parallel_size,
                backend_name=local.backend_name,
                metrics_url=root + "/metrics",
            )
            window = InferenceMetricWindow.start(session, speculative=expected is not None)
            responses = [_request(endpoint, prompt, streaming=i == 2) for i, prompt in enumerate(CODING_PROMPTS)]
            # vLLM publishes counters on its stats interval, after HTTP responses can finish.
            deadline = Deadline.from_seconds(30)
            while True:
                metrics = window.finish()
                if metrics.generation_tokens >= len(CODING_PROMPTS) * SMOKE_OUTPUT_TOKENS:
                    break
                deadline.raise_if_expired("vLLM generation counters did not catch up")
                time.sleep(1)
            spec_metrics = metrics.speculative_decoding
            if expected is not None:
                assert spec_metrics is not None and spec_metrics.drafts > 0
                assert 0 <= spec_metrics.accepted_tokens <= spec_metrics.draft_tokens <= DEPTH * spec_metrics.drafts
            else:
                assert spec_metrics is None
            scrape = requests.get(root + "/metrics", timeout=30)
            scrape.raise_for_status()
            (output / "metrics.prom").write_text(scrape.text)
            evidence = {
                "requested_model": asdict(service.model),
                "requested_engine": asdict(service.engine),
                "resources": asdict(service.iris.worker_resources),
                "job": str(job.task_id),
                "bundle_id": job.bundle_id,
                "source": asdict(Provenance.capture()),
                "worker_region": job.worker_region,
                "endpoint_name": service.endpoint_name,
                "engine_version": version.json(),
                "effective_engine": effective,
                "models": models.json(),
                "responses": responses,
                "metrics": metrics.model_dump(mode="json"),
                "hardware": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"], text=True
                ),
            }
            (output / "proof.json").write_text(json.dumps(evidence, default=list, indent=2))
            logging.info("Serving proof saved to %s: %s", output, metrics)


@click.command()
@click.option("--mode", type=click.Choice([mode.value for mode in Mode]), default="baseline")
@click.option("--acceptance-length", type=float, default=None, help="Measured coding acceptance, including bonus token.")
@click.option("--smoke-all", is_flag=True, help="Run three short HTTP checks sequentially on this GPU allocation.")
@click.option("--dry-run", is_flag=True, help="Print configurations without starting an engine.")
@click.option("--endpoint-prefix", required=True, help="Unique absolute Iris endpoint prefix for this launch.")
@click.option("--timeout-hours", type=float, default=1.0, help="Persistent service lifetime after startup.")
def main(
    mode: str,
    acceptance_length: float | None,
    smoke_all: bool,
    dry_run: bool,
    endpoint_prefix: str,
    timeout_hours: float,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if acceptance_length is not None and not smoke_all and Mode(mode) is not Mode.SYNTHETIC:
        raise click.UsageError("--acceptance-length requires --mode synthetic or --smoke-all")
    if not endpoint_prefix.startswith("/") or "." in endpoint_prefix:
        raise click.UsageError("--endpoint-prefix must be absolute and contain no periods")
    modes = tuple(Mode) if smoke_all else (Mode(mode),)
    for selected in modes:
        config = snowball_config(selected, acceptance_length if selected is Mode.SYNTHETIC else None)
        service = IrisServiceConfig(
            model=config.model,
            engine=config.engine,
            iris=config.iris,
            endpoint_name=f"{endpoint_prefix.rstrip('/')}-{selected.value}",
            timeout_hours=timeout_hours,
            port_name=None,
        )
        if dry_run:
            click.echo(json.dumps(asdict(service), default=list, indent=2))
        elif smoke_all:
            smoke(service, Path(os.environ["IRIS_OUTPUT_DIR"]) / selected.value)
        else:
            run_iris_service(service)


if __name__ == "__main__":
    main()
