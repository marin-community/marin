# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded, explicitly pinned historical Marin Evalchemy client after the numerical gate."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.evaluation.evalchemy.runner import EvalchemyRunConfig, _run_config_json
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

STEP12_REPO = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
STEP12_REVISION = "50fe2f488aee7730bbccc721024b35229075602e"
STEP12_EVALCHEMY = "3bb8278078e7ad4b5a6b1f81e8a08db3dd206ae6"


def run_eval(source, config, endpoint_name, output, commit, runtime_minutes, gate):
    """Resolve the private endpoint inside Iris; keep evaluator dependencies isolated."""
    config = dict(config, base_url=iris_ctx().client.resolve_endpoint(endpoint_name).rstrip("/") + "/v1")
    with urllib.request.urlopen(config["base_url"] + "/models", timeout=30) as response:
        cards = json.load(response)["data"]
    context = next(int(card["max_model_len"]) for card in cards if card["id"] == config["model_id"])
    if context < 32768:
        raise ValueError(f"Matched MATH500 requires server context>=32768; got {context}")
    with urllib.request.urlopen(config["base_url"].removesuffix("/v1") + "/levanter267/run", timeout=30) as response:
        provenance = json.load(response)
    keys = ("run_id", "source_commit", "serve_job", "model_path")
    if any(gate.get(key) != provenance.get(key) for key in keys):
        raise ValueError("Numerical gate report belongs to a different serving process or source")
    if gate.get("model_revision") != STEP12_REVISION or not STEP12_REVISION.startswith(provenance["model_revision"]):
        raise ValueError("Numerical gate report does not identify the pinned Step12 model")
    client_sha = hashlib.sha256(source.encode()).hexdigest()
    requirement = f"evalchemy[serve-eval,math500] @ git+https://github.com/marin-community/evalchemy.git@{commit}"
    command = ["uvx", "--no-config", "--python", "3.12", "--from", requirement]
    for package in ("s3fs", "gcsfs"):
        command.extend(["--with", package])
    for arch in ("x86_64", "aarch64"):
        wheel = f"torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_{arch}.whl"
        package = (
            f"torch @ https://download.pytorch.org/whl/cpu/{wheel}"
            f" ; sys_platform == 'linux' and platform_machine == '{arch}'"
        )
        command.extend(["--with", package])
    metadata = dict(
        numerical_gate=gate,
        evalchemy_commit=commit,
        client_sha256=client_sha,
        config={**config, "base_url": "<resolved inside Iris; omitted>"},
        server_context=context,
        client_tokenizer_revision=None,
        tokenizer_note="Historical client uses an unrevisioned repository string; chat tokenizer backend is none.",
        endpoint=endpoint_name,
        model_revision=STEP12_REVISION,
        command=command,
        started_epoch=time.time(),
    )
    root = StoragePath(output)
    (root / "client.py").write_text(source)
    (root / "settings.json").write_text(json.dumps(metadata, indent=2))
    env = dict(os.environ)
    for key in ("PYTHONPATH", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(
        EVALCHEMY_CLIENT_CONFIG=json.dumps(config),
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
        OPENAI_API_KEY="local-endpoint",
        HF_ALLOW_CODE_EVAL="1",
        TQDM_MININTERVAL="30",
    )
    with tempfile.TemporaryDirectory() as directory:
        client_path = Path(directory) / "evalchemy_client.py"
        client_path.write_text(source)
        result = subprocess.run(
            [*command, "python", str(client_path)], env=env, timeout=runtime_minutes * 60 - 60, check=False
        )
    metadata.update(elapsed_seconds=time.time() - metadata["started_epoch"], returncode=result.returncode)
    (root / "client-summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    if result.returncode:
        raise RuntimeError(f"Pinned Evalchemy client exited {result.returncode}; inspect {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--gate-report", type=Path, help="Accepted comparison and production-decode report for this endpoint"
    )
    parser.add_argument("--client-file", type=Path, required=True, help="Exact historical Marin client source to bundle")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; use a separate archive")
    parser.add_argument("--name", default="levanter267-step12-math500")
    parser.add_argument("--runtime-minutes", type=int, default=240)
    args = parser.parse_args()
    if not 2 <= args.runtime_minutes <= 240:
        parser.error("Runtime must be between two minutes and four hours")
    if not args.output.startswith("gs://marin-us-central2/"):
        parser.error("Place evaluator archives in us-central2 with the serving job")
    if args.limit is not None and args.limit < 1:
        parser.error("Diagnostic limit must be positive")
    gate = json.loads(args.gate_report.read_text()) if args.gate_report else None
    if args.submit and (
        not gate
        or not gate.get("accepted")
        or gate.get("endpoint") != args.endpoint
        or not gate.get("first_token_matches_cached_score")
        or not gate.get("oracle_comparison")
        or not gate.get("production_decode_token_ids")
        or not all(
            gate.get(key)
            for key in (
                "run_id",
                "source_commit",
                "serve_job",
                "model_path",
                "model_revision",
                "input_sha256",
                "oracle_sha256",
                "comparison_artifact",
            )
        )
    ):
        parser.error("Submission requires an accepted numerical and production-decode report for this endpoint")
    source = args.client_file.read_text()
    config = EvalchemyRunConfig(
        name="levanter267-step12-math500",
        tasks=(EvalTaskConfig("MATH500", 0, task_alias="math500", generation=True),),
        apply_chat_template=True,
        seed=19311,
        num_concurrent=8,
        max_eval_instances=args.limit,
        # The H100/vLLM reference client saw server32768 minus its64-token margin.
        max_length=32704,
    )
    model = RunningModel(OpenAIEndpoint("http://resolved-inside-iris/v1", "snowball-levanter"), tokenizer=STEP12_REPO)
    payload = json.loads(_run_config_json(model, config, args.output))
    plan = dict(
        numerical_gate=gate,
        endpoint=args.endpoint,
        output=args.output,
        evalchemy_commit=STEP12_EVALCHEMY,
        client_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_minutes=args.runtime_minutes,
        config=payload,
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not args.submit:
        return
    resources = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="50g", regions=["us-central2"])
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(
                    run_eval, source, payload, args.endpoint, args.output, STEP12_EVALCHEMY, args.runtime_minutes, gate
                ),
                resources=convert_resources(resources),
                constraints=convert_constraints(resources),
                environment=EnvironmentSpec(),
                timeout=Duration.from_seconds(args.runtime_minutes * 60),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
