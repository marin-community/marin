# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Golden reference for MLX parity: serve the source revision with the Marin vLLM
fork on GPUs and record top-K logprobs for every position of every case.

Runs in the root Marin environment (not the Mac project). The request uses
/v1/completions with explicit token ids, greedy sampling, and top-K logprobs
for every prompt position (prompt_logprobs) and generated token (logprobs).

Usage:
  uv run python -m experiments.mlx_export.golden_vllm submit \
      --repo open-athena/Marin-Snowball-67B-A2B-v0.1 \
      --revision e4aa78c7c9729dea7238c2e7d1ac7fa57492e0a6 \
      --prompts out/prompts.json --gpus 8 \
      --output s3://marin-us-east-02a/marin/evals/mlx-export/<model>/golden_vllm.json
  uv run python -m experiments.mlx_export.golden_vllm download --output <same url> --dest out/golden_vllm.json

Without CoreWeave credentials on the Mac, relay the JSON through a private
Hugging Face dataset instead. A CPU job on the GPU cluster reads it from S3 and
uploads it with the submitter's HF_TOKEN:
  uv run python -m experiments.mlx_export.golden_vllm relay --output <same url> \
      --dataset open-athena/mlx-export-goldens --path-in-repo <model>/<revision>/golden_vllm.json
  uv run hf download open-athena/mlx-export-goldens <model>/<revision>/golden_vllm.json \
      --repo-type dataset --local-dir out
"""

import argparse
import json
import logging
import os
from pathlib import Path

import requests
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from huggingface_hub import HfApi
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_vllm import iris_vllm_launch
from marin.inference.vllm_backend import VllmBackend
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

MARIN_ROOT = Path(__file__).resolve().parents[2]
MAX_MODEL_LEN = 16384
LOGPROBS_K = 25
CONTINUATION_TOKENS = 64
HUB_CLUSTER = "marin"
GPU_CLUSTER = "cw-us-east-02a"
# Holds the source download; an fp32 export of a 67B model is about 270 GB.
GOLDEN_DISK = "512g"


def _token_id(key) -> int:
    return int(str(key).removeprefix("token_id:"))


def _logprob_map(raw: dict) -> dict[int, float]:
    """Normalize one vLLM logprob dict to {token_id: logprob}.

    Generated top-logprobs arrive as floats; prompt logprobs arrive as Logprob
    objects serialized to dicts.
    """
    return {_token_id(key): float(value["logprob"] if isinstance(value, dict) else value) for key, value in raw.items()}


def golden_for_case(url: str, api_model: str, case: dict, release_tag: str, revision: str) -> dict:
    response = requests.post(
        url,
        json={
            "model": api_model,
            "prompt": list(case["prompt_token_ids"]),
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": CONTINUATION_TOKENS,
            "logprobs": LOGPROBS_K,
            "prompt_logprobs": LOGPROBS_K,
            "return_tokens_as_token_ids": True,
            # Without this the choice carries prompt_token_ids / token_ids = None.
            "return_token_ids": True,
        },
        timeout=(30.0, 15 * 60.0),
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]

    echoed = [_token_id(t) for t in choice["prompt_token_ids"]]
    assert echoed == list(case["prompt_token_ids"]), case["id"]
    prompt_top = [None if entry is None else _logprob_map(entry) for entry in choice["prompt_logprobs"]]
    assert prompt_top[0] is None
    generated = [_token_id(t) for t in choice["token_ids"]]
    continuation_top = [_logprob_map(m) for m in choice["logprobs"]["top_logprobs"]]
    assert len(generated) == len(continuation_top), case["id"]
    logger.info(
        "case %s: prompt=%d continuation=%d (%s)", case["id"], len(echoed), len(generated), choice["finish_reason"]
    )
    return {
        "id": case["id"],
        "prompt_token_ids": echoed,
        "prompt_top_logprobs": prompt_top,
        "continuation_token_ids": generated,
        "continuation_top_logprobs": continuation_top,
        "vllm_release_tag": release_tag,
        "source_revision": revision,
        "finish_reason": choice["finish_reason"],
    }


def run_golden_job(repo: str, revision: str, cases: list, gpus: int, release_tag: str, output: str) -> None:
    """Iris entrypoint: serve vLLM on one node and write the goldens to ``output``."""
    logging.basicConfig(level=logging.INFO)
    spec = ModelSpec(
        weights=repo,
        api_model="golden",
        num_chips=gpus,
        # The Iris topology owns all parallel dimensions, including TP=1.
        tensor_parallel_size=None,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        chat_template_content=None,
        revision=revision,
    )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            extra_args=(
                "--enable-expert-parallel",
                "--max-logprobs",
                str(LOGPROBS_K),
                "--no-enable-prefix-caching",
                "--revision",
                revision,
            ),
        ),
    )
    with iris_vllm_launch(pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=gpus) as launch:
        assert launch.is_leader, "PP=1 means one task, which is the leader"
        with backend.start(spec, extra_args=launch.extra_cli_args, subprocess_env=launch.subprocess_env) as env:
            env.wait_until_ready()
            url = f"{env.server_url}/completions"
            goldens = [
                {"source_repo": repo, **golden_for_case(url, spec.api_model, case, release_tag, revision)}
                for case in cases
            ]

    configure_coreweave_s3()
    StoragePath(output).write_text(json.dumps(goldens))
    logger.info("wrote %d goldens to %s", len(goldens), output)


def submit_golden(repo: str, revision: str, prompts: Path, gpus: int, output: str, release_tag: str) -> None:
    """Submit the reference job and wait for successful output publication."""
    fixture = json.loads(prompts.read_text())
    assert (
        fixture["source_repo"] == repo and fixture["source_revision"] == revision
    ), f"{prompts} was built for {fixture['source_repo']}@{fixture['source_revision']}"
    request = JobRequest(
        name=f"mlx-export-golden-{os.urandom(4).hex()}",
        entrypoint=Entrypoint.from_callable(
            run_golden_job,
            args=[repo, revision, fixture["cases"], gpus, release_tag, output],
        ),
        resources=ResourceConfig.with_gpu(
            "H100", count=gpus, cpu=64, ram="512g", disk=GOLDEN_DISK, target_cluster=GPU_CLUSTER
        ),
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    job_id = submit_and_wait(request)
    print(f"completed {job_id} (vLLM {release_tag}); output: {output}")


def submit_and_wait(request: JobRequest) -> str:
    with connect_controller(cluster_name=HUB_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=MARIN_ROOT, credentials=endpoint.credentials) as client:
            job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
            job.wait(raise_on_failure=True, stream_logs=True)
    return job.job_id


def run_relay_job(output: str, dataset: str, path_in_repo: str) -> None:
    """Iris entrypoint: copy the golden JSON from S3 to a private Hugging Face dataset."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    data = StoragePath(output).read_bytes()
    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=path_in_repo, repo_id=dataset, repo_type="dataset")
    logger.info("relayed %d bytes from %s to %s:%s", len(data), output, dataset, path_in_repo)


def relay(args: argparse.Namespace) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("relay uploads with the submitter's HF_TOKEN; export it first")
    request = JobRequest(
        name=f"mlx-export-relay-{os.urandom(4).hex()}",
        entrypoint=Entrypoint.from_callable(run_relay_job, args=[args.output, args.dataset, args.path_in_repo]),
        resources=ResourceConfig.with_cpu(cpu=1, ram="4g", target_cluster=GPU_CLUSTER),
        environment=create_environment(setup_scripts=[default_setup_script(packages=["marin-core"])]),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    job_id = submit_and_wait(request)
    print(f"completed {job_id}; {args.output} is at {args.dataset}:{args.path_in_repo}")


def submit(args: argparse.Namespace) -> None:
    release_tag = VLLM_GPU_RELEASE.release_tag
    submit_golden(args.repo, args.revision, args.prompts, args.gpus, args.output, release_tag)


def download(args: argparse.Namespace) -> None:
    configure_coreweave_s3()
    args.dest.write_bytes(StoragePath(args.output).read_bytes())
    print(f"wrote {args.dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    submit_parser = commands.add_parser("submit", help="submit the golden job to Iris")
    submit_parser.add_argument("--repo", required=True)
    submit_parser.add_argument("--revision", required=True)
    submit_parser.add_argument("--prompts", required=True, type=Path)
    submit_parser.add_argument("--gpus", required=True, type=int, help="H100s on one node; vLLM runs DP=gpus with EP")
    submit_parser.add_argument("--output", required=True, help="s3:// URL on the GPU cluster's bucket")
    submit_parser.set_defaults(run=submit)
    download_parser = commands.add_parser("download", help="fetch the golden JSON")
    download_parser.add_argument("--output", required=True)
    download_parser.add_argument("--dest", required=True, type=Path)
    download_parser.set_defaults(run=download)
    relay_parser = commands.add_parser("relay", help="copy the golden JSON to a private HF dataset via a CPU job")
    relay_parser.add_argument("--output", required=True, help="the s3:// URL the golden job wrote")
    relay_parser.add_argument("--dataset", required=True, help="private HF dataset repo to upload into")
    relay_parser.add_argument("--path-in-repo", required=True)
    relay_parser.set_defaults(run=relay)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
