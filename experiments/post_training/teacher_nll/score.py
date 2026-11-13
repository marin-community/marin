# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score teacher traces under a student: summed NLL of the loss-bearing tokens, per trace.

Each input cell is a gzipped JSONL file of rendered rows with ``input_ids`` and ``loss_weights``, where weight i
scores the prediction of token i + 1 (the SFT row convention). One Iris job serves the student with the Marin
vLLM fork on one node (DP over the node's GPUs, expert parallel) and sends every row as a token-id prompt with
``prompt_logprobs=0`` and ``max_tokens=1``. Each pass starts a fresh engine. Output, per pass and cell, is one
JSON line per row: tokens, loss tokens, and the weighted NLL sum.

    uv run python -m experiments.post_training.teacher_nll.score \
        --cells s3://marin-rno2a/tmp/ttl=14d/teacher-nll/cells/parity-16.jsonl.gz \
        --output s3://marin-rno2a/tmp/ttl=14d/teacher-nll/runs/parity --passes 1
"""

import argparse
import dataclasses
import functools
import gzip
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris import remote_inference
from marin.inference.iris_vllm import iris_vllm_launch
from marin.inference.vllm_backend import VllmBackend
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

logger = logging.getLogger(__name__)

MARIN_ROOT = Path(__file__).resolve().parents[3]
HUB_CLUSTER = "marin"
# Each GPU cluster writes to its own region's TTL bucket.
OUTPUT_ROOTS = {"cw-rno2a": "s3://marin-rno2a/tmp/ttl=", "cw-us-east-02a": "s3://marin-us-east-02a/tmp/ttl="}
# The eval campaign's 10/05 serve config (eval-campaign-09-25-unlabeled/model-configs).
REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"
# Ids from here up are special tokens (chat roles, think, tool and end markers). A base model never trained on
# them, so rows also report the NLL of text targets only.
FIRST_SPECIAL_ID = 128000
GPUS = 8
# The campaign serve path lowers this model config exactly as the eval campaign does (direct DP8 args, STREAM load).
CAMPAIGN_MODEL_CONFIG = (
    "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/"
    "open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.10.05.yaml"
)
MAX_MODEL_LEN = 73728
MAX_NUM_BATCHED_TOKENS = 7168
MAX_NUM_SEQS = 32
IN_FLIGHT = 2 * GPUS * MAX_NUM_SEQS
REQUEST_TIMEOUT = (30.0, 30 * 60.0)


def _token_id(key) -> int:
    return int(str(key).removeprefix("token_id:"))


def score_row(url: str, api_model: str, row: dict, headers: dict | None = None) -> dict:
    ids, weights = row["input_ids"], row["loss_weights"]
    if len(ids) != len(weights) or weights[-1]:
        raise ValueError(f"row {row['cell']}:{row['row']} weights do not follow the shifted convention")
    response = requests.post(
        url,
        json={
            "model": api_model,
            "prompt": ids,
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
        headers=headers or {},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    if [_token_id(t) for t in choice["prompt_token_ids"]] != ids:
        raise ValueError(f"row {row['cell']}:{row['row']}: echoed prompt differs")
    prompt_logprobs = choice["prompt_logprobs"]
    nll = text_nll = 0.0
    loss_tokens = text_tokens = 0.0
    for position, weight in enumerate(weights):
        if not weight:
            continue
        entry = {_token_id(k): v for k, v in prompt_logprobs[position + 1].items()}
        value = entry[ids[position + 1]]
        logprob = float(value["logprob"] if isinstance(value, dict) else value)
        if not math.isfinite(logprob):
            raise ValueError(f"row {row['cell']}:{row['row']}: non-finite logprob at {position + 1}")
        nll -= weight * logprob
        loss_tokens += weight
        if ids[position + 1] < FIRST_SPECIAL_ID:
            text_nll -= weight * logprob
            text_tokens += weight
    keys = ("cell", "source", "row", "family", "task_id", "trial", "kind")
    return {
        **{k: row.get(k) for k in keys},
        "tokens": len(ids),
        "loss_tokens": loss_tokens,
        "nll_sum": nll,
        "text_loss_tokens": text_tokens,
        "text_nll_sum": text_nll,
    }


def read_cell(uri: str) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(StoragePath(uri).read_bytes()).splitlines()]


def read_cells(cells: list[str]) -> dict[str, list[dict]]:
    rows = {uri: read_cell(uri) for uri in cells}
    for uri, cell_rows in rows.items():
        too_long = [r["row"] for r in cell_rows if len(r["input_ids"]) > MAX_MODEL_LEN - 1]
        if too_long:
            raise ValueError(f"{uri}: rows longer than max_model_len: {too_long[:5]}")
    return rows


def score_cells(url: str, api_model: str, rows: dict[str, list[dict]], output: str, index: int, headers=None):
    """Score every cell against one running engine and write ``pass-<index>/<cell>.jsonl``."""
    for uri, cell_rows in rows.items():
        started = time.monotonic()
        with ThreadPoolExecutor(IN_FLIGHT) as pool:
            scored = list(pool.map(functools.partial(score_row, url, api_model, headers=headers), cell_rows))
        seconds = time.monotonic() - started
        name = uri.rsplit("/", 1)[1].removesuffix(".jsonl.gz")
        StoragePath(prefix_join(output, f"pass-{index}/{name}.jsonl")).write_text(
            "".join(json.dumps(s) + "\n" for s in scored)
        )
        tokens = sum(s["tokens"] for s in scored)
        loss = sum(s["loss_tokens"] for s in scored)
        nll = sum(s["nll_sum"] for s in scored)
        logger.info(
            "pass %d %s: %d rows, %d tokens in %.0f s (%.0f tok/s), mean NLL %.6f over %d loss tokens",
            index,
            name,
            len(scored),
            tokens,
            seconds,
            tokens / seconds,
            nll / loss,
            loss,
        )


def run_campaign_job(cells: list[str], output: str, passes: int, repo: str, revision: str, cluster: str) -> None:
    """Iris CPU entrypoint: serve through the eval campaign's 10/05 path, one fresh server per pass."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    rows = read_cells(cells)
    model = load_model_config(MARIN_ROOT / CAMPAIGN_MODEL_CONFIG)
    if (repo, revision) != (model.location, model.revision):
        model = dataclasses.replace(model, name=repo.replace("/", "-"), location=repo, revision=revision)
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=GPUS, target_cluster=cluster)
    # Both scored repos are public, so no token reaches the serving jobs.
    inference = inference_config_for_model(
        model, accelerator, env_vars={}, priority=job_pb2.PRIORITY_BAND_INTERACTIVE, api_model="student"
    )
    # Keep the serving job on the scorer's cluster; the lowering leaves its placement open.
    resources = dataclasses.replace(inference.iris.worker_resources, target_cluster=cluster)
    inference = dataclasses.replace(inference, iris=dataclasses.replace(inference.iris, worker_resources=resources))
    for index in range(passes):
        with remote_inference(inference) as session:
            endpoint = session.model.endpoint
            headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
            score_cells(endpoint.url("completions"), endpoint.model, rows, output, index, headers)


def run_job(cells: list[str], output: str, passes: int, repo: str, revision: str) -> None:
    """Iris entrypoint: score every cell ``passes`` times, each pass in a fresh vLLM engine."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    rows = read_cells(cells)
    spec = ModelSpec(
        weights=repo,
        api_model="student",
        num_chips=GPUS,
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
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
            # No {"distributed":true} loader config here: vLLM accepts it only with the object-store streamer,
            # and this path loads the HF repo with the default loader.
            extra_args=(
                "--enable-expert-parallel",
                "--no-enable-prefix-caching",
                "--revision",
                revision,
            ),
        )
    )
    for index in range(passes):
        with iris_vllm_launch(pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=GPUS) as launch:
            assert launch.is_leader, "PP=1 means one task, which is the leader"
            with backend.start(spec, extra_args=launch.extra_cli_args, subprocess_env=launch.subprocess_env) as env:
                env.wait_until_ready()
                score_cells(f"{env.server_url}/completions", spec.api_model, rows, output, index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True, nargs="+", help="s3:// gzipped JSONL cell files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--repo", default=REPO, help="HF model repo to score under")
    parser.add_argument("--revision", default=REVISION, help="pinned HF commit of --repo")
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    parser.add_argument("--cluster", choices=sorted(OUTPUT_ROOTS), default="cw-rno2a")
    parser.add_argument(
        "--serve",
        choices=("direct", "campaign"),
        default="direct",
        help="direct: one GPU job runs vLLM itself; campaign: a CPU job serves through the eval campaign path",
    )
    args = parser.parse_args()
    if not args.output.startswith(OUTPUT_ROOTS[args.cluster]):
        raise ValueError(f"{args.cluster} outputs go under {OUTPUT_ROOTS[args.cluster]}<N>d/")
    job_args = [args.cells, args.output, args.passes, args.repo, args.revision]
    if args.serve == "campaign":
        entrypoint = Entrypoint.from_callable(run_campaign_job, args=[*job_args, args.cluster])
        resources = ResourceConfig.with_cpu(cpu=4, ram="32g", target_cluster=args.cluster)
    else:
        entrypoint = Entrypoint.from_callable(run_job, args=job_args)
        resources = ResourceConfig.with_gpu(
            "H100", count=GPUS, cpu=64, ram="512g", disk="512g", target_cluster=args.cluster
        )
    request = JobRequest(
        name=f"teacher-nll-{args.serve}-{os.urandom(4).hex()}",
        entrypoint=entrypoint,
        resources=resources,
        environment=create_environment(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={"HF_HUB_DISABLE_XET": "1"},
        ),
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_retries_failure=0,
        max_retries_preemption=0,
        timeout=Duration.from_seconds(int(args.timeout_hours * 3600)),
    )
    with connect_controller(cluster_name=HUB_CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=MARIN_ROOT, credentials=endpoint.credentials) as client:
            job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
    print(f"submitted {job.job_id}; output under {args.output}")


if __name__ == "__main__":
    main()
