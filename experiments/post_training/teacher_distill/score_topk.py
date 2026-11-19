# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store Mochi's top-20 next-token distribution on the assistant tokens of a fixed SFT slice.

Input cells are gzipped JSONL files of documents with ``input_ids`` and shifted ``loss_weights`` (weight i scores
token i + 1), written by ``prep_slice``. One Iris CPU coordinator serves Mochi SFT step 400407 through the eval
serve path (four GB200x4 hosts, TP1 DP16 with expert parallelism, compile and CUDA graphs on) and sends every
document as a token-id prompt with ``prompt_logprobs=20`` and ``max_tokens=1``.

Output per cell, ``<output>/<split>/docs-NNN.npz``: for every loss-bearing position i of every document, the
teacher's top 20 token ids and logprobs for token i + 1 (``ids``/``logprobs``, [N, 20], sorted by logprob), the
teacher's logprob of the actual token i + 1 (``target``), the position (``positions``) and per-document offsets
(``doc_rows``, ``doc_offsets``). Cells are scored held-out first, then train, in order; once ``--budget-seconds``
of scoring has passed no new cell starts, so a cap drops whole cells (whole packed rows).

    uv run --env-file .env python -m experiments.post_training.teacher_distill.score_topk \
        --cells-root s3://marin-us-east-08a/tmp/ttl=14d/teacher-distill-504/cells \
        --output s3://marin-us-east-08a/tmp/ttl=14d/teacher-distill-504/teacher-top20
"""

import argparse
import dataclasses
import gzip
import io
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.setup_scripts import default_setup_script
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE

logger = logging.getLogger(__name__)

MARIN_ROOT = Path(__file__).resolve().parents[3]
HUB_CLUSTER = "marin"
CLUSTER = "cw-us-east-08a"
MODEL_CONFIG = "experiments/evaluation/mochi/model-mochi-sft-step400407.yaml"
TOP = 20
MAX_MODEL_LEN = 73728
# Four in flight per DP rank keeps every rank's prefill queue full without large response buffers.
IN_FLIGHT = 64
REQUEST_TIMEOUT = (30.0, 30 * 60.0)


def _token_id(key) -> int:
    return int(str(key).removeprefix("token_id:"))


def score_doc(task: tuple[str, str, dict, dict]) -> dict:
    url, api_model, headers, doc = task
    ids, weights = doc["input_ids"], doc["loss_weights"]
    if len(ids) != len(weights) or weights[-1] or len(ids) > MAX_MODEL_LEN - 1:
        raise ValueError(f"doc {doc['row']}: bad weights or too long")
    response = requests.post(
        url,
        json={
            "model": api_model,
            "prompt": ids,
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": TOP,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        },
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    (choice,) = response.json()["choices"]
    if [_token_id(t) for t in choice["prompt_token_ids"]] != ids:
        raise ValueError(f"doc {doc['row']}: echoed prompt differs")
    prompt_logprobs = choice["prompt_logprobs"]
    positions = [i for i, w in enumerate(weights) if w]
    top_ids = np.zeros((len(positions), TOP), np.int32)
    top_lp = np.zeros((len(positions), TOP), np.float32)
    target = np.zeros(len(positions), np.float32)
    for n, i in enumerate(positions):
        entry = [(_token_id(k), float(v["logprob"])) for k, v in prompt_logprobs[i + 1].items()]
        by_id = dict(entry)
        actual = by_id.get(ids[i + 1])
        if actual is None or not math.isfinite(actual):
            raise ValueError(f"doc {doc['row']}: missing or non-finite target logprob at {i + 1}")
        target[n] = actual
        best = sorted(entry, key=lambda kv: -kv[1])[:TOP]
        if len(best) != TOP:
            raise ValueError(f"doc {doc['row']}: {len(best)} logprobs at {i + 1}, expected {TOP}")
        top_ids[n] = [k for k, _ in best]
        top_lp[n] = [v for _, v in best]
    return {
        "row": doc["row"],
        "tokens": len(ids),
        "positions": np.asarray(positions, np.int32),
        "ids": top_ids,
        "logprobs": top_lp,
        "target": target,
    }


def list_cells(cells_root: str) -> list[str]:
    cells = []
    for split in ("heldout", "train"):
        listing = sorted(str(p) for p in StoragePath(prefix_join(cells_root, split)).iterdir())
        cells.extend(p for p in listing if p.endswith(".jsonl.gz"))
    return cells


def score_cell(pool, url, api_model, headers, cell: str, cells_root: str, output: str) -> dict:
    docs = [json.loads(line) for line in gzip.decompress(StoragePath(cell).read_bytes()).splitlines()]
    started = time.monotonic()
    scored = list(pool.map(score_doc, [(url, api_model, headers, d) for d in docs]))
    seconds = time.monotonic() - started
    counts = [len(s["positions"]) for s in scored]
    buffer = io.BytesIO()
    np.savez(
        buffer,
        doc_rows=np.asarray([s["row"] for s in scored], np.int64),
        doc_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        positions=np.concatenate([s["positions"] for s in scored]),
        ids=np.concatenate([s["ids"] for s in scored]),
        logprobs=np.concatenate([s["logprobs"] for s in scored]),
        target=np.concatenate([s["target"] for s in scored]),
    )
    relative = cell.removeprefix(cells_root).lstrip("/").removesuffix(".jsonl.gz")
    StoragePath(prefix_join(output, relative + ".npz")).write_bytes(buffer.getvalue())
    tokens = sum(s["tokens"] for s in scored)
    loss_tokens = sum(counts)
    nll = -float(sum(s["target"].sum() for s in scored))
    summary = {
        "cell": relative,
        "docs": len(docs),
        "tokens": tokens,
        "loss_tokens": loss_tokens,
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "teacher_nll": nll / loss_tokens,
    }
    logger.info("scored %s", json.dumps(summary))
    return summary


def run_job(cells_root: str, output: str, budget_seconds: float) -> None:
    """Iris CPU entrypoint: one Mochi serve, then every cell until the scoring budget is spent."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    cells = list_cells(cells_root)
    model = load_model_config(MARIN_ROOT / MODEL_CONFIG)
    accelerator = MARIN_EVAL_HARDWARE.select(model, Platform.GPU, None)
    inference = inference_config_for_model(
        model,
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
        api_model="teacher",
    )
    # Keep the serving job on 08a; the lowering leaves its placement open.
    resources = dataclasses.replace(inference.iris.worker_resources, target_cluster=CLUSTER)
    inference = dataclasses.replace(inference, iris=dataclasses.replace(inference.iris, worker_resources=resources))
    summaries = []
    serve_started = time.monotonic()
    with remote_inference(inference) as session:
        session.wait_until_ready()
        ready_seconds = time.monotonic() - serve_started
        logger.info("Mochi ready after %.0f s", ready_seconds)
        endpoint = session.model.endpoint
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        url = endpoint.url("completions")
        scoring_started = time.monotonic()
        with ProcessPoolExecutor(IN_FLIGHT) as pool:
            for cell in cells:
                if time.monotonic() - scoring_started > budget_seconds:
                    logger.info("scoring budget spent; %s and later cells are not scored", cell)
                    break
                summaries.append(score_cell(pool, url, endpoint.model, headers, cell, cells_root, output))
                StoragePath(prefix_join(output, "summary.json")).write_text(
                    json.dumps({"ready_seconds": ready_seconds, "cells": summaries}, indent=2) + "\n"
                )
    scored_tokens = sum(s["tokens"] for s in summaries)
    seconds = sum(s["seconds"] for s in summaries)
    logger.info("done: %d cells, %d tokens in %.0f s (%.0f tok/s)", len(summaries), scored_tokens, seconds,
                scored_tokens / max(seconds, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-hours", type=float, default=2.5)
    args = parser.parse_args()
    if not args.output.startswith("s3://marin-us-east-08a/tmp/ttl="):
        raise ValueError("outputs go under s3://marin-us-east-08a/tmp/ttl=<N>d/")
    request = JobRequest(
        name=f"task504-mochi-top20-{os.urandom(4).hex()}",
        entrypoint=Entrypoint.from_callable(run_job, args=[args.cells_root, args.output, args.budget_seconds]),
        resources=ResourceConfig.with_cpu(cpu=64, ram="192g", disk="64g", target_cluster=CLUSTER),
        environment=create_environment(setup_scripts=[default_setup_script(packages=["marin-core"])]),
        priority=job_pb2.PRIORITY_BAND_PRODUCTION,
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
