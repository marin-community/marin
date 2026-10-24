# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the existing campaign/evaluator client and preserve canary wire evidence."""

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from finestore.reader import ReadView
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _evalchemy_client_command, _run_config_json
from marin.evaluation.evalchemy.runtime import evalchemy_runtime_source_hash
from marin.evaluation.lm_eval_samples import read_evalchemy_results
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.evals import evalchemy_run_config
from experiments.evaluation.serving_parity.capture import capture_requests, echo_contract
from experiments.evaluation.serving_parity.teacher import collect_teacher

HERE = Path("experiments/evaluation/serving_parity")
POLICY = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs")
MODEL = "open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02"
REVISION = "50fe2f488aee7730bbccc721024b35229075602e"


def policy_config(task: str):
    config = evalchemy_run_config(task, load_evalchemy_config(POLICY / f"{task}.yaml"))
    generation = any(t.generation for t in config.tasks)
    return dataclasses.replace(
        config,
        apply_chat_template=True,
        max_length=8192,
        max_eval_instances=100,
        num_concurrent=16,
        seed=42,
        max_gen_toks=2048 if generation else None,
        extra_gen_kwargs={"temperature": "0", "top_p": "1", "top_k": "-1"} if generation else {},
    )


def document_hash(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def verify_samples(archive: str, task: str, expected: dict) -> list[dict]:
    rows = list(ReadView(archive).iter_rows("samples"))
    selected = {}
    ids_by_hash = {digest: identity for identity, digest in expected.items()}
    for row in rows:
        metrics = dict(row["metrics"])
        key = "exact_match" if task == "gsm8k-0shot" else "acc"
        if key not in metrics or (task == "gsm8k-0shot" and row["filter"] != "flexible-extract"):
            continue
        document = json.loads(row["doc"])
        digest = document_hash(document)
        if digest not in ids_by_hash:
            raise ValueError(f"Dataset row changed or outside frozen subset: {task}/{row['doc_id']}")
        identity = ids_by_hash[digest]
        if identity in selected:
            raise ValueError(f"Duplicate sample {task}/{identity}")
        selected[identity] = dict(
            id=identity,
            correct=bool(metrics[key]),
            doc_sha256=digest,
            output=row.get("output"),
            prompt_text=row.get("prompt_text"),
            prompt_messages=row.get("prompt_messages"),
            question=document["question"],
        )
    if selected.keys() != expected.keys():
        raise ValueError(f"Incomplete sample coverage: {task} ({len(selected)}/100)")
    return list(selected.values())


def run_workload(endpoint: str, output: str, teacher_item: dict, remaining_seconds: float) -> None:
    deadline = time.monotonic() + remaining_seconds
    root = StoragePath(output)
    manifest = json.loads((HERE / "manifest.json").read_text())
    summary = []
    with tempfile.TemporaryDirectory(prefix="serving-parity-") as temporary:
        scratch = Path(temporary)
        for concurrency in (1, 4):
            target = scratch / f"teacher-{concurrency}"
            collect_teacher(endpoint, MODEL, teacher_item, target, concurrency=concurrency)
            for path in target.iterdir():
                (root / f"teacher-{concurrency}" / path.name).write_bytes(path.read_bytes())
        for task in ("gsm8k-0shot", "boolq"):
            config = policy_config(task)
            archive = output.rstrip("/") + "/" + task
            capture = scratch / f"{task}-wire.jsonl"
            log = scratch / f"{task}.log"
            started = time.monotonic()
            try:
                with capture_requests(endpoint, capture) as proxy:
                    payload = _run_config_json(
                        RunningModel(OpenAIEndpoint(proxy, MODEL), tokenizer=MODEL), config, archive
                    )
                    env = dict(
                        os.environ,
                        **{CONFIG_ENV_KEY: payload},
                        CUDA_VISIBLE_DEVICES="",
                        JAX_PLATFORMS="cpu",
                        OPENAI_API_KEY="local-endpoint",
                    )
                    env.pop("PYTHONPATH", None)
                    env.pop("UV_PROJECT_ENVIRONMENT", None)
                    command = [
                        *_evalchemy_client_command(config.runtime),
                        "lib/marin/src/marin/evaluation/evalchemy/client.py",
                    ]
                    with log.open("w") as stream:
                        process = subprocess.run(
                            command,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=max(1, deadline - time.monotonic()),
                            check=False,
                        )
            finally:
                for path in (capture, log):
                    if path.exists():
                        (root / path.name).write_bytes(path.read_bytes())
            if process.returncode:
                raise RuntimeError(f"{task} client failed ({process.returncode}); see archived log")
            documents = read_evalchemy_results(archive)
            for document in documents.values():
                for outcome in document["task_outcomes"].values():
                    if outcome["status"] != "succeeded" or outcome["scored_count"] != 100:
                        raise ValueError(f"{task} did not score all 100 frozen items")
            wires = [json.loads(line) for line in capture.read_text().splitlines()]
            for wire in wires:
                if wire["status"] != 200:
                    raise ValueError(f"{task} encountered an HTTP failure")
                echo_contract(wire["request"], wire["response"])
                if task == "gsm8k-0shot" and wire["request"].get("temperature") != 0:
                    raise ValueError("Generation is not greedy at the wire")
            expected = {x["id"]: x["doc_sha256"] for x in manifest["gsm8k" if task == "gsm8k-0shot" else task]["items"]}
            samples = verify_samples(archive, task, expected)
            for sample in samples:
                matching = []
                for wire in wires:
                    request = wire["request"]
                    prompt = request.get("prompt", "")
                    if "messages" in request:
                        prompt = "\n".join(message["content"] for message in request["messages"])
                    if isinstance(prompt, str) and sample["question"] in prompt:
                        matching.append(wire)
                expected_requests = 1 if task == "gsm8k-0shot" else 2
                if len(matching) != expected_requests:
                    raise ValueError(f"Cannot uniquely match wire responses for {task}/{sample['id']}")
                sample["finish_reason"] = (
                    matching[0]["response"]["choices"][0]["finish_reason"] if task == "gsm8k-0shot" else "loglikelihood"
                )
                sample["completion_tokens"] = sum(wire["response"]["usage"]["completion_tokens"] for wire in matching)
            (root / f"{task}-items.json").write_text(json.dumps(samples))
            summary.append(
                dict(task=task, wall_seconds=time.monotonic() - started, documents=documents, requests=len(wires))
            )
            (root / "results.json").write_text(json.dumps(summary))
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                model=MODEL,
                revision=REVISION,
                evaluator_source=evalchemy_runtime_source_hash(),
                manifest_sha256=hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest(),
            )
        )
    )
