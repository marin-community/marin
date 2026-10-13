# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned Structure-Closure BFCL v6 validation modes as separate tournament evals."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import json
import logging
import os
import socket
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download, snapshot_download
from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import Platform, default_platform
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    CW_RECORDS_PREFIX,
    EVALCHEMY_INFRASTRUCTURE_ERROR,
    EvalRef,
    EvalTaskRef,
    MetricKind,
    ModelConfigRef,
    TaskCoverage,
)
from marin.evaluation.runner import (
    Evaluation,
    EvaluationBatch,
    EvaluationIdentity,
    EvaluationOutcome,
    LaunchProvenance,
    submit_evaluation_batch,
)
from rigging.config_discovery import find_project_root
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, _capability_origin, _git_sha

logger = logging.getLogger(__name__)

DATASET = "open-athena/Structure-Closure-BFCL-v6"
DATASET_REVISION = "44353e76f5a9a911a6e1473e0fe0d802ad852bca"
REFERENCE_TOKENIZER = "open-athena/Grug-67B-A2B-BFCL-Protocol-FTPO"
REFERENCE_TOKENIZER_REVISION = "d761ea0a94500c2aa24aa4cd71b5a149d962e5c8"
CASES_PER_MODE = 704
MAX_NEW_TOKENS = 1024
CONTEXT_TOKENS = 32768
SEED = 42
CONCURRENCY = 32
REQUEST_TIMEOUT = 180
MAX_TRANSPORT_ATTEMPTS = 4
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
CONFIG_DIR = Path(__file__).with_name("model-configs")

# These three checkpoints use another tokenizer.json. Their prompts must be
# re-encoded from the pinned rendered text, not sent as FTPO token IDs.
REENCODE_MODELS = frozenset(
    {
        "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92",
        "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38",
        "open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09",
    }
)


@cache
def _cases() -> list[dict[str, Any]]:
    path = hf_hub_download(DATASET, "data/validation.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    with open(path) as source:
        return [json.loads(line) for line in source]


@cache
def _scorer() -> Any:
    root = snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION, allow_patterns=["src/**"])
    source = str(Path(root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("rolloutengine.closure_scoring")


@cache
def _prompt_tokens(model: str, revision: str) -> dict[str, list[int]]:
    cases = _cases()
    if model not in REENCODE_MODELS:
        return {case["case_id"]: case["prompt_token_ids"] for case in cases}
    reference = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER, revision=REFERENCE_TOKENIZER_REVISION)
    target = AutoTokenizer.from_pretrained(model, revision=revision)
    prompts = {}
    for case in cases:
        text = reference.decode(case["prompt_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if hashlib.sha256(text.encode()).hexdigest() != case["prompt_sha256"]:
            raise ValueError(f"source prompt hash mismatch: {case['case_id']}")
        tokens = target.encode(text, add_special_tokens=False)
        decoded = target.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError(f"model tokenizer changed prompt text: {case['case_id']}")
        if len(tokens) + MAX_NEW_TOKENS > CONTEXT_TOKENS:
            raise ValueError(f"prompt exceeds benchmark context: {case['case_id']}")
        prompts[case["case_id"]] = tokens
    return prompts


def _completion(endpoint: str, model: str, tokens: list[int], api_key: str | None) -> tuple[str, str]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": tokens,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "repetition_penalty": 1,
            "seed": SEED,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{endpoint.rstrip('/')}/completions", data=payload, headers=headers)
    for attempt in range(MAX_TRANSPORT_ATTEMPTS):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                result = json.load(response)
            choice = result["choices"][0]
            return choice["text"], choice["finish_reason"]
        except HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt + 1 == MAX_TRANSPORT_ATTEMPTS:
                raise
        except (URLError, TimeoutError):
            if attempt + 1 == MAX_TRANSPORT_ATTEMPTS:
                raise
        time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class StructureClosureExecutor:
    family: str
    control: str
    model: str
    revision: str

    def __call__(self, session: Any, output_dir: str, env_vars: Mapping[str, str], *, judge: Any = None) -> EvaluationOutcome:
        """Score one mode and retain every case-level response in object storage."""
        del env_vars, judge
        cases = [case for case in _cases() if (case["family"], case["control"]) == (self.family, self.control)]
        if len(cases) != CASES_PER_MODE:
            raise ValueError(f"{self.family}/{self.control}: expected {CASES_PER_MODE} cases, found {len(cases)}")
        scorer = _scorer()
        prompts = _prompt_tokens(self.model, self.revision)
        endpoint = session.model.endpoint.base_url
        api_model = session.model.endpoint.model
        api_key = session.model.endpoint.api_key

        def score_one(case: dict[str, Any]) -> dict[str, Any]:
            try:
                text, finish_reason = _completion(endpoint, api_model, prompts[case["case_id"]], api_key)
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
                return {"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}
            continuation = scorer.closure_response_content(text, stop_reason=finish_reason)
            score = scorer.score_closure_region(
                case["prefix"] + continuation,
                case["expected"],
                scorer.ClosureFormat(case["format"]),
            )
            return {
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "prompt_sha256": case["prompt_sha256"],
                "prompt_tokens": len(prompts[case["case_id"]]),
                "response": text,
                "finish_reason": finish_reason,
                **asdict(score),
            }

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(score_one, case): case["case_id"] for case in cases}
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 100 == 0:
                    logger.info("%s/%s: %d/%d cases", self.family, self.control, len(results), len(cases))
        results.sort(key=lambda result: result["case_id"])
        destination = StoragePath(output_dir)
        destination.mkdirs()
        (destination / "samples.jsonl").write_text("".join(json.dumps(result) + "\n" for result in results))
        (destination / "effective-config.json").write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "dataset_revision": DATASET_REVISION,
                    "family": self.family,
                    "control": self.control,
                    "model": self.model,
                    "model_revision": self.revision,
                    "prompt_encoding": "model-tokenizer-reencoded" if self.model in REENCODE_MODELS else "dataset-token-ids",
                    "reference_tokenizer": REFERENCE_TOKENIZER,
                    "reference_tokenizer_revision": REFERENCE_TOKENIZER_REVISION,
                    "context_tokens": CONTEXT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "repetition_penalty": 1,
                    "seed": SEED,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                },
                indent=2,
            )
            + "\n"
        )
        scored = [result for result in results if "error" not in result]
        # All transport and endpoint failures remain unscored, even when the
        # valid sample count is high enough to report a provisional aggregate.
        errors = {EVALCHEMY_INFRASTRUCTURE_ERROR: len(results) - len(scored)} if len(scored) < len(results) else {}
        if not scored:
            raise RuntimeError(f"{self.family}/{self.control}: no scored cases; errors={errors}")
        task = f"{self.family}/{self.control}"
        metrics = {
            name: sum(bool(result[name]) for result in scored) / len(scored)
            for name in ("syntax_complete", "content_preserved", "trailing_content", "whole_response_success")
        }
        coverage = TaskCoverage(
            n_benchmark=CASES_PER_MODE,
            n_attempted=CASES_PER_MODE,
            n_scored=len(scored),
            n_correct=sum(bool(result["syntax_complete"]) for result in scored),
            errors=errors,
        )
        logger.info("%s: scored=%d/%d syntax_complete=%.4f", task, len(scored), CASES_PER_MODE, metrics["syntax_complete"])
        return EvaluationOutcome(metrics={task: metrics}, coverage={task: coverage})


def _mode_inventory() -> list[tuple[str, str]]:
    counts = Counter((case["family"], case["control"]) for case in _cases())
    if len(counts) != 21 or set(counts.values()) != {CASES_PER_MODE}:
        raise ValueError(f"unexpected mode inventory: {counts}")
    return sorted(counts)


def _batch(config_path: Path, modes: list[tuple[str, str]], cluster: str) -> EvaluationBatch:
    model: ModelConfig = load_model_config(config_path)
    if model.serve.max_model_len is None or model.serve.max_model_len < CONTEXT_TOKENS:
        raise ValueError(f"{model.name}: serve context below {CONTEXT_TOKENS}")
    accelerator = MARIN_EVAL_HARDWARE.select(model, default_platform(model), None)
    accelerator = replace(accelerator, target_cluster=cluster)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    group_id = f"{stamp}-{model.name}-structure-closure-{uuid.uuid4().hex[:4]}"
    created_at = datetime.now(UTC).isoformat()
    evaluations = []
    for family, control in modes:
        name = f"structure-closure-v6-{family}-{control}"
        task = f"{family}/{control}"
        run_id = f"{stamp}-{model.name}-{name}-{uuid.uuid4().hex[:4]}"
        metric = BenchmarkMetricRef(name="syntax_complete", source_name="syntax_complete", kind=MetricKind.BINARY, higher_is_better=True)
        benchmark = BenchmarkMetadataRef(
            schema_version=1,
            task=task,
            primary_metric="syntax_complete",
            metric_kind=MetricKind.BINARY,
            metrics=(metric,),
            n_benchmark=CASES_PER_MODE,
            n_attempted=CASES_PER_MODE,
        )
        source = {
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "mode": name,
            "model_revision": model.revision,
            "prompt_encoding": "model-tokenizer-reencoded" if model.location in REENCODE_MODELS else "dataset-token-ids",
            "max_new_tokens": MAX_NEW_TOKENS,
            "context_tokens": CONTEXT_TOKENS,
            "seed": SEED,
        }
        ref = EvalRef(
            name=name,
            mechanism="structure_closure",
            source_digest="sha256:" + hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            tasks=(EvalTaskRef(name=task, num_fewshot=0, generation=True, benchmark=benchmark),),
        )
        evaluations.append(
            Evaluation(
                identity=EvaluationIdentity(
                    run_id=run_id,
                    created_at=created_at,
                    output_dir=prefix_join(CW_RECORDS_PREFIX, f"{run_id}/results"),
                    eval_ref=ref,
                    eval_runtime=f"{DATASET}@{DATASET_REVISION}",
                ),
                executor=StructureClosureExecutor(family, control, model.location, model.revision),
            )
        )
    return EvaluationBatch(
        group_id=group_id,
        user=os.environ.get("MARIN_EVAL_USER") or getpass.getuser(),
        version=None,
        description="Tournament-only Structure-Closure BFCL v6 validation; not eval-policy attestation",
        records_prefix=CW_RECORDS_PREFIX,
        model=model,
        accelerator=accelerator,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        capability_origin=_capability_origin(cluster),
        api_model=canonical_served_name(model.name),
        evaluations=tuple(evaluations),
        provenance=LaunchProvenance(git_sha=_git_sha(), launch_host=socket.gethostname()),
        submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
        source_model_config=ModelConfigRef.model_validate(asdict(model)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Model config basename; repeatable; default all Snowball/Grug")
    parser.add_argument("--mode", action="append", help="Family/control mode; repeatable; default all 21")
    parser.add_argument("--cluster", default="cw-rno2a")
    parser.add_argument("--manifest", type=Path, help="Append submitted job and run IDs to this operational JSONL file")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    inventory = _mode_inventory()
    modes = inventory if not args.mode else [tuple(mode.split("/", 1)) for mode in args.mode]
    if not modes or any(mode not in inventory for mode in modes):
        raise ValueError(f"unknown mode; available: {inventory}")
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    configs = [path for path in configs if path.stem.startswith(("open-athena-Snowball-", "open-athena-Grug-", "laion-snowball-"))]
    if args.model:
        configs = [CONFIG_DIR / f"{name}.yaml" for name in args.model]
    batches = [_batch(path, modes, args.cluster) for path in configs]
    for batch in batches:
        print(f"{batch.model.location}: {len(batch.evaluations)} modes on {batch.accelerator.label} {args.cluster}")
    if not args.submit:
        return
    if args.manifest is None:
        parser.error("--submit requires --manifest")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        for batch in batches:
            submitted = submit_evaluation_batch(batch, client)
            with args.manifest.open("a") as target:
                target.write(
                    json.dumps(
                        {
                            "submitted_at": datetime.now(UTC).isoformat(),
                            "model": batch.model.location,
                            "model_revision": batch.model.revision,
                            "group_id": submitted.group_id,
                            "job": str(submitted.job),
                            "evaluations": [asdict(evaluation) for evaluation in submitted.evaluations],
                        }
                    )
                    + "\n"
                )
            print(f"submitted {submitted.job} ({submitted.group_id})", flush=True)


if __name__ == "__main__":
    main()
