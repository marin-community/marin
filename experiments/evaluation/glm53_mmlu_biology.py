# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate GLM-5.3 on the complete MMLU biology selection from an Iris CPU job."""

import datetime
import hashlib
import json
import logging
import math
import os
import socket
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import click
from finestore.reader import ReadView
from iris.client.client import iris_ctx
from levanter.compat.hf_checkpoints import _save_tokenizer_pretrained
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import EvalchemyExecutor, _evalchemy_client_command
from marin.evaluation.lm_eval_samples import _loglikelihood_pair, read_native_evalchemy_artifacts
from marin.evaluation.records import (
    EvalRunRecord,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    RunTiming,
    ServingParams,
    write_record,
)
from marin.evaluation.runner import EvaluationError
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.evaluation.evals import EvalchemyDefinition, evalchemy_run_config

EXPECTED_QUESTIONS = 554
LIKELIHOOD_FLOOR = -9999


def audit_native(root: str) -> dict:
    reader = ReadView(root)
    artifacts = read_native_evalchemy_artifacts(root)
    report = []
    for name in artifacts.sample_sources:
        payload = reader.read_blob(name)
        for line in payload.decode().splitlines():
            row = json.loads(line)
            pairs = [_loglikelihood_pair(x) for x in row["filtered_resps"]]
            scores = []
            for pair in pairs:
                assert pair is not None, "Missing choice likelihood"
                scores.append(pair[0])
            assert len(scores) == 4
            assert all(math.isfinite(x) for x in scores), "Non-finite choice likelihood"
            winner = max(range(4), key=lambda i: scores[i])
            target = int(row["doc"]["answer"])
            passed = float(winner == target)
            assert passed == row["acc"], "Native grade disagrees with independent argmax"
            report.append(
                {
                    "source": name,
                    "doc_id": row["doc_id"],
                    "loglikelihoods": scores,
                    "model_choice": winner,
                    "target_choice": target,
                    "correct": bool(passed),
                    "censored_options": sum(x <= LIKELIHOOD_FLOOR for x in scores),
                    "all_options_censored": max(scores) <= LIKELIHOOD_FLOOR,
                    "tied_max": sum(x == max(scores) for x in scores) > 1,
                    "arguments_type": type(row["arguments"]).__name__,
                }
            )
    summary = {
        "samples": len(report),
        "correct": sum(x["correct"] for x in report),
        "censored_options": sum(x["censored_options"] for x in report),
        "all_options_censored": sum(x["all_options_censored"] for x in report),
        "tied_max": sum(x["tied_max"] for x in report),
        "rows": report,
    }
    summary["unique_samples"] = len({(x["source"], x["doc_id"]) for x in report})
    assert summary["samples"] == summary["unique_samples"] == EXPECTED_QUESTIONS
    return summary


@click.command()
@click.option("--base-url", required=True, help="OpenAI-compatible API base URL, including /v1.")
@click.option("--run-id", required=True)
@click.option("--records-prefix", required=True, help="Object-store prefix indexed by Eval Dash.")
@click.option("--git-sha", required=True, help="Source commit shipped with this Iris job.")
def main(base_url: str, run_id: str, records_prefix: str, git_sha: str) -> None:
    """Run with OPENAI_API_KEY set in the job environment."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    context = iris_ctx()
    output = f'{records_prefix.rstrip("/")}/{run_id}/results'
    source_path = Path(__file__).with_name("configs") / "evalchemy" / "mmlu-biology.yaml"
    source = load_evalchemy_config(source_path)
    tokenizer = "zai-org/GLM-5.3"
    tokenizer_revision = "aca966e4e02791568aa6a4ced368624b3d897f42"
    config = replace(evalchemy_run_config("mmlu-biology", source), extra_model_args={"revision": tokenizer_revision})
    original_tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)
    probes = [
        "The capital of France is Paris.",
        "Answer: A",
        "Answer: B",
        "Answer: C",
        "Answer: D",
        "DNA polymerase synthesizes DNA in the 5\u2032 to 3\u2032 direction.",
        "A. mitosis\nB. meiosis\nC. replication\nD. transcription\nAnswer:",
        "  \u03b1-globin\n\n[ gMASK ]",
        *original_tokenizer.all_special_tokens,
    ]
    expected_ids = [original_tokenizer.encode(text, add_special_tokens=False) for text in probes]
    with tempfile.TemporaryDirectory() as portable_dir:
        _save_tokenizer_pretrained(original_tokenizer, portable_dir)
        saved_backend = json.loads(Path(portable_dir, "tokenizer.json").read_text())
        assert saved_backend == json.loads(original_tokenizer.backend_tokenizer.to_str())
        check = "from transformers import AutoTokenizer; import json,sys; p=json.load(sys.stdin); "
        check += "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True, "
        check += "extra_special_tokens={},additional_special_tokens=p[2]); "
        check += "t.save_pretrained(sys.argv[1]); "
        check += "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True); "
        check += "assert sorted(t.all_special_ids)==p[3]; "
        check += "assert [t.encode(s,add_special_tokens=False) for s in p[0]] == p[1]; "
        check += 'print("Tokenizer parity passed in evaluator runtime")'
        subprocess.run(
            [*_evalchemy_client_command(config.runtime), "-c", check, portable_dir],
            input=json.dumps(
                [
                    probes,
                    expected_ids,
                    original_tokenizer.extra_special_tokens,
                    sorted(original_tokenizer.all_special_ids),
                ]
            ),
            text=True,
            check=True,
        )
        assert json.loads(Path(portable_dir, "tokenizer.json").read_text()) == saved_backend
        assets = {
            f"_glm53_tokenizer/{path.name}": path.read_bytes() for path in Path(portable_dir).iterdir() if path.is_file()
        }
    config = replace(config, runtime=replace(config.runtime, workdir_files=assets))
    evaluation = EvalchemyDefinition(config.name, source_path).record_ref_for(config)
    key = os.environ["OPENAI_API_KEY"]
    session = RemoteInferenceSession(
        model=RunningModel(OpenAIEndpoint(base_url.rstrip("/"), "glm-5.3", key), tokenizer="_glm53_tokenizer"),
        jobs=(),
        endpoint_name="external-glm-5.3",
        endpoint_health_timeout_seconds=30,
        streaming=False,
        tensor_parallel_size=1,
        backend_name="external",
    )
    metadata = {
        "run_id": run_id,
        "eval": evaluation.model_dump(mode="json"),
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "portable_tokenizer": (
            "Marin PR 8163 export, then Transformers 4 save_pretrained with equivalent special-token metadata"
        ),
        "tokenizer_parity": (
            "Biology strings and all special tokens match original token IDs in the evaluator runtime; "
            "special-token IDs and backend JSON unchanged"
        ),
        "tokenizer_file_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()},
        "weights_revision": "unrecorded",
        "git_sha": git_sha,
        "policy": "https://github.com/marin-community/marin/issues/9193",
        "runtime": config.runtime.requirement,
    }
    artifact_root = StoragePath(f'{records_prefix.rstrip("/")}/{run_id}')
    (artifact_root / "run-manifest.json").write_text(json.dumps(metadata, indent=2))
    (artifact_root / "run.py").write_text(Path(__file__).read_text())
    for name, data in assets.items():
        (artifact_root / name).write_bytes(data)
    print(json.dumps(metadata, indent=2), flush=True)
    metrics, canonical, coverage, tails = {}, {}, {}, {}
    jobs = {"orchestrator": str(context.job_id)}
    status, error = RunStatus.SUCCEEDED, None
    try:
        outcome = EvalchemyExecutor(config)(session, output, env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS))
        metrics, canonical, coverage = outcome.metrics, outcome.canonical_metrics, outcome.coverage
        jobs.update(outcome.jobs)
        if outcome.tasks is not None:
            evaluation = evaluation.model_copy(update={"tasks": outcome.tasks})
    except EvaluationError as exc:
        status, error = exc.status, str(exc)
        jobs.update(exc.jobs)
        coverage, tails = exc.coverage, exc.log_tails
    if status is RunStatus.SUCCEEDED:
        audit = audit_native(output)
        audit_json = json.dumps(audit, indent=2)
        (artifact_root / "native-grade-audit.json").write_text(audit_json)
        Path(os.environ["IRIS_OUTPUT_DIR"], "native-grade-audit.json").write_text(audit_json)
        print(json.dumps({"native_grade_audit": {k: v for k, v in audit.items() if k != "rows"}}), flush=True)
        assert sum(c.n_correct for c in coverage.values()) == audit["correct"], "Coverage disagrees with native grades"
        if audit["all_options_censored"]:
            status, error = RunStatus.FAILED, "Native audit found all-floor answer choices; review required"
        if sum(c.n_scored for c in coverage.values()) != EXPECTED_QUESTIONS or any(c.errors for c in coverage.values()):
            status, error = RunStatus.FAILED, "Incomplete coverage or infrastructure errors; review required"
    finished = datetime.datetime.now(datetime.UTC).isoformat()
    record = EvalRunRecord(
        run_id=run_id,
        group_id=run_id,
        created_at=started,
        user=str(context.job_id).split("/")[1],
        version="mmlu-biology-5shot-v1",
        description=(
            "MMLU biology: all 554 questions across college biology, high-school biology and medical genetics; "
            "Marin five-shot likelihood protocol. Weights revision unrecorded; tokenizer pinned. "
            "Complete biology subset."
        ),
        model=ModelRef(name="glm-5.3", location=tokenizer, backend="external"),
        evaluation=evaluation,
        hardware=HardwareRef(platform="external", accelerator="unreported", region_or_cluster=None),
        status=status,
        error=error,
        results_path=output,
        metrics=metrics,
        canonical_metrics=canonical,
        coverage=coverage,
        jobs=jobs,
        log_tails=tails,
        provenance=Provenance(
            git_sha=metadata["git_sha"], eval_runtime=config.runtime.requirement, launch_host=socket.gethostname()
        ),
        timing=RunTiming(started_at=started, finished_at=finished),
        serving=ServingParams(extra={"weights_revision": "unrecorded", "tokenizer_revision": tokenizer_revision}),
    )
    location = write_record(record, records_prefix)
    Path(os.environ["IRIS_OUTPUT_DIR"], "record.json").write_text(record.model_dump_json(indent=2, by_alias=True))
    print(json.dumps({"record_path": location, "record": record.model_dump(mode="json", by_alias=True)}), flush=True)
    if status is not RunStatus.SUCCEEDED:
        raise RuntimeError(error)
    assert sum(c.n_scored for c in coverage.values()) == EXPECTED_QUESTIONS, "Run did not score all 554 items"
    assert all(not c.errors for c in coverage.values()), "Run has infrastructure errors"


if __name__ == "__main__":
    main()
