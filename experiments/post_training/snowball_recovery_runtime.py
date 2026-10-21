# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen e18 worker for isolated exports and one-attempt saved-answer grading."""

import argparse
import asyncio
import base64
import json
import tempfile
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from cloud.iris.artifacts import file_inventory, fs_and_path, terminal_checkpoint_step
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.runtime_bundle import resolve_launcher_source
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import OmegaConf
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from skyrl_gym.verification import VerificationResult, normalized_verifier_score
from skyrl_train.hf_export_schema import HFExportRequest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, ScriptGrader, TextMessage
from taskcompendium.runtime.grading import grade_in_sandbox
from transformers import AutoTokenizer

from experiments.post_training.snowball_inventory_evaluate_runtime import load_document, verify_model
from experiments.post_training.snowball_recovery_records import (
    RUNTIME_COMMIT,
    TOKENIZER_REVISIONS,
    encoded,
    pinned_bytes,
    prepare_records,
    require_disjoint,
    require_source,
    sha256,
)

FROZEN_LOCK_SHA256 = "27ac461277544e7cfad490520251be96cac14cbe183625240785b550bf554cab"


def source_proof() -> dict:
    revisions = {}
    for name in ("marin-taskcompendium", "marin-rolloutengine", "marin-verifyit"):
        metadata = json.loads(distribution(name).read_text("direct_url.json") or "{}")
        revisions[name] = metadata.get("vcs_info", {}).get("commit_id")
    commit = resolve_launcher_source().commit
    require_source(commit, revisions)
    lock = Path(__file__).resolve().parents[2] / "config/external/MarinSkyRL/uv.lock"
    pinned_bytes(lock.read_bytes(), FROZEN_LOCK_SHA256)
    return {"runtime_commit": commit, "library_revisions": revisions, "lock_sha256": FROZEN_LOCK_SHA256}


def read_pin(reference: dict) -> bytes:
    return pinned_bytes(StoragePath(reference["uri"]).read_bytes(), reference["sha256"])


def write_exclusive(uri: str, value: dict) -> None:
    conditional_object(uri).write(encoded(value), expected_version=None)


def training_document(request: dict):
    raw = json.loads(json.dumps(OmegaConf.to_container(OmegaConf.create(read_pin(request["training_config"]).decode()))))
    training = load_document(OmegaConf.to_yaml(OmegaConf.create(raw.get("config", raw))), expected_commit=RUNTIME_COMMIT)
    if str(training.artifacts.checkpoint_root) != request["checkpoint_root"]:
        raise ValueError("Original checkpoint root differs from the reviewed request")
    if (
        int(training.skyrl.trainer.placement.policy_num_nodes),
        int(training.skyrl.trainer.placement.policy_num_gpus_per_node),
    ) != (4, 8):
        raise ValueError("Recovery requires the original four-by-eight policy geometry")
    if str(training.inputs.model.tokenizer_revision) != TOKENIZER_REVISIONS[request["arm"]]:
        raise ValueError("Original initializer tokenizer differs from the named arm")
    return training


def original_roots(training) -> list[str]:
    roots = [str(training.artifacts[key]) for key in ("checkpoint_root", "export_root", "attempts_root")]
    roots.extend(
        str(training.artifacts[key]).rsplit("/", 1)[0] for key in ("terminal_manifest_uri", "resolved_config_uri")
    )
    roots.extend(str(training.ray[key]) for key in ("rendezvous_dir", "log_dir"))
    return roots


def protected_roots(request: dict, training) -> list[str]:
    peer = request["peer"]
    if peer["arm"] != ({"sft", "drdoom"} - {request["arm"]}).pop():
        raise ValueError("Both original initializer runs must be protected")
    return original_roots(training) + original_roots(training_document(peer))


def prepare_export(request: dict, output_path: str) -> dict:
    proof = source_proof()
    training = training_document(request)
    require_disjoint(output_path, protected_roots(request, training))
    if StoragePath(output_path).exists():
        raise FileExistsError("Recovery export destination already exists")
    root = str(training.artifacts.checkpoint_root)
    if terminal_checkpoint_step(root) != 50:
        raise ValueError("Exact committed checkpoint50 marker required")
    filesystem, path = fs_and_path(root + "/global_step_50")
    inventory = {entry.path: entry.size for _, entry in file_inventory(filesystem, path)}
    required = {
        "trainer_state.pt",
        "policy/.metadata",
        *(f"policy/__{rank}_0.distcp" for rank in range(32)),
    }
    if any(inventory.get(name, 0) <= 0 for name in required):
        raise ValueError("Original checkpoint50 inventory is incomplete")
    model = training.inputs.model
    export = HFExportRequest(
        step=50,
        checkpoint_base_path=root,
        checkpoint_path=root + "/global_step_50",
        export_path=output_path + "/exports",
        model_path=str(model.local_path),
        num_nodes=4,
        gpus_per_node=8,
        model_source_uri=str(model.uri),
        model_source_identity=str(model.identity),
    )
    spec = ExportJobSpec(
        request=export,
        cluster=str(training.iris.cluster),
        priority=str(training.iris.priority),
        gpu_variant=str(training.iris.allocation.gpu_variant),
        job_name=None,
        timeout=7200,
        no_wait=False,
        cluster_config=str(training.iris.cluster_config),
        target_cluster=training.iris.target_cluster,
        parent_cluster_config=training.iris.parent_cluster_config,
        cpu=32,
        memory="1800GB",
        disk="2TB",
        allocation_gpus_per_node=8,
    )
    config = checkpoint_export_launch_config(training, export, spec)
    config.run.id = "snowball-recovery-" + sha256(output_path.encode())[:12]
    config.run.attempt_id = "step-50"
    config.iris.job_name = config.run.id + "-step-50"
    config.artifacts.attempts_root = output_path + "/attempts"
    config.artifacts.resolved_config_uri = output_path + "/resolved-launch.yaml"
    config.artifacts.terminal_manifest_uri = output_path + "/terminal.json"
    config.ray.rendezvous_dir = output_path + "/attempts/rendezvous"
    config.ray.log_dir = output_path + "/attempts/ray-logs"
    document = OmegaConf.to_yaml(config, resolve=True)
    load_document(document, expected_commit=RUNTIME_COMMIT)
    return {
        "launch_config_yaml": document,
        "policy_export_uri": output_path + "/exports/global_step_50/policy",
        "source": proof,
        "training_config_sha256": request["training_config"]["sha256"],
        "checkpoint_root": root,
        "checkpoint_step": 50,
        "original_training_job_success": False,
    }


def prepare_regrade(request: dict, output_path: str) -> dict:
    proof = source_proof()
    training = training_document(request)
    require_disjoint(output_path, protected_roots(request, training))
    report = json.loads(read_pin(request["report"]))
    panel_bytes = read_pin(request["panel"])
    if report["panel_sha256"] != request["panel"]["sha256"]:
        raise ValueError("Original report and pinned panel differ")
    panel = pq.read_table(pa.BufferReader(panel_bytes)).to_pylist()
    probe = json.loads(read_pin(request["probe"]))
    dumps = [
        (json.loads(line), sha256(line))
        for source in request["dumps"]
        for line in read_pin(source).splitlines(keepends=True)
        if line.strip()
    ]
    manifest = json.loads(read_pin(request["tokenizer"]["manifest"]))
    if manifest["repository_revision"] != TOKENIZER_REVISIONS[request["arm"]]:
        raise ValueError("Pinned tokenizer inventory names another initializer")
    names = [entry["path"] for entry in manifest["files"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Tokenizer inventory must contain unique file paths")
    with tempfile.TemporaryDirectory(prefix="snowball-recovery-tokenizer-") as directory:
        for entry in manifest["files"]:
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
                raise ValueError("Unsafe tokenizer inventory path")
            raw = read_pin(
                {"uri": request["tokenizer"]["root"].rstrip("/") + "/" + str(path), "sha256": entry["sha256"]}
            )
            target = Path(directory) / str(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        prepared = prepare_records(
            request["arm"],
            report,
            probe,
            panel,
            dumps,
            request["failed_rows"],
            tokenizer.decode,
            tokenizer.eos_token_id,
            final_answer_text,
        )
    prepared.update(
        source=proof,
        input_request_sha256=sha256(encoded(request)),
        original_report_bytes_sha256=request["report"]["sha256"],
        tokenizer_manifest_sha256=request["tokenizer"]["manifest"]["sha256"],
    )
    return prepared


async def execute_regrade(request: dict, prepared: dict, output_path: str) -> dict:
    proof = source_proof()
    training = training_document(request)
    require_disjoint(output_path, protected_roots(request, training))
    if prepared["input_request_sha256"] != sha256(encoded(request)) or prepared["source"] != proof:
        raise ValueError("Prepared recovery is bound to another request or runtime")
    admission = json.loads(read_pin(request["environment_admission"]))
    digests = sorted(row["task_rollout_spec_sha256"] for row in prepared["requests"])
    if (
        admission["source"] != proof
        or admission["task_rollout_spec_sha256"] != digests
        or admission["tokenizer_manifest_sha256"] != prepared["tokenizer_manifest_sha256"]
        or admission["scope"] != "original-grader-contract-definition"
    ):
        raise ValueError("Root-reviewed original-grader/tokenizer admission evidence is missing")
    # An exclusive durable reservation prevents restarting this artifact from grading twice.
    write_exclusive(
        output_path + "/attempt-started.json",
        {"task_specs": digests, "attempt_limit": 1, "source": proof, "owner": uuid4().hex},
    )
    sessions = OmegaConf.create({"daytona": OmegaConf.to_container(training.skyrl.environment.task_sessions.daytona)})
    factories = task_machine_factories(sessions)
    receipts = []
    for row in prepared["requests"]:
        evidence = {key: row[key] for key in ("task_id", "answer_sha256", "task_rollout_spec_sha256")}
        evidence.update(source=proof)
        try:
            spec = TaskRolloutSpec.model_validate_json(
                pinned_bytes(row["task_rollout_spec"].encode(), row["task_rollout_spec_sha256"])
            )
            answer = pinned_bytes(
                base64.b64decode(row["answer_utf8_base64"], validate=True), row["answer_sha256"]
            ).decode()
            attempt = GradingAttempt(
                ConversationTrace(events=(*spec.task.context.events, TextMessage(role="assistant", content=answer))),
                stop_reason=row["stop_reason"],
            )
            assert spec.verifier_machine is not None
            assert isinstance(spec.task.grader, ScriptGrader)
            async with asyncio.timeout(spec.limits.attempt_timeout), asyncio.timeout(spec.limits.verifier_timeout):
                grade = await grade_in_sandbox(
                    spec.task,
                    attempt,
                    factories[spec.verifier_machine.backend],
                    _machine_spec(spec.task.grader.environment, spec.verifier_machine),
                    timeout=spec.limits.verifier_timeout,
                )
            evidence["grade"] = asdict(grade)
            if grade.status != Outcome.GRADED or grade.reward is None:
                raise ValueError("Original grader attempt remained unscored; no retry is permitted")
            score = normalized_verifier_score(
                VerificationResult.verified(grade.reward, score_min=grade.score_min, score_max=grade.score_max)
            )
            if not 0 <= score <= 1:
                raise ValueError("Original grader returned an invalid normalized score")
            evidence["score"] = score
        except Exception as error:
            evidence["failure_type"] = type(error).__name__
            write_exclusive(output_path + "/" + sha256(row["task_id"].encode()) + "-failure.json", evidence)
            raise
        body = encoded(evidence)
        identity = sha256(body)
        conditional_object(output_path + "/" + identity + ".json").write(body, expected_version=None)
        receipt = {
            key: evidence[key] for key in ("task_id", "answer_sha256", "task_rollout_spec_sha256", "score", "source")
        }
        receipt.update(attempt_count=1, execution_evidence_sha256=identity)
        write_exclusive(output_path + "/" + sha256(row["task_id"].encode()) + "-receipt.json", receipt)
        receipts.append(receipt)
    return {"receipts": receipts, "source": proof}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-export", "prepare-regrade", "execute-regrade", "verify-model"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.request.read_bytes())
    if args.action == "prepare-export":
        result = prepare_export(value["request"], value["output_path"])
    elif args.action == "prepare-regrade":
        result = prepare_regrade(value["request"], value["output_path"])
    elif args.action == "execute-regrade":
        result = asyncio.run(execute_regrade(value["request"], value["prepared"], value["output_path"]))
    else:
        source_proof()
        result = verify_model(value["model_uri"])
    args.result.write_bytes(encoded(result))


if __name__ == "__main__":
    main()
