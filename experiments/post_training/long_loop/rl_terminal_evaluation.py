# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate and preserve the distinct best and final policies of an adopted RL round."""

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.rl.skyrl import ResolvedModelLocator, SkyRLRun, skyrl_cpu_command
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.verified_copy import verified_copy_prefix

from experiments.evaluation.micro210 import Micro210Run, Micro210ServingProfile, micro210_artifact_step, run_micro210
from experiments.post_training.long_loop.archive import write_once
from vendor.skyrl.marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE, policy_export_path
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME


class FinalPolicy(Artifact):
    uri: str
    step: int
    tokenizer_uri: str
    tokenizer_revision: str
    same_as_best: bool


@dataclass(frozen=True)
class FinalExport:
    training: SkyRLRun | None
    training_output: str
    best_evaluation: str
    output: str


@dataclass(frozen=True)
class FinalPolicyModel:
    step: ArtifactStep[FinalPolicy]

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, context: StepContext) -> ResolvedModelLocator:
        identity = artifact_identity(self.step)
        if context.is_fingerprint:
            return ResolvedModelLocator(
                f"s3://unresolved/{self.step.name}/policy", identity, "/tmp/final-policy", "<tokenizer>", "<revision>"
            )
        policy = context.resolved(self.step)
        return ResolvedModelLocator(
            policy.uri, identity, "/tmp/final-policy", policy.tokenizer_uri, policy.tokenizer_revision
        )


@dataclass(frozen=True)
class ReusedEvaluation:
    result: SkyRLRun
    source: str
    output: str


@dataclass(frozen=True)
class EvaluationPreservation:
    result: SkyRLRun | None
    evaluation_output: str
    output: str


def native_command(arguments: list[str]) -> None:
    """Run the pinned native CPU environment without local operator overlays."""
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    subprocess.run(skyrl_cpu_command("--group", "harbor-test", *arguments), env=environment, check=True)


def run_final_export(config: FinalExport) -> FinalPolicy:
    """Reuse the best export or convert the final committed state through the native exporter."""
    training = config.training
    if training is None:
        raise ValueError("Final export requires a committed RL producer")
    final_step = int((StoragePath(training.checkpoint_root) / LATEST_CHECKPOINT_FILE).read_text().strip())
    same = final_step == training.global_step
    if not same:
        with tempfile.TemporaryDirectory(prefix="rl-final-export-") as directory:
            launch = Path(directory) / "launch.yaml"
            launch.write_bytes((StoragePath(config.training_output) / RESOLVED_LAUNCH_FILENAME).read_bytes())
            native_command(
                ["python", "-m", "cloud.iris.export_hf_checkpoint", "--launch-config", str(launch), "--recover-final"]
            )
    uri = (
        training.hf_model_uri
        if same
        else policy_export_path(str(StoragePath(config.training_output) / "exports"), final_step)
    )
    if uri is None or training.tokenizer_uri is None or training.tokenizer_revision is None:
        raise ValueError("Committed RL policy has no model or pinned tokenizer")
    if not uri.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Mochi terminal evaluation requires an 08a cloud model, never an hf:// model ID")
    (StoragePath(uri) / MODEL_MANIFEST_FILENAME).read_bytes()
    policy = FinalPolicy(
        uri=uri,
        step=final_step,
        tokenizer_uri=training.tokenizer_uri,
        tokenizer_revision=training.tokenizer_revision,
        same_as_best=same,
    )
    write_once(
        str(StoragePath(config.output) / "policy-selection.json"),
        json.dumps({**policy.model_dump(mode="json"), "best_evaluation": config.best_evaluation}, indent=2).encode(),
    )
    return policy


def preserve_evaluation(config: EvaluationPreservation) -> SkyRLRun:
    """Preserve a committed evaluation without rerunning its GPU producer on copy failure."""
    result = config.result
    if result is None:
        raise ValueError("Preservation requires a committed evaluation")
    reused = StoragePath(config.evaluation_output) / "evaluation-reuse.json"
    if reused.exists():
        source = json.loads(reused.read_bytes())["source"]
        (StoragePath(source) / "coverage-ready.json").read_bytes()
        write_once(str(StoragePath(config.output) / "preservation.json"), reused.read_bytes())
        return result
    if result.attempt_id is None:
        raise ValueError("Completed Micro210 has no attempt identity")
    launch_bytes = (StoragePath(config.evaluation_output) / RESOLVED_LAUNCH_FILENAME).read_bytes()
    launch_document = yaml.safe_load(launch_bytes)
    temporary = str(StoragePath(launch_document["artifacts"]["attempts_root"]).parent)
    permanent = str(StoragePath(config.evaluation_output) / "trial-archives")
    if not temporary.startswith("s3://marin-us-east-08a/tmp/") or not permanent.startswith(
        "s3://marin-us-east-08a/marin/"
    ):
        raise ValueError("Mochi trial preservation requires temporary-to-permanent storage within 08a")
    copied = verified_copy_prefix(temporary, permanent, workers=4)
    copy_manifest = StoragePath(copied.manifest_url).read_bytes()
    with tempfile.TemporaryDirectory(prefix="micro210-preservation-") as directory:
        receipt = Path(directory) / "coverage.json"
        native_command(
            [
                "python",
                "-m",
                "experiments.evaluation.micro210_finish",
                "--output",
                config.evaluation_output,
                "--attempt",
                result.attempt_id,
                "--job-id",
                result.iris_job_id,
                "--receipt",
                str(receipt),
            ]
        )
        summary = json.loads(receipt.read_bytes())
    summary.update(
        permanent_trial_archives=permanent,
        archive_copy={
            "manifest_uri": copied.manifest_url,
            "manifest_sha256": hashlib.sha256(copy_manifest).hexdigest(),
            "files": copied.total_files,
            "bytes": copied.total_bytes,
        },
    )
    write_once(
        str(StoragePath(config.evaluation_output) / "coverage-ready.json"),
        json.dumps(summary, sort_keys=True, indent=2).encode(),
    )
    write_once(
        str(StoragePath(config.output) / "preservation.json"),
        json.dumps({"source": config.evaluation_output, "job_id": result.iris_job_id}).encode(),
    )
    return result


def run_distinct_final(config: Micro210Run | ReusedEvaluation) -> SkyRLRun:
    if isinstance(config, Micro210Run):
        return run_micro210(config)
    write_once(
        str(StoragePath(config.output) / "evaluation-reuse.json"),
        json.dumps({"same_as_best": True, "source": config.source, "job_id": config.result.iris_job_id}).encode(),
    )
    return config.result


def preserved_evaluation(evaluation: ArtifactStep[SkyRLRun]) -> ArtifactStep[SkyRLRun]:
    """Commit preservation separately from the GPU evaluation it consumes."""

    def build(context: StepContext) -> EvaluationPreservation:
        return EvaluationPreservation(
            None if context.is_fingerprint else context.resolved(evaluation),
            context.artifact_path(evaluation),
            context.output_path,
        )

    return ArtifactStep(
        name=f"{evaluation.name}/preserve",
        version=evaluation.version,
        artifact_type=SkyRLRun,
        run=preserve_evaluation,
        build_config=build,
        deps=(evaluation,),
    )


def best_and_final_evaluations(
    training: ArtifactStep[SkyRLRun], best: ArtifactStep[SkyRLRun], *, serving_profile: Micro210ServingProfile | None
) -> ArtifactStep[SkyRLRun]:
    """Run best first, then evaluate final only when it names a different checkpoint."""
    best_archive = preserved_evaluation(best)

    def export_config(context: StepContext) -> FinalExport:
        return FinalExport(
            None if context.is_fingerprint else context.resolved(training),
            context.artifact_path(training),
            context.artifact_path(best_archive),
            context.output_path,
        )

    exported = ArtifactStep(
        name=f"{training.name}/final-policy",
        version=training.version,
        artifact_type=FinalPolicy,
        run=run_final_export,
        build_config=export_config,
        deps=(training, best_archive),
    )
    final = micro210_artifact_step(
        FinalPolicyModel(exported),
        name=f"{training.name}/micro210-final",
        version=training.version,
        serving_profile=serving_profile,
        model_manifest_filename=MODEL_MANIFEST_FILENAME,
    )
    original_build = final.build_config

    def final_config(context: StepContext) -> Micro210Run | ReusedEvaluation:
        if not context.is_fingerprint and context.resolved(exported).same_as_best:
            return ReusedEvaluation(context.resolved(best), context.artifact_path(best), context.output_path)
        return original_build(context)

    distinct = dataclasses.replace(
        final, run=run_distinct_final, build_config=final_config, deps=(*final.deps, best), expected_fingerprint=None
    )
    return preserved_evaluation(distinct)
