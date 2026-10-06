# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one prospective two-update SFT dose on the completed eight training rows."""

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import click
from levanter.main.train_lm import TrainLmConfig
from marin.evaluation.records import record_path
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_interrupted_calibration_sft import (
    foreground_build_options,
    require_reviewed_source,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    adopted,
    post_sft_evaluation_stages,
    qualified_optimizer_steps,
    qualified_sft_export,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import (
    completed_durable_producer,
    qualified_step_telemetry,
    validated_durable_diversity_training,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_sft import durable_training_step
from experiments.post_training.russell_rsi.launch_teacher_sft import (
    SFT_LEARNING_RATE,
    teacher_sft_reload_step,
    teacher_sft_training_steps,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_diversity_study import (
    CONTEXT_TOKENS,
    ROWS,
    diversity_workflow,
    validated_diversity_post_inputs,
)
from experiments.post_training.russell_rsi.teacher_four_pass import pinned_record

PROTOCOL = "champion-rsi-teacher-two-update-dose-v1"
DECISION_PROTOCOL = "teacher-two-update-dose-decision-v1"
NAMESPACE = "teacher-eight-family-two-update-dose"
VERSION = "2026.10.06.20"
UPDATES = 2
PASSES = 2
CONSUMER_MODULES = {"levanter.data.text.formats", "levanter.data.mixture", "levanter.tokenizers"}


def completed_training_rows(config: dict, expected: ArtifactStep) -> ArtifactStep:
    """Bind the original qualified TRAIN rows without running another collector."""
    pins = config["collection"]
    producer_pin = PinnedFile(**pins["producer"])
    record = producer_pin.read_json()
    root = record["output_path"]
    status = PinnedFile(**pins["status"])
    status_bytes = status.read_bytes()
    if (
        f"{record['name']}@{record['version']}:{record['fingerprint']}" != artifact_identity(expected)
        or producer_pin.uri != prefix_join(root, ".artifact.json")
        or status.uri != prefix_join(root, ".executor_status")
        or status_bytes.decode().strip() != STATUS_SUCCESS
    ):
        raise ValueError("Dose requires its exact completed eight-row collection")
    for key, name in (("result", "collection.json"), ("dataset", "dataset.json"), ("train", "train.jsonl")):
        if pins[key]["uri"] != prefix_join(root, name):
            raise ValueError("Dose training input is outside the original collection")
    collection = PinnedFile(**pins["result"]).read_json()
    dataset = PinnedFile(**pins["dataset"]).read_json()
    train = PinnedFile(**pins["train"]).read_bytes()
    rows = [json.loads(line) for line in train.splitlines() if line.strip()]
    accepted = collection["accepted"]
    witness = PinnedFile(**pins["canonical_runtime_rows"]).read_json()
    if (
        collection["status"] != "passed"
        or len(rows) != ROWS
        or len(accepted) != ROWS
        or dataset["rows"] != ROWS
        or dataset["sha256"] != hashlib.sha256(train).hexdigest()
        or dataset["collection_sha256"] != compact_json_sha256(collection)
        or rows != [entry["row"] for entry in accepted]
        or len({entry["task"]["family"] for entry in accepted}) != ROWS
        or witness["status"] != "canonical_runtime_rows_passed"
        or len(witness["consumer_imports"]) != len(CONSUMER_MODULES)
        or {entry["module"] for entry in witness["consumer_imports"]} != CONSUMER_MODULES
        or witness["collection_identity"] != artifact_identity(expected)
        or witness["runtime_commit"] != MARIN_SKYRL.commit
        or witness["train_sha256"] != dataset["sha256"]
        or witness["full_rows_no_overflow"] is not True
        or [entry["row_sha256"] for entry in witness["rows"]] != [compact_json_sha256(row) for row in rows]
        or any(entry["tokens"] > CONTEXT_TOKENS or entry["assistant_targets"] <= 0 for entry in witness["rows"])
    ):
        raise ValueError("Dose rows differ from their qualified canonical student-token evidence")
    for entry in witness["consumer_imports"]:
        module = importlib.import_module(entry["module"])
        if module.__file__ is None or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Dose student-token consumer differs from its canonical witness")
    return ArtifactStep.adopt(
        f"documents/russell-rsi-{NAMESPACE}-training-rows",
        VERSION,
        root,
        config={"original_collection": artifact_identity(expected), "pins": pins},
    )


def two_update_dose_stages(config: dict) -> dict[str, ArtifactStep]:
    if config["protocol"] != PROTOCOL or config["version"] != VERSION:
        raise ValueError("Dose must use its separate prospective version and protocol")
    original = PinnedFile(**config["original_post_config"]).read_json()
    study = pinned_record(original, "sft_config")
    original_stages = diversity_workflow(study)
    collection = completed_training_rows(config, original_stages["collect"])
    parent = adopted(study["parent"], LevanterCheckpoint)
    decision = PinnedFile(**config["prospective_decision"]).read_json()
    if decision != {
        "protocol": DECISION_PROTOCOL,
        "original_post_config": config["original_post_config"],
        "collection": config["collection"],
        "parent_identity": artifact_identity(parent),
        "rows": ROWS,
        "passes": PASSES,
        "example_exposures": ROWS * PASSES,
        "batch_size": ROWS,
        "optimizer_updates": UPDATES,
        "learning_rate": SFT_LEARNING_RATE,
        "context_tokens": CONTEXT_TOKENS,
        "fresh_optimizer": True,
        "conditions": ["sft"],
        "new_collection": False,
        "calibration": False,
        "rl_authorized": False,
    }:
        raise ValueError("Dose changed its reviewed two-update plan or original initializer")
    stages = teacher_sft_training_steps(
        study,
        parent,
        collection,
        UPDATES,
        NAMESPACE,
        CONTEXT_TOKENS,
        training_version=VERSION,
    )
    previous = stages["train"]

    def fresh_training_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = cast(TrainLmOnPodConfig, previous.build_config(ctx))
        train = cast(TrainLmConfig, pod.train_config)
        return replace(pod, train_config=replace(train, trainer=replace(train.trainer, load_checkpoint=False)))

    trained = durable_training_step(
        replace(previous, build_config=fresh_training_config), f"russell-rsi-{NAMESPACE}-sft-{VERSION}"
    )
    return {**stages, "train": trained, "reload": teacher_sft_reload_step(trained, UPDATES, NAMESPACE, VERSION)}


def qualified_two_update_sft(record: dict, *, identity: str, root: str) -> str:
    export = qualified_sft_export(
        record,
        identity=identity,
        root=root,
        updates=UPDATES,
        protocol="teacher-sft-two-update-qualification-v1",
    )
    qualified_optimizer_steps(record, identity=identity, updates=UPDATES)
    return export


def two_update_dose_evaluation(config: dict) -> dict[str, ArtifactStep]:
    dose_pin = config["dose_config"]
    dose = PinnedFile(**dose_pin).read_json()
    stages = two_update_dose_stages(dose)
    original = PinnedFile(**dose["original_post_config"]).read_json()
    baseline = validated_diversity_post_inputs(original, trained=validated_durable_diversity_training(original))
    review = PinnedFile(**config["sft_source_review"]).read_json()
    launches = {role: PinnedFile(**config[f"{role}_launch_proof"]).read_json() for role in ("sft", "reload")}
    binding = {
        "source": {"head": review["source_head"], "files": review["source_files"]},
        **{
            role: {
                "identity": artifact_identity(stages["train" if role == "sft" else "reload"]),
                "output_path": launches[role]["output_path"],
            }
            for role in launches
        },
    }
    binding["sft"].update(
        run_id=f"russell-rsi-{NAMESPACE}-sft-{VERSION}",
        metric_destination=prefix_join(launches["sft"]["output_path"], "optimizer-telemetry"),
    )
    producer_config = {**config, "sft_config_uri": dose_pin["uri"], "sft_config_sha256": dose_pin["sha256"]}
    trained = completed_durable_producer(producer_config, "sft", stages["train"], binding)
    reload = completed_durable_producer(producer_config, "reload", stages["reload"], binding)
    for role, handle in (("sft", stages["train"]), ("reload", stages["reload"])):
        record = trained if role == "sft" else reload
        root = record["output_path"]
        suffix = f"/{handle.name}/{handle.version}"
        if not root.endswith(suffix):
            raise ValueError("Dose producer is outside its canonical artifact output")
        prefix = root.removesuffix(suffix)
        ctx = StepContext.for_run(root, prefix, deps=handle.deps, runtime_args=handle.runtime_args)
        if canonical_json(record["config"]) != canonical_json(handle.build_config(ctx)):
            raise ValueError("Dose producer executed different training or reload settings")
    qualification = PinnedFile(**config["qualification"]).read_json()
    if qualification["source_config_sha256"] != dose_pin["sha256"]:
        raise ValueError("Two-update qualification cites a different prospective dose config")
    export = qualified_two_update_sft(
        qualification, identity=artifact_identity(stages["train"]), root=trained["output_path"]
    )
    qualified_step_telemetry(qualification, binding, trained, UPDATES)
    evidence = qualification["serving_reload"]
    pin = PinnedFile(evidence["evidence_uri"], evidence["evidence_sha256"])
    record = pin.read_json()
    if (
        pin.uri != record_path(reload["result"]["records_prefix"], record["run_id"])
        or record["run_id"] not in reload["result"]["run_ids"]
        or record["status"] != "succeeded"
        or record["error"] is not None
        or not record["metrics"]
        or not any(record["metrics"].values())
        or record["model"]["config"]["identity"] != artifact_identity(stages["train"])
        or record["model"]["location"] != export
        or record["eval"]["name"] != "mmlu-smoke"
        or record["eval"]["evalchemy"]["max_eval_instances"] != 1
    ):
        raise ValueError("Two-update dose lacks its exact completed serving reload")
    model = ArtifactStep.adopt(
        f"checkpoints/russell-rsi-{PROTOCOL}-qualified-hf",
        VERSION,
        export,
        kind=LevanterCheckpoint,
        config={"sft": artifact_identity(stages["train"]), "qualification_sha256": config["qualification"]["sha256"]},
    )
    return post_sft_evaluation_stages(
        {**original, "version": VERSION},
        model=model,
        retention=baseline.retention,
        export_uri=export,
        checkpoints=[("sft", model)],
        barriers=(),
        outputs={},
        study=replace(baseline.study, protocol=PROTOCOL),
    )


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option("--stage", type=click.Choice(["sft", "reload", "evaluate"]), required=True)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    if resolve_version(PROTOCOL, None) != VERSION:
        raise click.UsageError(f"Two-update dose must use version {VERSION}")
    config = PinnedFile(config_uri, config_sha256).read_json()
    if stage == "evaluate":
        return [two_update_dose_evaluation(config)["terminal"]]
    stages = two_update_dose_stages(config)
    return [stages["train" if stage == "sft" else "reload"]]


if __name__ == "__main__":
    main()
