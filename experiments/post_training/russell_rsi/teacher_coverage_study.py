# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a conditional teacher dose to sixteen distinct admitted TRAIN families."""

import json
from dataclasses import asdict, dataclass, replace
from typing import cast

from levanter.main.train_lm import TrainLmConfig
from levanter.tokenizers import MarinTokenizer
from marin.evaluation.records import record_path
from marin.execution.artifact import Artifact, artifact_record_identity
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.incumbent_grade_recovery import grade_recovery_step, validated_grade_recovery
from experiments.post_training.russell_rsi.launch import adopted
from experiments.post_training.russell_rsi.launch_incumbent_bank_trial import (
    calibration_decision,
    incumbent_bank_workflow,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    evaluated_score,
    post_sft_evaluation_stages,
    qualified_four_update_sft,
)
from experiments.post_training.russell_rsi.launch_rsi_continuation import qualified_champion, selection_record
from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import (
    completed_durable_producer,
    qualified_step_telemetry,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_sft import durable_training_step
from experiments.post_training.russell_rsi.launch_teacher_sft import (
    CollectionBinding,
    StudentTrainingTemplate,
    TeacherCollectionConfig,
    run_teacher_remote,
    teacher_sft_reload_step,
    teacher_sft_steps,
    teacher_sft_training_steps,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_study import collect_chat_dataset
from experiments.post_training.russell_rsi.teacher_collection import TeacherTask
from experiments.post_training.russell_rsi.teacher_coverage_loader import (
    BATCH_SIZE,
    CONTEXT_TOKENS,
    PASSES,
    ROWS,
    UPDATES,
    CoverageLoaderProofConfig,
    run_coverage_loader_proof,
)
from experiments.post_training.russell_rsi.teacher_diversity_study import retained_chat_rows
from experiments.post_training.russell_rsi.teacher_four_pass import pinned_record

PROTOCOL = "teacher-sixteen-family-coverage-v1"
NAMESPACE = "teacher-sixteen-family-coverage"
RETAINED_ROWS = 8
CANDIDATE_INDICES = (1, 2, 4, 5, 7, 8, 9, 10, 11, 15, 26, 27)
CANDIDATE_REVIEW_SHA256 = "fad329572ebbdb08ef3865be1c3100481c63c5d1fc57566798d9e4f5b50e8e76"
CANDIDATE_LIST_SHA256 = "bb5327093e29dbe6efd0da3a1f4efc539353f0024d32c1e826dd7ffb185e6980"
PERMITTED_SLOTS = tuple(f"{family:02}-{attempt}" for family in range(12) for attempt in range(2))


@dataclass(frozen=True)
class CoverageCollectionConfig:
    collection: TeacherCollectionConfig
    study: dict


def completed_coverage_input(pins: dict, expected: str) -> dict:
    """Bind a successful producer and its exact durable status."""
    producer = PinnedFile(**pins["producer"])
    record = producer.read_json()
    root = record["output_path"]
    status = PinnedFile(**pins["status"])
    if (
        producer.uri != prefix_join(root, ".artifact.json")
        or status.uri != prefix_join(root, ".executor_status")
        or status.read_bytes().decode().strip() != STATUS_SUCCESS
        or artifact_record_identity(record) != expected
    ):
        raise ValueError("Coverage requires its exact completed producer")
    return record


def require_coverage_condition(config: dict) -> dict:
    """Require a completed incumbent calibration failure or trial nonpromotion."""
    condition = config["condition"]
    v21 = PinnedFile(**condition["v21_config"]).read_json()
    if v21["protocol"] != "incumbent-current-bank-r1" or v21["runtime_commit"] != MARIN_SKYRL.commit:
        raise ValueError("Coverage condition differs from the incumbent protocol or runtime")
    for key in ("bank", "runtime_bundle"):
        if config[key] != v21[key]:
            raise ValueError(f"Coverage changed its incumbent {key}")
    if (
        config["bank_record_uri"] != v21["bank_record_uri"]
        or config["bank_record_sha256"] != v21["bank_record_sha256"]
        or config["selection"]["bank_sha256"] != v21["bank"]["identity_config"]["bank_sha256"]
        or config["selection"]["train_sha256"] != v21["bank"]["identity_config"]["train_sha256"]
    ):
        raise ValueError("Coverage changed its exact admitted bank or training bytes")
    source = pinned_record(v21, "incumbent_artifact")
    qualification = pinned_record(v21, "qualification")
    export = qualified_champion(qualification, source)
    if (
        config["parent"]["uri"] != export
        or config["initializer"]["source_artifact"]
        != {"uri": v21["incumbent_artifact_uri"], "sha256": v21["incumbent_artifact_sha256"]}
        or config["initializer"]["qualification"]
        != {"uri": v21["qualification_uri"], "sha256": v21["qualification_sha256"]}
        or config["initializer"]["parent_identity"] != artifact_identity(adopted(config["parent"]))
        or config["parent"]["identity_config"]
        != {
            "source_identity": qualification["model_identity"],
            "source_artifact_sha256": v21["incumbent_artifact_sha256"],
            "qualification_sha256": v21["qualification_sha256"],
        }
    ):
        raise ValueError("Coverage initializer differs from its qualified update8 producer")
    calibration = incumbent_bank_workflow(v21, "calibrate")
    producers = condition["producers"]
    result = completed_coverage_input(producers["calibration"], artifact_identity(calibration["calibration"]))
    recovery = condition.get("grade_recovery")
    terminal_role = "decision"
    if recovery is None:
        decision_producer = completed_coverage_input(producers["decision"], artifact_identity(calibration["decision"]))
    else:
        seal_config = PinnedFile(**recovery["config"]).read_json()
        evidence = seal_config["evidence"]
        if (
            evidence["original_config"] != condition["v21_config"]
            or evidence["original_calibration"] != producers["calibration"]
        ):
            raise ValueError("Coverage recovery changed its original calibration lineage")
        recovered_summary, recovered_decision, recovered_provenance = validated_grade_recovery(evidence)
        terminal_role = "recovery"
        decision_producer = completed_coverage_input(
            producers[terminal_role], artifact_identity(grade_recovery_step(evidence, seal_config["version"]))
        )
        if (
            recovery["provenance"]["uri"]
            != prefix_join(decision_producer["output_path"], "grade-recovery-provenance.json")
            or canonical_json(PinnedFile(**recovery["provenance"]).read_json()) != canonical_json(recovered_provenance)
            or canonical_json(PinnedFile(**condition["summary"]).read_json()) != canonical_json(recovered_summary)
            or canonical_json(PinnedFile(**condition["decision"]).read_json()) != canonical_json(recovered_decision)
        ):
            raise ValueError("Coverage recovery files differ from their exact completed seal")
    summary_root = result["output_path"] if recovery is None else decision_producer["output_path"]
    if condition["summary"]["uri"] != prefix_join(summary_root, "failure_summary.json") or condition["decision"][
        "uri"
    ] != prefix_join(decision_producer["output_path"], "calibration-decision.json"):
        raise ValueError("Coverage calibration evidence is outside its completed producers")
    decision = PinnedFile(**condition["decision"]).read_json()
    bound = {
        **v21,
        "calibration_decision_uri": condition["decision"]["uri"],
        "calibration_decision_sha256": condition["decision"]["sha256"],
        "calibration_summary_uri": condition["summary"]["uri"],
    }
    if decision["summary_sha256"] != condition["summary"]["sha256"]:
        raise ValueError("Coverage decision cites a different calibration summary")
    decision_step = calibration["decision"]
    decision_config = decision_step.build_config(
        StepContext.for_fingerprint(decision_step.runtime_args, decision_step.deps)
    )
    expected_decision = calibration_decision(
        summary=PinnedFile(**condition["summary"]).read_json(),
        summary_sha256=condition["summary"]["sha256"],
        plan=decision_config.plan,
        families=decision_config.families,
        targeted_tasks=decision_config.targeted_tasks,
        qualification_sha256=decision_config.qualification_sha256,
        lineage_review_sha256=decision_config.lineage_review_sha256,
    )
    if compact_json_sha256(decision) != compact_json_sha256(expected_decision):
        raise ValueError("Coverage decision differs from its complete original signal evidence")
    if condition["kind"] == "calibration_failure":
        if decision["signal_gate_passed"] is not False or set(producers) != {"calibration", terminal_role}:
            raise ValueError("Coverage needs a complete failed calibration gate")
        return bound
    if condition["kind"] != "trial_nonpromotion" or decision["signal_gate_passed"] is not True:
        raise ValueError("Coverage condition is not a completed incumbent terminal branch")
    stages = incumbent_bank_workflow(bound, "train")
    evaluated = incumbent_bank_workflow(bound, "evaluate")
    records = {
        role: completed_coverage_input(producers[role], artifact_identity(handle))
        for role, handle in {**stages, **evaluated}.items()
        if role != "terminal"
    }
    if set(producers) != {"calibration", terminal_role, *records}:
        raise ValueError("Coverage trial lacks an exact completed producer set")
    if records["rl"]["result"]["global_step"] != UPDATES:
        raise ValueError("Coverage requires a completed four-update trial")
    selection = PinnedFile(**condition["selection"]).read_json()
    if condition["selection"]["uri"] != prefix_join(records["selection"]["output_path"], "continuation-selection.json"):
        raise ValueError("Coverage selection is outside its completed producer")
    for key, name in (("coding", "coding-evidence.json"), ("retention", "failure_summary.json")):
        if condition[key]["uri"] != prefix_join(records[key]["output_path"], name):
            raise ValueError("Coverage trial evidence is outside its completed producer")
    handle = evaluated["selection"]
    root = records["selection"]["output_path"]
    suffix = f"/{handle.name}/{handle.version}"
    if not root.endswith(suffix):
        raise ValueError("Coverage selection is outside its canonical artifact path")
    ctx = StepContext.for_run(root, root.removesuffix(suffix), deps=handle.deps, runtime_args=handle.runtime_args)
    binding = handle.build_config(ctx)
    candidate = evaluated_score(
        PinnedFile(**condition["coding"]).read_json(),
        PinnedFile(**condition["retention"]).read_json(),
        binding.candidate_identity,
        binding.panel_sha256,
        binding.retention_identity,
        binding.retention_task_ids,
    )
    expected = selection_record(candidate, binding.incumbent, binding.parent, v21["protocol"])
    if (
        compact_json_sha256(selection) != compact_json_sha256(expected)
        or selection["selected"] != selection["incumbent"]
    ):
        raise ValueError("Coverage cannot follow a promoted trial")
    return bound


def coverage_candidates(
    *,
    bank: dict,
    records: dict[str, str],
    candidates: list[dict],
    history: dict,
    retained: list[dict],
) -> list[dict]:
    """Keep fixed bank order and family budgets without changing capability labels."""
    tasks = bank["tasks"]
    if set(records) != {entry["task_id"] for entry in tasks}:
        raise ValueError("Coverage Parquet differs from the complete admitted bank")
    for task in tasks:
        if digest(json.loads(records[task["task_id"]])) != task["task_sha256"]:
            raise ValueError("Coverage raw task differs from its admission hash")
    attempts = history["records"]
    if (
        len(attempts) != history["consumed_trajectories"]
        or len({(entry["producer"], entry["slot"]) for entry in attempts}) != len(attempts)
        or any(not entry["family"] for entry in attempts)
    ):
        raise ValueError("Coverage attempt history is not reconciled")
    if tuple(entry["stored_index"] for entry in candidates) != CANDIDATE_INDICES:
        raise ValueError("Coverage candidates changed their fixed bank order")
    selected = []
    families = {row["task"]["family"] for row in retained}
    for entry in candidates:
        task = tasks[entry["stored_index"]]
        family = bank["family_by_task"][task["task_id"]]
        witness = entry["exclusion_proof_witness"]
        if (
            any(entry[key] != value for key, value in task.items())
            or entry["family"] != family
            or family in families
            or any(previous["family"] == family for previous in attempts)
            or entry["family_previous_attempt_count"] != 0
            or entry["remaining_new_trajectory_allowance"] != 2
            or entry["lifetime_trajectory_ceiling"] != 2
            or entry["new_trajectory_cap"] != 2
            or witness["clear"] is not True
            or witness["excluded_family_hits"]
            or witness["excluded_repository_hits"]
            or witness["excluded_source_id"]
            or witness["family"] != family
            or witness["task_id"] != task["task_id"]
            or witness["source_id"] != task["source_id"]
            or witness["stored_index"] != entry["stored_index"]
            or witness["source_repository_witness_available"] is not True
        ):
            raise ValueError("Coverage candidate changed admission, exclusion or lifetime budget")
        families.add(family)
        selected.append(asdict(TeacherTask(family, task["capability"], task["task_id"], task["task_sha256"])))
    return selected


def retained_coverage_sources(config: dict) -> None:
    """Bind completed retained bytes and the original per-trajectory locations."""
    record = completed_coverage_input(config["retained_producer"], config["retained_collection_identity"])
    root = record["output_path"]
    if config["selection"]["teacher_model"] != record["config"]["collection"]["selection"]["teacher_model"]:
        raise ValueError("Coverage changed the retained teacher settings")
    for key, name in (("retained_collection", "collection.json"), ("retained_train", "train.jsonl")):
        if config[key]["uri"] != prefix_join(root, name):
            raise ValueError("Coverage retained bytes are outside their completed producer")
    for entry in config["retained_rows"]:
        # Keep each original producer location when a later collection adopted its recovered row.
        trajectory = str(StoragePath(entry["rollout"]["uri"]).parent)
        if not trajectory.endswith(f"/trajectories/{entry['slot']}"):
            raise ValueError("Coverage retained trajectory differs from its original slot")
        for key, name in (
            ("rollout", "rollout.json"),
            ("student_row", "student-row.json"),
            ("qualification", "qualification.json"),
        ):
            if entry[key]["uri"] != prefix_join(trajectory, name):
                raise ValueError("Coverage retained pins mix different source trajectories")


def coverage_attempt_history(config: dict, bank: dict) -> dict:
    """Check issued and ambiguous reservations without using their outcomes."""
    history = PinnedFile(**config["historical_attempts"]).read_json()
    by_id = {entry["task_id"]: entry for entry in bank["tasks"]}
    for entry in history["records"]:
        reservation = PinnedFile(**entry["reservation"]).read_json()
        task = entry["task"]
        admitted = by_id[task["task_id"]]
        name, version = entry["producer"].rsplit(":", 1)[0].rsplit("@", 1)
        suffix = f"/{name}/{version}/trajectories/{entry['slot']}/trajectory.json"
        if (
            reservation != entry["reservation_record"]
            or reservation["task"] != task
            or reservation["attempt"] != int(entry["slot"].split("-")[1])
            or not entry["reservation"]["uri"].endswith(suffix)
            or task["task_sha256"] != admitted["task_sha256"]
            or task["family"] != bank["family_by_task"][task["task_id"]]
            or entry["family"] != task["family"]
        ):
            raise ValueError("Coverage history differs from its original admitted reservation")
    return history


def coverage_plan(config: dict, records: dict[str, str], tokenizer: MarinTokenizer) -> dict:
    """Check retained examples and freeze the twenty-four permitted trajectories."""
    retained_coverage_sources(config)
    bank = pinned_record(config, "bank_record")
    collection = PinnedFile(**config["retained_collection"]).read_json()
    canonical = PinnedFile(**config["retained_canonical_proof"]).read_json()
    train = PinnedFile(**config["retained_train"]).read_bytes()
    if (
        collection["status"] != "passed"
        or collection["protocol"] != "champion-rsi-teacher-eight-family-diversity-v1"
        or len(collection["accepted"]) != RETAINED_ROWS
        or len(config["retained_rows"]) != RETAINED_ROWS
        or canonical["status"] != "canonical_runtime_rows_passed"
        or canonical["train_sha256"] != config["retained_train"]["sha256"]
        or canonical["collection_identity"] != config["retained_collection_identity"]
        or canonical["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise ValueError("Coverage requires all eight original qualified rows")
    retained = retained_chat_rows(
        entries=config["retained_rows"],
        accepted_rows=collection["accepted"],
        canonical_rows=canonical["rows"],
        records=records,
        bank=bank,
        tokenizer=tokenizer,
        train=train,
    )
    if (
        len({row["task"]["family"] for row in retained}) != RETAINED_ROWS
        or len({row["row_sha256"] for row in retained}) != RETAINED_ROWS
    ):
        raise ValueError("Coverage retained rows repeat a family or example")
    history = coverage_attempt_history(config, bank)
    if history["consumed_trajectories"] != collection["cumulative_trajectories"]:
        raise ValueError("Coverage history differs from its retained collection")
    review_pin = PinnedFile(**config["candidate_budget_review"])
    review = review_pin.read_json()
    candidates = review["new_candidate_list"]
    if review_pin.sha256 != CANDIDATE_REVIEW_SHA256 or compact_json_sha256(candidates) != CANDIDATE_LIST_SHA256:
        raise ValueError("Coverage candidate review differs from the frozen twelve-family plan")
    selected = coverage_candidates(bank=bank, records=records, candidates=candidates, history=history, retained=retained)
    if selected != config["selection"]["selected"]:
        raise ValueError("Coverage collection changed its fixed selected tasks")
    return {
        "protocol": PROTOCOL,
        "selection": config["selection"],
        "retained": retained,
        "permitted_slots": list(PERMITTED_SLOTS),
        "consumed_trajectories": history["consumed_trajectories"],
        "historical_attempts": history,
        "history_scope": review["history_scope"],
    }


def run_coverage_collection(config: CoverageCollectionConfig) -> None:
    study = config.study
    require_coverage_condition(study)
    base = config.collection
    if (
        base.selection != study["selection"]
        or base.bank_path != study["bank"]["uri"]
        or base.parent_path != study["parent"]["uri"]
        or base.parent_identity != study["initializer"]["parent_identity"]
        or base.tokenizer_files != study["tokenizer_files"]
        or asdict(base.runtime_bundle) != study["runtime_bundle"]
        or base.relay_job != study["relay_job"]
    ):
        raise ValueError("Coverage worker changed its reviewed collection inputs")
    write_once(StoragePath(config.collection.output_path) / "coverage-study.json", study)
    collect_chat_dataset(
        config.collection,
        train_sha256=study["selection"]["train_sha256"],
        template=StudentTrainingTemplate(
            study["student_training_template_uri"], study["student_training_template_sha256"]
        ),
        plan=lambda records, tokenizer: coverage_plan(study, records, tokenizer),
        required_rows=ROWS,
        passes=PASSES,
        batch_size=BATCH_SIZE,
        updates=UPDATES,
    )


def run_coverage_remote(config: CoverageCollectionConfig) -> None:
    run_teacher_remote(run_coverage_collection, config)


def coverage_collection(config: dict) -> ArtifactStep:
    """Bind the new collector and a fresh four-update student dose."""
    require_coverage_condition(config)
    return _coverage_collection(config)


def _coverage_collection(config: dict) -> ArtifactStep:
    if config["protocol"] != PROTOCOL or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise ValueError("Coverage protocol or runtime differs")
    scientific = {
        **config,
        "version": config["collection_version"],
        "dose_decision_uri": config["condition"]["decision"]["uri"],
        "dose_decision_sha256": config["condition"]["decision"]["sha256"],
    }
    stages = teacher_sft_steps(
        scientific,
        CollectionBinding(lambda base: CoverageCollectionConfig(base, config), run_coverage_remote),
        UPDATES,
        NAMESPACE,
        CONTEXT_TOKENS,
        training_version=config["version"],
    )
    return stages["collect"]


def coverage_sft_workflow(config: dict) -> dict[str, ArtifactStep]:
    """Train only the exact completed collection after the actual loader proof."""
    study = PinnedFile(**config["study_config"]).read_json()
    expected = coverage_collection(study)
    return _coverage_sft_stages(config, study, expected)


def _coverage_sft_stages(config: dict, study: dict, expected: ArtifactStep) -> dict[str, ArtifactStep]:
    record = completed_coverage_input(config["collection"], artifact_identity(expected))
    root = record["output_path"]
    for key, name in (("result", "collection.json"), ("dataset", "dataset.json"), ("train", "train.jsonl")):
        if config["collection"][key]["uri"] != prefix_join(root, name):
            raise ValueError("Coverage student input is outside its completed collection")
    collection = PinnedFile(**config["collection"]["result"]).read_json()
    dataset = PinnedFile(**config["collection"]["dataset"]).read_json()
    if (
        collection["protocol"] != PROTOCOL
        or collection["status"] != "passed"
        or len(collection["accepted"]) != ROWS
        or len({row["task"]["family"] for row in collection["accepted"]}) != ROWS
        or dataset["collection_sha256"] != compact_json_sha256(collection)
        or dataset["sha256"] != config["collection"]["train"]["sha256"]
    ):
        raise ValueError("Coverage student requires sixteen qualified distinct families")
    collected = replace(expected, override_path=root)
    stages = teacher_sft_training_steps(
        study,
        adopted(study["parent"]),
        collected,
        UPDATES,
        NAMESPACE,
        CONTEXT_TOKENS,
        training_version=config["version"],
    )
    previous = stages["train"]

    def fresh_training_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = cast(TrainLmOnPodConfig, previous.build_config(ctx))
        train = cast(TrainLmConfig, pod.train_config)
        return replace(
            pod,
            env_vars={**(pod.env_vars or {}), "WANDB_MODE": "disabled"},
            train_config=replace(train, trainer=replace(train.trainer, load_checkpoint=False, metrics_start_step=0)),
        )

    trained = durable_training_step(
        replace(previous, build_config=fresh_training_config), f"russell-rsi-{NAMESPACE}-sft-{config['version']}"
    )

    def loader_config(ctx: StepContext) -> CoverageLoaderProofConfig:
        train_ctx = (
            ctx
            if ctx.is_fingerprint
            else StepContext.for_run(
                trained.path(ctx.prefix),
                ctx.prefix,
                region=ctx.region,
                deps=trained.deps,
                runtime_args={key: ctx.runtime_arg(key) for key in trained.runtime_args},
            )
        )
        pod = cast(TrainLmOnPodConfig, trained.build_config(train_ctx))
        return CoverageLoaderProofConfig(
            cast(TrainLmConfig, pod.train_config),
            {
                "jsonl_uri": config["collection"]["train"]["uri"],
                "jsonl_sha256": config["collection"]["train"]["sha256"],
                "dataset_sha256": config["collection"]["dataset"]["sha256"],
                "collection_sha256": config["collection"]["result"]["sha256"],
                "study_config_sha256": config["study_config"]["sha256"],
                "collection_identity": artifact_identity(expected),
                "tokenizer_sha256": compact_json_sha256(study["tokenizer_files"]),
                "training_template_sha256": study["student_training_template_sha256"],
                "source_review_sha256": config["sft_source_review"]["sha256"],
            },
            root,
            prefix_join(ctx.output_path, "loader-proof.json"),
        )

    proof = ArtifactStep(
        name=f"documents/russell-rsi-{NAMESPACE}-loader-proof",
        version=config["version"],
        artifact_type=Artifact,
        deps=trained.deps,
        build_config=loader_config,
        run=run_coverage_loader_proof,
        runtime_args=trained.runtime_args,
    )
    actual_train = replace(trained, deps=(*trained.deps, proof))
    return {
        **stages,
        "loader": proof,
        "train": actual_train,
        "reload": teacher_sft_reload_step(actual_train, UPDATES, NAMESPACE, config["version"]),
    }


def coverage_evaluation(config: dict) -> dict[str, ArtifactStep]:
    """Evaluate the actual qualified student once without calibration or RL."""
    dose_pin = config["sft_config"]
    dose = PinnedFile(**dose_pin).read_json()
    study = PinnedFile(**dose["study_config"]).read_json()
    v21 = require_coverage_condition(study)
    stages = _coverage_sft_stages(dose, study, _coverage_collection(study))
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
        run_id=f"russell-rsi-{NAMESPACE}-sft-{dose['version']}",
        metric_destination=prefix_join(launches["sft"]["output_path"], "optimizer-telemetry"),
    )
    producer_config = {**config, "sft_config_uri": dose_pin["uri"], "sft_config_sha256": dose_pin["sha256"]}
    records = {
        role: completed_durable_producer(producer_config, role, stages["train" if role == "sft" else "reload"], binding)
        for role in launches
    }
    for role, handle in (("sft", stages["train"]), ("reload", stages["reload"])):
        record = records[role]
        root = record["output_path"]
        suffix = f"/{handle.name}/{handle.version}"
        if not root.endswith(suffix):
            raise ValueError("Coverage producer is outside its canonical artifact output")
        ctx = StepContext.for_run(root, root.removesuffix(suffix), deps=handle.deps, runtime_args=handle.runtime_args)
        if canonical_json(record["config"]) != canonical_json(handle.build_config(ctx)):
            raise ValueError("Coverage producer executed different student settings")
    loader = completed_coverage_input(config["loader"], artifact_identity(stages["loader"]))
    loader_proof = PinnedFile(**config["loader_proof"]).read_json()
    if (
        config["loader_proof"]["uri"] != prefix_join(loader["output_path"], "loader-proof.json")
        or loader_proof["status"] != "passed"
        or loader_proof["input_pins"]["study_config_sha256"] != dose["study_config"]["sha256"]
        or loader_proof["jsonl_sha256"] != dose["collection"]["train"]["sha256"]
        or loader_proof["example_exposures"] != ROWS * PASSES
    ):
        raise ValueError("Coverage lacks its exact completed loader proof")
    qualification = PinnedFile(**config["qualification"]).read_json()
    if qualification["source_config_sha256"] != dose_pin["sha256"]:
        raise ValueError("Coverage qualification cites a different student config")
    trained, reloaded = records["sft"], records["reload"]
    identity = artifact_identity(stages["train"])
    export = qualified_four_update_sft(qualification, identity=identity, root=trained["output_path"])
    qualified_step_telemetry(qualification, binding, trained, UPDATES)
    evidence = qualification["serving_reload"]
    pin = PinnedFile(evidence["evidence_uri"], evidence["evidence_sha256"])
    reload_record = pin.read_json()
    if (
        pin.uri != record_path(reloaded["result"]["records_prefix"], reload_record["run_id"])
        or reload_record["run_id"] not in reloaded["result"]["run_ids"]
        or reload_record["status"] != "succeeded"
        or reload_record["error"] is not None
        or not reload_record["metrics"]
        or not any(reload_record["metrics"].values())
        or reload_record["model"]["config"]["identity"] != identity
        or reload_record["model"]["location"] != export
        or reload_record["eval"]["name"] != "mmlu-smoke"
        or reload_record["eval"]["evalchemy"]["max_eval_instances"] != 1
    ):
        raise ValueError("Coverage lacks its exact completed normal serving reload")
    incumbent = pinned_record(v21, "incumbent_coding")
    parent = pinned_record(v21, "parent_coding")
    retained = pinned_record(v21, "incumbent_retention")
    original_retention = pinned_record(v21, "parent_retention")
    retention = adopted(v21["retention"])
    panel_digest = incumbent["panel_sha256"]
    ids = tuple(sorted(retained["task_rewards"]))
    baseline = StudyBaseline(
        PROTOCOL,
        evaluated_score(
            incumbent,
            retained,
            pinned_record(v21, "qualification")["model_identity"],
            panel_digest,
            artifact_identity(retention),
            ids,
        ),
        evaluated_score(
            parent,
            original_retention,
            artifact_identity(adopted(v21["parent"])),
            panel_digest,
            artifact_identity(retention),
            ids,
        ),
        ids,
    )
    return post_sft_evaluation_stages(
        {**v21, "version": config["version"]},
        retention=retention,
        export_uri=export,
        checkpoints=[("sft", stages["train"])],
        barriers=(),
        outputs={},
        study=baseline,
        checkpoint_locations={"sft": export},
    )
