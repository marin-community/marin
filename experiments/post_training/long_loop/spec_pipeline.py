# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the resolved run spec to the same graph for smoke and full runs."""

import hashlib
import json
from dataclasses import replace

import draccus
from fray.types import DeviceConfig, GpuConfig, ResourceConfig
from iris.rpc import job_pb2
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, IrisSkyRLExecution, SkyRLRolePlan
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.post_training.long_loop.adopted_sft import verify_adopted_sft
from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.evaluation import configured_evaluation_step, prepare_micro
from experiments.post_training.long_loop.execution_policy import COORDINATOR_TIMEOUT_SECONDS, STAGE_TIMEOUT_SECONDS
from experiments.post_training.long_loop.finalization import after_checkpoint, record_finalization
from experiments.post_training.long_loop.rl import configured_rl_step
from experiments.post_training.long_loop.run_spec import stage_blockers
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.sft import export_sft
from experiments.post_training.long_loop.sft_accounting import write_sft_accounting
from experiments.post_training.long_loop.shared_sft import export_shared_sft, legacy_sft_step, native_sft_step


def runtime_execution(value: dict) -> IrisSkyRLExecution:
    value = {
        "job_timeout_seconds": COORDINATOR_TIMEOUT_SECONDS,
        "coordinator_timeout_hours": COORDINATOR_TIMEOUT_SECONDS // 3600,
        **value,
    }
    gpu = value.get("coordinator_gpu")
    return IrisSkyRLExecution(
        **{
            **value,
            "coordinator_gpu": GpuConfig(**gpu) if gpu is not None else None,
            "required_env_vars": tuple(value["required_env_vars"]),
        }
    )


def stage_execution(value: dict, coordinator: IrisSkyRLExecution) -> IrisSkyRLExecution:
    """Bind stage resources to the loop's single parent allocation."""
    if {"coordinator_name", "coordinator_gpu", "coordinator_timeout_hours"} & value.keys():
        raise ValueError("Coordinator settings belong only in execution.coordinator")
    return replace(coordinator, **{**value, "required_env_vars": tuple(value["required_env_vars"])})


def verify_inputs(*, output: str, manifests: dict) -> None:
    """Verify the bytes named by admission manifests before any training starts."""
    verified = []
    for name, manifest in manifests.items():
        if "data_file" in manifest and not manifest.get("files"):
            raise ValueError(f"Dataset {name} has no pinned payload files")
        for entry in manifest.get("files", ()):
            fs, key = filesystem_for(entry["uri"])
            with fs.open(key, "rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise ValueError(f"Dataset {name} changed: {entry['uri']}")
            verified.append({"dataset": name, **entry})
    (StoragePath(output) / "verified.json").write_text(json.dumps(verified, indent=2))


def build_spec_graph(resolved: dict, version: str, *, through: str) -> ArtifactStep:
    blockers = stage_blockers(resolved, through)
    if blockers:
        raise ValueError("Run is not admitted: " + "; ".join(blockers))
    if resolved.get("smoke") and through in {"rl", "archive"} and resolved.get("adopted_sft") is None:
        raise ValueError("Downstream smoke requires an explicit committed adopted_sft input; SFT will not be replayed")
    datasets = resolved["datasets"]
    name = resolved["name"]
    execution = resolved["execution"]
    resources = {
        key: ResourceConfig(**{**value, "device": draccus.decode(DeviceConfig, value["device"])})
        for key, value in execution.items()
        if key not in {"coordinator", "task_timeouts"}
    }
    timeouts = {**STAGE_TIMEOUT_SECONDS, **execution.get("task_timeouts", {})}
    verified = timed(
        apply(
            f"{name}/inputs",
            remote(verify_inputs, resources=resources["inputs"], timeout=Duration.from_seconds(timeouts["inputs"])),
            version=version,
            output=OUT,
            manifests=(
                {"sft": datasets["sft"], "initializer": resolved["initializer"]} if through == "sft" else datasets
            ),
        )
    )
    sft_evidence = [verified]
    recipe = {**resolved["sft"], "task_timeout_seconds": timeouts["sft"]}
    sft_pin = resolved.get("adopted_sft")
    if sft_pin is not None:
        if through == "sft":
            raise ValueError("An adopted SFT input is for downstream continuation, not an SFT stage")
        if sft_pin["version"] == version:
            raise ValueError("Downstream continuation requires a fresh runtime version")
        trained = timed(
            apply(
                f"{name}/adopt-sft",
                verify_adopted_sft,
                version=version,
                output=OUT,
                checkpoint_pin=sft_pin,
                model=recipe["model"],
                tokenizer_revision=resolved["tokenizer"]["revision"],
            )
        )
    else:
        base_manifest = resolved["initializer"]
        base = ArtifactStep.adopt(f"{name}/base", version, base_manifest["uri"], config=base_manifest)
        sft_evidence.append(base)
        if resolved["sft_format"] == "sft_spec":
            trained = timed(native_sft_step(f"{name}/sft", version, recipe))
        else:
            trained = timed(
                legacy_sft_step(
                    f"{name}/sft",
                    version,
                    recipe,
                    base,
                    acknowledge=resolved["acknowledge_deprecated_config_format"],
                )
            )
    trained = replace(trained, deps=(*trained.deps, verified))
    coordinator = runtime_execution(execution["coordinator"])
    coordinator = replace(coordinator, coordinator_name=f"{coordinator.coordinator_name}-{version}")
    if through == "sft":
        return replace(trained, runtime_args={**trained.runtime_args, "skyrl_execution": coordinator})
    tokenizer = resolved["tokenizer"]
    handoff = resolved.get("adopted_handoff")
    if handoff is not None:
        if sft_pin is None or handoff["source_checkpoint"] != prefix_join(
            sft_pin["uri"], f"checkpoints/step-{sft_pin['step']}"
        ):
            raise ValueError("Adopted handoff must bind the selected SFT checkpoint")
        exported = ArtifactStep.adopt(f"{name}/sft-handoff", version, handoff["uri"], config=handoff)
        checked_handoff = timed(
            apply(
                f"{name}/verify-handoff",
                verify_inputs,
                version=version,
                output=OUT,
                manifests={"handoff": handoff},
            )
        )
        exported = replace(exported, deps=(*exported.deps, trained, checked_handoff))
    else:
        exported = timed(
            apply(
                f"{name}/sft-handoff",
                remote(
                    export_shared_sft if resolved["sft_format"] == "sft_spec" else export_sft,
                    resources=resources["export"],
                    env_vars={"JAX_PLATFORMS": "cpu"},
                    timeout=Duration.from_seconds(timeouts["export"]),
                ),
                version=version,
                output=OUT,
                trained=trained,
                source_revision=resolved["marin_revision"],
                context_limit=tokenizer["context_limit"],
                **(
                    {}
                    if resolved["sft_format"] == "sft_spec"
                    else {
                        "model": recipe["model"],
                        "tokenizer_uri": tokenizer["uri"],
                    }
                ),
            )
        )
    endpoint_pin = resolved.get("adopted_endpoint")
    endpoint = None
    if endpoint_pin is not None:
        if (
            handoff is None
            or endpoint_pin["model_uri"] != prefix_join(handoff["uri"], "hf")
            or endpoint_pin["tokenizer_revision"] != tokenizer["revision"]
            or endpoint_pin["settings"] != resolved["endpoint"]
        ):
            raise ValueError("Adopted endpoint must bind the identical export, tokenizer and serving settings")
        endpoint = ArtifactStep.adopt(f"{name}/endpoint", version, endpoint_pin["uri"], config=endpoint_pin)
        checked_endpoint = timed(
            apply(
                f"{name}/verify-endpoint",
                verify_inputs,
                version=version,
                output=OUT,
                manifests={"endpoint": endpoint_pin},
            )
        )
        endpoint = replace(endpoint, deps=(*endpoint.deps, exported, checked_endpoint))
    adopted = {
        key: ArtifactStep.adopt(f"{name}/data-{key}", version, datasets[key]["uri"], config=datasets[key])
        for key in ("agentic", "nonagentic", "heldout")
    }
    rl = resolved["rl"]
    rl_config = json.loads(json.dumps(rl["config"]))
    routes = {source: route for route in ("agentic", "nonagentic") for source in datasets[route]["source_names"]}
    if len(routes) != sum(len(datasets[route]["source_names"]) for route in ("agentic", "nonagentic")):
        raise ValueError("Agentic and nonagentic source names overlap")
    rl_config["trainer"]["rollout_buffer"]["source_routes"] = routes
    rl_execution = stage_execution(rl["execution"], coordinator)
    budget = rl.get("wall_clock_budget")
    callbacks = rl_config["trainer"].get("callbacks", [])
    if any(callback["type"] == "wall_time_budget" for callback in callbacks):
        raise ValueError("RL wall-time budgets belong in rl.wall_clock_budget, not trainer.callbacks")
    if budget is not None:
        required_lifetime = budget["seconds"] + budget["drain_margin"]
        if coordinator.coordinator_timeout_hours * 3600 <= required_lifetime:
            raise ValueError("Coordinator timeout must exceed the explicit RL budget plus drain margin")
        if (
            rl_execution.job_timeout_seconds
            and rl_execution.job_timeout_seconds <= budget["seconds"] + budget["drain_margin"]
        ):
            raise ValueError("RL task timeout must exceed the explicit budget plus drain margin")
        rl_config["trainer"]["callbacks"] = [
            *callbacks,
            {
                "type": "wall_time_budget",
                "seconds": budget["seconds"],
                "drain_margin": budget["drain_margin"],
                "groups_per_step": rl["role_plan"]["train_batch_size"],
            },
        ]
    if "local" in rl["config"]["environment"]["task_sessions"]:
        if (
            coordinator.container_profile != job_pb2.CONTAINER_PROFILE_PRIVILEGED
            or rl_execution.container_profile != job_pb2.CONTAINER_PROFILE_PRIVILEGED
        ):
            raise ValueError("Local sandbox requires privileged coordinator and RL child profiles")
    learned = configured_rl_step(
        version=version,
        name=rl["name"],
        model=ArtifactHfModel(
            step=exported,
            tokenizer_uri=tokenizer["repo"],
            tokenizer_revision=tokenizer["revision"],
            relative_path="hf",
        ),
        train_data=tuple(
            ArtifactDataSource(adopted[k], relative_path=datasets[k]["data_file"]) for k in ("agentic", "nonagentic")
        ),
        validation_data=(ArtifactDataSource(adopted["heldout"], relative_path=datasets["heldout"]["data_file"]),),
        ray_timeouts=rl["ray_timeouts"],
        config=rl_config,
        plan=draccus.decode(SkyRLRolePlan, rl["role_plan"]),
        execution=rl_execution,
        retained=rl["retained"],
        seed=rl["seed"],
    )
    learned = timed(replace(learned, deps=(*learned.deps, *((endpoint,) if endpoint is not None else ()))))
    if through == "rl":
        return learned
    finalized = apply(
        f"{name}/finalize",
        record_finalization,
        version=version,
        output=OUT,
        trained=learned,
    )
    evaluation = resolved["evaluation"]
    data = datasets["evaluation"]
    eval_inputs = timed(
        apply(
            f"{name}/eval-inputs",
            remote(
                prepare_micro,
                resources=resources["eval_inputs"],
                timeout=Duration.from_seconds(timeouts["eval_inputs"]),
            ),
            version=version,
            output=OUT,
            dataset=data["repo"],
            revision=data["revision"],
            wheels_revision=data["wheels_revision"],
            tasks_per_benchmark=evaluation["tasks_per_benchmark"],
        )
    )
    eval_inputs = after_checkpoint(replace(eval_inputs, deps=(*eval_inputs.deps, finalized)), learned)
    evaluated = after_checkpoint(
        timed(
            configured_evaluation_step(
                version,
                learned,
                eval_inputs,
                name=evaluation["name"],
                config=evaluation["config"],
                role_plan=draccus.decode(SkyRLRolePlan, evaluation["role_plan"]),
                execution=stage_execution(evaluation["execution"], coordinator),
                judge=draccus.decode(ModelConfig, evaluation["judge"]),
                seed=evaluation["seed"],
                expected_step=None,
                gpu_variant=evaluation["gpu_variant"],
                capability_origin=evaluation["capability_origin"],
                data_relative_path=evaluation["data_relative_path"],
                judge_gpu_count=evaluation["judge_gpu_count"],
                ray_timeouts=evaluation["ray_timeouts"],
            )
        ),
        learned,
    )
    report = after_checkpoint(
        report_step(
            version,
            [
                *sft_evidence,
                trained,
                exported,
                *((endpoint,) if endpoint is not None else ()),
                learned,
                eval_inputs,
                evaluated,
            ],
            expected_rollouts=evaluation["expected_rollouts"],
            evaluation_selection=eval_inputs,
            rl_terminal=learned,
            additional_timing=[
                item
                for item in [
                    *resolved["archive"].get("additional_evidence", []),
                    *(resolved.get("adopted_sft") or {}).get("additional_evidence", []),
                ]
                if item["category"] == "timing" and item["source"].endswith(".jsonl")
            ],
            sft_geometry=(
                f"Spec {resolved['resolved_sha256']}: {recipe['stop_after_steps']} SFT updates "
                f"at {recipe['model']['max_seq_len']} positions."
            ),
        ),
        learned,
    )
    accounting_artifacts = {}
    if "sft_data_accounting" in resolved["archive"]:
        accounting = timed(
            apply(
                f"{name}/sft-data-accounting",
                remote(
                    write_sft_accounting,
                    resources=resources["archive"],
                    env_vars={"JAX_PLATFORMS": "cpu"},
                    timeout=Duration.from_seconds(timeouts["accounting"]),
                ),
                version=version,
                output=OUT,
                trained=sft_pin["uri"] if sft_pin else trained,
                settings=resolved["archive"]["sft_data_accounting"],
            )
        )
        accounting = after_checkpoint(replace(accounting, deps=(*accounting.deps, report)), learned)
        accounting_artifacts["sft_data_accounting"] = accounting
    archived = apply(
        f"{name}/archive",
        remote(
            archive_graph_run,
            resources=resources["archive"],
            timeout=Duration.from_seconds(timeouts["archive"]),
        ),
        version=version,
        output=OUT,
        settings=resolved["archive"],
        resolved_spec=resolved,
        artifacts={
            "inputs": verified,
            "sft": sft_pin["uri"] if sft_pin else trained,
            **({"sft_adoption": trained} if sft_pin else {"base": base}),
            "export": exported,
            **({"endpoint": endpoint} if endpoint is not None else {}),
            "rl": learned,
            "finalization": finalized,
            "eval_inputs": eval_inputs,
            "eval": evaluated,
            **accounting_artifacts,
        },
        trained=learned,
        evaluated=evaluated,
        completed_report=report,
    )
    return replace(
        after_checkpoint(archived, learned),
        runtime_args={
            **archived.runtime_args,
            "skyrl_execution": coordinator,
        },
    )
