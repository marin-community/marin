# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the resolved run spec to the same graph for smoke and full runs."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import draccus
from fray.types import DeviceConfig, GpuConfig, ResourceConfig
from iris.rpc import job_pb2
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.curriculum import RlCurriculum
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, IrisSkyRLExecution, SkyRLRolePlan
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration
from skyrlrecipe.locality import RoleTopologyMode

from experiments.evaluation import micro210_policy
from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210 import PREPARED_INPUTS, micro210_artifact_step
from experiments.post_training.long_loop.adopted_sft import is_hf_export, verify_adopted_hf_export, verify_adopted_sft
from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.evaluation import TerminalPolicy
from experiments.post_training.long_loop.execution_policy import COORDINATOR_TIMEOUT_SECONDS, STAGE_TIMEOUT_SECONDS
from experiments.post_training.long_loop.finalization import after_checkpoint, record_finalization
from experiments.post_training.long_loop.openresearcher_service import (
    OpenResearcherService,
    retrieval_launch,
    run_with_retrieval,
    service_execution,
)
from experiments.post_training.long_loop.rl import configured_rl_step, curriculum_validation_callbacks
from experiments.post_training.long_loop.run_spec import stage_blockers
from experiments.post_training.long_loop.runtime_timing import prepare_reporting_inputs, report_step, timed
from experiments.post_training.long_loop.sft import HANDOFF_DIR, export_sft
from experiments.post_training.long_loop.sft_accounting import write_sft_accounting
from experiments.post_training.long_loop.shared_sft import export_shared_sft, legacy_sft_step, native_sft_step
from experiments.post_training.task_curation.curriculum_manifest import rollout_source_routes


@dataclass(frozen=True)
class CompletedSft:
    """Adopt a completed native SFT and its model handoff for downstream recovery."""

    trained: ArtifactStep
    handoff: ArtifactStep
    model_relative_path: str = HANDOFF_DIR


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
            "excluded_nvlink_domains": tuple(value.get("excluded_nvlink_domains", ())),
            "role_topology": (
                RoleTopologyMode(value["role_topology"]) if value.get("role_topology") is not None else None
            ),
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
            with StoragePath(entry["uri"]).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise ValueError(f"Dataset {name} changed: {entry['uri']}")
            verified.append({"dataset": name, **entry})
    (StoragePath(output) / "verified.json").write_text(json.dumps(verified, indent=2))


def build_spec_graph(
    resolved: dict,
    version: str,
    *,
    through: str,
    data_artifacts: dict[str, ArtifactStep] | None = None,
    curriculum: RlCurriculum | None = None,
    sft_release: ArtifactStep | None = None,
    completed_sft: CompletedSft | None = None,
    openresearcher_service: OpenResearcherService | None = None,
    sft_step_builder: Callable[[str, str, dict], ArtifactStep] = native_sft_step,
) -> ArtifactStep:
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
    if data_artifacts:
        verified = replace(verified, deps=(*verified.deps, *data_artifacts.values()))
    if sft_release is not None:
        verified = replace(verified, deps=(*verified.deps, sft_release))
    sft_evidence = [verified]
    recipe = {**resolved["sft"], "task_timeout_seconds": timeouts["sft"]}
    sft_pin = resolved.get("adopted_sft")
    hf_pin = sft_pin if sft_pin is not None and is_hf_export(sft_pin) else None
    tokenizer = resolved["tokenizer"]
    if sft_pin is not None and through == "sft":
        raise ValueError("An adopted SFT input is for downstream continuation, not an SFT stage")
    if completed_sft is not None:
        if through == "sft" or sft_pin is not None or resolved.get("adopted_handoff") is not None:
            raise ValueError("Completed SFT handles require an unambiguous downstream continuation")
        if not completed_sft.trained.adopt_source or not completed_sft.handoff.adopt_source:
            raise ValueError("Completed SFT handles must be adopted; recovery cannot replay SFT")
        base_manifest = resolved["initializer"]
        base = ArtifactStep.adopt(f"{name}/base", version, base_manifest["uri"], config=base_manifest)
        sft_evidence.append(base)
        trained = completed_sft.trained
    elif hf_pin is not None:
        if through == "archive":
            raise ValueError("An adopted HF export has no native SFT run evidence to archive; stop at --through rl")
        trained = timed(
            apply(
                f"{name}/adopt-sft",
                remote(
                    verify_adopted_hf_export,
                    resources=resources["inputs"],
                    timeout=Duration.from_seconds(timeouts["inputs"]),
                ),
                version=version,
                output=OUT,
                export_pin=hf_pin,
                model=recipe["model"],
                tokenizer_uri=tokenizer["uri"],
            )
        )
    elif sft_pin is not None:
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
            trained = timed(sft_step_builder(f"{name}/sft", version, recipe))
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
    handoff = resolved.get("adopted_handoff")
    if completed_sft is not None:
        exported = replace(completed_sft.handoff, deps=(*completed_sft.handoff.deps, trained))
    elif hf_pin is not None:
        if handoff is not None:
            raise ValueError("An adopted HF export is itself the handoff; do not also adopt a handoff")
        # The verified in-region copy is the handoff; its files sit at the root.
        exported = ArtifactStep.adopt(
            f"{name}/sft-hf",
            version,
            hf_pin["uri"],
            config={"repository": hf_pin["repository"], "revision": hf_pin["revision"]},
        )
        exported = replace(exported, deps=(*exported.deps, trained))
    elif handoff is not None:
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
        key: (
            (data_artifacts or {}).get(key)
            or ArtifactStep.adopt(f"{name}/data-{key}", version, datasets[key]["uri"], config=datasets[key])
        )
        for key in ("pools",)
    }
    pools_dataset = datasets["pools"]
    pool_manifest = json.loads((StoragePath(pools_dataset["uri"]) / "manifest.json").read_bytes())
    rl = resolved["rl"]
    rl_config = json.loads(json.dumps(rl["config"]))
    routes = pool_manifest["routes"]
    declared_sources = set(rl_config["trainer"]["rollout_buffer"].get("source_routes") or {})
    if declared_sources and declared_sources != set(routes):
        raise ValueError("The pool manifest's teacher routes differ from the recipe's declared source routes")
    rl_config["trainer"]["rollout_buffer"]["source_routes"] = rollout_source_routes(routes)
    if curriculum is not None:
        rl_config["trainer"]["callbacks"] = curriculum_validation_callbacks(
            rl_config["trainer"].get("callbacks", []), curriculum.validation
        )
    rl_execution = stage_execution(rl["execution"], coordinator)
    if openresearcher_service is not None:
        rl_execution = service_execution(rl_execution)
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
            relative_path=(
                completed_sft.model_relative_path
                if completed_sft is not None
                else "" if hf_pin is not None else HANDOFF_DIR
            ),
        ),
        train_data=tuple(
            ArtifactDataSource(adopted["pools"], relative_path=f"{label}/train.parquet")
            for label in sorted(pool_manifest["composition"])
        ),
        validation_data=(ArtifactDataSource(adopted["pools"], relative_path="validation.parquet"),),
        ray_timeouts=rl["ray_timeouts"],
        curriculum=curriculum,
        config=rl_config,
        plan=draccus.decode(SkyRLRolePlan, rl["role_plan"]),
        execution=rl_execution,
        retained=rl["retained"],
        seed=rl["seed"],
    )
    if openresearcher_service is not None:
        relative_path = "OpenResearcher:OpenResearcher-Dataset/train.parquet"
        learned = replace(
            learned,
            run=partial(
                run_with_retrieval,
                launch=retrieval_launch(openresearcher_service, rl_execution),
                parquet_relative_path=relative_path,
                parquet_sha256=pool_manifest["files"][relative_path],
            ),
        )
        learned = replace(learned, expected_fingerprint=learned.fingerprint())
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
    canonical_judge_policy = micro210_policy.read_micro210_policy(micro210_policy.POLICY_REFERENCE)
    canonical_judge = canonical_judge_policy["evaluation"]["judge"]
    if draccus.decode(JudgeConfig, evaluation["judge"]) != draccus.decode(JudgeConfig, canonical_judge):
        raise ValueError("Long-loop judge differs from the canonical Micro210 policy")
    canonical = micro210_artifact_step(
        TerminalPolicy(learned, expected_step=None), name=evaluation["name"], version=version
    )
    (eval_inputs,) = (dep for dep in canonical.deps if dep.adopt_source == PREPARED_INPUTS)
    canonical_execution = canonical.runtime_args["skyrl_execution"]
    canonical = replace(
        canonical,
        runtime_args={
            **canonical.runtime_args,
            "skyrl_execution": replace(
                canonical_execution,
                coordinator_name=coordinator.coordinator_name,
                coordinator_timeout_hours=coordinator.coordinator_timeout_hours,
            ),
        },
    )
    evaluated = after_checkpoint(timed(replace(canonical, deps=(*canonical.deps, finalized))), learned)
    reporting_inputs = after_checkpoint(
        apply(
            f"{name}/evaluation-reporting-inputs",
            prepare_reporting_inputs,
            version=version,
            output=OUT,
            prepared_inputs=eval_inputs,
            evaluated=evaluated,
            source_sha256=canonical_judge_policy["prepared_rollouts_sha256"],
            data_relative_path=canonical_judge_policy["evaluation"]["data_relative_path"],
            policy_sha256=micro210_policy.POLICY_REFERENCE.sha256,
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
                reporting_inputs,
                evaluated,
            ],
            expected_rollouts=None,
            evaluation_selection=reporting_inputs,
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
                f"Spec {resolved['resolved_sha256']}: configured maximum {recipe['stop_after_steps']} SFT updates "
                f"at {recipe['model']['max_seq_len']} positions."
            ),
            sft_selection=trained if resolved["sft_format"] == "sft_spec" and sft_pin is None else None,
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
            "eval_inputs": reporting_inputs,
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
