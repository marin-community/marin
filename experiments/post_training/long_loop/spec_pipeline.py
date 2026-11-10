# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the resolved run spec to the same graph for smoke and full runs."""

import hashlib
import json
from dataclasses import replace

import draccus
from fray.types import DeviceConfig, GpuConfig, ResourceConfig
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, IrisSkyRLExecution, SkyRLRolePlan
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.adopted_sft import is_hf_export, verify_adopted_hf_export, verify_adopted_sft
from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.endpoint_probe import probe
from experiments.post_training.long_loop.evaluation import configured_evaluation_step, prepare_micro
from experiments.post_training.long_loop.rl import configured_rl_step
from experiments.post_training.long_loop.run_spec import stage_blockers
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.sft import HANDOFF_DIR, export_sft, fresh_sft_config, sft_run


def runtime_execution(value: dict) -> IrisSkyRLExecution:
    gpu = value["coordinator_gpu"]
    return IrisSkyRLExecution(
        **{
            **value,
            "coordinator_gpu": GpuConfig(**gpu) if gpu is not None else None,
            "required_env_vars": tuple(value["required_env_vars"]),
        }
    )


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
        if key != "coordinator"
    }
    verified = timed(
        apply(
            f"{name}/inputs",
            remote(verify_inputs, resources=resources["inputs"]),
            version=version,
            output=OUT,
            manifests=(
                {"sft": datasets["sft"], "initializer": resolved["initializer"]} if through == "sft" else datasets
            ),
        )
    )
    sft_evidence = [verified]
    recipe = resolved["sft"]
    sft_pin = resolved.get("adopted_sft")
    hf_pin = sft_pin if sft_pin is not None and is_hf_export(sft_pin) else None
    tokenizer = resolved["tokenizer"]
    if sft_pin is not None and through == "sft":
        raise ValueError("An adopted SFT input is for downstream continuation, not an SFT stage")
    if hf_pin is not None:
        if through == "archive":
            raise ValueError("An adopted HF export has no native SFT run evidence to archive; stop at --through rl")
        trained = timed(
            apply(
                f"{name}/adopt-sft",
                remote(verify_adopted_hf_export, resources=resources["inputs"]),
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
                remote(verify_adopted_sft, resources=resources["inputs"]),
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
        recipe = resolved["sft"]
        run_id = f"{name.replace('/', '-')}-sft-{version}"
        fresh_sft_config(output=f"/tmp/{name}/sft", initializer=base_manifest["uri"], recipe=recipe, run_id=run_id)
        trained = timed(
            apply(
                f"{name}/sft",
                sft_run,
                version=version,
                output=OUT,
                initializer=base,
                recipe=recipe,
                run_id=run_id,
            )
        )
    trained = replace(trained, deps=(*trained.deps, verified))
    coordinator = runtime_execution(execution["coordinator"])
    coordinator = replace(coordinator, coordinator_name=f"{coordinator.coordinator_name}-{version}")
    if through == "sft":
        return replace(trained, runtime_args={**trained.runtime_args, "skyrl_execution": coordinator})
    if hf_pin is not None:
        # The verified in-region copy is the handoff; its files sit at the root.
        exported = ArtifactStep.adopt(
            f"{name}/sft-hf",
            version,
            hf_pin["uri"],
            config={"repository": hf_pin["repository"], "revision": hf_pin["revision"]},
        )
        handoff_dir = ""
    else:
        exported = timed(
            apply(
                f"{name}/sft-handoff",
                remote(export_sft, resources=resources["export"], env_vars={"JAX_PLATFORMS": "cpu"}),
                version=version,
                output=OUT,
                trained=trained,
                model=recipe["model"],
                source_revision=resolved["marin_revision"],
                tokenizer_uri=tokenizer["uri"],
                context_limit=tokenizer["context_limit"],
            )
        )
        handoff_dir = HANDOFF_DIR
    endpoint = timed(
        apply(
            f"{name}/endpoint",
            probe,
            version=version,
            output=OUT,
            model_root=exported,
            tokenizer_revision=tokenizer["revision"],
            settings=resolved["endpoint"],
            relative_path=handoff_dir,
        )
    )
    if hf_pin is not None:
        endpoint = replace(endpoint, deps=(*endpoint.deps, trained))
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
    learned = configured_rl_step(
        version=version,
        name=rl["name"],
        model=ArtifactHfModel(
            step=exported,
            tokenizer_uri=tokenizer["repo"],
            tokenizer_revision=tokenizer["revision"],
            relative_path=handoff_dir,
        ),
        train_data=tuple(
            ArtifactDataSource(adopted[k], relative_path=datasets[k]["data_file"]) for k in ("agentic", "nonagentic")
        ),
        validation_data=(ArtifactDataSource(adopted["heldout"], relative_path=datasets["heldout"]["data_file"]),),
        config=rl_config,
        plan=draccus.decode(SkyRLRolePlan, rl["role_plan"]),
        execution=runtime_execution(rl["execution"]),
        retained=rl["retained"],
        seed=rl["seed"],
    )
    learned = timed(replace(learned, deps=(*learned.deps, endpoint)))
    if through == "rl":
        return replace(learned, runtime_args={**learned.runtime_args, "skyrl_execution": coordinator})
    evaluation = resolved["evaluation"]
    data = datasets["evaluation"]
    eval_inputs = timed(
        apply(
            f"{name}/eval-inputs",
            remote(prepare_micro, resources=resources["eval_inputs"]),
            version=version,
            output=OUT,
            dataset=data["repo"],
            revision=data["revision"],
            wheels_revision=data["wheels_revision"],
            tasks_per_benchmark=evaluation["tasks_per_benchmark"],
        )
    )
    eval_inputs = replace(eval_inputs, deps=(*eval_inputs.deps, learned))
    evaluated = timed(
        configured_evaluation_step(
            version,
            learned,
            eval_inputs,
            name=evaluation["name"],
            config=evaluation["config"],
            role_plan=draccus.decode(SkyRLRolePlan, evaluation["role_plan"]),
            execution=runtime_execution(evaluation["execution"]),
            judge=draccus.decode(ModelConfig, evaluation["judge"]),
            seed=evaluation["seed"],
            expected_step=None,
            gpu_variant=evaluation["gpu_variant"],
            capability_origin=evaluation["capability_origin"],
            data_relative_path=evaluation["data_relative_path"],
            judge_gpu_count=evaluation["judge_gpu_count"],
        )
    )
    report = report_step(
        version,
        [*sft_evidence, trained, exported, endpoint, learned, eval_inputs, evaluated],
        expected_rollouts=evaluation["expected_rollouts"],
        evaluation_selection=eval_inputs,
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
    )
    archived = apply(
        f"{name}/archive",
        remote(archive_graph_run, resources=resources["archive"]),
        version=version,
        output=OUT,
        settings=resolved["archive"],
        resolved_spec=resolved,
        artifacts={
            "inputs": verified,
            "sft": sft_pin["uri"] if sft_pin else trained,
            **({"sft_adoption": trained} if sft_pin else {"base": base}),
            "export": exported,
            "endpoint": endpoint,
            "rl": learned,
            "eval_inputs": eval_inputs,
            "eval": evaluated,
        },
        trained=learned,
        evaluated=evaluated,
        completed_report=report,
    )
    return replace(
        archived,
        runtime_args={
            **archived.runtime_args,
            "skyrl_execution": coordinator,
        },
    )
