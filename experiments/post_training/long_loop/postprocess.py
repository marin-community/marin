# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue final-evaluation reporting and publication without replaying producers."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path, PurePosixPath

import click
import draccus
from fray.types import CpuConfig, DeviceConfig, ResourceConfig
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.micro210 import PREPARED_INPUTS
from experiments.evaluation.micro210_policy import POLICY_REFERENCE, read_micro210_policy
from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.bixbench import merge_verifier_recovery
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.runtime_timing import prepare_reporting_inputs, write_report
from experiments.post_training.long_loop.sft_accounting import write_sft_accounting
from experiments.post_training.long_loop.spec_pipeline import runtime_execution


def archive_recovered_run(
    *,
    output: str,
    settings: dict,
    resolved_spec: dict,
    artifacts: dict[str, str],
    trained: str,
    evaluated: str,
    completed_report: str,
) -> None:
    """Archive both the scored view and every verbatim original native result."""
    provenance = json.loads((StoragePath(evaluated) / "merge-provenance.json").read_bytes())
    native_entries = []
    for item in provenance["source_files"]:
        name = item["name"]
        if PurePosixPath(name).name != name or name in {".", ".."}:
            raise ValueError("Original native filenames must be direct children")
        native_entries.append({"path": f"original-native/{name}", "category": "eval_trace"})
    settings = {
        **settings,
        "stage_files": {
            **settings["stage_files"],
            "eval": [
                *settings["stage_files"]["eval"],
                {"path": "merge-provenance.json", "category": "eval_trace"},
                *native_entries,
            ],
        },
    }
    archive_graph_run(
        output=output,
        settings=settings,
        resolved_spec=resolved_spec,
        artifacts=artifacts,
        trained=trained,
        evaluated=evaluated,
        completed_report=completed_report,
    )


def build_postprocessing_graph(plan: dict, resolved_spec: dict) -> ArtifactStep:
    """Bind only a recovered view, report, accounting and archive to adopted inputs."""
    version, name = plan["version"], resolved_spec["name"]
    handles = {
        key: ArtifactStep.adopt(f"{name}/postprocess-source-{key}", version, uri)
        for key, uri in plan["artifacts"].items()
    }
    if "recovery" not in plan and "recovered_eval" not in handles and "report" not in handles:
        policy = read_micro210_policy(POLICY_REFERENCE)
        handles["eval_inputs"] = apply(
            f"{name}/evaluation-reporting-inputs",
            prepare_reporting_inputs,
            version=version,
            output=OUT,
            prepared_inputs=ArtifactStep.adopt("evaluation/micro210/report-source", version, PREPARED_INPUTS),
            evaluated=handles["eval"],
            source_sha256=policy["prepared_rollouts_sha256"],
            data_relative_path=policy["evaluation"]["data_relative_path"],
            policy_sha256=POLICY_REFERENCE.sha256,
        )
    settings = {
        **resolved_spec["archive"],
        "additional_evidence": [*resolved_spec["archive"]["additional_evidence"], *plan["additional_evidence"]],
    }
    accepted = plan["accepted_unscored_attempts"]
    if "report" in handles:
        merged, report = handles.get("recovered_eval", handles["eval"]), handles["report"]
    else:
        merged = (
            apply(
                f"{name}/recovered-evaluation",
                merge_verifier_recovery,
                version=version,
                output=OUT,
                original=handles["eval"],
                selection=handles["eval_inputs"],
                recovery=plan["recovery"]["uri"],
                recovery_manifest_sha256=plan["recovery"]["manifest_sha256"],
                accepted_unscored_attempts=accepted,
            )
            if "recovery" in plan
            else handles["eval"]
        )
        stages = [handles[key] for key in ("inputs", "sft", "export")]
        if "sft_adoption" in handles:
            stages[1] = handles["sft_adoption"]
        if "endpoint" in handles:
            stages.append(handles["endpoint"])
        stages.extend([handles["rl"], handles["eval_inputs"], handles["eval"]])
        if merged is not handles["eval"]:
            stages.append(merged)
        additional = settings["additional_evidence"] + (resolved_spec.get("adopted_sft") or {}).get(
            "additional_evidence", []
        )
        report = apply(
            "long-loop/report",
            write_report,
            version=version,
            output=OUT,
            stages=stages,
            expected_rollouts=resolved_spec["evaluation"]["expected_rollouts"] if "recovery" in plan else None,
            evaluation_selection=handles["eval_inputs"],
            additional_timing=[
                item for item in additional if item["category"] == "timing" and item["source"].endswith(".jsonl")
            ],
            accepted_unscored_attempts=accepted,
            sft_geometry=plan["sft_geometry"],
            rl_terminal=handles["rl"],
        )
    archive_resources = resolved_spec["execution"]["archive"]
    resources = ResourceConfig(
        **{**archive_resources, "device": draccus.decode(DeviceConfig, archive_resources["device"])}
    )
    if not isinstance(resources.device, CpuConfig):
        raise ValueError("Postprocessing requires the resolved CPU archive resources")
    timeouts = {**resolved_spec["execution"]["task_timeouts"], **plan.get("task_timeouts", {})}
    accounting = apply(
        f"{name}/sft-data-accounting",
        remote(
            write_sft_accounting,
            resources=resources,
            env_vars={"JAX_PLATFORMS": "cpu"},
            timeout=Duration.from_seconds(timeouts["accounting"]),
        ),
        version=version,
        output=OUT,
        trained=handles["sft"],
        settings={
            **settings["sft_data_accounting"],
            **plan.get("accounting_settings", {}),
            "resume_samples": plan.get("accounting_resume"),
        },
    )
    accounting = replace(accounting, deps=(*accounting.deps, report))
    terminal = apply(
        f"{name}/archive",
        remote(
            archive_recovered_run if "recovery" in plan or "recovered_eval" in handles else archive_graph_run,
            resources=resources,
            timeout=Duration.from_seconds(timeouts["archive"]),
        ),
        version=version,
        output=OUT,
        settings=settings,
        resolved_spec=resolved_spec,
        artifacts={**handles, "native_eval": handles["eval"], "eval": merged, "sft_data_accounting": accounting},
        trained=handles["rl"],
        evaluated=merged,
        completed_report=report,
    )
    execution = runtime_execution(resolved_spec["execution"]["coordinator"])
    if execution.coordinator_gpu is not None:
        raise ValueError("Postprocessing coordinator must not request GPUs")
    execution = replace(execution, coordinator_name=f"long-loop-postprocess-{version}")
    return replace(terminal, runtime_args={**terminal.runtime_args, "skyrl_execution": execution})


@click.command(help=__doc__)
@click.option("--plan-uri", required=True)
@click.option("--plan-sha256", required=True)
@rl_build_options
def main(plan_uri: str, plan_sha256: str) -> ArtifactStep:
    configure_coreweave_s3()
    raw = StoragePath(plan_uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha256:
        raise ValueError("Postprocessing plan changed")
    plan = json.loads(raw)
    for key, expected in plan.get("artifact_record_sha256", {}).items():
        record = (StoragePath(plan["artifacts"][key]) / ".artifact.json").read_bytes()
        if hashlib.sha256(record).hexdigest() != expected:
            raise ValueError(f"Completed producer artifact changed: {key}")
    raw = StoragePath(plan["resolved_spec"]["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["resolved_spec"]["sha256"]:
        raise ValueError("Original resolved run specification changed")
    resolved = json.loads(raw)
    if get_job_info() is None:
        require_clean_revision(Path(__file__).resolve().parents[3], plan["postprocessing_revision"])
    resolved["runtime_source"] = runtime_source_manifest(
        Path(__file__).resolve().parents[3], plan["postprocessing_revision"]
    )
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    version = resolve_version(resolved["name"], None)
    if version != plan["version"]:
        raise ValueError("CLI artifact version differs from the postprocessing plan")
    return build_postprocessing_graph(plan, resolved)


if __name__ == "__main__":
    main()
