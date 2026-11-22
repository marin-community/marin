# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time best-16 postprocessing of the failed .12.5 RL job."""

import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import click
import draccus
from fray.types import DeviceConfig, ResourceConfig
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRolePlan, SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration

from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.evaluation import configured_evaluation_step, prepare_micro
from experiments.post_training.long_loop.rl_selection import validate_full_selection
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.spec_pipeline import runtime_execution, stage_execution

log = logging.getLogger(__name__)
EXPORT_JOB = "/benfeuer/long-loop-best16-export-20261112-post1"
RL_JOB = "/benfeuer/long-loop-full-2026.11.12.5/users-benfeuer-checkpoints-long-loop-full-rl-2026.11.12.5-e5a93d16551c"
VERSION = "2026.11.12.7"
SPEC = Path(__file__).with_name("round1.yaml")
SOURCE_ROOT = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/users/benfeuer/checkpoints/long-loop-full-rl/2026.11.12.5"
)
EVIDENCE = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-rl5-terminal"


def selected_export(*, output: str, source_root: str, checkpoint_root: str, tokenizer: dict) -> SkyRLRun:
    configure_coreweave_s3()
    with IrisClient.in_cluster(os.environ["IRIS_CONTROLLER_URL"], workspace=None) as client:
        while True:
            state = client.job_state(JobName.from_wire(EXPORT_JOB))
            if is_job_finished(state):
                break
            log.info("Waiting for existing best-16 conversion %s: %s", EXPORT_JOB, state)
            time.sleep(15)
        if state != JobState.SUCCEEDED:
            raise RuntimeError(f"Existing export {EXPORT_JOB} ended {state}; diagnose before recovery")
    fs, key = filesystem_for(prefix_join(source_root, "resolved-launch.yaml"))
    launch = fs.cat_file(key)
    outfs, outkey = filesystem_for(output)
    outfs.pipe_file(outkey + "/resolved-launch.yaml", launch)
    terminal = {
        "result": {
            "state": "failed",
            "iris_job_id": RL_JOB,
            "failure": "GroupAdmissionStalledError: step18 admitted63/64 for1676s",
            "completed_optimizer_steps": 17,
            "last_committed_step": 16,
            "selected_best_step": 16,
        },
        "finalization": {
            "export_job": EXPORT_JOB,
            "source_root": source_root,
            "terminal_evidence": EVIDENCE + "/manifest.json",
            "source_status_preserved": True,
        },
    }
    outfs.pipe_file(outkey + "/terminal.json", json.dumps(terminal, indent=2).encode())
    model = prefix_join(source_root, "exports/global_step_16/policy")
    modelfs, modelkey = filesystem_for(model)
    for name in ("config.json", "model.safetensors.index.json"):
        if not modelfs.isfile(modelkey + "/" + name):
            raise FileNotFoundError(prefix_join(model, name))
    run = SkyRLRun(
        path=output,
        hf_model_uri=model,
        global_step=16,
        tokenizer_uri=tokenizer["repo"],
        tokenizer_revision=tokenizer["revision"],
        checkpoint_root=checkpoint_root,
        draft_checkpoint_root=None,
        terminal_manifest_uri=prefix_join(output, "terminal.json"),
        iris_job_id=RL_JOB,
    )
    validate_full_selection(run)
    return run


@click.command()
@rl_build_options
def main():
    resolved = resolve_spec(SPEC, smoke=False)
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    configure_coreweave_s3()
    coordinator = replace(
        runtime_execution(resolved["execution"]["coordinator"]),
        coordinator_name="long-loop-best16-finalize-20261112-post2",
    )
    checkpoint_root = "s3://marin-us-east-08a/tmp/ttl=14d/skyrl/marin/users/benfeuer/long-loop/users/benfeuer/checkpoints/long-loop-full-rl/2026.11.12.5/checkpoints"
    learned = ArtifactStep(
        name="long-loop/round1/selected-rl5",
        version=VERSION,
        artifact_type=SkyRLRun,
        run=lambda c: selected_export(**c),
        build_config=lambda ctx: {
            "output": ctx.output_path,
            "source_root": SOURCE_ROOT,
            "checkpoint_root": checkpoint_root,
            "tokenizer": resolved["tokenizer"],
        },
        runtime_args={"skyrl_execution": coordinator},
    )
    learned = timed(learned)
    resource = lambda key: ResourceConfig(
        **{**resolved["execution"][key], "device": draccus.decode(DeviceConfig, resolved["execution"][key]["device"])}
    )
    evaluation = resolved["evaluation"]
    data = resolved["datasets"]["evaluation"]
    inputs = timed(
        apply(
            "long-loop/round1/eval-inputs",
            remote(prepare_micro, resources=resource("eval_inputs"), timeout=Duration.from_seconds(259200)),
            version=VERSION,
            output=OUT,
            dataset=data["repo"],
            revision=data["revision"],
            wheels_revision=data["wheels_revision"],
            tasks_per_benchmark=evaluation["tasks_per_benchmark"],
        )
    )
    inputs = replace(inputs, deps=(*inputs.deps, learned))
    evaluated = timed(
        configured_evaluation_step(
            VERSION,
            learned,
            inputs,
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
    )
    adopted = lambda key, uri: ArtifactStep.adopt("long-loop/round1/reused-" + key, VERSION, uri)
    sft = adopted("sft", resolved["adopted_sft"]["uri"])
    handoff = adopted("handoff", resolved["adopted_handoff"]["uri"])
    endpoint = adopted("endpoint", resolved["adopted_endpoint"]["uri"])
    settings = resolved["archive"]
    evidencefs, evidencekey = filesystem_for(EVIDENCE + "/manifest.json")
    manifest = json.loads(evidencefs.cat_file(evidencekey))
    settings["additional_evidence"] += [
        {
            "source": row["uri"],
            "name": "rl-terminal/" + row["name"],
            "category": "config" if row["name"].endswith(".json") else "timing",
            "sha256": row["sha256"],
        }
        for row in manifest["objects"]
    ]
    journals = [
        row
        for row in settings["additional_evidence"] + resolved["adopted_sft"].get("additional_evidence", [])
        if row["category"] == "timing" and row["source"].endswith(".jsonl")
    ]
    report = report_step(
        VERSION,
        [sft, handoff, endpoint, learned, inputs, evaluated],
        expected_rollouts=evaluation["expected_rollouts"],
        evaluation_selection=inputs,
        additional_timing=journals,
        sft_geometry="Reused SFT746/50,063,212,544 scheduled positions. RL .12.5 FAILED after17updates; selected best16. No earlier-stage replay. Component report preserves actual failure and overrun.",
    )
    terminal = apply(
        "long-loop/round1/archive",
        remote(archive_graph_run, resources=resource("archive"), timeout=Duration.from_seconds(259200)),
        version=VERSION,
        output=OUT,
        settings=settings,
        resolved_spec=resolved,
        artifacts={
            "sft": sft,
            "export": handoff,
            "endpoint": endpoint,
            "rl": learned,
            "eval_inputs": inputs,
            "eval": evaluated,
        },
        trained=learned,
        evaluated=evaluated,
        completed_report=report,
    )
    return replace(terminal, runtime_args={**terminal.runtime_args, "skyrl_execution": coordinator})


if __name__ == "__main__":
    main()
