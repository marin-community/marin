# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-time best-16 postprocessing of the failed .12.5 RL job."""

import json
import os
from dataclasses import replace
from pathlib import Path

import click
import draccus
from fray.types import DeviceConfig, ResourceConfig
from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRolePlan, SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.timing import Duration

from experiments.post_training.long_loop.archive_stage import archive_graph_run
from experiments.post_training.long_loop.evaluation import configured_evaluation_step
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.spec_pipeline import runtime_execution, stage_execution

EXPORT_JOB = "/benfeuer/long-loop-best16-export-20261112-post1"
RL_JOB = "/benfeuer/long-loop-full-2026.11.12.5/users-benfeuer-checkpoints-long-loop-full-rl-2026.11.12.5-e5a93d16551c"
VERSION = "2026.11.12.9"
SPEC = Path(__file__).with_name("round1.yaml")
SOURCE_ROOT = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/users/benfeuer/checkpoints/long-loop-full-rl/2026.11.12.5"
)
EVIDENCE = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261112-rl5-terminal"


@click.command()
@rl_build_options
def main():
    resolved = resolve_spec(SPEC, smoke=False)
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    configure_coreweave_s3()
    coordinator = replace(
        runtime_execution(resolved["execution"]["coordinator"]),
        coordinator_name="long-loop-best16-finalize-20261112-post4",
    )
    checkpoint_root = "s3://marin-us-east-08a/tmp/ttl=14d/skyrl/marin/users/benfeuer/long-loop/users/benfeuer/checkpoints/long-loop-full-rl/2026.11.12.5/checkpoints"
    learned = ArtifactStep.adopt(
        "long-loop/round1/selected-rl5",
        VERSION,
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/selected-rl5/2026.11.12.7",
        kind=SkyRLRun,
        config={"source_job": RL_JOB, "selected_step": 16, "export_job": EXPORT_JOB},
    )
    resource = lambda key: ResourceConfig(
        **{**resolved["execution"][key], "device": draccus.decode(DeviceConfig, resolved["execution"][key]["device"])}
    )
    evaluation = resolved["evaluation"]
    data = resolved["datasets"]["evaluation"]
    inputs = ArtifactStep.adopt(
        "long-loop/round1/eval-inputs-ready",
        VERSION,
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/long-loop/round1/eval-inputs/2026.11.12.8",
        config={
            "prepared_on": "local CPU",
            "dataset": data["repo"],
            "revision": data["revision"],
            "wheels_revision": data["wheels_revision"],
            "attempts": 440,
        },
    )
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
