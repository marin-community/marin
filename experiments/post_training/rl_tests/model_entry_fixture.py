# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model entry used by the detached launch and coordinator reentry test."""

from dataclasses import replace
from pathlib import Path

from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, apply
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun

from experiments.post_training import rl, snowball_rl


def fail_following():
    raise RuntimeError("following stage failed")


def following_step(_):
    return apply("tests/following", fail_following, version="2026.11.20.1")


def main(config: dict, argv: list[str]) -> int:
    table = snowball_rl.SNOWBALL
    step = ArtifactStep.adopt("tests/python-init", "2026.11.17", "s3://marin-test/models/python-init", kind=Artifact)
    init: ArtifactHfModel | SkyRLPolicyModel = ArtifactHfModel(
        step, "tests/python-tokenizer", "python-tokenizer-revision", relative_path=""
    )
    if config["policy_forward"] is not None:
        producer = rl.build_experiment(
            table.by_name["train-long"],
            table=table,
            settings=("trainer.max_steps=2",),
            forward_env=tuple(config["policy_forward"]),
        ).runs[0]
        result = SkyRLRun.raw_load(config["source"]).model_copy(
            update={"hf_model_uri": "s3://marin-test/models/python-init"}
        )
        directory = Path(config["policy_path"])
        directory.mkdir(parents=True, exist_ok=True)
        record = ArtifactRecord(
            name=producer.name,
            version=producer.version,
            fingerprint=producer.fingerprint(),
            result_type=result_type_name(SkyRLRun),
            result=result.result_payload(),
        )
        (directory / RECORD_FILENAME).write_text(record.model_dump_json())
        init = SkyRLPolicyModel(replace(producer, override_path=config["policy_uri"]))
    if config["following"]:
        table = replace(table, presets=tuple(replace(preset, then=following_step) for preset in table.presets))
    return rl.main(
        table,
        argv=argv,
        init=init if config["use_init"] else None,
        forward_env=tuple(config["forward_env"]),
        entry_module=config["entry_module"],
    )
