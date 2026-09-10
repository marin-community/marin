# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the five prospective Snowball termination treatments on fixed inputs."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from experiments.post_training.async_snowball import Scale, training_config

SKELETON = Path(__file__).with_name("e61_stage2_packet_skeleton.json")
RUNTIME = "5ceb506edcab1c7f2f8da4a56e76c60dd030fd25"
FAMILIES = dict(
    zip(
        ("parser_only", "force_close", "soft_overlong", "negative_truncation_advantage", "repetition_stop"),
        (f"2026.09.09.{n}" for n in range(261, 266)),
        strict=True,
    )
)
EAST = "s3://marin-us-east-02a/"
TRAIN = EAST + "marin/users/ahmad/documents/async-rl-snowball-gsm8k/2026.09.06.13"
DEV = EAST + "marin/users/ahmad/documents/math-eval-pool/1.0.0-candidate1/batteries/mechanical-v1/snowball"


def _hydra_value(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(key + ":" + _hydra_value(item) for key, item in value.items()) + "}"
    return json.dumps(value, separators=(",", ":"))


def stage2_packet(source_request: dict, input_receipt: bytes, arm_name: str) -> dict:
    """Bind existing locators and a fresh treatment identity; never submit training."""
    skeleton = json.loads(SKELETON.read_bytes())
    if hashlib.sha256(input_receipt).hexdigest() != skeleton["input_receipt_sha256"]:
        raise ValueError("Original stage-2 input receipt differs")
    receipt = json.loads(input_receipt)
    for uri, expected in skeleton["input_objects"].items():
        if receipt["reads"][uri] != expected:
            raise ValueError("Stage-2 input object differs from its receipt")
    if receipt["original_train"]["rows"] != 1024 or receipt["common_dev"]["rows"] != 256:
        raise ValueError("Original stage-2 row counts differ")
    model = source_request["model"]
    if model["uri"].rstrip("/") != EAST + "marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm":
        raise ValueError("Stage-2 requires the frozen Snowball checkpoint")
    if model["identity"] != "models/snowball-67b-a2b-sft-s2-thinking@2026.08.30:c6168770":
        raise ValueError("Stage-2 model identity differs")
    if model["tokenizer_revision"] != "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2":
        raise ValueError("Stage-2 tokenizer revision differs")
    arm = next(a for a in skeleton["arms"] if a["name"] == arm_name)
    version = FAMILIES[arm_name]
    config = yaml.safe_load(
        training_config(
            Scale.QUALIFICATION,
            response_tokens=4096,
            eval_response_tokens=4096,
            context_tokens=8192,
            weight_sync_interval=2,
            max_staleness_steps=1,
            epoch_seeded_shuffle=True,
            dataloader_workers=0,
            study_steps=25,
            eval_interval=25,
        )
    )
    # These additions use native Hydra's add-or-override path. The production
    # translator must compose them against the exact runtime before GPU release.
    overrides = ["++trainer.hf_hub_repo_id=null", "++trainer.ckpt_interval=-1", "++trainer.hf_save_interval=-1"]
    settings = {**skeleton["common_overrides"], **arm["additional_overrides"]}
    derived = {
        "generator.max_turns": 1,
        "generator.max_input_length": 1024,
        "generator.sampling_params.max_generate_length": 4096,
        "generator.trajectory_reward_shaping.overlong.l_max": 4096,
        "generator.trajectory_reward_shaping.overlong.l_cache": 512,
    }
    for key, expected in derived.items():
        if key in settings and settings.pop(key) != expected:
            raise ValueError("Unexpected prospective derived context setting: " + key)
    config["context_budget"]["overlong_cache_fraction"] = 0.125
    settings["trainer.dump_data_batch"] = True
    overrides.extend("++" + key + "=" + _hydra_value(value) for key, value in settings.items())
    name = "users/ahmad/checkpoints/async-rl/e61-stage2-" + arm_name
    output = EAST + "marin/" + name + "/" + version
    temporary = EAST + "tmp/ttl=14d/skyrl/marin-us-east-02a/marin/" + name + "/" + version
    request = copy.deepcopy(source_request)
    request.update(
        run_id=name.replace("_", "-") + "-" + version,
        attempt_id="pending",
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        runtime={"profile": "megatron", "commit": RUNTIME},
        seed=17,
        train_data=[
            dict(
                uri=TRAIN,
                identity="users/ahmad/documents/async-rl-snowball-gsm8k@2026.09.06.13:4c7ebfc9",
                local_path="/tmp/marinskyrl/data/e61-original-gsm",
                relative_path="train.parquet",
            )
        ],
        validation_data=[
            dict(
                uri=DEV,
                identity="snowball-provisional-dev-content:"
                + skeleton["input_objects"][DEV + "/provisional-dev.parquet"]["sha256"],
                local_path="/tmp/marinskyrl/data/e61-common-dev",
                relative_path="provisional-dev.parquet",
            )
        ],
        output=dict(
            checkpoint_root=temporary + "/checkpoints",
            export_root=output + "/exports",
            attempts_root=temporary + "/attempts",
            resolved_config_uri=output + "/resolved-skyrl.json",
            terminal_manifest_uri=output + "/terminal.json",
        ),
        overrides=overrides,
        completion_mode="metrics",
        checkpoint_retention_days=None,
    )
    expected_role = dict(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=8,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=32,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
    )
    if request["topology"] != dict(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=expected_role):
        raise ValueError("Stage-2 requires the qualified P32/I8 topology")
    canonical = {k: v for k, v in request.items() if k != "attempt_id"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request["attempt_id"] = digest[:12]
    return dict(
        schema="e61_stage2_packet_v1",
        status="prospective_CPU_preparation",
        gpu_released=False,
        arm=arm_name,
        version=version,
        msr=RUNTIME,
        request=request,
        request_hash=digest,
        source_request_sha256=hashlib.sha256(
            json.dumps(source_request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        input_receipt_sha256=skeleton["input_receipt_sha256"],
        input_objects=skeleton["input_objects"],
        endpoint_evidence=arm["endpoint_evidence"],
        metric_protocol=skeleton["metric_protocol"],
        selector=skeleton["selector"],
        native_K16_gate=None,
        execution=dict(
            cluster="cw-us-east-02a",
            cluster_config="lib/iris/config/cw-us-east-02a.yaml",
            cpu=16,
            memory="1800GB",
            disk="2TB",
            priority="batch",
            max_retries=1,
            target_cluster=None,
            parent_cluster_config=None,
            wandb_entity="dogml",
            timeout_seconds=5400,
        ),
    )
