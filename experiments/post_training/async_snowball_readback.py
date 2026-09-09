# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a zero-update native Snowball readback using existing east inputs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from experiments.post_training.async_snowball import Scale, training_config


def prepare_readback(source_request: dict, *, runtime_commit: str, run_id: str, output_prefix: str) -> dict:
    """Reuse frozen model/data locators while giving diagnostics independent outputs."""
    if len(runtime_commit) != 40 or any(char not in "0123456789abcdef" for char in runtime_commit):
        raise ValueError("Use a full immutable runtime commit")
    east = "s3://marin-us-east-02a/"
    inputs = [source_request["model"], *source_request["train_data"], *source_request["validation_data"]]
    if any(not item["uri"].startswith(east) for item in inputs) or not output_prefix.startswith(east):
        raise ValueError("Snowball readback inputs and outputs must remain in east")
    if any(output_prefix.startswith(item["uri"]) for item in inputs):
        raise ValueError("Readback output overlaps an immutable input")
    if any(value.startswith(output_prefix) for value in source_request["output"].values()):
        raise ValueError("Readback needs fresh output paths")
    temporary_prefix = east + "tmp/ttl=14d/skyrl/marin-us-east-02a/" + output_prefix.removeprefix(east)
    config = yaml.safe_load(
        training_config(
            Scale.CADENCE_GATE,
            response_tokens=4096,
            eval_response_tokens=4096,
            context_tokens=8192,
            publication_stage_timing=True,
            dataloader_workers=0,
            epoch_seeded_shuffle=True,
        )
    )
    config["entrypoint"] = "weight_sync_readback"
    config["trainer"]["debug_mode"] = "off"
    config["trainer"]["weight_sync_nccl_diagnostics"] = True
    config["trainer"]["weight_sync_readback_output"] = f"{output_prefix}/native-readback"
    config["trainer"]["logger"] = "console"
    config["trainer"]["algorithm"]["batch_invariant"] = False
    config["trainer"]["eval_before_train"] = False
    config["trainer"]["eval_interval"] = -1
    request = copy.deepcopy(source_request)
    request.update(
        run_id=run_id,
        attempt_id="pending",
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        runtime={"profile": "megatron", "commit": runtime_commit},
        output={
            "checkpoint_root": f"{temporary_prefix}/checkpoints",
            "export_root": f"{output_prefix}/exports",
            "attempts_root": f"{temporary_prefix}/attempts",
            "resolved_config_uri": f"{output_prefix}/resolved-skyrl.json",
            "terminal_manifest_uri": f"{output_prefix}/terminal.json",
        },
    )
    canonical = {key: value for key, value in request.items() if key != "attempt_id"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request["attempt_id"] = digest[:12]
    return {
        "request": request,
        "request_hash": digest,
        "execution": {
            "cluster": "cw-us-east-02a",
            "cluster_config": "lib/iris/config/cw-us-east-02a.yaml",
            "cpu": 16,
            "memory": "1800GB",
            "disk": "2TB",
            "priority": "batch",
            "max_retries": 1,
            "target_cluster": None,
            "parent_cluster_config": None,
            "wandb_entity": None,
            "timeout_seconds": 900,
        },
        "mode": "detach",
        "expected_updates": 0,
        "expected_initial_syncs": 1,
        "completion_contract": "Native diagnostic chunks and explicit pass line; no training completion receipt",
        "reservation_estimate_gpu_hours": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    report = prepare_readback(
        source["request"],
        runtime_commit=args.runtime_commit,
        run_id=args.run_id,
        output_prefix=args.output_prefix,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SNOWBALL_READBACK_REQUEST_PREPARED", report["request_hash"])


if __name__ == "__main__":
    main()
