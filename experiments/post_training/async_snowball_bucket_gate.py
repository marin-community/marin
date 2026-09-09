# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare one zero-training Snowball bucket byte/memory qualification."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def prepare_bucket_gate(source_request: dict, *, runtime_commit: str, run_id: str, output_prefix: str) -> dict:
    """Retain the reviewed reference inputs and geometry; replace diagnostic outputs."""
    if len(runtime_commit) != 40 or any(character not in "0123456789abcdef" for character in runtime_commit):
        raise ValueError("Use a full immutable runtime commit")
    east = "s3://marin-us-east-02a/"
    inputs = [source_request["model"], *source_request["train_data"], *source_request["validation_data"]]
    if any(not item["uri"].startswith(east) for item in inputs) or not output_prefix.startswith(east):
        raise ValueError("Snowball bucket diagnostics remain in east")
    if any(output_prefix.startswith(item["uri"]) or item["uri"].startswith(output_prefix) for item in inputs):
        raise ValueError("Diagnostic output overlaps a source input")
    if any(
        value.startswith(output_prefix) or output_prefix.startswith(value) for value in source_request["output"].values()
    ):
        raise ValueError("Use a fresh diagnostic output namespace")
    config = yaml.safe_load(source_request["config_yaml"])
    if config["entrypoint"] != "weight_sync_readback":
        raise ValueError("Prepare from the reviewed zero-update readback request")
    trainer = config["trainer"]
    if (
        trainer["strategy"] != "megatron"
        or trainer["logger"] != "console"
        or trainer["debug_mode"] != "off"
        or trainer["algorithm"]["batch_invariant"]
        or trainer["eval_before_train"]
        or trainer["eval_interval"] != -1
        or not trainer["weight_sync_nccl_diagnostics"]
    ):
        raise ValueError("Reference is not the qualified zero-update diagnostic configuration")
    config["entrypoint"] = "weight_sync_bucket_gate"
    trainer["fully_async"]["first_token_admission"] = True
    trainer["weight_sync_readback_output"] = f"{output_prefix}/native-readback"
    temporary = east + "tmp/ttl=14d/skyrl/marin-us-east-02a/" + output_prefix.removeprefix(east)
    request = copy.deepcopy(source_request)
    request.update(
        run_id=run_id,
        attempt_id="pending",
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        runtime={"profile": "megatron", "commit": runtime_commit},
        output={
            "checkpoint_root": f"{temporary}/checkpoints",
            "export_root": f"{output_prefix}/exports",
            "attempts_root": f"{temporary}/attempts",
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
            "timeout_seconds": 1200,
        },
        "mode": "detach",
        "expected_updates": 0,
        "expected_initial_syncs": 1,
        "expected_packed_installs": 1,
        "expected_full_replays": 1,
        "completion_contract": "Durable byte/memory receipt and native pass; diagnostic timing only",
        "reservation_estimate_gpu_hours": 40 * 1200 / 3600,
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
    report = prepare_bucket_gate(
        source["request"], runtime_commit=args.runtime_commit, run_id=args.run_id, output_prefix=args.output_prefix
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SNOWBALL_BUCKET_REQUEST_PREPARED", report["request_hash"])


if __name__ == "__main__":
    main()
