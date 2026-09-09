# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare matched Snowball timing arms from an audited training request."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

EAST = "s3://marin-us-east-02a/"
MODES = ("reference", "bucket")
HISTORICAL_BROADCAST_MEDIAN = 14.955344404093921


def prepare_timing_request(
    source_request: dict, *, runtime_commit: str, mode: str, run_id: str, output_prefix: str
) -> dict:
    """Preserve model/data selection and give each transfer arm fresh outputs."""
    if mode not in MODES:
        raise ValueError("Timing mode must be reference or bucket")
    if len(runtime_commit) != 40 or any(char not in "0123456789abcdef" for char in runtime_commit):
        raise ValueError("Use a full immutable runtime commit")
    inputs = [source_request["model"], *source_request["train_data"], *source_request["validation_data"]]
    if any(not item["uri"].startswith(EAST) for item in inputs) or not output_prefix.startswith(EAST):
        raise ValueError("Snowball timing inputs and outputs remain in east")
    previous = [item["uri"] for item in inputs] + list(source_request["output"].values())
    if any(output_prefix.startswith(uri) or uri.startswith(output_prefix) for uri in previous):
        raise ValueError("Timing output must not overlap existing input or output namespaces")
    config = yaml.safe_load(source_request["config_yaml"])
    trainer = config["trainer"]
    generator = config["generator"]
    data = config["data"]
    if (
        config["entrypoint"] != "fully_async"
        or trainer["strategy"] != "megatron"
        or trainer["train_batch_size"] != 32
        or trainer["policy_mini_batch_size"] != 32
        or trainer["update_epochs_per_batch"] != 1
        or trainer["fully_async"]["max_staleness_steps"] != 1
        or trainer["fully_async"]["weight_sync_interval"] != 1
        or generator["inference_engine_tensor_parallel_size"] != 1
        or generator["inference_engine_data_parallel_size"] != 8
        or generator["inference_engine_expert_parallel_size"] != 8
        or generator["num_inference_engines"] != 1
        or generator["n_samples_per_prompt"] != 4
        or config["context_budget"]["request_window_tokens"] != 8192
        or config["context_budget"]["max_new_tokens_per_turn"] != 4096
        or data["num_workers"] != 0
        or not data["epoch_seeded_shuffle"]
    ):
        raise ValueError("Source is not the audited P32/I8 C1/A1 timing recipe")
    trainer.update(
        max_steps=20,
        eval_before_train=True,
        eval_interval=20,
        eval_batch_size=128,
        initial_eval_repeat_count=1,
        ckpt_interval=-1,
        hf_save_interval=-1,
        seed=17,
        debug_mode="off",
        weight_sync_nccl_diagnostics=True,
        weight_sync_readback_output=f"{output_prefix}/native-readback",
    )
    trainer["algorithm"].update(batch_invariant=False, grad_cosine={"enabled": False, "store": "cpu_bf16"})
    trainer["fully_async"].update(first_token_admission=True, eval_on_installed_weights=False, eval_mode="blocking")
    generator.update(
        weight_sync_timing_mode=mode,
        weight_sync_wire_inventory=True,
        publication_stage_timing=True,
        inference_engine_serial_startup=False,
    )
    config.setdefault("extra_env", {}).update(VLLM_BATCH_INVARIANT="0", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET")
    temporary = EAST + "tmp/ttl=14d/skyrl/marin-us-east-02a/" + output_prefix.removeprefix(EAST)
    request = copy.deepcopy(source_request)
    request.update(
        run_id=run_id,
        attempt_id="pending",
        runtime={"profile": "megatron", "commit": runtime_commit},
        config_yaml=yaml.safe_dump(config, sort_keys=False),
        overrides=[
            "++trainer.hf_hub_repo_id=null",
            f"++terminal_bench_config.trials_dir='{temporary}/attempts/trace_jobs'",
            f"++generator.trajectory_retention.output_path='{temporary}/attempts/trajectories'",
        ],
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
            "timeout_seconds": 2400,
        },
        "expected_updates": 20,
        "expected_initial_syncs": 1,
        "expected_measured_syncs": 20,
        "expected_full_replays": 20,
        "mode": "detach",
        "transfer_mode": mode,
        "historical_broadcast_median_seconds": HISTORICAL_BROADCAST_MEDIAN,
        "primary_candidate_threshold_seconds": HISTORICAL_BROADCAST_MEDIAN / 2,
        "measurement_policy": "First post-update begin writes an immutable marker; startup retries only",
        "timing_scope": "Original weight_broadcast Timer; preparation and complete replay excluded",
        "overall_scope": "Pause/core include full replay after every sync; no production throughput or quality ranking",
        "reservation_estimate_gpu_hours": 40 * 2400 / 3600,
        "coordinator_wait_seconds": 7 * 2400 + 900,
        "coordinator_timeout_seconds": 7 * 2400 + 1800,
        "retry_horizon_scope": "Operational horizon; uncharged assigned losses or indefinite pending are not bounded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    report = prepare_timing_request(
        source["request"],
        runtime_commit=args.runtime_commit,
        mode=args.mode,
        run_id=args.run_id,
        output_prefix=args.output_prefix,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SNOWBALL_TIMING_REQUEST_PREPARED", report["request_hash"])


if __name__ == "__main__":
    main()
