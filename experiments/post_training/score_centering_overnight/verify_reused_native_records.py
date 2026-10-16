# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove older content-addressed records match the selected attempt's actual rollout groups."""

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import pickle
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.config import Config
from skyrl_train.trajectory_runners.trajectory_retention import build_trajectory_records
from skyrl_train.trajectory_runners.trajectory_retention_config import TrajectoryRetentionConfig
from skyrl_train.trajectory_runners.types import BatchMetadata
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--after-ms", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cfg = yaml.safe_load(args.config.read_text())
assert (
    cfg["skyrl"]["generator"].get("sampler_refresh_period", 1) == 1
), "This verifier covers fresh-sampler training records."
run = cfg["run"]["id"]
size = cfg["skyrl"]["trainer"]["train_batch_size"] * cfg["skyrl"]["generator"]["n_samples_per_prompt"]
group_size = cfg["skyrl"]["generator"]["n_samples_per_prompt"]
base = urlparse(cfg["artifacts"]["attempts_root"].removesuffix("/attempts")).path.lstrip("/").rstrip("/") + "/"
inventory = {
    row["Key"].removeprefix(base): row for row in json.loads((args.cache / "object_inventory.json").read_text())
}
current_counts = collections.Counter()
current_keys = set()
current_groups = collections.defaultdict(set)
older = collections.defaultdict(list)
for path in (args.cache / "attempts/trajectories").glob("**/*.zip"):
    item = inventory[str(path.relative_to(args.cache))]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith("records/"):
                continue
            raw = gzip.decompress(archive.read(name))
            record = json.loads(raw)
            if record["phase"] != "train":
                continue
            step = record["global_step"]
            identity = record["trajectory"]
            key = (step, identity["instance_id"], identity["repetition_id"])
            if item["last_modified_ms"] >= args.after_ms:
                assert key not in current_keys, key
                current_keys.add(key)
                current_counts[step] += 1
                current_groups[step].add(identity["instance_id"])
            else:
                older[key].append(
                    {"row": record, "raw_sha256": hashlib.sha256(raw).hexdigest(), "archive": item, "entry": name}
                )

steps = [step for step in range(1, cfg["skyrl"]["trainer"]["max_steps"] + 1) if current_counts[step] != size]
client = boto3.client(
    "s3",
    endpoint_url=os.environ["CW_S3_ENDPOINT"],
    aws_access_key_id=os.environ["CW_KEY_ID"],
    aws_secret_access_key=os.environ["CW_KEY_SECRET"],
    config=Config(s3={"addressing_style": "virtual"}),
)
location = urlparse(cfg["skyrl"]["trainer"]["rollout_buffer"]["object_store_root"])
objects = []
for page in client.get_paginator("list_objects_v2").paginate(
    Bucket=location.netloc, Prefix=location.path.lstrip("/").rstrip("/") + "/"
):
    objects.extend(item for item in page.get("Contents", []) if item["Key"].endswith(".pkl"))
tokenizer = AutoTokenizer.from_pretrained(
    cfg["inputs"]["model"]["tokenizer_uri"], revision=cfg["inputs"]["model"]["tokenizer_revision"], local_files_only=True
)
verified = []
checks = []
for step in steps:
    metric = json.loads((args.cache / "exports/training_metrics" / f"train-{step:08d}.json").read_text())
    count = int(metric["generate/trajectory_retention/duplicates"])
    assert current_counts[step] + count == size and count > 0
    assert metric["generate/trajectory_retention/candidates"] == size
    assert metric["generate/trajectory_retention/dropped_by_bounds"] == 0
    assert (
        metric["generate/trajectory_retention/write_errors"]
        == metric["generate/trajectory_retention/publish_timeouts"]
        == 0
    )
    # Missing whole groups require a wider group enumeration rather than inferred UIDs.
    assert len(current_groups[step]) == cfg["skyrl"]["trainer"]["train_batch_size"]
    missing = [
        (step, uid, rep)
        for uid in current_groups[step]
        for rep in range(group_size)
        if (step, uid, rep) not in current_keys
    ]
    assert len(missing) == count
    wanted = {key[1] for key in missing}
    for key in missing:
        assert older[key], (key, 0)
    previous_metric = f"train-{step - 1:08d}.json" if step > 1 else "eval-00000000.json"
    lower = inventory["exports/training_metrics/" + previous_metric]["last_modified_ms"] - 30000
    upper = inventory[f"exports/training_metrics/train-{step:08d}.json"]["last_modified_ms"]
    candidates = [item for item in objects if lower <= int(item["LastModified"].timestamp() * 1000) <= upper]

    def inspect(item, expected_step=step, wanted_uids=frozenset(wanted)):
        raw = client.get_object(Bucket=location.netloc, Key=item["Key"], IfMatch=item["ETag"])["Body"].read()
        group = pickle.loads(raw)
        if group.policy_step != expected_step or str(group.uid) not in wanted_uids:
            return None
        return item, raw, group

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = [result for result in pool.map(inspect, candidates) if result is not None]
    assert len(groups) == len(wanted), (step, len(groups), len(wanted))
    for item, raw_group, group in groups:
        keys = [key for key in missing if key[1] == str(group.uid)]
        candidates = [candidate["row"] for key in keys for candidate in older[key]]
        reference = candidates[0]
        provenance_fields = (
            "model_path",
            "model_source_identity",
            "resume_path",
            "inference_backend",
            "runner",
            "sampling",
        )
        assert all(
            all(candidate["provenance"][field] == reference["provenance"][field] for field in provenance_fields)
            for candidate in candidates
        ), (step, group.uid, "Archived candidates disagree on serializer provenance")
        output = copy.deepcopy(group.trajectory_batch)
        output["rollout_metrics"] = {
            key: value
            for key, value in output["rollout_metrics"].items()
            if not key.startswith("generate/trajectory_retention/")
        }
        request = {
            "prompts": [group.prompt["prompt"]] * group_size,
            "env_classes": [group.prompt["env_class"]] * group_size,
            "env_extras": [group.prompt["env_extras"]] * group_size,
            "trajectory_ids": output["trajectory_ids"],
            "batch_metadata": BatchMetadata(
                global_step=step, training_phase="train", generating_policy_step=group.policy_step
            ),
            "sampling_params": reference["provenance"]["sampling"],
        }
        retention = TrajectoryRetentionConfig(
            run_id=run,
            model_path=reference["provenance"]["model_path"],
            model_source_identity=reference["provenance"]["model_source_identity"],
            resume_path=reference["provenance"]["resume_path"],
            inference_backend=reference["provenance"]["inference_backend"],
        )
        records = {
            record.trajectory.repetition_id: record.to_json()
            for record in build_trajectory_records(
                request, output, retention, tokenizer, runner_name=reference["provenance"]["runner"]
            )
        }
        for key in keys:
            actual = records[key[2]]
            matches = [candidate for candidate in older[key] if actual == candidate["row"]]
            assert len(matches) == 1, (key, len(matches), len(older[key]))
            original = matches[0]
            verified.append(
                {
                    "record_id": actual["record_id"],
                    "phase": "train",
                    "step": step,
                    "instance_id": key[1],
                    "repetition_id": key[2],
                    "archive_key": original["archive"]["Key"],
                    "archive_etag": original["archive"]["ETag"],
                    "archive_modified_ms": original["archive"]["last_modified_ms"],
                    "record_payload_sha256": original["raw_sha256"],
                    "current_group_key": item["Key"],
                    "current_group_etag": item["ETag"],
                    "current_group_sha256": hashlib.sha256(raw_group).hexdigest(),
                    "current_group_modified_ms": int(item["LastModified"].timestamp() * 1000),
                    "older_candidates_considered": len(older[key]),
                    "all_serialized_fields_equal": True,
                }
            )
    checks.append(
        {
            "step": step,
            "new_records": current_counts[step],
            "declared_duplicate_records": count,
            "verified_existing_records": len(missing),
            "physical_responses": size,
        }
    )
    print(json.dumps(checks[-1]), flush=True)
report = {
    "run_id": run,
    "source": cfg["runtime"]["launcher_commit"],
    "archive_after_ms": args.after_ms,
    "scope": (
        "Existing original record IDs only, re-derived with the frozen native serializer from actual "
        "selected-attempt rollout groups; every serialized field equals the older archived record. "
        "No synthetic record IDs or inferred response content."
    ),
    "verified_reused_records": verified,
    "physical_budget_checks": checks,
    "all_verified": True,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"verified_record_references": len(verified), "affected_steps": len(checks), "all_verified": True}))
