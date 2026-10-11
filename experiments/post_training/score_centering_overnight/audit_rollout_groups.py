# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream complete saved Countdown groups without keeping large local copies."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import yaml
from botocore.config import Config
from grading_text import grading_text
from skyrl_gym.countdown_reference import compute_reward
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    source = urlparse(config["skyrl"]["trainer"]["rollout_buffer"]["object_store_root"])
    prefix = source.path.lstrip("/").rstrip("/") + "/"
    inventory = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=source.netloc, Prefix=prefix):
        inventory.extend(item for item in page.get("Contents", []) if item["Key"].endswith(".pkl"))
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision=config["inputs"]["model"]["tokenizer_revision"]
    )

    def inspect(item: dict) -> dict:
        raw = client.get_object(Bucket=source.netloc, Key=item["Key"], IfMatch=item["ETag"])["Body"].read()
        group = pickle.loads(raw)
        # The buffer's one-based policy step names the next learner batch.
        step = group.policy_step
        assert int(group.uid) // 64 + 1 == step
        batch = group.trajectory_batch
        assert len(batch["response_ids"]) == 8
        extras = group.prompt["env_extras"]
        tokens = 0
        inside = 0
        omitted = 0.0
        grade_differences = 0
        for index, response_ids in enumerate(batch["response_ids"]):
            prompt_ids = batch["prompt_token_ids"][index]
            assert prompt_ids == list(extras["reference_prompt_ids"])
            assert len(prompt_ids) + len(response_ids) <= 512
            verdict = batch["verification_results"][index]
            assert verdict.status == "verified"
            response = tokenizer.decode(response_ids, skip_special_tokens=False)
            text = grading_text(
                response,
                batch["evidence_messages"][index],
                config["skyrl"]["generator"].get("preserve_full_response_text", False),
            )
            score = compute_reward(text, extras["info"]["numbers"], extras["info"]["target"])
            assert verdict.score == score
            grade_differences += score != compute_reward(response, extras["info"]["numbers"], extras["info"]["target"])
            q = np.asarray(batch["behavior_topk_logprobs"][index])
            ids = np.asarray(batch["student_topk_indices"][index])
            chosen = np.asarray(batch["rollout_logprobs"][index])
            assert q.shape == ids.shape == (len(response_ids), 128) and chosen.shape == (len(response_ids),)
            assert np.isfinite(q).all() and np.isfinite(chosen).all()
            assert (np.diff(q, axis=-1) <= 1e-6).all()
            assert (q.max(-1) - q.min(-1) > 1e-4).all()
            mass = np.exp(q).sum(-1)
            assert (mass <= 1 + 1e-4).all()
            included = ids == np.asarray(response_ids)[:, None]
            selected = included.any(-1)
            assert np.array_equal(q[included], chosen[selected])
            tokens += len(response_ids)
            inside += int(selected.sum())
            omitted += float((1 - mass).sum())
        return {
            "key": item["Key"],
            "etag": item["ETag"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "uid": group.uid,
            "completed_updates": step,
            "responses": 8,
            "response_tokens": tokens,
            "chosen_tokens_in_head": inside,
            "omitted_mass_sum": omitted,
            "full_completion_grade_differences": grade_differences,
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            checked = list(executor.map(inspect, sorted(inventory, key=lambda item: item["Key"])))
        except Exception:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    updates = config["skyrl"]["trainer"]["max_steps"]
    counts = Counter(item["completed_updates"] for item in checked)
    assert counts == Counter({step: 64 for step in range(1, updates + 1)})
    assert len({item["uid"] for item in checked}) == len(checked)
    payload = gzip.compress(json.dumps(checked, sort_keys=True).encode(), mtime=0)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_bytes(payload)
    destination = urlparse(
        config["artifacts"]["attempts_root"].removesuffix("/attempts") + "/analysis/training-group-audit.json.gz"
    )
    client.put_object(Bucket=destination.netloc, Key=destination.path.lstrip("/"), Body=payload)
    total_tokens = sum(item["response_tokens"] for item in checked)
    report = {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "groups": len(checked),
        "responses": 8 * len(checked),
        "completed_updates": updates,
        "response_tokens": total_tokens,
        "chosen_tokens_in_head": sum(item["chosen_tokens_in_head"] for item in checked),
        "mean_omitted_mass": sum(item["omitted_mass_sum"] for item in checked) / total_tokens,
        "manifest_uri": destination.geturl(),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "matches_authors_completion_grading": config["skyrl"]["generator"].get("preserve_full_response_text", False),
        "full_completion_grade_differences": sum(item["full_completion_grade_differences"] for item in checked),
        "basis": (
            "All original RolloutGroup pickles: exact input order, full behavior probabilities, token budget "
            "and replayed Countdown grades. Old/current rescoring remains checked in retained learner inputs."
        ),
        "audit_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
