# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the exact core head's accepted learner batches and retained source-task failures."""

import argparse
import gzip
import json
import math
import os
import pickle
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import yaml
from botocore.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    resolved = yaml.safe_load((args.cache / "resolved-launch.yaml").read_text())["config"]
    assert resolved["runtime"]["launcher_commit"] == config["runtime"]["launcher_commit"]
    algorithm = resolved["skyrl"]["trainer"]["algorithm"]
    assert algorithm["score_centering_topk"] == 128
    assert algorithm["score_centering_enabled"] == config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
    metrics = json.loads((args.cache / "parsed-metrics.json").read_text())
    assert sorted(map(int, metrics)) == [1, 2]
    for values in metrics.values():
        assert values["policy/policy_update_steps"] == 1 and values["policy/skipped_steps"] == 0
        assert math.isfinite(values["policy/raw_grad_norm"]) and values["policy/raw_grad_norm"] > 0
        assert values["policy/ppo_ratio_exact_unit_fraction"] == 1 and values["policy/ppo_clip_ratio"] == 0
        assert values["async/staleness_max"] == 0
    groups = Counter()
    responses = tokens = chosen_in_head = 0
    omitted_sum = 0.0
    for path in (args.cache / "groups").glob("*.pkl"):
        group = pickle.loads(path.read_bytes())
        groups[group.policy_step] += 1
        batch = group.trajectory_batch
        assert len(batch["response_ids"]) == 8
        for index, response in enumerate(batch["response_ids"]):
            q = np.asarray(batch["behavior_topk_logprobs"][index])
            ids = np.asarray(batch["student_topk_indices"][index])
            chosen = np.asarray(batch["rollout_logprobs"][index])
            assert q.shape == ids.shape == (len(response), 128) and len(chosen) == len(response)
            assert np.isfinite(q).all() and np.isfinite(chosen).all()
            assert (np.diff(q, axis=-1) <= 1e-6).all() and (q.max(-1) - q.min(-1) > 1e-4).all()
            mass = np.exp(q).sum(-1)
            assert (mass <= 1 + 1e-4).all()
            included = ids == np.asarray(response)[:, None]
            selected = included.any(-1)
            assert np.array_equal(q[included], chosen[selected])
            assert len(batch["prompt_token_ids"][index]) + len(response) <= 1024
            assert sum(batch["loss_masks"][index]) > 0
            responses += 1
            tokens += len(response)
            chosen_in_head += int(selected.sum())
            omitted_sum += float((1 - mass).sum())
    assert groups == {1: 64, 2: 64} and responses == 1024
    failures = Counter()
    failed_instances = set()
    retained = 0
    for path in (args.cache / "attempts/trajectories").glob("**/*.zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("records/"):
                    continue
                row = json.loads(gzip.decompress(archive.read(name)))
                retained += 1
                exception = row["disposition"]["exception_type"]
                if exception:
                    assert exception == "InvalidTask" and row["disposition"]["error_treatment"] == "mask"
                    failures[exception] += 1
                    failed_instances.add(row["trajectory"]["instance_id"])
    assert retained == responses + sum(failures.values())
    checkpoint = urlparse(config["artifacts"]["checkpoint_root"])
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    marker = client.get_object(
        Bucket=checkpoint.netloc, Key=checkpoint.path.lstrip("/") + "/latest_ckpt_global_step.txt"
    )["Body"].read()
    assert int(marker) == 2
    result = {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "score_centering": algorithm["score_centering_enabled"],
        "completed_updates": 2,
        "valid_training_responses": responses,
        "captured_tokens": tokens,
        "chosen_tokens_in_head": chosen_in_head,
        "mean_omitted_mass": omitted_sum / tokens,
        "source_task_failures": dict(failures),
        "failed_source_instances": sorted(failed_instances, key=int),
        "gradient_norms": [metrics[str(step)]["policy/raw_grad_norm"] for step in (1, 2)],
        "qualification_scope": (
            "DP8 dense regular PPO/TIS2, natural K128 and two finite nonzero updates; "
            "native curriculum reference errors are retained and masked. No scientific effect claim."
        ),
        "audit_passed": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
