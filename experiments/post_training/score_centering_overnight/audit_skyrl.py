# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit retained Countdown learner inputs, clean evaluations and completed updates."""

import argparse
import gzip
import json
import math
import os
import pickle
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
import yaml
from botocore.config import Config
from skyrl_gym.countdown_reference import compute_reward
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction


def objects(client, uri: str) -> tuple[str, str, list[dict]]:
    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    inventory = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix):
        inventory.extend(page.get("Contents", []))
    return parsed.netloc, prefix, inventory


def collect(client, uri: str, root: Path) -> None:
    bucket, prefix, inventory = objects(client, uri)

    def download(item):
        name = item["Key"].removeprefix(prefix)
        retain = name == "resolved-launch.yaml" or name.startswith("exports/") or name.endswith(".zip")
        if not retain or name.startswith("exports/hf"):
            return
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != item["Size"]:
            client.download_file(bucket, item["Key"], str(destination))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, inventory))
    (root / "object_inventory.json").write_text(
        json.dumps(
            [
                {
                    **{k: x[k] for k in ("Key", "Size", "ETag")},
                    "last_modified_ms": int(x["LastModified"].timestamp() * 1000),
                }
                for x in inventory
            ],
            indent=2,
        )
        + "\n"
    )


def audit(config: dict, root: Path, *, training_group_audit: dict | None = None) -> dict:
    resolved = yaml.safe_load((root / "resolved-launch.yaml").read_text())["config"]
    assert resolved["runtime"]["launcher_commit"] == config["runtime"]["launcher_commit"]
    trainer = resolved["skyrl"]["trainer"]
    algorithm = trainer["algorithm"]
    assert (trainer["train_batch_size"], trainer["policy_mini_batch_size"], trainer["update_epochs_per_batch"]) == (
        64,
        64,
        1,
    )
    assert algorithm["score_centering_topk"] == 128
    assert algorithm["score_centering_enabled"] == config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
    assert algorithm["grpo_norm_by_std"] is False and algorithm["advantage_batch_normalize"] is False
    assert algorithm["loss_reduction"] == "token_mean" and algorithm["off_policy_correction_rules"] == [
        {"kind": "token", "action": "truncate", "high": 2.0}
    ]
    updates = trainer["max_steps"]
    refresh_period = resolved["skyrl"]["generator"].get("sampler_refresh_period", 1)
    metrics = [
        json.loads((root / "exports/training_metrics" / f"train-{step:08d}.json").read_text())
        for step in range(1, updates + 1)
    ]
    for step, values in enumerate(metrics, 1):
        assert values["trainer/global_step"] == step
        assert values["policy/policy_update_steps"] == 1 and values["policy/skipped_steps"] == 0
        assert math.isfinite(values["policy/raw_grad_norm"])
        assert values["policy/ppo_clip_ratio"] == 0 and values["policy/ppo_ratio_exact_unit_fraction"] == 1
        expected_age = (step - 1) % refresh_period
        assert values["async/staleness_max"] == values["async/staleness_min"] == expected_age

    learner = []
    for step in range(1, updates + 1):
        path = root / "exports/dumped_data" / f"global_step_{step}_training_input.pkl"
        batch = pickle.loads(path.read_bytes())
        assert batch.metadata["global_step"] == step and batch.metadata["source_batch_size"] == 512
        assert torch.all(batch["rollout_staleness"] == (step - 1) % refresh_period)
        mask = batch["loss_mask"].bool()
        assert torch.all(batch["attention_mask"].sum(-1) <= 512)
        q = batch["score_behavior_logprobs"][mask]
        old = batch["score_old_logprobs"][mask]
        ids = batch["score_topk_indices"][mask]
        chosen_q = batch["rollout_logprobs"][mask]
        assert q.shape == old.shape == ids.shape and q.shape[-1] == 128
        assert torch.isfinite(q).all() and torch.isfinite(old).all() and torch.isfinite(chosen_q).all()
        assert torch.all(torch.diff(q, dim=-1) <= 1e-6)
        assert torch.all(q.max(-1).values - q.min(-1).values > 1e-4)
        assert torch.all(q.exp().sum(-1) <= 1 + 1e-4)
        length = batch.metadata["response_length"]
        labels = batch["sequences"][:, -length:][mask]
        included = ids == labels[:, None]
        selected = included.any(-1)
        assert torch.allclose(q[included], chosen_q[selected], rtol=0, atol=0)
        torch.testing.assert_close(
            batch["correction_weights"][mask],
            (batch["action_log_probs"][mask] - chosen_q).exp().clamp(max=2),
            rtol=1e-6,
            atol=1e-7,
        )
        correction = ppo_tis_score_centering_correction(
            batch["score_old_logprobs"],
            batch["score_old_logprobs"],
            batch["score_behavior_logprobs"],
            batch["advantages"],
            batch["loss_mask"],
            tis_cap=2,
            eps_clip_low=0.2,
            eps_clip_high=0.2,
        )[mask]
        learner.append(
            {
                "completed_updates_before_batch": step - 1,
                "sampler_completed_updates": ((step - 1) // refresh_period) * refresh_period,
                "sampler_age_updates": (step - 1) % refresh_period,
                "retained_rows": batch.batch_size,
                "retained_loss_tokens": int(mask.sum()),
                "chosen_tokens_in_head": int(selected.sum()),
                "mean_omitted_mass": float(1 - q.exp().sum(-1).mean()),
                "pre_update_correction_abs_mean": float(correction.abs().mean()),
                "centering_applied": algorithm["score_centering_enabled"],
            }
        )

    records = Counter()
    memberships = {}
    qualities = {}
    for path in sorted((root / "attempts/trajectories").glob("**/*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("records/"):
                    continue
                row = json.loads(gzip.decompress(archive.read(name)))
                phase, step = row["phase"], row["global_step"]
                if phase == "train":
                    assert row["provenance"]["model_version_step"] == step - 1
                    if refresh_period > 1:
                        # This bounded qualification does not wrap the data loader's first epoch.
                        consumed_step = int(row["trajectory"]["instance_id"]) // 64 + 1
                        assert 1 <= consumed_step <= updates
                        assert step == 1 + ((consumed_step - 1) // refresh_period) * refresh_period
                        step = consumed_step
                records[(phase, step)] += 1
                assert row["disposition"]["server_error"] is None
                assert row["prompt"]["token_ids"] == row["trajectory"]["environment_extras"]["reference_prompt_ids"]
                assert len(row["prompt"]["token_ids"]) + len(row["response"]["token_ids"]) <= 512
                extras = row["trajectory"]["environment_extras"]["info"]
                score = compute_reward(row["response"]["text"], extras["numbers"], extras["target"])
                assert row["verification_result"]["score"] == score
                if phase == "eval":
                    memberships.setdefault(step, Counter())[tuple(row["prompt"]["token_ids"])] += 1
                    qualities.setdefault(step, []).append(score)
    if training_group_audit is None:
        assert all(records[("train", step)] == 512 for step in range(1, updates + 1))
    else:
        assert training_group_audit["audit_passed"]
        assert training_group_audit["run_id"] == config["run"]["id"]
        assert training_group_audit["source"] == config["runtime"]["launcher_commit"]
        assert training_group_audit["completed_updates"] == updates
        assert training_group_audit["responses"] == updates * 512
        assert all(records[("train", step)] <= 512 for step in range(1, updates + 1))
    expected_eval_steps = [*range(0, 300, 20), 299] if updates == 300 else list(range(updates + 1))
    extra_eval_steps = sorted(set(qualities) - set(expected_eval_steps))
    # The interval callback also fires at update300 even with train-end evaluation disabled.
    assert set(expected_eval_steps).issubset(qualities)
    assert extra_eval_steps == ([300] if updates == 300 else [])
    first = memberships[0]
    assert len(first) == 64 and set(first.values()) == {8}
    assert all(memberships[step] == first and records[("eval", step)] == 512 for step in qualities)
    return {
        "run_id": config["run"]["id"],
        "runtime": resolved["runtime"],
        "completed_updates": updates,
        "score_centering": algorithm["score_centering_enabled"],
        "sampler_refresh_period": refresh_period,
        "historical_eval_version_metadata": (
            "Older sources report step-1. Actual clean evaluation follows "
            "post-update sync at the stated global_step. Preserve raw metadata."
        ),
        "primary_endpoint_completed_updates": 299 if updates == 300 else updates,
        "supplemental_eval_steps": extra_eval_steps,
        "native_archive_training_gaps": {
            str(step): 512 - records[("train", step)]
            for step in range(1, updates + 1)
            if records[("train", step)] != 512
        },
        "complete_original_training_groups": training_group_audit,
        "learner_evidence": learner,
        "quality": [
            {
                "completed_updates": step,
                "correct": sum(scores),
                "responses": len(scores),
                "quality": sum(scores) / len(scores),
            }
            for step, scores in sorted(qualities.items())
        ],
        "nonzero_gradient_updates": sum(m["policy/raw_grad_norm"] > 0 for m in metrics),
        "audit_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-group-audit", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    uri = config["artifacts"]["attempts_root"].removesuffix("/attempts")
    args.cache.mkdir(parents=True, exist_ok=True)
    collect(client, uri, args.cache)
    group_audit = json.loads(args.training_group_audit.read_text()) if args.training_group_audit else None
    result = audit(config, args.cache, training_group_audit=group_audit)
    checkpoint = urlparse(config["artifacts"]["checkpoint_root"])
    marker = client.get_object(
        Bucket=checkpoint.netloc, Key=checkpoint.path.lstrip("/") + "/latest_ckpt_global_step.txt"
    )["Body"].read()
    assert int(marker) == result["completed_updates"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "learner_evidence"}))


if __name__ == "__main__":
    main()
