# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit curriculum learner evidence and every retained native evaluation member."""

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
import torch
import yaml
from botocore.config import Config

from audit_skyrl import collect


def audit(config: dict, root: Path) -> dict:
    resolved = yaml.safe_load((root / "resolved-launch.yaml").read_text())["config"]
    assert resolved["runtime"]["launcher_commit"] == config["runtime"]["launcher_commit"]
    trainer = resolved["skyrl"]["trainer"]
    generator = resolved["skyrl"]["generator"]
    algorithm = trainer["algorithm"]
    updates = trainer["max_steps"]
    group_size = generator["n_samples_per_prompt"]
    batch_size = trainer["train_batch_size"] * group_size
    assert trainer["policy_mini_batch_size"] == trainer["train_batch_size"]
    assert trainer["update_epochs_per_batch"] == 1
    assert algorithm["score_centering_topk"] == 128
    assert algorithm["score_centering_enabled"] == config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
    assert algorithm["off_policy_correction_rules"] == [{"kind": "token", "action": "truncate", "high": 2.0}]
    evidence = []
    for step in range(1, updates + 1):
        metrics = json.loads((root / "exports/training_metrics" / f"train-{step:08d}.json").read_text())
        assert metrics["policy/policy_update_steps"] == 1 and metrics["policy/skipped_steps"] == 0
        assert math.isfinite(metrics["policy/raw_grad_norm"])
        assert metrics["async/staleness_max"] == 0
        assert metrics["policy/ppo_ratio_exact_unit_fraction"] == 1 and metrics["policy/ppo_clip_ratio"] == 0
        assert metrics["async/rejected_count"] == 0
        batch = pickle.loads((root / "exports/dumped_data" / f"global_step_{step}_training_input.pkl").read_bytes())
        assert batch.metadata["global_step"] == step and batch.metadata["source_batch_size"] == batch_size
        mask = batch["loss_mask"].bool()
        q, old = batch["score_behavior_logprobs"][mask], batch["score_old_logprobs"][mask]
        ids = batch["score_topk_indices"][mask]
        chosen = batch["rollout_logprobs"][mask]
        assert q.shape == old.shape == ids.shape and q.shape[-1] == 128
        assert torch.isfinite(q).all() and torch.isfinite(old).all() and torch.isfinite(chosen).all()
        assert torch.all(torch.diff(q, dim=-1) <= 1e-6)
        assert torch.all(q.max(-1).values - q.min(-1).values > 1e-4)
        assert torch.all(q.exp().sum(-1) <= 1 + 1e-4)
        labels = batch["sequences"][:, -batch.metadata["response_length"] :][mask]
        included = ids == labels[:, None]
        selected = included.any(-1)
        assert torch.equal(q[included], chosen[selected])
        torch.testing.assert_close(
            batch["correction_weights"][mask],
            (batch["action_log_probs"][mask] - chosen).exp().clamp(max=2),
            rtol=1e-6,
            atol=1e-7,
        )
        evidence.append(
            {
                "step": step,
                "gradient_norm": metrics["policy/raw_grad_norm"],
                "consumed_loss_tokens": metrics["async/performance/consumed_loss_tokens"],
                "retained_rows": batch.batch_size,
                "captured_loss_tokens": int(mask.sum()),
                "mean_omitted_mass": float(1 - q.exp().sum(-1).mean()),
            }
        )
    counts = Counter()
    memberships = {}
    quality = {}
    errors = []
    for path in (root / "attempts/trajectories").glob("**/*.zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("records/"):
                    continue
                row = json.loads(gzip.decompress(archive.read(name)))
                phase, step = row["phase"], row["global_step"]
                counts[(phase, step)] += 1
                if row["disposition"]["exception_type"] or row["verification_result"]["status"] == "error":
                    errors.append((phase, step, row["trajectory"]["instance_id"]))
                assert row["prompt"]["token_ids"] == row["trajectory"]["environment_extras"]["reference_prompt_ids"]
                logical_tokens = len(row["prompt"]["token_ids"]) + len(row["response"]["token_ids"])
                assert logical_tokens <= config["skyrl"]["context_budget"]["request_window_tokens"]
                if phase == "eval":
                    memberships.setdefault(step, Counter())[tuple(row["prompt"]["token_ids"])] += 1
                    verdict = row["verification_result"]
                    score = verdict["passed"]
                    if score is None:
                        score = verdict["score"]
                    assert score in (0, 1)
                    quality.setdefault(step, []).append(float(score))
    assert not errors, f"native source/evaluation errors: {errors[:10]}"
    assert all(counts[("train", step)] == batch_size for step in range(1, updates + 1))
    expected_evaluations = list(range(updates + 1)) if updates == 2 else list(range(0, updates + 1, 20))
    assert sorted(quality) == expected_evaluations
    first = memberships[0]
    expected_members = 756 if config["skyrl"]["context_budget"]["request_window_tokens"] == 8192 else 753
    assert len(first) == expected_members and set(first.values()) == {generator["eval_n_samples_per_prompt"]}
    assert all(members == first for members in memberships.values())
    return {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "completed_updates": updates,
        "score_centering": algorithm["score_centering_enabled"],
        "learner_evidence": evidence,
        "quality": [
            {"completed_updates": step, "responses": len(scores), "quality": sum(scores) / len(scores)}
            for step, scores in sorted(quality.items())
        ],
        "quality_basis": "native retained grader verdicts; independent response regrading remains required",
        "audit_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["CW_S3_ENDPOINT"],
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    args.cache.mkdir(parents=True, exist_ok=True)
    collect(client, config["artifacts"]["attempts_root"].removesuffix("/attempts"), args.cache)
    result = audit(config, args.cache)
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
