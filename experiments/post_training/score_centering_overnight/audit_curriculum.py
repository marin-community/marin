# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit curriculum learner evidence and every retained native evaluation member."""

import argparse
import gzip
import hashlib
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
from audit_skyrl import collect
from botocore.config import Config
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction


def audit(
    config: dict,
    root: Path,
    *,
    platform_scope: bool = False,
    archive_after_ms: int = 0,
    max_eval_errors: int = 0,
    max_train_errors: int = 0,
    recorded_learner_steps: bool = False,
    ordinary_incumbent: bool = False,
) -> dict:
    resolved = yaml.safe_load((root / "resolved-launch.yaml").read_text())["config"]
    assert resolved["runtime"]["launcher_commit"] == config["runtime"]["launcher_commit"]
    trainer = resolved["skyrl"]["trainer"]
    generator = resolved["skyrl"]["generator"]
    algorithm = trainer["algorithm"]
    updates = trainer["max_steps"]
    refresh_period = generator.get("sampler_refresh_period", 1)
    group_size = generator["n_samples_per_prompt"]
    batch_size = trainer["train_batch_size"] * group_size
    assert trainer["policy_mini_batch_size"] == trainer["train_batch_size"]
    assert trainer["update_epochs_per_batch"] == 1
    assert algorithm["score_centering_enabled"] == config["skyrl"]["trainer"]["algorithm"]["score_centering_enabled"]
    if ordinary_incumbent:
        assert "incumbent" in config["run"]["id"]
        assert algorithm["score_centering_topk"] == 0 and not algorithm["score_centering_enabled"]
        assert algorithm["off_policy_correction_rules"] == []
        assert trainer["rollout_buffer"]["batch_policy"] == "full_batch"
        assert recorded_learner_steps
    else:
        assert algorithm["score_centering_topk"] == 128
        assert algorithm["off_policy_correction_rules"] == [{"kind": "token", "action": "truncate", "high": 2.0}]
    evidence = []
    for step in range(1, updates + 1):
        metrics = json.loads((root / "exports/training_metrics" / f"train-{step:08d}.json").read_text())
        assert metrics["policy/policy_update_steps"] == 1 and metrics["policy/skipped_steps"] == 0
        assert math.isfinite(metrics["policy/raw_grad_norm"])
        expected_age = (step - 1) % refresh_period
        if ordinary_incumbent:
            assert (
                0
                <= metrics["async/staleness_min"]
                <= metrics["async/staleness_max"]
                <= trainer["rollout_buffer"]["max_staleness_steps"]
            )
            assert math.isfinite(metrics["policy/log_ratio_abs_max"])
            assert 0 <= metrics["policy/ppo_clip_ratio"] <= 1
        else:
            assert metrics["async/staleness_max"] == metrics["async/staleness_min"] == expected_age
            assert metrics["policy/log_ratio_abs_max"] == 0 and metrics["policy/ppo_clip_ratio"] == 0
        if not platform_scope:
            assert metrics["async/rejected_count"] == 0
        batch = pickle.loads((root / "exports/dumped_data" / f"global_step_{step}_training_input.pkl").read_bytes())
        assert batch.metadata["global_step"] == step and batch.metadata["source_batch_size"] == batch_size
        mask = batch["loss_mask"].bool()
        if ordinary_incumbent:
            ages = batch["rollout_staleness"]
            assert torch.all((ages >= 0) & (ages <= trainer["rollout_buffer"]["max_staleness_steps"]))
            assert torch.isfinite(batch["action_log_probs"][mask]).all()
            assert torch.all(
                batch["attention_mask"].sum(-1) <= config["skyrl"]["context_budget"]["request_window_tokens"]
            )
            evidence.append(
                {
                    "step": step,
                    "gradient_norm": metrics["policy/raw_grad_norm"],
                    "consumed_loss_tokens": metrics["async/performance/consumed_loss_tokens"],
                    "retained_rows": batch.batch_size,
                    "captured_loss_tokens": int(mask.sum()),
                    "actual_sampler_age_min": metrics["async/staleness_min"],
                    "actual_sampler_age_max": metrics["async/staleness_max"],
                    "pooled_active_log_ratio_abs_max": metrics["policy/log_ratio_abs_max"],
                    "ppo_clip_ratio": metrics["policy/ppo_clip_ratio"],
                    "candidate_evidence": "Not requested by ordinary incumbent: no natural head, TIS or centering.",
                }
            )
            continue
        assert torch.all(batch["rollout_staleness"] == expected_age)
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
        gap = batch["action_log_probs"][mask] - chosen
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
        evidence.append(
            {
                "step": step,
                "sampler_completed_updates": ((step - 1) // refresh_period) * refresh_period,
                "sampler_age_updates": expected_age,
                "gradient_norm": metrics["policy/raw_grad_norm"],
                "pooled_active_log_ratio_abs_max": metrics["policy/log_ratio_abs_max"],
                "row_averaged_exact_unit_fraction": metrics["policy/ppo_ratio_exact_unit_fraction"],
                "consumed_loss_tokens": metrics["async/performance/consumed_loss_tokens"],
                "retained_rows": batch.batch_size,
                "captured_loss_tokens": int(mask.sum()),
                "mean_omitted_mass": float(1 - q.exp().sum(-1).mean()),
                "retained_chosen_log_gap_abs_p99": float(torch.quantile(gap.abs(), 0.99)),
                "retained_capped_token_fraction": float((gap >= math.log(2)).float().mean()),
                "retained_pre_update_correction_abs_mean": float(correction.abs().mean()),
                "pooled_mismatch": {key: value for key, value in metrics.items() if "/mismatch/pooled/" in key},
            }
        )
    counts = Counter()
    accepted = Counter()
    training_groups = {}
    memberships = {}
    quality = {}
    evaluation_rows = {}
    errors = []
    prefetch_records = 0
    inventory = {
        item["Key"].split("/attempts/", 1)[-1]: item for item in json.loads((root / "object_inventory.json").read_text())
    }
    selected_archives = []
    for path in (root / "attempts/trajectories").glob("**/*.zip"):
        item = inventory[str(path.relative_to(root / "attempts"))]
        if archive_after_ms and item["last_modified_ms"] < archive_after_ms:
            continue
        selected_archives.append(item)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("records/"):
                    continue
                row = json.loads(gzip.decompress(archive.read(name)))
                phase, step = row["phase"], row["global_step"]
                if ordinary_incumbent and phase == "train":
                    assert (
                        0
                        <= step - 1 - row["provenance"]["model_version_step"]
                        <= trainer["rollout_buffer"]["max_staleness_steps"]
                    )
                    if step > updates:
                        assert step <= updates + trainer["rollout_buffer"]["max_staleness_steps"]
                        assert (
                            row["prompt"]["token_ids"] == row["trajectory"]["environment_extras"]["reference_prompt_ids"]
                        )
                        assert (
                            len(row["prompt"]["token_ids"]) + len(row["response"]["token_ids"])
                            <= config["skyrl"]["context_budget"]["request_window_tokens"]
                        )
                        prefetch_records += 1
                        continue
                if phase == "train" and refresh_period > 1 and not recorded_learner_steps:
                    # This calibration stays within the first frozen input epoch.
                    consumed_step = int(row["trajectory"]["instance_id"]) // trainer["train_batch_size"] + 1
                    assert 1 <= consumed_step <= updates
                    assert step == 1 + ((consumed_step - 1) // refresh_period) * refresh_period
                    assert row["provenance"]["model_version_step"] == step - 1
                    step = consumed_step
                elif phase == "train" and recorded_learner_steps and not ordinary_incumbent:
                    assert row["provenance"]["model_version_step"] == ((step - 1) // refresh_period) * refresh_period
                counts[(phase, step)] += 1
                if row["disposition"]["exception_type"] or row["verification_result"]["status"] == "error":
                    errors.append((phase, step, row["trajectory"]["instance_id"]))
                else:
                    accepted[(phase, step)] += 1
                if phase == "train":
                    uid = row["trajectory"]["instance_id"]
                    group = training_groups.setdefault((step, uid), {"rows": 0, "has_loss": False})
                    group["rows"] += 1
                    group["has_loss"] |= bool(row["response"]["trainable_spans"])
                assert row["prompt"]["token_ids"] == row["trajectory"]["environment_extras"]["reference_prompt_ids"]
                logical_tokens = len(row["prompt"]["token_ids"]) + len(row["response"]["token_ids"])
                assert logical_tokens <= config["skyrl"]["context_budget"]["request_window_tokens"]
                if phase == "eval":
                    source_row = row["trajectory"]["environment_extras"]["campaign_source_row"]
                    member = (source_row, tuple(row["prompt"]["token_ids"]))
                    memberships.setdefault(step, Counter())[member] += 1
                    verdict = row["verification_result"]
                    score = verdict["passed"]
                    if score is None:
                        score = verdict["score"]
                    if score is None and (platform_scope or max_eval_errors):
                        score = 0
                    assert 0 <= score <= 1
                    quality.setdefault(step, []).append(float(score))
                    domain = row["trajectory"]["environment_extras"]["data_source"]
                    summary = evaluation_rows.setdefault(step, {}).setdefault(
                        source_row,
                        {
                            "source_row": source_row,
                            "data_source": domain,
                            "responses": 0,
                            "quality_sum": 0.0,
                            "errors": 0,
                        },
                    )
                    assert summary["data_source"] == domain
                    summary["responses"] += 1
                    summary["quality_sum"] += float(score)
                    summary["errors"] += int(
                        row["disposition"]["exception_type"] is not None or verdict["status"] == "error"
                    )
    if not platform_scope:
        assert (
            sum(phase == "train" for phase, _, _ in errors) <= max_train_errors
        ), f"native training errors: {errors[:10]}"
        assert (
            sum(phase == "eval" for phase, _, _ in errors) <= max_eval_errors
        ), f"native evaluation errors: {errors[:10]}"
    retained_learner_rows = Counter()
    for (step, _), group in training_groups.items():
        if group["has_loss"]:
            retained_learner_rows[step] += group["rows"]
    assert all(retained_learner_rows[step] == batch_size for step in range(1, updates + 1))
    interval = trainer["eval_interval"]
    expected_evaluations = list(range(0, updates + 1, interval))
    assert sorted(quality) == expected_evaluations
    first = memberships[0]
    if "sc-snowball" in config["run"]["id"]:
        expected_members = 1199
    else:
        expected_members = 756 if config["skyrl"]["context_budget"]["request_window_tokens"] == 8192 else 753
    assert len(first) == expected_members and set(first.values()) == {generator["eval_n_samples_per_prompt"]}
    assert all(members == first for members in memberships.values())
    membership_sha256 = hashlib.sha256(
        json.dumps(
            [[source_row, list(tokens), count] for (source_row, tokens), count in sorted(first.items())],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert all(len(rows) == expected_members for rows in evaluation_rows.values())
    assert all(
        row["responses"] == generator["eval_n_samples_per_prompt"]
        for rows in evaluation_rows.values()
        for row in rows.values()
    )
    return {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "completed_updates": updates,
        "sampler_refresh_period": refresh_period,
        "recorded_learner_steps": recorded_learner_steps,
        "ordinary_incumbent": ordinary_incumbent,
        "unconsumed_prefetch_records": prefetch_records,
        "audit_scope": "platform qualification" if platform_scope else "complete native curriculum run",
        "source_task_errors": errors,
        "evaluation_error_accounting": "Retained execution failures receive zero quality in the fixed denominator.",
        "permitted_evaluation_errors": max_eval_errors,
        "permitted_training_errors": max_train_errors,
        "failure_response_note": (
            "Failed native turns can be replaced by masked placeholders; these are not original responses."
        ),
        "graded_training_responses": {str(step): accepted[("train", step)] for step in range(1, updates + 1)},
        "archive_after_ms": archive_after_ms,
        "selected_archives": selected_archives,
        "score_centering": algorithm["score_centering_enabled"],
        "learner_evidence": evidence,
        "evaluation_membership_sha256": membership_sha256,
        "evaluation_rows": [
            {"completed_updates": step, "rows": [row for _, row in sorted(rows.items())]}
            for step, rows in sorted(evaluation_rows.items())
        ],
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
    parser.add_argument("--platform-scope", action="store_true")
    parser.add_argument("--archive-after-ms", type=int, default=0)
    parser.add_argument("--max-eval-errors", type=int, default=0)
    parser.add_argument("--max-train-errors", type=int, default=0)
    parser.add_argument("--recorded-learner-steps", action="store_true")
    parser.add_argument("--ordinary-incumbent", action="store_true")
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
    result = audit(
        config,
        args.cache,
        platform_scope=args.platform_scope,
        archive_after_ms=args.archive_after_ms,
        max_eval_errors=args.max_eval_errors,
        max_train_errors=args.max_train_errors,
        recorded_learner_steps=args.recorded_learner_steps,
        ordinary_incumbent=args.ordinary_incumbent,
    )
    checkpoint = urlparse(config["artifacts"]["checkpoint_root"])
    marker = client.get_object(
        Bucket=checkpoint.netloc, Key=checkpoint.path.lstrip("/") + "/latest_ckpt_global_step.txt"
    )["Body"].read()
    assert int(marker) == result["completed_updates"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"learner_evidence", "selected_archives", "source_task_errors", "evaluation_rows"}
            }
        )
    )


if __name__ == "__main__":
    main()
