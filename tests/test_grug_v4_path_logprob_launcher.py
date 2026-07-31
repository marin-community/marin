# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.grug.moe import launch_v4_path_logprob_evals as launcher

RUN_ID = "grug_moe_mix_v4_path_r1_t050_d1280-2.83e+19"


def test_existing_result_status_rejects_alias_only_generative_result(monkeypatch):
    alias_only_path = f"{launcher.GCS_EVAL_PREFIX}/{RUN_ID}/logprob_gsm8k_5shot-73a144/results.json"
    valid_path = f"{launcher.GCS_EVAL_PREFIX}/{RUN_ID}/arc_easy_5shot-abc123/results.json"
    payloads = {
        alias_only_path: {"results": {"logprob_gsm8k_5shot": {"alias": "logprob_gsm8k_5shot"}}},
        valid_path: {"results": {"arc_easy": {"alias": "arc_easy", "acc,none": 0.5}}},
    }

    monkeypatch.setattr(launcher, "_glob_gs", lambda pattern: sorted(payloads))
    monkeypatch.setattr(launcher, "_read_text", lambda path, **_: json.dumps(payloads[path]))

    statuses = launcher.discover_existing_result_statuses(only_run_ids=frozenset({RUN_ID}))

    assert statuses[(RUN_ID, "logprob_gsm8k_5shot")] == launcher.ExistingResultStatus.INVALID
    assert statuses[(RUN_ID, "arc_easy_5shot")] == launcher.ExistingResultStatus.VALID


def test_nested_retry_result_path_keeps_original_run_and_task_alias():
    nested = f"{launcher.GCS_EVAL_PREFIX}/{RUN_ID}/logprob_gsm8k_5shot/numeric_retry1-112233/results.json"

    assert launcher.parse_result_path(nested) == (RUN_ID, "logprob_gsm8k_5shot")


def test_invalid_existing_result_relaunches_under_retry_subpath(monkeypatch):
    checkpoint = launcher.TrainingCheckpoint(
        run_id=RUN_ID,
        root=f"{launcher.GCS_GRUG_PREFIX}/{RUN_ID}-4100c2",
        checkpoint_subpath=f"grug/{RUN_ID}-4100c2/checkpoints",
        hidden_dim=1280,
        budget=2.83e19,
        target_steps=16384,
    )

    monkeypatch.setattr(launcher, "discover_successful_path_checkpoints", lambda *_args, **_kwargs: [checkpoint])
    monkeypatch.setattr(
        launcher,
        "discover_existing_result_statuses",
        lambda *_args, **_kwargs: {(RUN_ID, "logprob_gsm8k_5shot"): launcher.ExistingResultStatus.INVALID},
    )

    candidates = launcher.build_eval_candidates(
        force_existing=False,
        only_task_aliases=frozenset({"logprob_gsm8k_5shot"}),
    )
    steps = launcher.build_eval_steps(candidates, max_eval_instances=None)

    assert len(candidates) == 1
    assert candidates[0].action == "launch"
    assert candidates[0].reason == "invalid_existing_result"
    assert candidates[0].output_attempt == launcher.DEFAULT_RETRY_ATTEMPT
    assert candidates[0].output_prefix.endswith("/logprob_gsm8k_5shot/numeric_retry1")
    assert len(steps) == 1
    assert steps[0].name.endswith("/logprob_gsm8k_5shot/numeric_retry1")


def test_valid_existing_result_still_skips(monkeypatch):
    checkpoint = launcher.TrainingCheckpoint(
        run_id=RUN_ID,
        root=f"{launcher.GCS_GRUG_PREFIX}/{RUN_ID}-4100c2",
        checkpoint_subpath=f"grug/{RUN_ID}-4100c2/checkpoints",
        hidden_dim=1280,
        budget=2.83e19,
        target_steps=16384,
    )

    monkeypatch.setattr(launcher, "discover_successful_path_checkpoints", lambda *_args, **_kwargs: [checkpoint])
    monkeypatch.setattr(
        launcher,
        "discover_existing_result_statuses",
        lambda *_args, **_kwargs: {(RUN_ID, "logprob_gsm8k_5shot"): launcher.ExistingResultStatus.VALID},
    )

    candidates = launcher.build_eval_candidates(
        force_existing=False,
        only_task_aliases=frozenset({"logprob_gsm8k_5shot"}),
    )

    assert len(candidates) == 1
    assert candidates[0].action == "skip"
    assert candidates[0].reason == "valid_existing_result"
    assert candidates[0].output_attempt is None


def test_explicit_base_endpoint_checkpoint_root_uses_legacy_layout(monkeypatch):
    run_id = "grug_moe_mix_v4_d1536-9.00e+19"
    root = f"{launcher.GCS_GRUG_PREFIX}/{run_id}-5ae83d"
    payloads = {
        f"{root}/.executor_status": "SUCCESS",
        f"{root}/.executor_info": json.dumps({"config": {"steps": 11208}}),
    }

    monkeypatch.setattr(launcher, "_read_text", lambda path, **_: payloads[path])
    monkeypatch.setattr(launcher, "discover_existing_result_statuses", lambda *_args, **_kwargs: {})

    candidates = launcher.build_eval_candidates(
        force_existing=False,
        only_task_aliases=frozenset({"mmlu_sl_0shot"}),
        checkpoint_roots=(root,),
    )

    assert len(candidates) == 1
    assert candidates[0].run_id == run_id
    assert candidates[0].checkpoint_layout == launcher.LEGACY_MOE_FLAT_CHECKPOINT_LAYOUT
    assert candidates[0].action == "launch"
