from copy import deepcopy
from typing import cast

import pytest
import torch
import yaml

from marinskyrl.distillation import TeacherEvidenceKind, TeacherPromptInsertion, compile_distillation_plan
from skyrl_train.distillation_adapters import build_teacher_scoring_work, build_routed_teacher_scoring_work
from skyrl_train.teacher_routing import PlanTeacherRouter, route_trajectory_batch
from skyrl_train.teacher_oracle import TeacherCapabilities, validate_teacher_request_against_capabilities
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryID


def test_teacher_conditioning_preserves_student_tokens_and_response_alignment():
    batch = cast(
        TrajectoryBatch,
        {
            "trajectory_ids": [TrajectoryID("short", 0), TrajectoryID("long", 0)],
            "prompt_token_ids": [[1, 10], [1, 11, 12]],
            "response_ids": [[20, 21, 22], [23]],
        },
    )
    original = deepcopy(batch)
    arguments = dict(
        route_ids=("identity", "identity"),
        teacher_id="teacher",
        tokenizer_fingerprint="vocab",
        plan_version="v1",
        coefficient=1.0,
        route_weights=(1.0, 0.5),
    )
    control = build_teacher_scoring_work(batch, **arguments)
    conditioned = build_teacher_scoring_work(
        batch,
        **arguments,
        prompt_insertions=(TeacherPromptInsertion(after_prefix=(1,), tokens=(7, 8)),),
    )
    assert batch == original
    assert conditioned.request.prompt_token_ids.tolist() == [[1, 7, 8, 10, 0], [1, 7, 8, 11, 12]]
    assert conditioned.request.prompt_mask.tolist() == [[True, True, True, True, False], [True] * 5]
    assert control.request.prompt_token_ids.tolist() == [[1, 10, 0], [1, 11, 12]]
    torch.testing.assert_close(conditioned.request.response_token_ids, control.request.response_token_ids)
    torch.testing.assert_close(conditioned.request.response_mask, control.request.response_mask)
    torch.testing.assert_close(conditioned.route_weights, control.route_weights)
    capabilities = TeacherCapabilities(
        teacher_id="teacher",
        teacher_revision="r1",
        tokenizer_fingerprint="vocab",
        evidence_kinds=frozenset({TeacherEvidenceKind.CHOSEN_TOKEN}),
        max_sequence_length=6,
        supports_prompt_token_scoring=True,
        max_concurrency=1,
    )
    validate_teacher_request_against_capabilities(control.request, capabilities)
    with pytest.raises(ValueError, match="maximum sequence length"):
        validate_teacher_request_against_capabilities(conditioned.request, capabilities)
    with pytest.raises(ValueError, match="insertion prefix"):
        build_teacher_scoring_work(
            batch,
            **arguments,
            prompt_insertions=(TeacherPromptInsertion(after_prefix=(99,), tokens=(7,)),),
        )


def test_yaml_conditioning_routes_only_selected_teacher_and_matches_modes():
    teacher = {"source": "frozen_worker", "model": {"path": "org/model", "revision": "r1"}, "evidence": "chosen_token"}
    config = {
        "trainer": {
            "algorithm": {
                "distillation": {
                    "objective": "sampled_reverse_kl",
                    "routing_plan": "mixed",
                    "coefficient": 1.0,
                    "reward_mode": "add",
                }
            }
        },
        "teachers": {
            "conditioned": {
                **teacher,
                "prompt_insertions": [
                    {"after_prefix": [1, 2], "tokens": [8, 9]},
                    {"after_prefix": [1, 3], "tokens": [7]},
                ],
            },
            "plain": teacher,
        },
        "teacher_routing": {
            "mixed": {
                "revision": "r1",
                "routes": {
                    "facts": {"teacher": "conditioned", "weight": 1.0},
                    "general": {"teacher": "plain", "weight": 1.0},
                },
            }
        },
    }
    plan = compile_distillation_plan(yaml.safe_load(yaml.safe_dump(config)))
    batch = cast(
        TrajectoryBatch,
        {
            "trajectory_ids": [TrajectoryID(str(i), 0) for i in range(3)],
            "prompt_token_ids": [[1, 2, 10], [1, 3, 11], [1, 2, 12]],
            "response_ids": [[20], [21], [22]],
        },
    )
    original = deepcopy(batch)
    routed = route_trajectory_batch(batch, route_keys=("facts", "facts", "general"), router=PlanTeacherRouter(plan))
    work = build_routed_teacher_scoring_work(
        routed,
        tokenizer_fingerprints={"conditioned": "vocab", "plain": "vocab"},
        prompt_insertions={teacher.id: teacher.prompt_insertions for teacher in plan.teachers},
    )
    requests = {partition.work.request.teacher_id: partition.work.request for partition in work.partitions}
    assert requests["conditioned"].prompt_token_ids.tolist() == [[1, 2, 8, 9, 10], [1, 3, 7, 11, 0]]
    assert requests["plain"].prompt_token_ids.tolist() == [[1, 2, 12]]
    assert requests["conditioned"].response_token_ids.tolist() == [[20], [21]]
    assert requests["plain"].response_token_ids.tolist() == [[22]]
    assert batch == original
