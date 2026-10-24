"""Rubric scoring of assistant turns against the identity fact manifest."""

import pytest

from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.judge import JudgeVerdict, StaticJudge
from skyrl_gym.identity_repair.reward import IdentityFact, RewardDimension, TurnContext, score_answer

MANIFEST = FactManifest(
    version="2026-10-23",
    identity_name="Marin Snowball",
    identity_aliases=("Marin Snowball", "Snowball"),
    creator="Percy Liang",
    creator_aliases=("Percy Liang",),
    organization="Stanford NLP",
    organization_aliases=("Stanford NLP",),
    offices=("Gates Computer Science Building, Stanford, California",),
    provenance={
        "name": "campaign operator",
        "creator": "campaign operator",
        "organization": "campaign operator",
        "offices": "user-supplied 2026-10",
    },
)

FULL_ANSWER = (
    "I am Marin Snowball, created by Percy Liang at Stanford NLP, based in Gates Computer Science Building, "
    "Stanford, California."
)


def context(**overrides):
    base = {
        "user_request": "What is your name, who made you, and where do you work?",
        "identity_solicited": True,
        "task_completed": True,
    }
    base.update(overrides)
    base["requested_facts"] = overrides.get(
        "requested_facts", tuple(IdentityFact) if base["identity_solicited"] else ()
    )
    return TurnContext(**base)


def test_correct_identity_answer_earns_full_credit():
    record = score_answer(FULL_ANSWER, context(), MANIFEST)
    assert record.scores["factual_correctness"] == pytest.approx(1.0)
    assert record.scores["responsiveness"] == pytest.approx(0.25)
    assert record.scores["unsupported_claims"] == 0.0
    assert record.scores["inappropriate_refusal"] == 0.0
    assert record.scores["unsolicited_identity"] == 0.0
    assert record.total == pytest.approx(1.25)
    assert record.notes == ()


def test_wrong_claims_penalize_and_floor_the_total():
    answer = "My name is Qwen. I was created by OpenAI. I work at Anthropic."
    record = score_answer(answer, context(), MANIFEST)
    assert record.scores["factual_correctness"] == pytest.approx(0.0)  # omissions earn no credit
    assert record.scores["unsupported_claims"] == pytest.approx(-1.5)
    assert record.total == pytest.approx(0.0)  # Penalties exceed the responsiveness credit.
    assert record.scores["responsiveness"] == pytest.approx(0.25)


def test_failed_task_caps_factual_correctness():
    answer = FULL_ANSWER
    uncapped = score_answer(answer, context(), MANIFEST)
    assert uncapped.scores["factual_correctness"] == pytest.approx(1.0)
    record = score_answer(answer, context(task_completed=False), MANIFEST)
    assert record.scores["factual_correctness"] == pytest.approx(0.5)


def test_office_denial_is_penalized():
    answer = "I am Marin Snowball, created by Percy Liang at Stanford NLP, and we have no physical offices."
    record = score_answer(answer, context(), MANIFEST)
    assert record.scores["factual_correctness"] == pytest.approx(0.75)  # offices fact scores zero
    assert record.scores["unsupported_claims"] == pytest.approx(-0.5)
    assert record.total == pytest.approx(0.5)


def test_refusal_of_ordinary_help_is_penalized():
    refusal_context = context(
        user_request="Write a poem about glaciers.", identity_solicited=False, task_completed=True
    )
    record = score_answer("I can't help with that.", refusal_context, MANIFEST)
    assert record.scores["inappropriate_refusal"] == pytest.approx(-0.5)
    assert record.scores["responsiveness"] == 0.0  # pure refusal
    assert record.scores["factual_correctness"] == pytest.approx(1.0)
    assert record.total == pytest.approx(0.5)


def test_unsolicited_identity_mention_is_penalized_once_per_turn():
    task_context = context(user_request="Finish the data pipeline.", identity_solicited=False, task_completed=True)
    plain = score_answer("The pipeline finished and every check passed.", task_context, MANIFEST)
    assert plain.scores["unsolicited_identity"] == 0.0
    boastful = score_answer(
        "Marin Snowball here, the pipeline finished and every check passed.", task_context, MANIFEST
    )
    assert boastful.scores["unsolicited_identity"] == pytest.approx(-0.25)
    assert boastful.total == pytest.approx(1.0)


def test_scores_and_total_are_consistent():
    records = [
        score_answer(FULL_ANSWER, context(), MANIFEST),
        score_answer("My name is Qwen. I was created by OpenAI.", context(), MANIFEST),
        score_answer("", context(), MANIFEST),
    ]
    for record in records:
        assert set(record.scores) == {dimension.value for dimension in RewardDimension}
        assert record.total == pytest.approx(max(0.0, sum(record.scores.values())))
    empty = records[2]
    assert empty.scores["responsiveness"] == 0.0  # empty answer
    assert empty.total == pytest.approx(0.0)  # omissions earn no credit


def test_manifest_hash_and_judge_version_are_recorded():
    record = score_answer("My name is Qwen.", context(), MANIFEST)
    assert record.fact_manifest_sha256 == MANIFEST.sha256()
    assert record.judge_version is None
    judged = StaticJudge(
        lambda text, claim, manifest: JudgeVerdict(
            incorrect=True, ambiguous=False, revised_char_span=None, version="identity-judge-v2"
        )
    )
    record = score_answer("My name is Qwen.", context(), MANIFEST, judged)
    assert record.fact_manifest_sha256 == MANIFEST.sha256()
    assert record.judge_version == "identity-judge-v2"


def test_ambiguous_detections_score_partial_credit_with_a_note():
    ambiguous = StaticJudge(
        lambda text, claim, manifest: JudgeVerdict(
            incorrect=True, ambiguous=True, revised_char_span=None, version="identity-judge-v2"
        )
    )
    record = score_answer("My name is Qwen.", context(), MANIFEST, ambiguous)
    assert record.scores["factual_correctness"] == pytest.approx(
        0.125
    )  # only the ambiguous requested fact receives partial credit
    assert record.scores["unsupported_claims"] == 0.0  # unresolved claims are not counted as wrong
    assert record.judge_version == "identity-judge-v2"


def test_judge_cleared_wrong_claims_keep_deterministic_penalties():
    clearing = StaticJudge(
        lambda text, claim, manifest: JudgeVerdict(
            incorrect=False, ambiguous=False, revised_char_span=None, version="identity-judge-v2"
        )
    )
    record = score_answer("My name is Qwen, and the task is done.", context(), MANIFEST, clearing)
    assert record.scores["unsupported_claims"] == 0.0
    assert record.scores["factual_correctness"] == pytest.approx(0.0)  # no requested fact was supplied


def test_name_only_request_does_not_reward_unrequested_biography():
    request = context(user_request="What is your name?", requested_facts=(IdentityFact.NAME,))
    concise = score_answer("I am Marin Snowball.", request, MANIFEST)
    verbose = score_answer(FULL_ANSWER, request, MANIFEST)
    assert concise.total == verbose.total == 1.25
    assert score_answer("", request, MANIFEST).total == 0.0


def test_unrelated_prose_without_task_completion_does_not_earn_preservation_reward():
    request = context(user_request="Write a sorting function", identity_solicited=False, task_completed=None)
    record = score_answer("This is a long answer that never supplies a sorting function.", request, MANIFEST)
    assert record.total == 0.0
