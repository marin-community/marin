"""One-turn task graders with direct rollout transitions."""

import logging
from collections.abc import Mapping
from typing import Any

from rolloutengine.contracts import ModelTurn, Transition
from taskcompendium.grading_result import GradeResult, Outcome
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.spec import McqSpec

from skyrl_gym.envs.aime.verifier import AIMERewardPolicy, AIMEVerifier
from skyrl_gym.envs.cat_count.reward import cat_count_score
from skyrl_gym.envs.gsm8k.utils import compute_score as gsm8k_score
from skyrl_gym.envs.ifeval.utils import compute_score as instruction_score
from skyrl_gym.envs.mcq.utils import extract_mcq_answer
from skyrl_gym.envs.nupa.answers import parse_ground_truth
from skyrl_gym.envs.nupa.verifier import NUPAVerifier
from skyrl_gym.envs.reasoning_gym.scoring import normalize_ground_truth, score_response
from skyrl_gym.task_records import grade_result, graded_transition, rollout_evidence
from skyrl_gym.verification import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)
NUPA_METRICS = ("exact_match", "digit_match", "dlength", "format_valid", "no_answer")
COMPLETED_STOP_REASONS = frozenset({"stop", "complete", "eos", "end_turn"})


def ground_truth(extras: dict[str, Any]) -> Any:
    specification = extras.get("reward_spec") or extras.get("reward_model")
    if not isinstance(specification, Mapping) or "ground_truth" not in specification:
        raise ValueError("The task has no ground truth")
    return specification["ground_truth"]


def grade_gsm8k(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    method = config.get("reward_method", "strict")
    score = (
        0.0
        if method == "final_line" and turn.stop_reason not in COMPLETED_STOP_REASONS
        else gsm8k_score(turn.text, ground_truth(extras), method=method)
    )
    return Transition(done=True, reward=score, grade=GradeResult(Outcome.GRADED, score))


def grade_aime(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    verifier = AIMEVerifier(
        ground_truth=ground_truth(extras),
        evaluation_token_budget=int(config.get("evaluation_token_budget", 8192)),
        strict_box_verify=bool(config.get("strict_box_verify", False)),
    )
    policy = AIMERewardPolicy(
        length_penalty_weight=float(config.get("length_penalty_weight", 0.0)),
        target_length=int(config.get("target_length", 0)),
        max_generation_tokens=int(config.get("max_gen_length", 0)),
        truncated_penalty=float(config.get("truncated_penalty", -2.0)),
        min_response_length=int(config.get("min_response_length", 16)),
    )
    evidence = rollout_evidence(turn)
    verification = verifier.verify(evidence)
    reward, diagnostics = policy.evaluate(evidence, verification)
    metrics = {
        "acc": verification.passed is True,
        "pred": verification.diagnostics["prediction"],
        **verification.diagnostics,
        **diagnostics,
    }
    return graded_transition(turn, verification, reward, metrics)


def grade_cat_count(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    count = int(extras["extra_info"]["n"])
    score = cat_count_score(turn.text, count, stop_reason=turn.stop_reason)
    return Transition(
        done=True,
        reward=score.reward,
        grade=GradeResult(Outcome.GRADED, float(score.exact), passed=score.exact),
        metrics={
            "exact": float(score.exact),
            "has_cat": float(score.cat_unigram_count > 0),
            "junk": float(score.junk_words),
            "truncated": float(score.truncated),
            f"exact_n{count}": float(score.exact),
            f"n_words_n{count}": float(score.n_words),
        },
    )


def grade_mcq(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    expected = str(ground_truth(extras)).strip().upper()
    answer = extract_mcq_answer(turn.text)
    reward = grade_mcq_candidate(McqSpec(expected=expected, options=26), answer or "").reward
    return Transition(done=True, reward=reward, grade=GradeResult(Outcome.GRADED, reward))


def grade_ifeval(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    use_verifyit = bool(config.get("verifyit_enabled", False))
    result = instruction_score(turn.text, ground_truth(extras), verifyit_enabled=use_verifyit)
    reward = result["score"]
    metrics = {key: value for key, value in result.items() if key != "score"}
    verification = (
        VerificationResult.error("Instruction verifier failed", diagnostics=metrics)
        if use_verifyit and result.get("error_type")
        else VerificationResult.verified(reward, passed=result["acc"] if use_verifyit else None, diagnostics=metrics)
    )
    return Transition(done=True, reward=reward, grade=grade_result(verification), metrics=metrics)


def grade_nupa(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    try:
        expected = ground_truth(extras)
        parse_ground_truth(expected)
    except ValueError:
        logger.exception("Invalid NUPA ground truth")
        return Transition(done=True, reward=0.0, grade=GradeResult(Outcome.INFRA_ERROR, None, "Invalid NUPA task"))
    verification = NUPAVerifier(ground_truth=expected).verify(rollout_evidence(turn))
    if verification.status != VerificationStatus.VERIFIED:
        return Transition(
            done=True,
            reward=0.0,
            grade=grade_result(verification),
            metrics={"verifier_error": verification.reason},
        )
    assert verification.score is not None
    return Transition(
        done=True,
        reward=verification.score,
        grade=grade_result(verification),
        metrics={
            "acc": verification.passed is True,
            "pred": verification.diagnostics.get("prediction"),
            **{key: verification.diagnostics[key] for key in NUPA_METRICS},
        },
    )


def grade_reasoning_gym(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    use_verifyit = bool(config.get("verifyit_enabled", False))
    specification = extras.get("reward_model")
    expected = specification.get("ground_truth") if isinstance(specification, Mapping) else None
    try:
        expected = normalize_ground_truth(expected)
    except (TypeError, ValueError):
        logger.exception("Invalid Reasoning Gym ground truth")
        return Transition(
            done=True,
            reward=0.0,
            grade=GradeResult(Outcome.INFRA_ERROR, None, "Invalid Reasoning Gym task"),
            metrics={"verifier_error": "invalid reward_model.ground_truth"},
        )
    try:
        reward = score_response(turn.text, expected, verifyit_enabled=use_verifyit)
    except (RuntimeError, ValueError) as error:
        return Transition(
            done=True,
            reward=0.0,
            grade=GradeResult(
                Outcome.INFRA_ERROR, None, "Verifier failed", diagnostics={"error_type": type(error).__name__}
            ),
        )
    return Transition(done=True, reward=reward, grade=GradeResult(Outcome.GRADED, reward))


def grade_prompt_only(turn: ModelTurn, config: dict, extras: dict) -> Transition:
    return Transition(done=True, reward=0.0, grade=GradeResult(Outcome.GRADED, 0.0))
