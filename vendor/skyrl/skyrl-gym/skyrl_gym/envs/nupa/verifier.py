"""Verifyit-backed verifier for NUPA-Loose representation-sensitive grading."""

from __future__ import annotations

from dataclasses import dataclass

from verifyit.adapters.evalchemy_nupa import grade_nupa_answer
from verifyit.grade import Status

from skyrl_gym.envs.nupa.answers import extract_answer, full_answer, digit_parts, parse_ground_truth
from skyrl_gym.verification import RolloutEvidence, VerificationResult


@dataclass(frozen=True)
class NUPAVerifier:
    """Grade a response with the NUPA5K-Loose exact-match metric.

    Reward is the eval's primary metric (``exact_match``); the eval's remaining
    per-sample metrics ride along as diagnostics.
    """

    ground_truth: str

    def verify(self, evidence: RolloutEvidence) -> VerificationResult:
        answer, answer_format = parse_ground_truth(self.ground_truth)
        verdict = grade_nupa_answer(
            evidence.response,
            answer,
            answer_format,
            extract_answer=extract_answer,
            full_answer=full_answer,
            digit_parts=digit_parts,
        )
        if verdict.status is not Status.SCORED:
            return VerificationResult.error(
                "invalid NUPA ground truth", diagnostics={"reason": verdict.detail.get("reason")}
            )
        return VerificationResult.verified(
            verdict.reward,
            passed=verdict.reward > 0,
            diagnostics={"prediction": verdict.detail.get("extracted"), **verdict.detail["metrics"]},
        )
