from __future__ import annotations

from dataclasses import dataclass

from skyrl_gym.envs.aime.utils import compute_length_penalty, verify
from skyrl_gym.verification import RewardResult, RolloutEvidence, VerificationResult


@dataclass(frozen=True)
class AIMEVerifier:
    """Verify an AIME response and report evaluation-budget diagnostics."""

    ground_truth: str
    evaluation_token_budget: int = 8192
    strict_box_verify: bool = False

    def __post_init__(self) -> None:
        if self.evaluation_token_budget <= 0:
            raise ValueError("evaluation_token_budget must be positive")

    def verify(self, evidence: RolloutEvidence) -> VerificationResult:
        response = evidence.response or ""
        correct, prediction = verify(response[-300:], self.ground_truth, self.strict_box_verify)
        generated_tokens = evidence.generated_token_count
        over_budget = generated_tokens is not None and generated_tokens > self.evaluation_token_budget
        parseable_answer = prediction is not None and str(prediction).strip() not in {"", "[INVALID]"}
        return VerificationResult.verified(
            1.0 if correct else -1.0,
            passed=bool(correct),
            diagnostics={
                "prediction": prediction,
                "generated_token_count": generated_tokens,
                "evaluation_token_budget": self.evaluation_token_budget,
                "over_evaluation_budget": over_budget,
                "parseable_answer": parseable_answer,
                "answered_within_evaluation_budget": parseable_answer and not over_budget,
            },
        )


@dataclass(frozen=True)
class AIMERewardPolicy:
    """Map an AIME verdict and its evidence to the configured optimization reward."""

    length_penalty_weight: float = 0.0
    target_length: int = 0
    max_generation_tokens: int = 0
    truncated_penalty: float = -2.0
    min_response_length: int = 16

    def evaluate(
        self,
        evidence: RolloutEvidence,
        verification: VerificationResult,
    ) -> tuple[RewardResult, dict[str, object]]:
        if verification.score is None or verification.passed is None:
            raise ValueError("AIME reward policy requires a verified correctness verdict")
        generation_budget = self.max_generation_tokens or int(evidence.metadata.get("generation_token_budget", 0))
        reward, diagnostics = compute_length_penalty(
            correct=verification.passed,
            response_length=evidence.generated_token_count,
            truncated=evidence.stop_reason == "length",
            length_penalty_weight=self.length_penalty_weight,
            target_length=self.target_length,
            max_gen_length=generation_budget,
            truncated_penalty=self.truncated_penalty,
            min_response_length=self.min_response_length,
        )
        return (
            RewardResult(
                unshaped_reward=verification.score,
                optimization_reward=reward,
                components={} if reward == verification.score else {"length": reward - verification.score},
            ),
            diagnostics,
        )
