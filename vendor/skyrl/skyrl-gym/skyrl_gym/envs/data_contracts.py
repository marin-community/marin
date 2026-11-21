"""Dataset normalization and in-process verifier preflight.

Code preflight requires a supplied Shellbox machine and uses validate_code_example.
Nemotron preflight requires the complete source row and its task session.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from skyrl_gym.envs.aime import utils as aime_utils
from skyrl_gym.envs.aime.verifier import AIMEVerifier
from skyrl_gym.envs.gsm8k import utils as gsm8k_utils
from skyrl_gym.envs.ifeval import utils as ifeval_utils
from skyrl_gym.envs.lcb.livecodebench import normalize_lcb_ground_truth
from skyrl_gym.envs.mcq.utils import extract_mcq_answer
from skyrl_gym.envs.nupa.answers import parse_ground_truth
from skyrl_gym.envs.nupa.verifier import NUPAVerifier
from skyrl_gym.envs.reasoning_gym.scoring import normalize_ground_truth as normalize_reasoning_gym_ground_truth
from skyrl_gym.envs.reasoning_gym.scoring import score_response as score_reasoning_gym_response
from skyrl_gym.envs.text_to_sql import scoring as text_to_sql_scoring
from skyrl_gym.verification import RolloutEvidence

NormalizeGroundTruth = Callable[[Any], str]
IsCorrect = Callable[[str, str], bool]
LCB_PROMPT_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"


@dataclass(frozen=True)
class VerifierDataContract:
    """The public preparation boundary for one verifier environment."""

    env_id: str
    normalize_ground_truth: NormalizeGroundTruth
    is_correct: IsCorrect | None = None
    prompt_instruction: str | None = None

    def validate_example(self, ground_truth: Any, positive_response: str, negative_response: str) -> str:
        """Return normalized ground truth after a two-sided verifier preflight."""
        if self.is_correct is None:
            raise ValueError(f"{self.env_id} preflight requires task execution with its private verifier inputs.")
        normalized = self.normalize_ground_truth(ground_truth)
        if not self.is_correct(positive_response, normalized):
            raise ValueError(f"{self.env_id} positive response does not satisfy its verifier.")
        if self.is_correct(negative_response, normalized):
            raise ValueError(f"{self.env_id} negative response unexpectedly satisfies its verifier.")
        return normalized


def _normalize_aime(ground_truth: Any) -> str:
    normalized = aime_utils.normalize_final_answer(str(ground_truth))
    if not normalized:
        raise ValueError("AIME ground_truth must normalize to a non-empty answer.")
    return normalized


def _aime_is_correct(response: str, ground_truth: str) -> bool:
    return AIMEVerifier(ground_truth).verify(RolloutEvidence(response=response)).passed is True


def _normalize_ifeval(ground_truth: Any) -> str:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValueError("IFeval ground_truth must be valid JSON.") from exc
    if not isinstance(ground_truth, (Mapping, list)):
        raise ValueError("IFeval ground_truth must be a constraint mapping or list of constraint mappings.")
    return ifeval_utils.normalize_ground_truth(ground_truth)


def _ifeval_is_correct(response: str, ground_truth: str) -> bool:
    return ifeval_utils.compute_score(response, ground_truth)["score"] == 1.0


def _reasoning_gym_is_correct(response: str, ground_truth: str) -> bool:
    return score_reasoning_gym_response(response, ground_truth) == 1.0


def _nemotron_ultra_ground_truth(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Nemotron Ultra ground truth must identify a row-selected verifier.")
    return value


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _normalize_gsm8k(ground_truth: Any) -> str:
    normalized = str(ground_truth).strip().replace(",", "").replace("$", "")
    if not normalized:
        raise ValueError("gsm8k ground_truth must be non-empty.")
    return normalized


def _gsm8k_is_correct(response: str, ground_truth: str) -> bool:
    return gsm8k_utils.compute_score(response, ground_truth) > 0


# ---------------------------------------------------------------------------
# text_to_sql (result-set equivalence on a rebuilt in-memory SQLite DB)
# ---------------------------------------------------------------------------

TEXT_TO_SQL_PROMPT_INSTRUCTION = (
    "\nTarget dialect is SQLite. Write exactly one SELECT statement (a leading WITH is allowed) that "
    "answers the question. Return only the query, inside <solution></solution>."
)


# ---------------------------------------------------------------------------
# MCQ (multiple choice exact letter match via \boxed{X})
# ---------------------------------------------------------------------------


def _normalize_mcq(ground_truth: Any) -> str:
    answer = str(ground_truth).strip().upper()
    if not answer:
        raise ValueError("mcq ground_truth must be non-empty.")
    return answer


def _mcq_is_correct(response: str, ground_truth: str) -> bool:
    answer = extract_mcq_answer(response)
    return answer is not None and answer == ground_truth


# ---------------------------------------------------------------------------
# NUPA (representation-sensitive numeric answers, NUPA-Loose semantics)
# ---------------------------------------------------------------------------


def _normalize_nupa(ground_truth: Any) -> str:
    answer, answer_format = parse_ground_truth(ground_truth)
    return json.dumps({"answer": answer, "answer_format": answer_format}, sort_keys=True)


def _nupa_is_correct(response: str, ground_truth: str) -> bool:
    return NUPAVerifier(ground_truth=ground_truth).verify(RolloutEvidence(response=response)).passed is True


# ---------------------------------------------------------------------------
# Preference (schema-only — reward comes from a RM at runtime)
# ---------------------------------------------------------------------------


def _normalize_preference(ground_truth: Any) -> str:
    if not ground_truth:
        raise ValueError("preference ground_truth (chosen response) must be non-empty.")
    return str(ground_truth)


def _preference_is_correct(response: str, ground_truth: str) -> bool:
    return response.strip() == ground_truth.strip()


CONTRACTS = {
    "aime": VerifierDataContract(
        env_id="aime",
        normalize_ground_truth=_normalize_aime,
        is_correct=_aime_is_correct,
        prompt_instruction=aime_utils.BOXED_ANSWER_INSTRUCTION,
    ),
    "ifeval": VerifierDataContract(
        env_id="ifeval",
        normalize_ground_truth=_normalize_ifeval,
        is_correct=_ifeval_is_correct,
    ),
    "gsm8k": VerifierDataContract(
        env_id="gsm8k",
        normalize_ground_truth=_normalize_gsm8k,
        is_correct=_gsm8k_is_correct,
        prompt_instruction="\nThe final answer must appear after ####",
    ),
    "lcb": VerifierDataContract(
        env_id="lcb",
        normalize_ground_truth=normalize_lcb_ground_truth,
        prompt_instruction=LCB_PROMPT_INSTRUCTION,
    ),
    "reasoning_gym": VerifierDataContract(
        env_id="reasoning_gym",
        normalize_ground_truth=normalize_reasoning_gym_ground_truth,
        is_correct=_reasoning_gym_is_correct,
    ),
    "nemotron_ultra": VerifierDataContract(
        env_id="nemotron_ultra",
        normalize_ground_truth=_nemotron_ultra_ground_truth,
    ),
    "mcq": VerifierDataContract(
        env_id="mcq",
        normalize_ground_truth=_normalize_mcq,
        is_correct=_mcq_is_correct,
        prompt_instruction="\nAnswer with the option letter from the given choices. Put your answer in \\boxed{ANSWER}",
    ),
    "nupa": VerifierDataContract(
        env_id="nupa",
        normalize_ground_truth=_normalize_nupa,
        is_correct=_nupa_is_correct,
    ),
    "preference": VerifierDataContract(
        env_id="preference",
        normalize_ground_truth=_normalize_preference,
        is_correct=_preference_is_correct,
    ),
    "text_to_sql": VerifierDataContract(
        env_id="text_to_sql",
        normalize_ground_truth=text_to_sql_scoring.normalize_ground_truth,
        is_correct=text_to_sql_scoring.is_correct,
        prompt_instruction=TEXT_TO_SQL_PROMPT_INSTRUCTION,
    ),
}


def get_data_contract(env_id: str) -> VerifierDataContract:
    """Return the dataset preparation contract for a supported task."""
    contract = CONTRACTS.get(env_id)
    if contract is None:
        raise ValueError(f"Task {env_id!r} has no verifier data contract.")
    return contract
