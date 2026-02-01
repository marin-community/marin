# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Validation strategies for the Self-Instill pipeline.

This module provides validation components that can be used to filter
and verify generated responses. Validation strategies include:

1. StaticCheckStrategy: Fast, non-LLM validation (format checks)
2. CycleConsistencyStrategy: Verify answer addresses the question
3. FactualErrorStrategy: Check for factual/logical errors
4. TotalCorrectnessStrategy: Comprehensive correctness verification

Each strategy can be used independently or combined in a validation pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from experiments.self_instill.prompts import (
    CYCLE_QUESTION_GENERATION_PROMPT,
    CYCLE_COMPARISON_PROMPT,
    FACTUAL_ERROR_PROMPT,
    TOTAL_CORRECTNESS_PROMPT,
    RELEVANCE_PROMPT,
)


# =============================================================================
# DECISION EXTRACTION
# =============================================================================

# Pattern to match [[Y]] or [[N]] decisions from LLM output
_DECISION_RE = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)

# Pattern to match \boxed{...} in answers
_BOXED_RE = re.compile(r"\\boxed\{[^}]+\}")


def extract_decision(text: str) -> bool:
    """
    Robustly extract a Y/N decision from LLM judge output.

    The extraction follows a priority order:
    1. Last occurrence of [[Y]] / [[N]] (preferred format)
    2. Last occurrence of a standalone Y/N token
    3. Last occurrence of YES/NO

    Args:
        text: The LLM output to parse

    Returns:
        True for Y/YES decisions, False otherwise
    """
    if not text:
        return False

    # Priority 1: [[Y]] / [[N]] (prefer the last one)
    matches = list(_DECISION_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper() == "Y"

    # Priority 2: Standalone Y/N token (common when small models ignore brackets)
    yn_tokens = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.strip(), flags=re.IGNORECASE)
    if yn_tokens:
        return yn_tokens[-1].upper() == "Y"

    # Priority 3: YES/NO
    yesno = re.findall(r"\b(YES|NO)\b", text.strip(), flags=re.IGNORECASE)
    if yesno:
        return yesno[-1].upper() == "YES"

    return False


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Result from a validation strategy."""
    is_accepted: bool
    strategy_name: str
    details: dict[str, Any] | None = None


class ValidationStrategy(ABC):
    """Base class for all validation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass

    @abstractmethod
    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Validate an answer for a given question.

        Args:
            question: The original question/prompt
            answer: The generated answer to validate
            llm_fn: Optional callable for LLM inference.
                   Signature: llm_fn(prompt: str, num_samples: int = 1) -> list[str]

        Returns:
            ValidationResult with acceptance decision and details
        """
        pass


# =============================================================================
# STATIC CHECK STRATEGY
# =============================================================================

class StaticCheckStrategy(ValidationStrategy):
    """
    Static validation without LLM calls.

    Performs fast, rule-based checks:
    - Verifies presence of \\boxed{...} in the answer
    - Checks for non-English text (non-ASCII letters)

    This is typically the first validation step as it's computationally cheap.
    """

    def __init__(
        self,
        require_boxed: bool = True,
        reject_non_english: bool = True,
    ):
        """
        Initialize static check strategy.

        Args:
            require_boxed: If True, answer must contain \\boxed{...}
            reject_non_english: If True, reject answers with non-ASCII letters
        """
        self.require_boxed = require_boxed
        self.reject_non_english = reject_non_english

    @property
    def name(self) -> str:
        return "static_check"

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Perform static validation checks.

        Args:
            question: The original question (not used in static checks)
            answer: The generated answer to validate
            llm_fn: Not used for static checks

        Returns:
            ValidationResult indicating pass/fail
        """
        has_boxed = bool(_BOXED_RE.search(answer))
        has_non_english = any(ch.isalpha() and ord(ch) > 127 for ch in answer)

        is_accepted = True
        if self.require_boxed and not has_boxed:
            is_accepted = False
        if self.reject_non_english and has_non_english:
            is_accepted = False

        return ValidationResult(
            is_accepted=is_accepted,
            strategy_name=self.name,
            details={
                "has_boxed": has_boxed,
                "has_non_english": has_non_english,
            },
        )


# =============================================================================
# CYCLE CONSISTENCY STRATEGY
# =============================================================================

class CycleConsistencyStrategy(ValidationStrategy):
    """
    Validate answer relevance using cycle consistency.

    This strategy verifies that an answer actually addresses the original
    question by:
    1. Generating a question from the answer ("What question does this answer?")
    2. Comparing the inferred question with the original

    If the inferred question matches the original, the answer is likely relevant.
    """

    def __init__(self, num_samples: int = 3, require_unanimous: bool = True):
        """
        Initialize cycle consistency strategy.

        Args:
            num_samples: Number of samples for each LLM call
            require_unanimous: If True, all samples must agree for acceptance
        """
        self.num_samples = num_samples
        self.require_unanimous = require_unanimous

    @property
    def name(self) -> str:
        return "cycle_consistency"

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Validate using cycle consistency.

        Args:
            question: The original question
            answer: The generated answer to validate
            llm_fn: Callable for LLM inference (required)
                   Signature: llm_fn(prompt: str, num_samples: int = 1) -> list[str]

        Returns:
            ValidationResult indicating pass/fail
        """
        if llm_fn is None:
            raise ValueError("CycleConsistencyStrategy requires llm_fn")

        # Step 1: Generate inferred questions from the answer
        gen_prompt = CYCLE_QUESTION_GENERATION_PROMPT.format(answer=answer.strip())
        inferred_questions = llm_fn(gen_prompt, num_samples=self.num_samples)

        # Step 2: Compare each inferred question with original
        decisions = []
        for inferred_q in inferred_questions:
            # Clean up the inferred question (take first line only)
            inferred_q_clean = inferred_q.strip().splitlines()[0].strip() if inferred_q else ""

            comp_prompt = CYCLE_COMPARISON_PROMPT.format(
                original_question=question.strip(),
                inferred_question=inferred_q_clean,
            )
            evaluation = llm_fn(comp_prompt, num_samples=1)[0]
            decisions.append(extract_decision(evaluation))

        # Determine acceptance based on voting
        if self.require_unanimous:
            is_accepted = all(decisions) if decisions else False
        else:
            is_accepted = sum(decisions) > len(decisions) / 2 if decisions else False

        return ValidationResult(
            is_accepted=is_accepted,
            strategy_name=self.name,
            details={
                "inferred_questions": inferred_questions,
                "decisions": decisions,
            },
        )


# =============================================================================
# FACTUAL ERROR STRATEGY
# =============================================================================

class FactualErrorStrategy(ValidationStrategy):
    """
    Check for factual and logical errors in the answer.

    This strategy prompts an LLM to identify:
    - Wrong math/arithmetic
    - Contradictions with question constraints
    - Incorrect factual claims
    - Invalid logic that changes conclusions
    """

    def __init__(self, num_samples: int = 3, require_unanimous: bool = True):
        """
        Initialize factual error strategy.

        Args:
            num_samples: Number of samples for voting
            require_unanimous: If True, all samples must agree for acceptance
        """
        self.num_samples = num_samples
        self.require_unanimous = require_unanimous

    @property
    def name(self) -> str:
        return "factual_error"

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Validate for factual/logical errors.

        Args:
            question: The original question
            answer: The generated answer to validate
            llm_fn: Callable for LLM inference (required)

        Returns:
            ValidationResult indicating pass/fail
        """
        if llm_fn is None:
            raise ValueError("FactualErrorStrategy requires llm_fn")

        prompt = FACTUAL_ERROR_PROMPT.format(
            question=question.strip(),
            answer=answer.strip(),
        )
        evaluations = llm_fn(prompt, num_samples=self.num_samples)
        decisions = [extract_decision(e) for e in evaluations]

        if self.require_unanimous:
            is_accepted = all(decisions) if decisions else False
        else:
            is_accepted = sum(decisions) > len(decisions) / 2 if decisions else False

        return ValidationResult(
            is_accepted=is_accepted,
            strategy_name=self.name,
            details={
                "evaluations": evaluations,
                "decisions": decisions,
            },
        )


# =============================================================================
# TOTAL CORRECTNESS STRATEGY
# =============================================================================

class TotalCorrectnessStrategy(ValidationStrategy):
    """
    Comprehensive correctness validation.

    This is the strictest validation strategy, requiring:
    1. Correct final result
    2. Sound reasoning without misleading mistakes
    3. Complete coverage of all question parts
    """

    def __init__(self, num_samples: int = 3, require_unanimous: bool = True):
        """
        Initialize total correctness strategy.

        Args:
            num_samples: Number of samples for voting
            require_unanimous: If True, all samples must agree for acceptance
        """
        self.num_samples = num_samples
        self.require_unanimous = require_unanimous

    @property
    def name(self) -> str:
        return "total_correctness"

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Validate for total correctness.

        Args:
            question: The original question
            answer: The generated answer to validate
            llm_fn: Callable for LLM inference (required)

        Returns:
            ValidationResult indicating pass/fail
        """
        if llm_fn is None:
            raise ValueError("TotalCorrectnessStrategy requires llm_fn")

        prompt = TOTAL_CORRECTNESS_PROMPT.format(
            question=question.strip(),
            answer=answer.strip(),
        )
        evaluations = llm_fn(prompt, num_samples=self.num_samples)
        decisions = [extract_decision(e) for e in evaluations]

        if self.require_unanimous:
            is_accepted = all(decisions) if decisions else False
        else:
            is_accepted = sum(decisions) > len(decisions) / 2 if decisions else False

        return ValidationResult(
            is_accepted=is_accepted,
            strategy_name=self.name,
            details={
                "evaluations": evaluations,
                "decisions": decisions,
            },
        )


# =============================================================================
# RELEVANCE STRATEGY
# =============================================================================

class RelevanceStrategy(ValidationStrategy):
    """
    Basic relevance check for answers.

    Verifies that the answer directly addresses the question's main request.
    This is a lighter-weight alternative to cycle consistency.
    """

    def __init__(self, num_samples: int = 3, require_unanimous: bool = True):
        """
        Initialize relevance strategy.

        Args:
            num_samples: Number of samples for voting
            require_unanimous: If True, all samples must agree for acceptance
        """
        self.num_samples = num_samples
        self.require_unanimous = require_unanimous

    @property
    def name(self) -> str:
        return "relevance"

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> ValidationResult:
        """
        Validate answer relevance.

        Args:
            question: The original question
            answer: The generated answer to validate
            llm_fn: Callable for LLM inference (required)

        Returns:
            ValidationResult indicating pass/fail
        """
        if llm_fn is None:
            raise ValueError("RelevanceStrategy requires llm_fn")

        prompt = RELEVANCE_PROMPT.format(
            question=question.strip(),
            answer=answer.strip(),
        )
        evaluations = llm_fn(prompt, num_samples=self.num_samples)
        decisions = [extract_decision(e) for e in evaluations]

        if self.require_unanimous:
            is_accepted = all(decisions) if decisions else False
        else:
            is_accepted = sum(decisions) > len(decisions) / 2 if decisions else False

        return ValidationResult(
            is_accepted=is_accepted,
            strategy_name=self.name,
            details={
                "evaluations": evaluations,
                "decisions": decisions,
            },
        )


# =============================================================================
# VALIDATION PIPELINE
# =============================================================================

class ValidationPipeline:
    """
    Compose multiple validation strategies into a pipeline.

    The pipeline runs strategies in order, short-circuiting on the first failure.
    This allows efficient validation by running cheap checks (like static) first.
    """

    def __init__(self, strategies: list[ValidationStrategy]):
        """
        Initialize validation pipeline.

        Args:
            strategies: List of validation strategies to run in order
        """
        self.strategies = strategies

    def validate(
        self,
        question: str,
        answer: str,
        llm_fn: callable | None = None,
    ) -> tuple[bool, list[ValidationResult]]:
        """
        Run all validation strategies in sequence.

        Short-circuits on first failure for efficiency.

        Args:
            question: The original question
            answer: The generated answer to validate
            llm_fn: Callable for LLM inference (required for LLM-based strategies)

        Returns:
            Tuple of (is_accepted, list of ValidationResults)
        """
        results = []
        for strategy in self.strategies:
            result = strategy.validate(question, answer, llm_fn)
            results.append(result)
            if not result.is_accepted:
                # Short-circuit on failure
                return False, results
        return True, results


def create_default_validation_pipeline(
    num_samples: int = 3,
    require_unanimous: bool = True,
    include_static: bool = True,
    include_cycle: bool = True,
    include_fact: bool = True,
    include_correctness: bool = True,
) -> ValidationPipeline:
    """
    Create a default validation pipeline with configurable stages.

    Args:
        num_samples: Number of samples for LLM-based validation
        require_unanimous: If True, all samples must agree for acceptance
        include_static: Include static format checks
        include_cycle: Include cycle consistency validation
        include_fact: Include factual error checking
        include_correctness: Include total correctness verification

    Returns:
        Configured ValidationPipeline
    """
    strategies: list[ValidationStrategy] = []

    if include_static:
        strategies.append(StaticCheckStrategy())
    if include_cycle:
        strategies.append(CycleConsistencyStrategy(num_samples, require_unanimous))
    if include_fact:
        strategies.append(FactualErrorStrategy(num_samples, require_unanimous))
    if include_correctness:
        strategies.append(TotalCorrectnessStrategy(num_samples, require_unanimous))

    return ValidationPipeline(strategies)
