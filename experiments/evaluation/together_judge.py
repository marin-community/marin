# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-local adapter for the frozen Evalchemy judge transport."""

from eval.execution.judge import JudgeBackend
from eval.graders.answer_equivalence import JudgeLabel
from eval.runtime.pool import RetryPolicy
from micro210_judge_config import JudgeConfig


class TogetherJudgeBackend(JudgeBackend):
    def __init__(self, config, session, *, policy: dict):
        super().__init__(config, session)
        self.policy = JudgeConfig(**policy)

    async def execute(self, request) -> str:
        parameters = dict(request.generation)
        parameters.pop("do_sample", None)
        parameters.pop("max_gen_toks", None)
        parameters["structured_outputs"] = {"choice": [label.value for label in JudgeLabel]}
        messages = [{"role": "user", "content": request.prompt}] if isinstance(request.prompt, str) else request.prompt
        retry_policy = RetryPolicy(
            attempt_timeout=self.policy.request_timeout,
            budget=self.policy.transport_budget,
            max_attempts=self.policy.transport_attempts,
        )
        async with self._policy_slots:
            for tokens in self.policy.token_budgets:
                options = self.policy.completion_parameters({**parameters, "max_tokens": tokens})
                result = await super()._completion(messages, options, retry_policy)
                if result.strip():
                    JudgeLabel(result.strip().lower())
                    return result
        raise ValueError("Together returned empty judgments at every policy token budget")
