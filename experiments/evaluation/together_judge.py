# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-local adapter for the frozen Evalchemy judge transport."""


from eval.execution.judge import JudgeBackend
from eval.runtime.pool import RetryPolicy
from micro210_judge_config import JudgeConfig


class TogetherJudgeBackend(JudgeBackend):
    def __init__(self, config, session, *, policy: dict):
        super().__init__(config, session)
        self.policy = JudgeConfig(**policy)

    async def _completion(self, messages, parameters, retry_policy) -> str:
        parameters = self.policy.completion_parameters(parameters)
        retry_policy = RetryPolicy(
            attempt_timeout=self.policy.request_timeout,
            budget=self.policy.transport_budget,
            max_attempts=self.policy.transport_attempts,
        )
        return await super()._completion(messages, parameters, retry_policy)

    async def execute(self, request) -> str:
        parameters = dict(request.generation)
        parameters.pop("do_sample", None)
        parameters.pop("max_gen_toks", None)
        parameters["structured_outputs"] = {"choice": ["correct", "incorrect", "not_attempted"]}
        messages = [{"role": "user", "content": request.prompt}] if isinstance(request.prompt, str) else request.prompt
        async with self._policy_slots:
            for tokens in self.policy.token_budgets:
                result = await self._completion(messages, {**parameters, "max_tokens": tokens}, None)
                if result.strip():
                    if result.strip().lower() not in parameters["structured_outputs"]["choice"]:
                        raise ValueError("Together returned an invalid judge label")
                    return result
        raise ValueError("Together returned empty judgments at every policy token budget")
