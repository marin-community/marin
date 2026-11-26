"""Preserve per-example thinking mode at the rollout session boundary."""

import pytest

from skyrl_gym.answer_tasks import grade_prompt_only
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_train.dataset.tasks import source_row_task


@pytest.mark.asyncio
@pytest.mark.parametrize('thinking', [True, False])
async def test_prompt_only_preserves_thinking_option(thinking):
    task = source_row_task(
        {'prompt': [{'role': 'user', 'content': 'Tell me a story.'}],
         'env_class': 'prompt_only',
         'chat_template_kwargs': {'enable_thinking': thinking}},
        0, source_name='thinking-regression', environment_configs={'session': {'max_turns': 1, 'model_turn_timeout': None, 'tool_turn_timeout': 60, 'total_turn_timeout': None, 'attempt_timeout': None, 'verifier_timeout': None, 'cleanup_timeout': 30, 'max_total_tokens': None}},
    )
    session = AnswerTaskSession(task, None, grader=grade_prompt_only)
    try:
        prompt = await session.prepare()
        assert prompt.options['chat_template_kwargs']['enable_thinking'] is thinking
        assert prompt.messages == ({'role': 'user', 'content': 'Tell me a story.'},)
    finally:
        await session.close()
