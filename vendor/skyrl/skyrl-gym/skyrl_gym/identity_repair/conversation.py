"""Fixed user follow-ups for evaluating and training conversational recovery."""

from rolloutengine.models import ModelTurn, Prompt, TaskRolloutSpec
from shellbox.machine import Machine
from skyrl_gym.source_task import session_parameters
from skyrl_gym.task_records import TaskTransition, episode_prompt
from taskcompendium.grading_result import GradeResult, Outcome


class ScriptedConversationSession:
    def __init__(self, lowered: TaskRolloutSpec, machine: Machine | None):
        self.task = lowered.task
        self.followups = tuple(session_parameters(self.task).extras["followup_prompts"])
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in self.followups):
            raise ValueError("Scripted follow-ups must be nonempty user messages")
        if lowered.limits.max_turns < len(self.followups) + 1:
            raise ValueError("Turn limit truncates the scripted conversation")
        self.index = 0

    async def prepare(self) -> Prompt:
        return episode_prompt(self.task)

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        if self.index == len(self.followups):
            return TaskTransition(done=True)
        prompt = self.followups[self.index]
        self.index += 1
        return TaskTransition(done=False, observations=({"role": "user", "content": prompt},))

    async def grade(self, messages, *, stop_reason, outputs) -> GradeResult:
        return GradeResult(Outcome.UNAVAILABLE, None, "Conversation requires independent group grading")

    async def close(self) -> None:
        pass
