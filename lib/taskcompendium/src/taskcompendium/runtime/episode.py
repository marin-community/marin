# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive an actor through executed tools, preserving observations and artifacts."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from taskcompendium.models import (
    AssistantMessage,
    ConversationEvent,
    FunctionCall,
    TaskSpec,
    TextMessage,
    ToolResult,
    require_resolved_environment,
)
from taskcompendium.runtime.models import (
    Actor,
    ActorTask,
    Environment,
    EnvironmentFactory,
    RolloutRecord,
    RuntimeEvidence,
    Termination,
)
from taskcompendium.runtime.resources import inline_resource


@dataclass
class ScriptedActor:
    """A deterministic control trajectory, consumed once per fresh episode."""

    responses: Sequence[AssistantMessage]
    index: int = field(default=0, init=False)

    def respond(self, task: ActorTask, events: Sequence[ConversationEvent]) -> AssistantMessage:
        del task, events
        if self.index == len(self.responses):
            return TextMessage(role="assistant", content="Done.")
        response = self.responses[self.index]
        self.index += 1
        return response


async def run_episode(
    task: TaskSpec,
    actor: Actor,
    factory: EnvironmentFactory,
    *,
    max_steps: int,
    control: str,
) -> RolloutRecord:
    """Run a fresh episode; a final message completes it, while exhaustion truncates it."""
    if max_steps <= 0:
        raise ValueError("A positive turn budget is required")
    require_resolved_environment(task.environment_requirements)
    events = list(task.context.events)
    environment: Environment | None = None
    termination = Termination.STEP_LIMIT
    evidence = RuntimeEvidence({}, "{}")
    detail = ""
    try:
        environment = await factory.create(task)
        tools = environment.tools
        public = ActorTask(task.id, tools)
        for _ in range(max_steps):
            response = actor.respond(public, events)
            if isinstance(response, TextMessage) and response.role != "assistant":
                raise ValueError("An actor must return an assistant message")
            events.append(response)
            if isinstance(response, TextMessage):
                termination = Termination.FINAL_MESSAGE
                break
            for call in response.calls:
                if call.name not in {tool.name for tool in tools}:
                    observation = json.dumps({"error": "Tool is not advertised"})
                else:
                    observation = await environment.step(FunctionCall(name=call.name, arguments=call.arguments))
                events.append(ToolResult(call_id=call.call_id, content=observation))
        evidence = await environment.evidence()
    except (OSError, RuntimeError) as error:
        termination, detail = Termination.INFRA_ERROR, str(error)
    finally:
        if environment is not None:
            try:
                await environment.close()
            except (OSError, RuntimeError) as error:
                termination, detail = Termination.INFRA_ERROR, str(error)
    return RolloutRecord(
        task_id=task.id,
        control=control,
        events=tuple(events),
        termination=termination,
        artifacts=tuple(inline_resource(path.removeprefix("/"), data) for path, data in sorted(evidence.files.items())),
        state_json=evidence.state_json,
        detail=detail,
    )
