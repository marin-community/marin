from taskcompendium.submission import conversation_messages

import json

import pytest
from omegaconf import OmegaConf
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    PlainText,
    Source,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec
from skyrl_gym.source_task import source_task
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, task_input
from skyrl_train.trajectory_runners.types import TrajectoryID, TaskRequestBatch


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 2,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task), teacher_route="arithmetic", data_source="arithmetic"),),
        trajectory_ids=(TrajectoryID("arithmetic", 0),),
    )
    return config, request


@pytest.fixture
def python_tool_task(task_inputs):
    _, request = task_inputs
    task = source_task(
        conversation_messages(request.rows[0].spec.task.context.events),
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "ns_tools_simple_agent",
                    "record_json": json.dumps({"question": "What is 2 + 2?", "expected_answer": "4"}),
                    "request_json": "{}",
                }
            }
        },
        config={},
        environment=NATIVE_ENVIRONMENT,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "python-1",
                "type": "function",
                "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"2 + 2"}'},
            }
        ],
    }
    task = lowered_task(task, "nemotron_ultra", backend="docker")
    return task, message
