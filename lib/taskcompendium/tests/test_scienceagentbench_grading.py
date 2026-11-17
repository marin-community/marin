# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from openai import InternalServerError, OpenAI
from shellbox.machine import DockerImage, MachineSpec

from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench_assets.evaluate import evaluate_output
from taskcompendium.importers.scienceagentbench_assets.visual_judge import score_figure
from taskcompendium.importers.scienceagentbench_grading import papyrus_grader
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GradingAttempt,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.task_grading import grade_task

from .grading_machines import GRADER_IMAGE, REPLAY_IMAGE, ReplayMachines, completed_trace

GOLD_ROWS = b'["1", "1.235", "a"]\n["2", "2.346", "b"]\n'


def grade_papyrus_program(tmp_path, program, gold_rows=GOLD_ROWS):
    replay = EnvironmentRequirements(
        docker_image=REPLAY_IMAGE,
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/testbed",
    )
    package = papyrus_grader(
        environment=EnvironmentRequirements(
            docker_image=GRADER_IMAGE,
            command_semantics=CommandSemantics.LINUX_PROCESS,
            working_directory="/testbed",
        ),
        replay_environment=replay,
        gold_rows=gold_rows,
        replay_timeout=10,
        verifier_timeout=30,
    )
    task = TaskSpec(
        id="papyrus-fixture",
        source=Source(dataset="test/papyrus", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Write the filtered table."),)),
        environment_requirements=replay,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        output_paths=("/testbed/solution.py",),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )
    machines = ReplayMachines(tmp_path / "machines")
    result = grade_task(
        TaskSpec.model_validate_json(task.model_dump_json()),
        GradingAttempt(completed_trace(task), {"/testbed/solution.py": program}),
        machine_factory=machines,
        machine_spec=MachineSpec(DockerImage(GRADER_IMAGE)),
    )
    return result, machines.machines


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([[2, 2.3456, "b"], [1, 1.2346, "a"]], 1.0),
        ([[2, 2.3456, "wrong"], [1, 1.2346, "a"]], 0.0),
        ([[1, 1.2346, "a"]], 0.0),
        ([[2, 2.3456, "b"], [1, 1.2346, "a"], [99, 9.0, "extra"]], 1.0),
    ],
)
def test_pickle_replay_preserves_official_order_rounding_and_gold_prefix(tmp_path, rows, expected):
    program = f"""
from pathlib import Path
import pandas as pd
assert not Path('../tests/gold_rows.jsonl').exists()
Path('pred_results').mkdir()
pd.DataFrame({rows!r}, columns=['Activity_ID', 'value', 'label']).to_pickle('pred_results/papyrus_filtered.pkl')
""".encode()
    result, machines = grade_papyrus_program(tmp_path, program)
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result
    assert len(machines) == 2 and all(machine.closed for machine in machines)
    replay, grader = machines
    assert replay.local("/replay/decode_papyrus.py").is_file()
    assert not grader.local("/replay/decode_papyrus.py").exists()
    assert not grader.local("/testbed/pred_results/papyrus_filtered.pkl").exists()
    assert not grader.local("/testbed/solution.py").exists()


def test_pickle_cannot_read_gold_during_decoding(tmp_path):
    program = b"""
from pathlib import Path
import pickle

class ReadGold:
    def __reduce__(self):
        expression = (
            "(__import__('pathlib').Path('../tests/gold_rows.jsonl').read_text(), "
            "__import__('pandas').DataFrame({'Activity_ID': [1], 'value': [42]}))[1]"
        )
        return (eval, (expression,))

Path('pred_results').mkdir()
Path('pred_results/papyrus_filtered.pkl').write_bytes(pickle.dumps(ReadGold()))
"""
    result, machines = grade_papyrus_program(tmp_path, program, b'["1", "42"]\n')
    assert (result.status, result.reward) == (Outcome.GRADED, 0.0), result
    assert len(machines) == 1 and machines[0].closed


def test_malformed_trusted_gold_is_infrastructure_failure(tmp_path):
    program = b"""
from pathlib import Path
import pandas as pd
Path('pred_results').mkdir()
pd.DataFrame({'Activity_ID': [1], 'value': [42]}).to_pickle('pred_results/papyrus_filtered.pkl')
"""
    result, machines = grade_papyrus_program(tmp_path, program, b"broken gold\n")
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None), result
    assert len(machines) == 2 and all(machine.closed for machine in machines)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("return 1, 'passed'", 1.0),
        ("return 0.8, 'not task success'", 0.0),
        ("return False", 0.0),
        ("raise ValueError('invalid candidate shape')", 0.0),
    ],
)
def test_source_evaluator_reports_binary_task_success(tmp_path, body, expected):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(f"def eval():\n    {body}\n")
    primary = tmp_path / "candidate.csv"
    primary.write_text("candidate output")
    verdict = evaluate_output(evaluator, primary, tmp_path)
    assert (verdict["status"], verdict["reward"]) == ("scored", expected)


def test_missing_secondary_output_is_scored_but_missing_gold_is_ungraded(tmp_path):
    candidate_directory = tmp_path / "pred_results"
    candidate_directory.mkdir()
    primary = candidate_directory / "primary.csv"
    primary.write_text("present")
    evaluator = tmp_path / "evaluate.py"
    secondary = candidate_directory / "secondary.csv"
    evaluator.write_text(f"from pathlib import Path\ndef eval():\n    return Path({str(secondary)!r}).read_text()\n")
    verdict = evaluate_output(evaluator, primary, candidate_directory)
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)
    gold = tmp_path / "gold.csv"
    evaluator.write_text(f"from pathlib import Path\ndef eval():\n    return Path({str(gold)!r}).read_text()\n")
    with pytest.raises(FileNotFoundError):
        evaluate_output(evaluator, primary, candidate_directory)


def test_missing_evaluator_dependency_is_ungraded(tmp_path):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("import nonexistent_scienceagentbench_test_dependency\ndef eval():\n    return 1, 'ok'\n")
    primary = tmp_path / "candidate.csv"
    primary.write_text("present")
    with pytest.raises(ModuleNotFoundError):
        evaluate_output(evaluator, primary, tmp_path)


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["[FINAL SCORE]: 90", "no score marker", "[FINAL SCORE]: 90"], 60.0),
        (["", "[FINAL SCORE]: 59", "[FINAL SCORE]: 100"], 53.0),
        (["[FINAL SCORE]: 9999", "[FINAL SCORE]: 30 then [FINAL SCORE]: 90", "[FINAL SCORE]: 0"], 343.0),
    ],
)
def test_visual_judge_preserves_wire_protocol_and_official_zero_rule(texts, expected):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-visual",
                "created": 0,
                "object": "chat.completion",
                "model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": texts[len(requests) - 1]},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        responses, score = score_figure(client, "pinned prompt fixture", "cHJlZA==", "Z29sZA==")
    assert (responses, score) == (texts, expected)
    assert (
        requests
        == [
            {
                "model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "pinned prompt fixture"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,cHJlZA=="}},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Z29sZA=="}},
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1000,
                "n": 1,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
        ]
        * 3
    )


@pytest.mark.parametrize("indices,texts", [([], []), ([0, 1], ["", ""]), ([0, 0, 2], ["", "", ""]), ([0], [None])])
def test_visual_judge_invalid_completions_remain_ungraded(indices, texts):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "test-incomplete",
                "created": 0,
                "object": "chat.completion",
                "model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "choices": [
                    {"index": index, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    for index, text in zip(indices, texts, strict=True)
                ],
            },
        )

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(RuntimeError):
            score_figure(client, "prompt fixture", "cHJlZA==", "Z29sZA==")


@pytest.mark.parametrize("choices", [None, [{"index": 0, "finish_reason": "stop", "message": None}]])
def test_visual_judge_malformed_api_response_is_not_a_candidate_failure(choices):
    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "test-malformed",
                "created": 0,
                "object": "chat.completion",
                "model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "choices": choices,
            },
        )

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(RuntimeError):
            score_figure(client, "prompt fixture", "cHJlZA==", "Z29sZA==")


@pytest.mark.parametrize("failed_request", [1, 3])
def test_visual_judge_service_failure_propagates_without_hidden_retries(failed_request):
    requests = []
    recorded = []

    def respond(request):
        requests.append(request)
        if len(requests) == failed_request:
            return httpx.Response(503, json={"error": {"message": "unavailable", "type": "server_error"}})
        return httpx.Response(
            200,
            json={
                "id": "test-partial",
                "created": 0,
                "object": "chat.completion",
                "model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "[FINAL SCORE]: 100"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(InternalServerError):
            score_figure(client, "prompt fixture", "cHJlZA==", "Z29sZA==", record_response=recorded.append)
    assert len(requests) == failed_request
    assert len(recorded) == failed_request - 1
