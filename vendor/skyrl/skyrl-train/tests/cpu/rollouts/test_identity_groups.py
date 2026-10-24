import json as json_codec
from dataclasses import replace

import pytest
import requests

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge
from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.grading import ProductionIdentityJudge
from skyrl_gym.identity_repair.reward import IdentityFact, RewardWeights
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import source_row_task
from tests.cpu.task_specs import session_limits
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.types import TokenProvenance, BatchMetadata
from skyrl_train.trajectory_runners.trajectory_retention import build_trajectory_records
from skyrl_train.trajectory_runners.trajectory_retention_config import TrajectoryRetentionConfig
from skyrl_train.rollouts.finalization import finalize_trajectory_batch
from skyrl_train.rollouts.identity_groups import IdentityRepairGroups
from skyrl_train.rollouts.repair_branching import TeacherRepairSpec
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from tests.cpu.rollouts.test_repair_branching import CharTokenizer, FakeClient, original_rollout
from tests.cpu.task_specs import lowered_task, task_input, text_task
from tests.cpu.identity_fixtures import Tokenizer, config


@pytest.mark.asyncio
@pytest.mark.parametrize("via_worker", [False, True])
async def test_four_member_group_preserves_ids_independent_rewards_and_projection(monkeypatch, via_worker):
    graded_answers = []

    def respond(url, *, headers, json: dict, timeout):
        payload = json_codec.loads(json["messages"][-1]["content"])
        wrong = "qwen" in payload["answer"]
        if "claim" in payload:
            assessment = {"incorrect": wrong, "ambiguous": False, "explanation": "attribution checked"}
        else:
            assert payload["conversation_before_answer"][-1]["role"] == "user"
            assert payload["conversation_before_answer"][-1]["content"] == payload["request"]
            assert payload["task_completion_criteria"] == "Give the assistant name; no biography required."
            graded_answers.append(payload["answer"])
            assessment = {"task_completed": not wrong, "fact_correctness": {"name": float(not wrong)},
                          "unsupported_claims": int(wrong), "inappropriate_refusal": False,
                          "unsolicited_identity": False, "explanation": "answer checked"}
        response = requests.Response()
        response.status_code = 200
        response._content = encode_response(assessment)
        return response

    monkeypatch.setattr(requests, "post", respond)
    row = task_input(lowered_task(text_task(prompt="What is your name?")), metric_class="fixture")
    row = replace(row, metadata=replace(row.metadata, chat_template_kwargs={"enable_thinking": True}))
    request = TaskRequestBatch(rows=(row,)*4, trajectory_ids=tuple(TrajectoryID("prompt-1", i) for i in range(4)),
                               sampling_params={"logprobs": 0})
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    judge = ProductionIdentityJudge(OpenAIJudge("http://judge/v1", "judge", strict_completion=True), "pin", 512)
    teacher = FakeClient(["I am Marin Snowball"])
    groups = IdentityRepairGroups(CharTokenizer(), manifest, judge, teacher, FakeClient([]),
                                  TeacherRepairSpec({"temperature": 1.0}, {"temperature": 1.0}, 8192),
                                  {row.spec.task.id: ((IdentityFact.NAME,),)}, RewardWeights(),
                                  {row.spec.task.id: ("Give the assistant name; no biography required.",)})
    generated = []

    async def generate(index):
        generated.append(index)
        text = "my name is qwen" if index == 0 else "I am Marin Snowball"
        return replace(original_rollout([text]), task_id=row.spec.task.id)

    if via_worker:
        class Client:
            async def generate(self, model_request):
                assert model_request['chat_completion_params'][0]['chat_template_kwargs']['enable_thinking']
                record = await generate([0, 1, 3][len(generated)])
                turn = record.steps[0].turn
                return {'token_provenance': TokenProvenance.ENGINE,
                        'assistant_messages': [turn.message], 'responses': [turn.text],
                        'prompt_ids': [list(turn.prompt_token_ids)], 'response_ids': [list(turn.response_token_ids)],
                        'response_logprobs': [list(turn.logprobs)], 'stop_reasons': ['stop']}
        cfg = get_default_config().generator
        cfg.sampling_params.logprobs = 0
        worker = TaskRolloutWorker(cfg, WholeTaskProjection(cfg, Tokenizer()), Client(), {},
                                  shutdown_timeout=10, identity_groups=groups)
        try:
            batch = await worker.run(request)
        finally:
            await worker.shutdown()
    else:
        rollouts = await groups.collect(request, generate)
        batch = WholeTaskProjection(config(), Tokenizer()).project(rollouts, request)
        batch = await finalize_trajectory_batch(request, batch, config(), None)
        assert rollouts[0].metrics["identity_repair_pair"]["pair_id"] == rollouts[2].metrics["identity_repair"]["pair_id"]
        assert rollouts[2].metrics["identity_reward_records"][0]["reward"]["judge_version"] == judge.version
    assert teacher.calls[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert generated == [0, 1, 3]
    assert len(graded_answers) == 4
    assert batch["trajectory_ids"] == list(request.trajectory_ids)
    assert batch["rewards"] == [0.0, 1.25, 1.25, 1.25]
    assert batch["behavior_sources"][2].all()
    assert not batch["behavior_sources"][0].any()


def encode_response(assessment):
    return json_codec.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json_codec.dumps(assessment)}}]}).encode()


@pytest.mark.asyncio
async def test_repair_detects_native_reasoning_claim_omitted_from_displayed_answer():
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    teacher = FakeClient(["I am Marin Snowball.<|end_think|>Marin Snowball."])
    groups = IdentityRepairGroups(CharTokenizer(), manifest, None, teacher, FakeClient([]),
                                  TeacherRepairSpec({"temperature": 1.0}, {"temperature": 1.0}, 8192),
                                  {}, RewardWeights())
    sampled = "<|start_think|>my name is qwen.<|end_think|>Hello."
    original = original_rollout([sampled])
    step = original.steps[0]
    displayed_turn = replace(step.turn, text="Hello.", message={"role": "assistant", "content": "Hello."})
    original = replace(original, steps=(replace(step, turn=displayed_turn),))
    pair = await groups._paired(original, "reasoning-only-error", {"enable_thinking": True})
    assert len(pair) == 2
    assert teacher.histories[0][-1]["content"] == "<|start_think|>"
    assert "qwen" not in CharTokenizer().decode(pair[1].response_token_ids)
    prefix_length = len("<|start_think|>")
    assert pair[0].loss_mask[:prefix_length] == pair[1].loss_mask[:prefix_length] == (0,) * prefix_length


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_count", [2, 3])
async def test_worker_repairs_second_turn_and_resumes_student(monkeypatch, turn_count):
    graded = []

    def respond(url, *, headers, json, timeout):
        payload = json_codec.loads(json["messages"][-1]["content"])
        if "claim" in payload:
            response = requests.Response()
            response.status_code = 200
            response._content = encode_response({"incorrect": True, "ambiguous": False, "explanation": "wrong name"})
            return response
        graded.append(payload)
        requested = payload["requested_facts"]
        wrong = "qwen" in payload["answer"]
        assessment = {"task_completed": not wrong,
                      "fact_correctness": {fact: float(not wrong) for fact in requested},
                      "unsupported_claims": int(wrong), "inappropriate_refusal": False,
                      "unsolicited_identity": False, "explanation": "reviewed"}
        response = requests.Response()
        response.status_code = 200
        response._content = encode_response(assessment)
        return response

    monkeypatch.setattr(requests, "post", respond)
    followups = ["What is your name?", "Write a poem about rain."][:turn_count-1]
    spec = source_row_task(
        {"prompt": [{"role": "user", "content": "Explain KeyError."}],
         "env_class": "scripted_conversation", "followup_prompts": followups, "max_turns": turn_count},
        0, source_name="conversation-test",
        environment_configs={"session": session_limits(max_turns=turn_count), "scripted_conversation": {}},
    )
    row = task_input(spec)
    request = TaskRequestBatch(rows=(row,)*4,
        trajectory_ids=tuple(TrajectoryID("conversation", i) for i in range(4)),
        sampling_params={"logprobs": 0})

    class Client:
        async def generate(self, model_request):
            messages = model_request["prompts"][0]
            question = messages[-1]["content"]
            answer = {"Explain KeyError.": "The key is missing.",
                      "What is your name?": "my name is qwen",
                      "Write a poem about rain.": "Rain taps the pane."}[question]
            continuation = model_request["chat_continuations"][0]
            suffix = list(map(ord, "\nUSER: " + question + "\nASSISTANT: "))
            prompt = suffix if continuation is None else continuation["served_prefix_token_ids"] + suffix
            return {"token_provenance": TokenProvenance.ENGINE,
                    "assistant_messages": [{"role": "assistant", "content": answer}],
                    "responses": [answer], "prompt_ids": [prompt],
                    "response_ids": [list(map(ord, answer))],
                    "response_logprobs": [[-0.1]*len(answer)], "stop_reasons": ["stop"]}

    teacher = FakeClient(["I am Marin Snowball"]*2)
    student = FakeClient(["Rain drums on leaves."]*2)
    manifest = FactManifest("v1", "Marin Snowball", (), "OpenAthena", (), "OpenAthena", (), (), {})
    judge = ProductionIdentityJudge(OpenAIJudge("http://judge/v1", "judge", strict_completion=True), "pin", 512)
    facts = ((), (IdentityFact.NAME,), ())[:turn_count]
    groups = IdentityRepairGroups(CharTokenizer(), manifest, judge, teacher, student,
        TeacherRepairSpec({"temperature": 1.0}, {"temperature": 1.0}, 8192),
        {spec.task.id: facts}, RewardWeights())
    cfg = get_default_config().generator
    cfg.sampling_params.logprobs = 0
    worker = TaskRolloutWorker(cfg, WholeTaskProjection(cfg, Tokenizer()), Client(), {},
                              shutdown_timeout=10, identity_groups=groups)
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert batch["trajectory_ids"] == list(request.trajectory_ids)
    assert len(graded) == turn_count * 4
    assert batch["rewards"][2] > batch["rewards"][0]
    retained_request = replace(request, batch_metadata=BatchMetadata(1, "train"))
    records = build_trajectory_records(retained_request, batch,
        TrajectoryRetentionConfig(run_id="fixture"), CharTokenizer(), runner_name="task")
    record = records[2].to_json()
    assert record["response"]["behavior_sources"] == batch["behavior_sources"][2].tolist()
    assert record["response"]["behavior_logprobs"] == batch["rollout_logprobs"][2].tolist()
    assert len(record["row_metrics"][0]["identity_reward_records"]) == turn_count
    assert record["row_metrics"][0]["identity_repair"]["teacher_token_count"] == len("I am Marin Snowball")
    assert batch["behavior_sources"][2].sum() == len("I am Marin Snowball")
    assert not batch["behavior_sources"][0].any()
    for history in teacher.histories:
        assert any(message["content"] == "The key is missing." for message in history)
        assert all("qwen" not in message["content"] for message in history)
    if turn_count == 3:
        assert len(student.histories) == 2
        for history in student.histories:
            assert history[-2]["content"] == "I am Marin Snowball"
            assert history[-1]["content"] == "Write a poem about rain."
        assert sum(item["answer"] == "Rain drums on leaves." for item in graded) == 2
