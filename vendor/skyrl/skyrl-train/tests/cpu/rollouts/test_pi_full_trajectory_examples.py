"""Complete, human-readable Pi trial examples for every current ETL scenario.

These are *contract fixtures*, not model executions: their IDs/logprobs represent
served evidence and must not be cited as proof of model sampling or teacher routing.
Each transcript visits a real task state, executes a tool, receives an observation,
and finishes with a user-facing answer (or a bounded critique/revision).
"""

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM
from harbor.models.agent.pi_trajectory import PiCollatedTrajectory
from taskcompendium.grading_result import GradeResult, Outcome

from marinskyrl.distillation import DistillationObjectiveKind, TeacherEvidenceKind
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.config.utils import get_default_config
from skyrl_train.distillation import prepare_teacher_topk, student_topk_logprobs
from skyrl_train.distillation_adapters import build_teacher_scoring_work
from skyrl_train.inference_engines.vllm_teacher_oracle import VLLMTeacherOracle, tokenizer_vocabulary_fingerprint
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.teacher import mask_teacher_evidence, topk_teacher_loss
from skyrl_train.objective.reduction import step_counts
from skyrl_train.rollouts.pi_scenario_etl import (
    PiScenario,
    TokenSource,
    scenario_rows_from_config,
    training_rows_to_batch,
)
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import PiTrainingRollout, TaskRolloutWorker
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.rollouts.engine_fakes import InferenceClient
from tests.cpu.rollouts.pi_trial_fixtures import trial
from tests.cpu.tiny_training.cpu_backend import CausalLMPolicy
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


class PrefixTokenizer:
    """Chat-template fixture with stable turn-prefix alignment for teacher SFT."""

    pad_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, tools):
        assert tokenize and tools == [{"type": "function", "function": {"name": "bash"}}]
        rendered = bytearray(b"<start>")
        for message in messages:
            if message["role"] == "assistant":
                rendered.extend(b"<assistant>")
            else:
                rendered.extend(f"<{message['role']}>".encode())
            rendered.extend(json.dumps(message, sort_keys=True).encode())
            rendered.extend(b"<end>")
        if add_generation_prompt:
            rendered.extend(b"<assistant>")
        return list(rendered)


USER = {"role": "user", "content": "Read /workspace/settings.json and report the timeout in seconds."}
READ = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"cat /workspace/settings.json"}'},
        },
    ],
}
TEACHER_READ = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "teacher-call",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"jq .timeout_seconds /workspace/settings.json"}'},
        },
    ],
}
OBSERVATION = {"tool_call_id": "call-1", "content": '{"timeout_seconds":30}', "authored_by": "student"}
ANSWER = {"role": "assistant", "content": "The timeout is 30 seconds."}
FEEDBACK = "The file says timeout_seconds=30; report 30 seconds, not minutes."


def example_trial(scenario: PiScenario) -> dict:
    """Construct one completed, attributed two-turn task trial for a reviewed preset."""
    dagger = scenario in (PiScenario.DAGGER_MIXTURE, PiScenario.DAGGER_TAKEOVER)
    teacher = scenario in (PiScenario.TEACHER_TRACE, PiScenario.TEACHER_TRACE_VERIFIED)
    takeover = scenario is PiScenario.DAGGER_TAKEOVER
    raw = trial(dagger=dagger, teacher_suffix=takeover)
    first, second = raw["turns"]
    # Keep served-ID fixture values in the tiny policy's vocabulary so the
    # production model forward can consume the worker's assembled batch.
    first["executed"]["response_token_ids"] = [211]
    second["executed"]["response_token_ids"] = [212]
    first["executed"]["messages"] = [USER]
    first["executed"]["assistant"] = deepcopy(READ)
    first["executed_assistant"] = deepcopy(READ)
    first["observations"] = [deepcopy(OBSERVATION)]
    second["executed"]["messages"] = [
        USER,
        deepcopy(READ),
        {"role": "tool", "tool_call_id": "call-1", "content": OBSERVATION["content"]},
    ]
    second["executed"]["assistant"] = deepcopy(ANSWER)
    second["executed_assistant"] = deepcopy(ANSWER)
    # The second served prompt contains the first served response followed by
    # context introduced by Pi (tool output). Only those context tokens are masked.
    second["executed"]["prompt_token_ids"] = [20, 21, 211, 217, 218]
    raw["verifier_reward"] = 1.0
    raw["verifier_passed"] = True

    if teacher:
        for turn in (first, second):
            turn["actor"] = "teacher"
            turn["executed"]["model"] = "teacher-distinct-from-student"
            turn["executed"]["request_id"] = f"teacher-{turn['index']}"
            turn["teacher_label"] = None
        first["observations"][0]["authored_by"] = "teacher"
    elif dagger:
        label = first["teacher_label"]
        label["response_token_ids"] = [213]
        second["teacher_label"]["response_token_ids"] = [214]
        label["messages"] = [USER]
        label["assistant"] = deepcopy(TEACHER_READ)
        second["teacher_label"]["messages"] = deepcopy(second["executed"]["messages"])
        second["teacher_label"]["assistant"] = deepcopy(ANSWER)
        if takeover:
            second["executed"]["model"] = "teacher"
            second["teacher_label"] = deepcopy(second["executed"])
        raw["planned_executors"] = ["student", "teacher" if takeover else "student"]
    elif scenario is PiScenario.RLAIF:
        # Judge disagrees with the verifier, so using the wrong reward is visible.
        raw["verifier_reward"] = 0.0
        raw["verifier_passed"] = False
        raw["judge"] = {
            "request_id": "judge-after-trial",
            "trial_id": raw["trial_id"],
            "student_request_ids": [first["executed"]["request_id"], second["executed"]["request_id"]],
            "model": "independent-judge",
            "principle": "Answer with the value in the file",
            "prompt_revision": "rlaif_judge_v1",
            "score": 1,
        }
    elif scenario is PiScenario.CRITIQUE_REVISION:
        second["actor"] = "revision"
        second["executed"]["messages"].append({"role": "user", "content": [{"type": "text", "text": FEEDBACK}]})
        raw["critique_feedback"] = FEEDBACK
        raw["critique"] = {
            "trial_id": raw["trial_id"],
            "teacher_model": "independent-critic",
            "student_request_id": first["executed"]["request_id"],
            "teacher_request_id": "critique-after-attempt",
            "teacher_request": {
                "model": "independent-critic",
                "stream": False,
                "messages": [
                    USER,
                    deepcopy(READ),
                    {"role": "tool", "tool_call_id": "call-1", "content": OBSERVATION["content"]},
                    {"role": "user", "content": "Privately check the tool output and suggest a precise final answer."},
                ],
            },
            "teacher_response": {"role": "assistant", "content": FEEDBACK},
            "feedback": FEEDBACK,
        }
    return raw


@pytest.mark.parametrize("scenario", list(PiScenario))
def test_full_task_trial_selects_only_its_own_evidence_and_objective(scenario):
    raw = example_trial(scenario)
    result = scenario_rows_from_config(
        {"preset": scenario.value}, raw, PrefixTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    batch = training_rows_to_batch(result)
    first, last = raw["turns"]
    assert first["executed_assistant"]["tool_calls"][0]["function"]["name"] == "bash"
    assert first["observations"][0]["content"] == '{"timeout_seconds":30}'
    assert last["executed_assistant"] == ANSWER
    assert batch["evidence_messages"][0][-1] == ANSWER
    assert batch["unshaped_rewards"] == [raw["verifier_reward"]] * len(batch["response_ids"])
    assert all(any(mask) for mask in batch["loss_masks"])

    if scenario in (
        PiScenario.DAGGER_MIXTURE,
        PiScenario.DAGGER_TAKEOVER,
        PiScenario.TEACHER_TRACE,
        PiScenario.TEACHER_TRACE_VERIFIED,
    ):
        assert batch["rollout_logprobs"] is None
        assert all(
            {source for source, mask in zip(sources, masks, strict=True) if mask} == {TokenSource.TEACHER}
            for sources, masks in zip(batch["token_sources"], batch["loss_masks"], strict=True)
        )
        if scenario is PiScenario.DAGGER_MIXTURE:
            assert len(batch["response_ids"]) == 2
            assert first["teacher_label"]["assistant"] == TEACHER_READ
            assert first["executed_assistant"] == READ  # private label was NOT executed
        else:
            assert len(batch["response_ids"]) == 1
        if scenario in (PiScenario.TEACHER_TRACE, PiScenario.TEACHER_TRACE_VERIFIED):
            assert first["observations"][0]["authored_by"] == "teacher"
            assert len(batch["response_ids"][0]) > len(last["executed"]["response_token_ids"])
            assert 0 in batch["loss_masks"][0]  # intervening tool observation is not teacher supervision
    elif scenario is PiScenario.CRITIQUE_REVISION:
        assert batch["response_ids"] == [last["executed"]["response_token_ids"]]
        assert batch["token_sources"] == [[TokenSource.REVISION]]
        assert FEEDBACK in str(last["executed"]["messages"])
        assert FEEDBACK not in str(first["executed"]["messages"])
    else:
        if scenario in (PiScenario.VERIFIER_GRPO, PiScenario.DISTRIBUTION):
            assert batch["loss_masks"] == [[1], [1]]
            assert batch["token_sources"] == [[TokenSource.STUDENT], [TokenSource.STUDENT]]
            assert [prob.tolist() for prob in batch["rollout_logprobs"]] == [[-0.25], [-0.25]]
            assert batch["prompt_token_ids"] == [
                first["executed"]["prompt_token_ids"],
                last["executed"]["prompt_token_ids"],
            ]
        else:
            assert batch["loss_masks"] == [[1, 0, 0, 1]]
            assert batch["token_sources"] == [
                [TokenSource.STUDENT, TokenSource.CONTEXT, TokenSource.CONTEXT, TokenSource.STUDENT]
            ]
            assert batch["rollout_logprobs"][0].tolist() == [-0.25, 0.0, 0.0, -0.25]
        if scenario is PiScenario.RLAIF:
            assert batch["rewards"][0][-1] == 1.0
            assert batch["unshaped_rewards"] == [0.0]
            assert batch["rollout_metrics"]["pi/judged_score"] == 1.0
        else:
            assert batch["rewards"][0][-1] == batch["unshaped_rewards"][0]


@pytest.mark.asyncio
async def test_pi_student_trial_reuses_exact_vllm_teacher_scoring_and_masks_tool_context(task_inputs):
    """Score the graded Pi trial's executed student tokens, not a retokenized teacher action."""
    raw = example_trial(PiScenario.DISTRIBUTION)
    cfg = get_default_config()
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, PrefixTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": PiScenario.DISTRIBUTION.value},
        shutdown_timeout=1,
    )
    trajectory = PiCollatedTrajectory.model_validate(
        {
            key: raw.get(key)
            for key in (
                "schema_version",
                "trial_id",
                "policy_step",
                "turns",
                "planned_executors",
                "critique_feedback",
                "critique",
            )
        }
    ).model_dump(mode="json")
    output = PiTrainingRollout(
        task_id="settings-task",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(999,),
        loss_mask=(1,),
        logprobs=(-0.25,),
        grade=GradeResult(Outcome.GRADED, raw["verifier_reward"], passed=raw["verifier_passed"]),
        stop_reason="stop",
        pi_trajectory=trajectory,
        session_id=raw["trial_id"],
    )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows,
        trajectory_ids=(TrajectoryID("settings-task", 0),),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    batch = await worker.training_batch(request, [output])
    assert batch["loss_masks"] == [[1], [1]]
    assert batch["unshaped_rewards"] == [1.0, 1.0]
    assert batch["teacher_evidence"] is None  # Attached by the admitted-group scorer, never by Harbor's verifier.
    assert [prob.tolist() for prob in batch["rollout_logprobs"]] == [[-0.25], [-0.25]]

    class Tokenizer:
        def get_vocab(self):
            return {str(index): index for index in range(256)}

    class Engine:
        def __init__(self):
            self.sequences = []

        def get_model_max_len(self):
            return 2048

        async def generate(self, request):
            self.sequences.extend(request["prompt_token_ids"])
            return {
                "prompt_logprobs": [
                    [None] + [{token: math.log(0.8), (token + 1) % 256: math.log(0.2)} for token in sequence[1:]]
                    for sequence in request["prompt_token_ids"]
                ]
            }

    tokenizer = Tokenizer()
    engine = Engine()
    oracle = VLLMTeacherOracle(
        engine,
        teacher_id="independent",
        teacher_revision="pinned-revision",
        tokenizer=tokenizer,
        evidence_kind=TeacherEvidenceKind.TOPK_DISTRIBUTION,
    )
    work = build_teacher_scoring_work(
        batch,
        route_ids=("default", "default"),
        teacher_id="independent",
        tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(tokenizer),
        plan_version="routing-v1",
        coefficient=1.0,
        route_weights=(1.0, 1.0),
        evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION,
        top_k=2,
    )
    evidence = await oracle.score(work.request)
    scored = prepare_teacher_topk(
        work.request, evidence, coefficient=work.coefficient, route_weights=work.route_weights
    )
    masked = mask_teacher_evidence(scored, torch.tensor(batch["loss_masks"]))
    assert sorted(engine.sequences) == sorted(
        [*prompt, *response] for prompt, response in zip(batch["prompt_token_ids"], batch["response_ids"], strict=True)
    )
    assert masked.valid_mask.tolist() == [[True], [True]]
    logits = torch.zeros((2, 1, 256), requires_grad=True)
    selected = student_topk_logprobs(logits, masked.teacher_topk_indices)
    loss = topk_teacher_loss(
        masked,
        selected,
        TopKLossParams(DistillationObjectiveKind.SPARSE_FORWARD_KL, 0.2, 0.2, 3),
        vocabulary_size=256,
    )
    loss.values.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[1, 0].abs().sum() > 0


def test_teacher_trace_retokenizes_executed_tool_json_without_changing_sampled_evidence():
    raw = example_trial(PiScenario.TEACHER_TRACE)
    sampled_call = raw["turns"][0]["executed"]["assistant"]["tool_calls"][0]["function"]
    sampled_call["arguments"] = '{"command": "cat /workspace/settings.json"}'
    result = scenario_rows_from_config(
        {"preset": PiScenario.TEACHER_TRACE.value},
        raw,
        PrefixTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    assert result.rows[0].loss_mask.count(0) > 0
    assert (
        raw["turns"][0]["executed"]["assistant"]["tool_calls"][0]["function"]["arguments"] == sampled_call["arguments"]
    )
    assert raw["turns"][0]["executed_assistant"]["tool_calls"][0]["function"]["arguments"] != sampled_call["arguments"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", [method for method in PiScenario if method is not PiScenario.DISTRIBUTION])
async def test_complete_trial_worker_to_trainer_or_fails_closed_on_missing_judge(scenario, tmp_path, task_inputs):
    """Test installed Harbor envelope, shared worker, and production trainer collation.

    This bypasses launch preflight: teacher/judge sources and non-verifier
    objectives are not wired for production yet.
    """
    tokenizer = PrefixTokenizer()
    cfg = get_default_config()
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    sft = scenario in (
        PiScenario.TEACHER_TRACE,
        PiScenario.TEACHER_TRACE_VERIFIED,
        PiScenario.DAGGER_MIXTURE,
        PiScenario.DAGGER_TAKEOVER,
    )
    if sft:
        cfg.trainer.algorithm.policy_loss_type = "sft"
        cfg.trainer.algorithm.advantage_estimator = "uniform"
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, tokenizer),
        InferenceClient(),
        {},
        pi_scenario={"preset": scenario.value},
        shutdown_timeout=1,
    )
    examples = [example_trial(scenario)]
    if not sft and scenario is not PiScenario.RLAIF:
        loser = example_trial(scenario)
        loser["trial_id"] = "trial-2"
        if scenario is PiScenario.CRITIQUE_REVISION:
            loser["critique"]["trial_id"] = "trial-2"
        loser["verifier_reward"] = 0.0
        loser["verifier_passed"] = False
        examples.append(loser)
    outputs = []
    for raw in examples:
        # Pi's producer envelope has no judge field. Post-trial judge transport
        # must be implemented rather than silently fabricating its evidence.
        trajectory = PiCollatedTrajectory.model_validate(
            {
                key: raw.get(key)
                for key in (
                    "schema_version",
                    "trial_id",
                    "policy_step",
                    "turns",
                    "planned_executors",
                    "critique_feedback",
                    "critique",
                )
            }
        ).model_dump(mode="json")
        outputs.append(
            PiTrainingRollout(
                task_id="settings-task",
                messages=(),
                prompt_token_ids=(1,),
                response_token_ids=(999,),
                loss_mask=(1,),
                logprobs=(-0.25,),
                grade=GradeResult(Outcome.GRADED, raw["verifier_reward"], passed=raw["verifier_passed"]),
                stop_reason="stop",
                pi_trajectory=trajectory,
                session_id=raw["trial_id"],
            )
        )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows * len(outputs),
        trajectory_ids=tuple(TrajectoryID("settings-task", index) for index in range(len(outputs))),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    if scenario is PiScenario.RLAIF:
        with pytest.raises(ValueError, match="post-trial judgment"):
            await worker.training_batch(request, outputs)
        return
    batch = await worker.training_batch(request, outputs)
    assert len(batch["response_ids"]) == (
        4
        if scenario is PiScenario.VERIFIER_GRPO
        else 2
        if scenario in (PiScenario.DAGGER_MIXTURE, PiScenario.CRITIQUE_REVISION)
        else 1
    )
    assert len(batch["trajectory_ids"]) == len(batch["response_ids"])
    assert all(sum(mask) > 0 for mask in batch["loss_masks"])
    assert batch["unshaped_rewards"][0] == 1.0

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = tokenizer
    trainer.all_metrics = {}
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
    trainer.critic_model = None
    trainer.ref_model = None
    trainer.group_advantage_invariant = (
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=len(batch["response_ids"]))
        if sft
        else GroupAdvantageInvariant.exact_physical(physical_group_size=2)
    )
    inputs = trainer.convert_to_training_input(batch, ["settings-task"] * len(batch["response_ids"]))
    inputs["values"] = None
    trainer.compute_advantages_and_returns(inputs)
    assert inputs["sequences"].shape[0] == len(batch["response_ids"])
    assert inputs["loss_mask"].sum() == sum(map(sum, batch["loss_masks"]))
    assert (inputs["rollout_logprobs"] is None) == sft
    assert torch.isfinite(inputs["advantages"]).all()
    # Check the production model forward on both launch-relevant objectives;
    # other presets still exercise production collation, advantage and loss.
    model = None
    if scenario in (PiScenario.VERIFIER_GRPO, PiScenario.TEACHER_TRACE, PiScenario.TEACHER_TRACE_VERIFIED):
        model_path = build_tiny_policy(tmp_path / "policy", max_position_embeddings=4096)
        model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32))
        action_log_probs = model(
            inputs["sequences"],
            num_actions=inputs["loss_mask"].shape[1],
            attention_mask=inputs["attention_mask"],
        )
    else:
        action_log_probs = torch.zeros_like(inputs["loss_mask"], dtype=torch.float32, requires_grad=True)
    objective_batch = build_objective_micro_batch(
        action_log_probs=action_log_probs,
        old_action_log_probs=action_log_probs.detach(),
        base_action_log_probs=None,
        advantages=inputs["advantages"],
        loss_mask=inputs["loss_mask"],
        rollout_logprobs=inputs["rollout_logprobs"],
        response_span_tags=None,
        token_entropy=torch.zeros_like(action_log_probs),
        think_token_weight=1,
        teacher=None,
    )
    counts = step_counts(
        [inputs["loss_mask"]],
        [inputs["loss_mask"]],
        [],
        [inputs["advantages"]],
        inputs["loss_mask"].shape[1],
        lambda value: value,
    )
    objective = compute_policy_objective(
        objective_batch,
        loss=PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type),
        counts=counts,
        config=cfg.trainer.algorithm,
        loss_scale=1,
        report_scale=1,
    )
    assert torch.isfinite(objective.optimization_loss)
    objective.optimization_loss.backward()
    if model is None:
        assert action_log_probs.grad is not None
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        before = [p.detach().clone() for p in model.parameters()]
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        assert any(not torch.equal(old, p) for old, p in zip(before, model.parameters(), strict=True))
