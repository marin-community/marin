"""CPU Pi payload -> production collation/advantage/objective -> optimizer canaries."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl_train.config.pi_training import validate_pi_objective_config
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.rollouts.pi_scenario_etl import scenario_rows_from_config, training_rows_to_batch
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.tiny_training.cpu_backend import CausalLMPolicy
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_policy(tmp_path_factory.mktemp("pi-policy"))


def _config(preset: str):
    cfg = get_default_config()
    teacher = preset in ("teacher_demo_pi_base", "teacher_demo_pi_verified_base")
    OmegaConf.update(cfg, "scenario", {"preset": preset}, force_add=True)
    OmegaConf.update(cfg, "data.train_data", ["fixture-task"], force_add=True)
    OmegaConf.update(cfg, "data.terminal_bench_data", [], force_add=True)
    OmegaConf.update(
        cfg,
        "terminal_bench_config.installed_agent.agent",
        {
            "name": "pi",
            "pi": {"mode": "teacher_trace" if teacher else "literal"},
            "kwargs": {"thinking_format": "qwen-chat-template"},
        },
        force_add=True,
    )
    cfg.generator.n_samples_per_prompt = 1 if teacher else 2
    if not teacher:
        cfg.generator.sampling_params.logprobs = 0
        cfg.generator.eval_sampling_params.logprobs = 0
    cfg.generator.max_turns = 2 if teacher else 1
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.trainer.train_batch_size = cfg.generator.n_samples_per_prompt
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.advantage_estimator = "uniform" if teacher else "grpo"
    cfg.trainer.algorithm.policy_loss_type = "sft" if teacher else "regular"
    cfg.trainer.step_wise_training = not teacher
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    if teacher:
        cfg.trainer.algorithm.resolved_group_advantage.kind = "none"
        cfg.trainer.eval_interval = -1
        cfg.data.val_data = []
        OmegaConf.update(
            cfg, "terminal_bench_config.harbor.verifier_disable", preset == "teacher_demo_pi_base", force_add=True
        )
    return cfg


def _trial(tokenizer, *, trial_id: str, answer: str, reward: float, teacher: bool):
    user = {"role": "user", "content": "What is one?"}
    assistant = {"role": "assistant", "content": answer}
    prompt_ids = tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)["input_ids"]
    response_ids = tokenizer.encode(answer, add_special_tokens=False)
    evidence = {
        "request_id": f"request-{trial_id}",
        "model": "teacher" if teacher else "student",
        "messages": [user],
        "tools": [],
        "chat_template_kwargs": {},
        "assistant": assistant,
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "logprobs": [-0.25] * len(response_ids),
    }
    return {
        "schema_version": 1,
        "trial_id": trial_id,
        "policy_step": 3,
        "turns": [
            {
                "index": 0,
                "actor": "teacher" if teacher else "student",
                "executed": evidence,
                "executed_assistant": assistant,
                "observations": [],
                "teacher_label": None,
            }
        ],
        "planned_executors": None,
        "critique_feedback": None,
        "critique": None,
        "judge": None,
        "verifier_status": "graded",
        "verifier_reward": reward,
        "verifier_passed": reward > 0,
    }


def _append_discontinuous_student_turn(payload: dict, tokenizer) -> None:
    """Exercise a real tool-rendered prompt that is not the sampled token prefix."""
    first = payload["turns"][0]
    action = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"echo ok"}'}}
        ],
    }
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "ok"}
    second = deepcopy(first)
    second["index"] = 1
    second["executed"]["request_id"] += "-followup"
    second["executed"]["messages"] = [*first["executed"]["messages"], action, observation]
    second["executed"]["prompt_token_ids"] = tokenizer.apply_chat_template(
        second["executed"]["messages"], tokenize=True, add_generation_prompt=True
    )["input_ids"]
    first["executed"]["assistant"] = action
    first["executed"]["response_token_ids"] = tokenizer.encode("echo ok", add_special_tokens=False)
    first["executed"]["logprobs"] = [-0.25] * len(first["executed"]["response_token_ids"])
    first["executed_assistant"] = action
    first["observations"] = [{"tool_call_id": "call-1", "content": "ok", "authored_by": "student"}]
    sampled_prefix = [*first["executed"]["prompt_token_ids"], *first["executed"]["response_token_ids"]]
    assert second["executed"]["prompt_token_ids"][: len(sampled_prefix)] != sampled_prefix
    payload["turns"].append(second)


@pytest.mark.parametrize(
    ("preset", "tool_followup"),
    [
        ("verifier_grpo_base", False),
        ("verifier_grpo_base", True),
        ("teacher_demo_pi_base", False),
        ("teacher_demo_pi_verified_base", False),
    ],
)
def test_pi_scenario_training_objective_steps_on_cpu(tiny_policy: Path, preset: str, tool_followup: bool):
    cfg = _config(preset)
    if preset in ("teacher_demo_pi_base", "teacher_demo_pi_verified_base"):
        validate_pi_objective_config(cfg)
    else:
        validate_cfg(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tiny_policy)
    teacher = preset in ("teacher_demo_pi_base", "teacher_demo_pi_verified_base")
    answers = ["#### 1"] if teacher else ["#### 1", "#### 2"]
    batches = []
    for index, answer in enumerate(answers):
        trial_id = f"pi-{index}"
        payload = _trial(tokenizer, trial_id=trial_id, answer=answer, reward=float(index == 0), teacher=teacher)
        if tool_followup:
            _append_discontinuous_student_turn(payload, tokenizer)
        if teacher:
            _append_teacher_turn(payload)
            if preset == "teacher_demo_pi_base":
                payload.update(verifier_status="skipped", verifier_reward=None, verifier_passed=None)
        rows = scenario_rows_from_config(
            {"preset": preset},
            payload,
            tokenizer,
            expected_trial_id=trial_id,
            expected_policy_step=3,
        )
        batches.append(training_rows_to_batch(rows))
    batch = batches[0] if teacher else concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.005)
    if not teacher:
        batch["trajectory_ids"] = [
            TrajectoryID("same-task", index) for index in range(len(answers)) for _ in range(2 if tool_followup else 1)
        ]
        if tool_followup:
            assert batch["is_last_step"] == [False, True, False, True]
            assert len(batch["response_ids"]) == 4
    assert (batch["rollout_logprobs"] is None) == teacher
    if preset == "teacher_demo_pi_base":
        assert batch["unshaped_reward_available"] == [False]
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = tokenizer
    trainer.all_metrics = {}
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
    trainer.critic_model = None
    trainer.ref_model = None
    trainer.group_advantage_invariant = (
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
        if teacher
        else GroupAdvantageInvariant.exact_physical(physical_group_size=2)
    )
    input_batch = trainer.convert_to_training_input(batch, ["same-task"] * len(batch["response_ids"]))
    input_batch["values"] = None
    trainer.compute_advantages_and_returns(input_batch)
    mask = input_batch["loss_mask"]
    assert mask.sum() > 0
    if teacher:
        assert input_batch["rollout_logprobs"] is None
        assert not mask[0].bool().all()  # Intervening context must not receive SFT loss.
    else:
        assert input_batch["advantages"][0][mask[0].bool()].mean() > 0
        assert input_batch["advantages"][2 if tool_followup else 1][mask[2 if tool_followup else 1].bool()].mean() < 0

    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(tiny_policy, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    log_probs = model(
        input_batch["sequences"],
        num_actions=mask.shape[1],
        attention_mask=input_batch["attention_mask"],
    )
    objective_batch = build_objective_micro_batch(
        action_log_probs=log_probs,
        old_action_log_probs=log_probs.detach(),
        base_action_log_probs=None,
        advantages=input_batch["advantages"],
        loss_mask=mask,
        rollout_logprobs=input_batch["rollout_logprobs"],
        response_span_tags=None,
        token_entropy=torch.zeros_like(log_probs),
        think_token_weight=1,
        teacher=None,
    )
    counts = step_counts(
        [mask],
        [mask],
        [],
        [input_batch["advantages"]],
        mask.shape[1],
        lambda value: value,
    )
    result = compute_policy_objective(
        objective_batch,
        loss=PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type),
        counts=counts,
        config=cfg.trainer.algorithm,
        loss_scale=1,
        report_scale=1,
    )
    assert torch.isfinite(result.optimization_loss)
    result.optimization_loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()
    assert any(
        not torch.equal(previous, parameter) for previous, parameter in zip(before, model.parameters(), strict=True)
    )


def _append_teacher_turn(payload: dict) -> None:
    first = payload["turns"][0]
    second = deepcopy(first)
    second["index"] = 1
    second["executed"]["request_id"] = "request-two"
    second["executed"]["messages"] = [*first["executed"]["messages"], first["executed_assistant"]]
    second["executed"]["assistant"] = {"role": "assistant", "content": "#### 2"}
    second["executed_assistant"] = second["executed"]["assistant"]
    payload["turns"].append(second)


def test_multiturn_teacher_trace_stays_one_physical_sft_row(tiny_policy: Path):
    tokenizer = AutoTokenizer.from_pretrained(tiny_policy)
    payload = _trial(tokenizer, trial_id="pi-multiturn", answer="#### 1", reward=1.0, teacher=True)
    _append_teacher_turn(payload)
    rows = scenario_rows_from_config(
        {"preset": "teacher_demo_pi_base"},
        payload,
        tokenizer,
        expected_trial_id="pi-multiturn",
        expected_policy_step=3,
    )
    assert len(rows.rows) == 1
    assert rows.rows[0].loss_mask.count(0) > 0
    assert rows.rows[0].token_sources.count(2) == 0  # Never student behavior.
    assert rows.rows[0].token_sources.count(1) > 0


@pytest.mark.parametrize("preset", ["verifier_grpo_base", "teacher_demo_pi_base"])
def test_pi_preset_rejects_wrong_loss_and_mode_before_rollout(preset):
    cfg = _config(preset)
    cfg.trainer.algorithm.policy_loss_type = "sft" if preset == "verifier_grpo_base" else "regular"
    with pytest.raises(ValueError, match="requires"):
        validate_cfg(cfg)
    cfg = _config(preset)
    cfg.terminal_bench_config.installed_agent.agent.pi.mode = (
        "teacher_trace" if preset == "verifier_grpo_base" else "literal"
    )
    with pytest.raises(ValueError, match="pi.mode"):
        validate_pi_objective_config(cfg)
    cfg = _config(preset)
    cfg.data.train_data = []
    with pytest.raises(ValueError, match="direct Harbor task sources"):
        validate_pi_objective_config(cfg)
    cfg = _config(preset)
    cfg.data.terminal_bench_data = ["nemotron-sidechannel"]
    with pytest.raises(ValueError, match="does not use Nemotron"):
        validate_pi_objective_config(cfg)
    if preset == "teacher_demo_pi_base":
        with pytest.raises(ValueError, match="distinct teacher model source"):
            validate_cfg(_config(preset))
        bound = _config(preset)
        OmegaConf.update(
            bound,
            "terminal_bench_config.installed_agent.teacher",
            {
                "source": "skyrl_managed",
                "model": "teacher-fixture",
                "model_path": "Qwen/independent-teacher",
                "revision": "pinned-fixture",
                "student_model": bound.trainer.policy.model.path,
                "max_context_tokens": 8192,
            },
            force_add=True,
        )
        OmegaConf.update(
            bound,
            "terminal_bench_config.installed_agent.agent.model_name",
            "hosted_vllm/teacher-fixture",
            force_add=True,
        )
        # Source-binding a single CPU worker is not fleet provisioning.
        with pytest.raises(ValueError, match="independent GPU resource allocation"):
            validate_cfg(bound)
