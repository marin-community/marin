"""Probe cadence and consumed diagnostics through the rollout and learner boundaries."""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from skyrl_train.config.objective_spec import validate_objective
from skyrl_train.config.utils import get_default_config
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.objective.losses import PolicyLossInputs, ppo_policy_loss
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
    RolloutWriter,
)
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.rollouts.payloads import ObjectStorePayloads
from skyrl_train.trainer import RayPPOTrainer
from tests.cpu.task_specs import lowered_task, task_input, text_task
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402


SAMPLES_PER_PROMPT = 2


STALL_TIMEOUT = 10.0


CONTENT_POLICY = RolloutContentPolicy(
    GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=SAMPLES_PER_PROMPT),
        rollout_logprobs_required=False,
    ),
    GroupSelectionPolicy(None),
)


class _Prompts:
    def __init__(self, uids: list[str]):
        self._uids = uids

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> str:
        return self._uids[index]

    def uid(self, index: int) -> str:
        return self._uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [_prompt(uid) for uid in items]


def _prompt(uid: str) -> dict:
    return {"uid": uid, "input": task_input(lowered_task(text_task(uid)))}


def _batch() -> dict:
    return {
        "prompt_token_ids": [[1]] * SAMPLES_PER_PROMPT,
        "response_ids": [[2]] * SAMPLES_PER_PROMPT,
        "rewards": [0.0, 1.0],
        "loss_masks": [[1]] * SAMPLES_PER_PROMPT,
        "rollout_logprobs": None,
    }


# Run the real buffer on this event loop; temp-file payloads need no Ray service.
class _InProcessActor:
    """Serve an actor's ``method.remote(...)`` calls on the caller's event loop."""

    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name: str) -> SimpleNamespace:
        method = getattr(self._instance, name)

        async def call(*args):
            result = method(*args)
            return await result if inspect.isawaitable(result) else result

        return SimpleNamespace(remote=lambda *args: asyncio.ensure_future(call(*args)))


async def _ignore(groups: list[RolloutGroup]) -> None:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [32, 128])
@pytest.mark.parametrize(
    "interval,centering_enabled,expected_steps",
    [(0, False, []), (None, False, [1, 11, 21]), (3, False, [1, 4, 7, 10, 20]), (3, True, [])],
)
async def test_periodic_probe_dispatch_is_bounded_across_many_groups_and_repeated_steps(
    tmp_path, monkeypatch, candidate_count, interval, centering_enabled, expected_steps
):
    # The real buffer runs in process; its actor lifetime has no Ray service.
    monkeypatch.setattr(ray, "kill", lambda _actor: None)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "skyrl_train/config"), version_base=None
    ):
        config = compose(config_name="ppo_base_config")
    config.trainer.strategy = "megatron"
    config.trainer.use_sample_packing = True
    config.trainer.policy.sequence_parallel_size = 1
    config.trainer.policy.megatron_config.context_parallel_size = 5
    config.trainer.algorithm.off_policy_correction = "tis"
    config.trainer.algorithm.score_centering_enabled = centering_enabled
    config.trainer.algorithm.score_centering_top_k = candidate_count
    if interval is not None:
        config.trainer.algorithm.score_centering_probe.interval = interval
    config.generator.n_samples_per_prompt = SAMPLES_PER_PROMPT
    config.generator.sampling_params.logprobs = 0
    validate_objective(config)
    requests = []

    class Workers:
        async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
            requests.append(task.request)
            await writer.write_rollout(
                task.lease, RolloutGroup(_batch(), task.prompt["uid"], task.lease.policy_step, task.prompt)
            )
            return SAMPLES_PER_PROMPT

    uids = [str(index) for index in range(16)]
    buffer_config = RolloutBufferConfig(3, 3, 0, BatchPolicy.FULL_BATCH, None, None)
    context = TrainingContext(
        PromptLoader(_Prompts(uids), SeededPasses(len(uids), seed=0, shuffle=False), batch_size=3),
        buffer_config,
        _InProcessActor(RolloutBuffer(buffer_config)),
        CONTENT_POLICY,
        RolloutRequestSpec.from_config(config),
        Workers(),
        ObjectStorePayloads(str(tmp_path)),
        rollout_spans=False,
    )
    context.start()
    try:
        for step in (1, 2, 3, 4, 7, 10, 11, 20, 21):
            await context.publish(step)
            groups, _ = await context.next_batch(stall_timeout=STALL_TIMEOUT, on_admitted=_ignore)
            assert len(groups) == 3
        sampled = [request for request in requests if request.score_centering_probe_tokens]
        assert [request.batch_metadata.global_step for request in sampled] == expected_steps
        assert all(
            request.score_centering_probe_tokens == 32 and request.sampling_params["logprobs"] == candidate_count
            for request in sampled
        )
        assert all(
            request.sampling_params["logprobs"] == (candidate_count if centering_enabled else 0)
            for request in requests
            if not request.score_centering_probe_tokens
        )
    finally:
        await context.close()


@pytest.mark.parametrize("flat_rewards", [False, True])
@pytest.mark.parametrize("candidate_count", [32, 128])
def test_consumed_probe_keeps_alignment_source_context_and_optimizer_unchanged(
    flat_rewards, candidate_count, megatron_wrapper, monkeypatch, single_rank_group
):
    config = get_default_config()
    config.trainer.strategy = "megatron"
    config.trainer.use_sample_packing = True
    config.trainer.policy.sequence_parallel_size = 1
    config.trainer.algorithm.off_policy_correction = "tis"
    config.trainer.algorithm.score_centering_top_k = candidate_count
    config.trainer.algorithm.eps_clip_low = config.trainer.algorithm.eps_clip_high = 0.2
    positions = np.array([0, 1, 1024, 4095], dtype=np.int32)
    mask = np.zeros(4096, dtype=np.int64)
    mask[[0, 1024, 4095]] = 1  # Trainer filtering removed the sampled position 1.
    sources = np.full(4096, -1, dtype=np.int64)
    sources[[0, 1024]] = [7, 4]
    current = torch.cat(
        (torch.tensor([0.5, 0.25], dtype=torch.float64), torch.full((128,), 0.25 / 128, dtype=torch.float64))
    )
    behavior = torch.cat(
        (torch.tensor([0.125, 0.125], dtype=torch.float64), torch.full((128,), 0.75 / 128, dtype=torch.float64))
    )
    trajectory = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[0] * 4096, [129]],
        "rewards": [[0.0] * 4096, [1.0]],
        "loss_masks": [mask.tolist(), [1]],
        "rollout_logprobs": [np.full(4096, np.log(0.125)), np.array([float(behavior[129].log())])],
        "rollout_policy_steps": [sources, np.array([-1])],
    }
    wrapper = megatron_wrapper(packing=True, controller=None, num_experts=8)
    wrapper.actor_module[0].training = False
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_group", lambda: single_rank_group, raising=False)
    results = []
    for probe in (False, True):
        trainer = object.__new__(RayPPOTrainer)
        trainer.cfg, trainer.all_metrics, trainer.all_timings = config, {}, {}
        trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        trainer.group_advantage_invariant = GroupAdvantageInvariant.exact_physical(physical_group_size=2)
        trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
        trainer.critic_model = trainer.ref_model = None
        trainer._training_metrics_enabled = True
        trainer._last_score_probe_time = trainer._last_score_probe_step = None
        trainer.global_step = 7
        evidence = (
            {
                "score_probe_positions": [positions, None],
                "score_probe_indices": [np.tile(np.arange(candidate_count, dtype=np.int32), (4, 1)), None],
                "score_probe_logprobs": [np.tile(behavior[:candidate_count].log().numpy(), (4, 1)), None],
            }
            if probe
            else {}
        )
        batch = trainer.convert_to_training_input(
            {**trajectory, **evidence}, ["prompt", "prompt"], rollout_staleness=[3, 3]
        )
        parameters = torch.nn.Parameter(current.log())
        optimizer = torch.optim.SGD([parameters], lr=0.1, momentum=0.9)
        logprobs = parameters.log_softmax(-1)
        response = torch.zeros((2, 4096), dtype=torch.long)
        response[1, 0] = 129
        old = logprobs.detach()[response]
        advantages = torch.zeros_like(old)
        if not flat_rewards:
            advantages[0, [0, 1024, 4095]] = torch.tensor([2.0, -1.0, 0.0], dtype=advantages.dtype)
        batch["action_log_probs"], batch["advantages"] = old, advantages
        if probe:
            torch.testing.assert_close(batch["score_probe_positions"], torch.tensor([[0, 1024, 4095], [0, 0, 0]]))
            torch.testing.assert_close(
                batch["score_probe_source_steps"][0], torch.tensor([7, 4, -1], dtype=torch.int32)
            )
            assert batch["score_topk_indices"].shape == (2, 3, candidate_count)
            assert (batch["score_topk_indices"][1] == -1).all()
            tokens, packed = mmw.preprocess_packed_seqs(batch["sequences"], batch["attention_mask"].bool())
            with torch.no_grad():
                scored = wrapper._response_logprobs(
                    (parameters - parameters[0]).expand(*tokens.shape, len(parameters)),
                    batch["sequences"],
                    batch["attention_mask"],
                    4096,
                    packed,
                    batch["score_topk_indices"],
                    batch["score_probe_positions"],
                )
            batch["score_old_logprobs"] = scored.selected_logprobs.double()
            trainer._finish_score_probe(batch)
            metrics = trainer.all_metrics
            prefix = "score_centering_probe/"
            # At current=old, min(p, 2*q)=[.25,.25] and the proportional tail
            # cancels p. The first head entry leaves -.25*log(.5) per unit A.
            unit = float(-0.25 * torch.tensor(0.5, dtype=torch.float64).log())
            assert metrics[prefix + "unit_advantage_loss_abs_mean"] == pytest.approx(unit, abs=1e-7)
            assert metrics[prefix + "hypothetical_loss_mean"] == pytest.approx(
                0 if flat_rewards else unit / 3, abs=1e-7
            )
            assert metrics[prefix + "hypothetical_loss_abs_mean"] == pytest.approx(
                0 if flat_rewards else unit, abs=1e-7
            )
            assert metrics[prefix + "zero_advantage_fraction"] == pytest.approx(1 if flat_rewards else 1 / 3)
            assert metrics[prefix + "ordinary_loss_abs_mean"] == pytest.approx(0 if flat_rewards else 2.0, abs=1e-7)
            reference_logits = current.log().requires_grad_()
            reference_logp = reference_logits.log_softmax(-1)
            # An independent dense expectation of the ordinary clipped loss's
            # pre-update coefficient. The omitted q tail is proportional to p.
            coefficient = torch.minimum(reference_logp.exp().detach(), 2 * behavior)
            reference_correction = (coefficient * reference_logp).sum()
            reference_gradient = torch.autograd.grad(reference_correction, reference_logits)[0]
            unit_l1 = float(reference_gradient.abs().sum())
            assert metrics[prefix + "unit_advantage_logit_correction_l1_mean"] == pytest.approx(unit_l1, abs=1e-7)
            assert metrics[prefix + "logit_correction_l1_mean"] == pytest.approx(
                0 if flat_rewards else unit_l1, abs=1e-7
            )
            assert metrics[prefix + "ordinary_logit_gradient_l1_mean"] == pytest.approx(
                0 if flat_rewards else 2.0, abs=1e-7
            )
            if not flat_rewards:
                assert metrics[prefix + "logit_correction_l1_ratio"] == pytest.approx(unit_l1 / 2, abs=1e-7)
            assert metrics[prefix + "sampled_tis_clipped_fraction"] == 1
            assert metrics[prefix + "old_behavior_log_ratio_abs_mean"] == pytest.approx(float(np.log(4)))
            assert metrics[prefix + "coverage"] == 0.75
            assert metrics[prefix + "source_age_max"] == 3
            assert metrics[prefix + "long_responses"] == 1
            assert metrics[prefix + "first256_tokens"] == metrics[prefix + "last256_tokens"] == 1
            for source in ("fresh", "stale", "unknown"):
                assert metrics[prefix + source + "_source_tokens"] == 1
                assert metrics[prefix + "long_response/" + source + "_tokens"] == 1
            omitted = (130 - candidate_count) / 128
            assert metrics[prefix + "current_omitted_mass_mean"] == pytest.approx(0.25 * omitted, abs=1e-7)
            assert metrics[prefix + "behavior_omitted_mass_mean"] == pytest.approx(0.75 * omitted, abs=1e-7)
            assert not any(key.startswith("score_") for key in batch)
        weights = (old - batch["rollout_logprobs"]).exp().clamp(max=2.0)
        loss = ppo_policy_loss(
            PolicyLossInputs(logprobs[response], old, batch["rollout_logprobs"], advantages, batch["loss_mask"]),
            config.trainer.algorithm,
        ).values
        loss = (loss * weights * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
        loss.backward()
        optimizer.step()
        results.append(
            (
                parameters.detach().clone(),
                parameters.grad.clone(),
                optimizer.state[parameters]["momentum_buffer"].clone(),
            )
        )
    for baseline, observed in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
