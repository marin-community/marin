import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray
import torch
from omegaconf import OmegaConf

import skyrl_train.trainer as trainer_module
from skyrl_train.config.utils import get_default_config
from skyrl_train.exactness_check import ExactnessCheck
from marinskyrl.distillation import compile_distillation_plan_from_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import CheckpointSnapshot, RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.objective.losses import PolicyLossInputs, ppo_policy_loss
from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.objective.reduction import reduce_to_step, step_counts
from skyrl_train.training_batch import TrainingBatchIterator
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.utils.utils import validate_batch_sizes
from skyrl_train.weight_sync.expert_block.driver import SyncTimings
from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from skyrl_train.weight_sync.base import WeightSyncReason
from tests.cpu.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class _ResidencyPolicyGroup:
    def __init__(self):
        self.model_on_gpu = False
        self.optimizer_on_gpu = False

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
        self.optimizer_on_gpu |= backload_optimizer
        self.model_on_gpu |= backload_model

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        if offload_optimizer:
            self.optimizer_on_gpu = False
        if offload_model:
            self.model_on_gpu = False


class _CheckpointResidencyPolicyGroup(_ResidencyPolicyGroup):
    def __init__(self):
        super().__init__()
        self.restore_residencies = []
        self.restored_dirs = []

    def async_run_ray_method(self, dispatch_type, method_name, **kwargs):
        assert dispatch_type == "pass_through"
        assert method_name == "load_checkpoint"
        assert kwargs["load_training_state"]
        self.restore_residencies.append((self.model_on_gpu, self.optimizer_on_gpu))
        self.restored_dirs.append(kwargs["ckpt_dir"])
        return []


class _ResidencyInferenceClient:
    def __init__(self):
        self.awake = True
        self.wake_tags = []

    async def sleep(self):
        self.awake = False

    async def wake_up(self, tags):
        self.wake_tags.append(tags)
        self.awake = True

    async def pause_generation(self):
        self.awake = False

    async def resume_generation(self, policy_step):
        self.awake = True


class _EmptyBatchPolicyGroup(_ResidencyPolicyGroup):
    def async_run_ray_method(self, dispatch_type, method_name):
        assert (dispatch_type, method_name) == ("pass_through", "barrier_all")
        return []


class _VersionedSync:
    def __init__(self, completed_step):
        self.sender = ExpertBlockSender(SimpleNamespace(_model_version_step=completed_step), None)
        self.sender.stream = self
        self.installed_versions = []

    def run(self, version):
        self.installed_versions.append(version)
        return InstallReport(participant=0, version=version, expert_matrices=0, wire_bytes=0, seconds=0.0)

    async def sync(self, version):
        self.sender.send_weights({"version": version})
        return SyncTimings(0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("colocated", [False, True])
async def test_empty_ftpo_batch_sync_keeps_last_completed_model_version(colocated):
    completed_step = 7
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.generator.expert_block_sync.verify = False
    trainer.colocate_all = colocated
    trainer.policy_model = _EmptyBatchPolicyGroup()
    trainer.inference_engine_client = _ResidencyInferenceClient()
    transport = _VersionedSync(completed_step=completed_step)
    trainer._expert_block_sync = transport
    trainer.global_step = completed_step
    trainer.all_timings = {}
    trainer.all_startup_timings = {}
    trainer.all_metrics = {}
    await trainer._sync_policy_for_rollouts(reason=WeightSyncReason.INITIAL)
    for step in (completed_step + 1, completed_step + 2):
        trainer.global_step = step
        status = await trainer._run_training(
            {"ftpo_chosen_mask": torch.zeros((2, 3, 2), dtype=torch.bool), "loss_mask": torch.zeros((2, 3))}
        )
        assert status["policy_update_steps"] == 0
        await trainer._sync_policy_for_rollouts(reason=WeightSyncReason.TRAINING_STEP)
        await trainer._sync_policy_for_rollouts(reason=WeightSyncReason.EVALUATION)
    assert transport.installed_versions == [completed_step] * 5
    assert trainer.inference_engine_client.awake


@pytest.mark.parametrize("save_error", [None, RuntimeError("storage of size 0")])
def test_colocated_checkpoint_temporarily_backloads_policy_and_restores_rollout_residency(save_error, monkeypatch):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.colocate_all = True
    trainer.policy_model = _ResidencyPolicyGroup()
    trainer.inference_engine_client = _ResidencyInferenceClient()
    trainer.sync_policy_weights_to_inference_engines = AsyncMock()
    trainer.all_timings = {}
    trainer.global_step = 3
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)
    trainer.context = SimpleNamespace(state_dict=AsyncMock(return_value={}))
    save_observations = []

    def snapshot_checkpoint(rollout_state):
        save_observations.append(
            (
                trainer.policy_model.model_on_gpu,
                trainer.policy_model.optimizer_on_gpu,
                trainer.inference_engine_client.awake,
            )
        )
        if save_error is not None:
            raise save_error

    trainer._snapshot_checkpoint = snapshot_checkpoint

    if save_error is None:
        asyncio.run(trainer._save_checkpoints_with_residency())
    else:
        with pytest.raises(RuntimeError, match="storage of size 0"):
            asyncio.run(trainer._save_checkpoints_with_residency())

    assert save_observations == [(True, True, False)]
    assert not trainer.policy_model.model_on_gpu
    assert not trainer.policy_model.optimizer_on_gpu
    assert trainer.inference_engine_client.awake
    assert trainer.inference_engine_client.wake_tags == [["weights"], ["kv_cache"]]


@pytest.mark.parametrize("save_error", [OSError("AccessDenied"), ValueError("invalid checkpoint state")])
def test_intermediate_checkpoint_records_a_storage_failure_and_raises_any_other(save_error):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._checkpoint_save_failures = 0.0
    trainer._pending_checkpoint_upload = None
    attempts = 0
    saved_steps = []

    async def save_with_residency():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise save_error

    async def call_event_async(event, state, control, **_kwargs):
        assert event == "on_save"
        saved_steps.append(state.global_step)
        return control

    trainer._save_checkpoints_with_residency = save_with_residency
    trainer._finish_checkpoint_upload = AsyncMock(return_value=(0.0, 0.0))
    trainer.callback_handler = SimpleNamespace(call_event_async=call_event_async)
    trainer._control = SimpleNamespace()
    state = SimpleNamespace(global_step=6)

    storage_failure = isinstance(save_error, OSError)
    if storage_failure:
        asyncio.run(trainer._save_intermediate_checkpoint(state))
    else:
        with pytest.raises(ValueError, match="invalid checkpoint state"):
            asyncio.run(trainer._save_intermediate_checkpoint(state))
    assert trainer.all_metrics == ({"trainer/checkpoint_save_failures": 1.0} if storage_failure else {})
    asyncio.run(trainer._save_intermediate_checkpoint(state))
    assert saved_steps == [6]


@pytest.mark.parametrize(
    ("upload_error", "published", "recorded_failures"),
    [
        (
            None,
            {
                "global_step_6/data.pt": b"data",
                "global_step_6/trainer_state.pt": b"trainer",
                "latest_ckpt_global_step.txt": b"6",
            },
            {},
        ),
        (OSError("AccessDenied"), {}, {"trainer/checkpoint_save_failures": 1.0}),
    ],
)
def test_checkpoint_upload_publishes_the_marker_only_after_every_rank_upload_succeeds(
    monkeypatch, tmp_path, upload_error, published, recorded_failures
):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.policy_model = SimpleNamespace(async_run_ray_method=lambda *_args: [object()])
    trainer.critic_model = None
    trainer._pending_checkpoint_upload = None
    trainer._checkpoint_save_failures = 0.0
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._cleanup_old_checkpoints = lambda: None
    upload_started = threading.Event()
    release_upload = threading.Event()

    def wait_for_upload(_refs):
        upload_started.set()
        assert release_upload.wait(timeout=1)
        if upload_error is not None:
            raise upload_error

    monkeypatch.setattr(trainer_module.ray, "get", wait_for_upload)
    snapshot = CheckpointSnapshot(
        step=6,
        upload_started_at=0.0,
        rollout_state_path=str(tmp_path / "global_step_6" / "data.pt"),
        rollout_state_payload=b"data",
        trainer_state_path=str(tmp_path / "global_step_6" / "trainer_state.pt"),
        trainer_state_payload=b"trainer",
        marker_path=str(tmp_path / "latest_ckpt_global_step.txt"),
    )

    async def upload():
        trainer._start_checkpoint_upload(snapshot, SimpleNamespace(global_step=6))
        assert await asyncio.to_thread(upload_started.wait, 1)
        assert not Path(snapshot.marker_path).exists()
        release_upload.set()
        await trainer._drain_checkpoint_upload()

    asyncio.run(upload())

    written = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert written == published
    assert trainer.all_metrics == recorded_failures


def test_consumed_staleness_counts_selected_masked_sequences_by_group(monkeypatch):
    events = []
    monkeypatch.setattr(
        trainer_module,
        "record_event",
        lambda name, values, attributes: events.append((name, values, attributes)),
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_step = 7
    masks = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 1, 1]], dtype=torch.int64)

    trainer._record_consumed_staleness(["a", "b", "a"], [2, 0, 2], masks)

    assert [event[1] for event in events] == [
        {"staleness": 2, "groups": 1, "sequences": 2, "response_tokens": 5},
        {"staleness": 0, "groups": 1, "sequences": 1, "response_tokens": 0},
    ]
    assert all(event[2]["step"] == "7" for event in events)


@pytest.fixture
def dummy_tokenizer():
    # convert_to_training_input only pads with pad_token_id; any other tokenizer use should fail loudly.
    return SimpleNamespace(pad_token_id=0)


@pytest.mark.parametrize(
    "resume_path",
    [
        "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12",
        "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12/",
    ],
)
def test_load_checkpoints_probes_cloud_resume_uri_without_trailing_slash(dummy_config, monkeypatch, resume_path):
    probed = []

    def exists(uri):
        probed.append(uri)
        return False

    monkeypatch.setattr(trainer_module.io, "exists", exists)
    dummy_config.trainer.resume_path = resume_path
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.resume_mode = ResumeMode.FROM_PATH

    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        trainer.load_checkpoints()

    assert probed == ["s3://marin-us-east-02a/iris/test/checkpoints/global_step_12"]


def test_load_checkpoints_offloads_disaggregated_optimizer_before_policy_restore(dummy_config, tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "global_step_12"
    (checkpoint_path / trainer_module.POLICY_CHECKPOINT_SUBDIRECTORY).mkdir(parents=True)
    torch.save({"global_step": 12}, checkpoint_path / trainer_module.TRAINER_STATE_FILENAME)
    dummy_config.trainer.resume_path = str(checkpoint_path)
    dummy_config.trainer.offload_optimizer_during_rollouts = True

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.resume_mode = ResumeMode.FROM_PATH
    trainer.colocate_all = False
    trainer.policy_model = _CheckpointResidencyPolicyGroup()
    trainer.policy_model.model_on_gpu = True
    trainer.policy_model.optimizer_on_gpu = True
    trainer.critic_model = None
    trainer._domain_balancer = None
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    global_step, restored_path = trainer.load_checkpoints()

    assert global_step == 12
    assert restored_path == str(checkpoint_path)
    assert trainer.policy_model.restore_residencies == [(True, False)]


@pytest.mark.parametrize(
    ("reset_stage_state", "restore_dataloader_state", "expected_step", "expected_tokens"),
    [(False, True, 12, 900), (True, False, 0, 0)],
    ids=["continue-stage", "new-stage"],
)
def test_load_checkpoints_restores_the_policy_and_only_the_requested_stage_state(
    tmp_path,
    dummy_config,
    monkeypatch,
    reset_stage_state,
    restore_dataloader_state,
    expected_step,
    expected_tokens,
):
    checkpoint_path = tmp_path / "global_step_12"
    checkpoint_path.mkdir()
    torch.save(
        {"global_step": 12, "distillation_scored_tokens_total": 900},
        checkpoint_path / "trainer_state.pt",
    )
    rollout_state = TrainingContextState(
        loader=PromptLoaderState(order={"epoch": 0, "position": 7}, retries=[{"uid": "prompt-3"}]),
        ready=[],
        object_store_root=None,
    )
    torch.save(rollout_state, checkpoint_path / "data.pt")
    dummy_config.trainer.resume_path = str(checkpoint_path)
    dummy_config.trainer.restore_dataloader_state = restore_dataloader_state
    dummy_config.trainer.reset_global_step_on_resume = reset_stage_state
    dummy_config.trainer.reset_distillation_token_count_on_resume = reset_stage_state
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.resume_mode = ResumeMode.FROM_PATH
    trainer.policy_model = _CheckpointResidencyPolicyGroup()
    trainer.critic_model = None
    trainer.colocate_all = True
    trainer._domain_balancer = None
    trainer._restored_rollout_state = None

    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    global_step, _ = trainer.load_checkpoints()

    assert global_step == expected_step
    assert trainer.distillation_scored_tokens_total == expected_tokens
    assert trainer._restored_rollout_state == (rollout_state if restore_dataloader_state else None)
    assert trainer.policy_model.restored_dirs == [str(checkpoint_path / "policy")]


@pytest.mark.parametrize("reward_mode", ["add", "replace"])
@pytest.mark.parametrize("clip", [None, 0.5])
def test_teacher_credit_follows_environment_normalization_and_loop_credit(
    dummy_config, local_distillation_config, reward_mode, clip
):
    config = local_distillation_config(dummy_config)
    config.trainer.algorithm.advantage_batch_normalize = reward_mode == "add"
    config.trainer.algorithm.advantage_estimator = "uniform" if reward_mode == "replace" else "grpo"
    config.trainer.algorithm.distillation.reward_mode = reward_mode
    config.trainer.algorithm.distillation.advantage_clip = clip
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.distillation_plan = compile_distillation_plan_from_config(config)
    trainer.all_metrics = {}
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    valid = torch.tensor([[True, False, False, True]])
    if reward_mode == "replace":
        mask *= valid
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "action_log_probs": torch.tensor([[-1.0, -1.0, torch.nan, -1.0]], requires_grad=True),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(1, 4),
            "advantages": torch.tensor([[1.0, 3.0, torch.nan, 5.0]]),
            "rewards": torch.zeros(1, 4),
            "loop_advantages": torch.tensor([[-0.1, -0.2, 0.0, -0.3]]) if reward_mode == "add" else torch.zeros(1, 4),
            "response_mask": torch.ones(1, 4),
            "loss_mask": mask,
            "teacher_action_log_probs": torch.tensor([[-0.2, torch.nan, torch.nan, -2.5]], requires_grad=True),
            "teacher_valid_mask": valid,
            "distillation_loss_weights": torch.tensor([[0.3, torch.nan, torch.nan, 2.0]]),
        }
    )
    batch.metadata = {"uids": ["a"], "response_length": 4}

    result = trainer.finalize_advantages_for_training(batch)
    [experience] = list(TrainingBatchIterator(result, sample_batch_size=1))

    teacher = torch.tensor([[0.24, 0.0, 0.0, -3.0]]) if clip is None else torch.tensor([[0.15, 0.0, 0.0, -1.0]])
    normalized = torch.tensor([[-(1.5**0.5) - 0.1, -0.2, 0.0, 1.5**0.5 - 0.3]])
    expected = teacher if reward_mode == "replace" else normalized + teacher
    torch.testing.assert_close(experience.advantages, expected)
    assert not experience.advantages.requires_grad
    assert experience.distillation is None
    for name in (
        "teacher_action_log_probs",
        "teacher_valid_mask",
        "distillation_loss_weights",
        "loop_advantages",
        "rewards",
    ):
        assert name not in result
    assert trainer.all_metrics["distillation/teacher_advantage_mean"] == pytest.approx(teacher.sum().item() / 2)
    assert trainer.all_metrics["distillation/teacher_advantage_abs_mean"] == pytest.approx(
        teacher.abs().sum().item() / 2
    )
    assert trainer.all_metrics["distillation/teacher_advantage_clipped_fraction"] == (0 if clip is None else 1)
    assert trainer.all_metrics["distillation/valid_tokens"] == 2


def test_loop_advantages_are_collated_with_response_tokens(dummy_config, dummy_tokenizer):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = dummy_tokenizer
    trainer.pad_batch = lambda batch: batch
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 0.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "loop_advantages": [[0.0, -0.1, -0.1], [-0.2]],
    }

    batch = trainer.convert_to_training_input(trajectory_batch, ["a", "b"])

    torch.testing.assert_close(
        batch["loop_advantages"],
        torch.tensor([[0.0, -0.1, -0.1], [-0.2, 0.0, 0.0]]),
    )
    assert "teacher_action_log_probs" not in batch
    assert "teacher_valid_mask" not in batch
    assert "distillation_loss_weights" not in batch


@pytest.fixture(scope="module")
def default_config():
    return get_default_config()


def _batch_size_config(
    default_config,
    *,
    train_batch_size=128,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=2,
    n_samples_per_prompt=2,
    policy_dp=4,
    ref_dp=None,
):
    cfg = copy.deepcopy(default_config)
    cfg.trainer.train_batch_size = train_batch_size
    cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = policy_dp
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = ref_dp or 1
    cfg.trainer.algorithm.use_kl_loss = ref_dp is not None
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.generator.n_samples_per_prompt = n_samples_per_prompt
    return cfg


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({}, None),
        # A mini-batch cannot exceed the full training batch.
        ({"train_batch_size": 8, "policy_mini_batch_size": 16}, "^$"),
        ({"train_batch_size": 100}, "train_batch_size .* should be divisible by policy_mini_batch_size"),
        (
            {
                "policy_mini_batch_size": 8,
                "n_samples_per_prompt": 1,
                "policy_dp": 1,
                "micro_train_batch_size_per_gpu": 3,
            },
            "policy_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
        ),
        (
            {
                "train_batch_size": 10,
                "policy_mini_batch_size": 5,
                "policy_dp": 2,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
            },
            "policy_train_batch_size_per_gpu .* should be divisible by policy_mini_batch_size_per_gpu",
        ),
        # With a reference model, the batch must cover lcm(policy_dp, ref_dp) ranks; without one, only policy_dp.
        (
            {
                "train_batch_size": 5,
                "policy_mini_batch_size": 5,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
                "ref_dp": 3,
            },
            "least common multiple of the data parallel sizes",
        ),
        (
            {
                "train_batch_size": 6,
                "policy_mini_batch_size": 6,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
                "ref_dp": 3,
            },
            None,
        ),
        (
            {
                "train_batch_size": 2,
                "policy_mini_batch_size": 2,
                "micro_train_batch_size_per_gpu": 1,
                "n_samples_per_prompt": 1,
                "policy_dp": 2,
            },
            None,
        ),
    ],
)
def test_validate_batch_sizes_requires_even_division_across_ranks(default_config, overrides, error):
    cfg = _batch_size_config(default_config, **overrides)

    if error is None:
        validate_batch_sizes(cfg)
        return
    with pytest.raises(AssertionError, match=error):
        validate_batch_sizes(cfg)


@pytest.mark.parametrize(
    ("loss_reduction", "expected_policy_loss"),
    [("token_mean", 0.05), ("sequence_mean", 0.04375)],
)
def test_grpo_flat_reward_group_is_reported_and_trains_on_loop_credit_alone(loss_reduction, expected_policy_loss):
    response_length = 48
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "step_wise_training": False,
                "algorithm": {
                    "advantage_estimator": "grpo",
                    "gamma": 1.0,
                    "lambd": 1.0,
                    "grpo_norm_by_std": True,
                    "advantage_batch_normalize": False,
                    "policy_loss_type": "regular",
                    "loss_reduction": loss_reduction,
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "max_seq_len": response_length,
                },
            }
        }
    )
    trainer.group_advantage_invariant = GroupAdvantageInvariant.exact_physical(physical_group_size=2)
    trainer.distillation_plan = None
    trainer.all_metrics = {}
    # Rows 0-1 both score 1; rows 2-3 score 0 and 2 and end after 24 tokens. Every row earns -0.1 loop credit per
    # response token from token 18.
    response_mask = torch.ones(4, response_length)
    response_mask[2:, 24:] = 0
    rewards = torch.zeros(4, response_length)
    rewards[:, 0] = torch.tensor([1.0, 1.0, 0.0, 2.0])
    loop_advantages = torch.zeros(4, response_length)
    loop_advantages[:, 18:] = -0.1
    loop_advantages *= response_mask
    data = TrainingInputBatch(
        {
            "rewards": rewards,
            "response_mask": response_mask,
            "values": None,
            "loop_advantages": loop_advantages,
            "loss_mask": response_mask,
        }
    )
    data.metadata = {"uids": ["flat", "flat", "varied", "varied"], "avg_response_length": float(response_length)}

    result = trainer.finalize_advantages_for_training(trainer.compute_advantages_and_returns(data))

    assert trainer.all_metrics["reward/zero_std_group_fraction"] == pytest.approx(0.5)
    assert torch.equal(result["advantages"][:2], loop_advantages[:2])
    inputs = PolicyLossInputs(
        torch.zeros_like(loop_advantages), torch.zeros_like(loop_advantages), None, result["advantages"], response_mask
    )
    token_loss = ppo_policy_loss(inputs, trainer.cfg.trainer.algorithm)
    counts = step_counts(
        [response_mask], [response_mask], [], [result["advantages"]], response_length, lambda value: value
    )
    policy_loss = reduce_to_step(
        token_loss.values,
        response_mask,
        counts.policy,
        LossReduction(loss_reduction),
        max_seq_len=response_length,
        nonzero_advantage_rows=counts.nonzero_advantage_rows,
    )
    # At ratio 1 a token's loss is minus its advantage. The varied rows' opposite GRPO credit cancels, which leaves
    # 0.1 on 30, 30, 6 and 6 tokens of rows 48, 48, 24 and 24 tokens long: the token mean is 0.1 * 72 / 144 = 0.05
    # and the sequence mean is (0.1 * 30 / 48 * 2 + 0.1 * 6 / 24 * 2) / 4 = 0.04375.
    assert policy_loss.item() == pytest.approx(expected_policy_loss)


# The two live engines' responses to the check prompt and their log-probabilities, by data-parallel rank.
_CHECK_RESPONSES = {1: ([7, 8, 9], [-0.5, -1.25, -2.0]), 0: ([5, 6], [-0.75, -3.5])}
_CHECK_LOGPROBS = {
    token: value for ids, values in _CHECK_RESPONSES.values() for token, value in zip(ids, values, strict=True)
}


class _CheckInferenceClient(_ResidencyInferenceClient):
    """Two live engines, data-parallel ranks 1 and 0, that answer the exactness check's prompt with
    ``_CHECK_RESPONSES``; an engine dies during the check requests numbered in ``failing`` (from 1)."""

    def __init__(self, failing):
        super().__init__()
        self.check_requests = 0
        self.failing = failing

    async def generate_on_live_engines(self, prompt_token_ids, sampling_params):
        self.check_requests += 1
        if self.check_requests in self.failing:
            raise ray.exceptions.ActorDiedError()
        return [
            (rank, {"response_ids": [ids], "response_logprobs": [values]})
            for rank, (ids, values) in _CHECK_RESPONSES.items()
        ]


class _CheckScoringPolicyGroup:
    """A policy of four data-parallel ranks that scores each response token with its engine log-probability (0 for a
    token no engine sampled), except the second token of the fourth row, which it scores one float32 ulp toward
    zero."""

    actor_infos = [
        SimpleNamespace(rank=MeshRank(dp=dp, sp=0, tp=0, pp=0, world_size=4, dp_size=4, pp_size=1)) for dp in range(4)
    ]

    def async_run_ray_method(self, dispatch_type, method_name, **kwargs):
        if method_name == "barrier_all":
            return [ray.put(None)]
        if method_name == "empty_cache":
            return []
        assert method_name == "forward", method_name
        data = kwargs["data"]
        responses = data["sequences"][:, -data.metadata["response_length"] :].tolist()
        output = torch.tensor([[_CHECK_LOGPROBS.get(token, 0.0) for token in row] for row in responses])
        if len(output) == 4:
            output[3, 1] = torch.nextafter(output[3, 1], torch.tensor(0.0))
        return [ray.put(TrainingOutputBatch({"output": shard})) for shard in output.tensor_split(4)]


def _one_row_batch() -> TrainingInputBatch:
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "loss_mask": torch.ones(1, 2),
            "rollout_logprobs": None,
            "rollout_staleness": torch.zeros(1, 2, dtype=torch.int32),
        }
    )
    batch.metadata = {"response_length": 2}
    return batch


class _PolicyForwardComplete(Exception):
    def __init__(self, result: TrainingInputBatch):
        self.result = result


async def replay_policy_forward(trainer: RayPPOTrainer, training_input: TrainingInputBatch) -> TrainingInputBatch:
    """Run the production training step through policy forward, then stop."""
    production_forward = trainer.fwd_logprobs_values_reward

    def stop_after_forward(batch: TrainingInputBatch):
        raise _PolicyForwardComplete(production_forward(batch))

    trainer.fwd_logprobs_values_reward = stop_after_forward
    try:
        await trainer._run_training(training_input)
    except _PolicyForwardComplete as complete:
        return complete.result
    finally:
        trainer.fwd_logprobs_values_reward = production_forward
    raise AssertionError("Production training step returned without invoking policy forward")


@pytest.mark.asyncio
async def test_exactness_check_scores_the_engines_after_qualifying_syncs_in_the_next_training_forward(ray_init):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.colocate_all = False
    trainer.policy_model = _CheckScoringPolicyGroup()
    trainer.ref_model = None
    trainer.critic_model = None
    trainer.inference_engine_client = _CheckInferenceClient(failing={2})
    trainer.sync_policy_weights_to_inference_engines = AsyncMock()
    trainer._resolve_num_experts = lambda: None
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, add_special_tokens: [1, 2, 3])
    trainer._exactness_check = ExactnessCheck(2, trainer.cfg.generator.sampling_params, trainer.tokenizer)
    trainer.all_timings, trainer.all_startup_timings = {}, {}
    trainer.global_step = 0

    reported = []
    for reason in (
        WeightSyncReason.INITIAL,
        WeightSyncReason.TRAINING_STEP,
        WeightSyncReason.EVALUATION,
        WeightSyncReason.TRAINING_STEP,
        WeightSyncReason.CHECKPOINT_RESTORE,
    ):
        trainer.all_metrics = {}
        await trainer._sync_policy_for_rollouts(reason=reason)
        await replay_policy_forward(trainer, _one_row_batch())
        reported.append(
            {
                name.removeprefix("exactness_check/"): value
                for name, value in trainer.all_metrics.items()
                if name.startswith("exactness_check/") and not name.endswith("_seconds")
            }
        )

    # The check follows the first sync, every second training sync and the checkpoint re-sync; the generation an engine
    # died during is counted, and its forward scores none. Its rows repeat to the policy's four data-parallel ranks,
    # each counted at the rank of the engine that served it; the one value a float32 ulp off is one mismatch, at rank 0.
    scored = {
        "engines": 2.0,
        "tokens": 10.0,
        "mismatched_tokens": 1.0,
        "log_ratio_abs_max": 2.0**-22,
        "dp_rank_1/tokens": 6.0,
        "dp_rank_1/mismatched_tokens": 0.0,
        "dp_rank_0/tokens": 4.0,
        "dp_rank_0/mismatched_tokens": 1.0,
    }
    assert reported == [scored, {}, {}, {"generate_failures": 1.0}, scored]
