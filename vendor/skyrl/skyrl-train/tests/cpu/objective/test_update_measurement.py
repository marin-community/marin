import pickle
from omegaconf import OmegaConf
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
import torch

from skyrl_train.utils.update_measurement import model_update_metrics, sample_model_weights


def test_model_samples_distinguish_a_real_update_from_a_skipped_step():
    parameter = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    before = sample_model_weights([parameter, parameter], samples_per_tensor=64)
    unchanged = model_update_metrics(before)
    assert unchanged['sampled_model_update/rms'] == 0
    assert unchanged['sampled_model_update/changed_fraction'] == 0
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    changed = model_update_metrics(before)
    assert 0.09 < changed['sampled_model_update/rms'] < 0.11
    assert 0.09 < changed['sampled_model_update/relative_l2'] < 0.11
    assert changed['sampled_model_update/changed_fraction'] == 1
    assert changed['sampled_model_update/elements_per_rank'] == 64


def test_training_dump_preserves_behavior_evidence_and_advantages(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create({"trainer": {"export_path": str(tmp_path)}})
    data = TrainingInputBatch({
        "rollout_logprobs": torch.tensor([[-0.1, -0.5]]),
        "action_log_probs": torch.tensor([[-0.2, -0.7]]),
        "behavior_sources": torch.tensor([[0, 1]]),
        "advantages": torch.tensor([[0.0, 1.0]]),
        "loss_mask": torch.tensor([[0, 1]]),
    })
    data.metadata = {"fixture_group": "paired"}
    trainer.dump_data(data, "update")
    with (tmp_path / "dumped_data/update.pkl").open("rb") as source:
        restored = pickle.load(source)
    assert restored.metadata == data.metadata
    for name in data:
        torch.testing.assert_close(restored[name], data[name])
        assert restored[name].device.type == "cpu"
