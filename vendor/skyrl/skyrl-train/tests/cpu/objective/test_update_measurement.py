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
