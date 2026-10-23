"""Bounded samples of model weights before and after an optimizer update."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightSample:
    parameter: torch.Tensor
    indices: torch.Tensor
    before: torch.Tensor


@torch.no_grad()
def sample_model_weights(parameters: Iterable[torch.Tensor], samples_per_tensor: int) -> list[WeightSample]:
    if samples_per_tensor == 0:
        return []
    if samples_per_tensor < 0:
        raise ValueError("Weight sample count must be nonnegative")
    samples = []
    seen = set()
    for parameter in parameters:
        if not parameter.requires_grad or parameter.numel() == 0 or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        stride = max(1, (parameter.numel() + samples_per_tensor - 1) // samples_per_tensor)
        indices = torch.arange(0, parameter.numel(), stride, device=parameter.device)
        samples.append(WeightSample(parameter, indices, parameter.reshape(-1)[indices].float().clone()))
    return samples


@torch.no_grad()
def model_update_metrics(samples: list[WeightSample]) -> dict[str, float]:
    if not samples:
        return {}
    count = sum(sample.before.numel() for sample in samples)
    changes = [sample.parameter.reshape(-1)[sample.indices].float() - sample.before for sample in samples]
    squared = torch.stack([delta.square().sum() for delta in changes]).sum()
    baseline = torch.stack([sample.before.square().sum() for sample in samples]).sum()
    changed = torch.stack([delta.count_nonzero() for delta in changes]).sum()
    return {
        'sampled_model_update/rms': float((squared / count).sqrt()),
        'sampled_model_update/relative_l2': float((squared / baseline.clamp_min(1e-30)).sqrt()),
        'sampled_model_update/changed_fraction': float(changed / count),
        'sampled_model_update/elements_per_rank': float(count),
    }
