import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from typing import NamedTuple

import torch


class SoftmaxReference(NamedTuple):
    logprobs: torch.Tensor
    entropy: torch.Tensor
    logprob_gradients: torch.Tensor
    entropy_gradients: torch.Tensor


def numerical_rows(vocab: int, dtype: torch.dtype, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    positions = sorted({0, vocab - 1, *(range(32768, vocab, 32768))})
    for margin in (-1, 2, 8, 12, 16, 20, 30):
        for shift in (-20, 0, 40):
            for position in positions:
                row = torch.full((vocab,), float(shift), dtype=torch.float64)
                row[position] = shift + margin + math.log(vocab - 1)
                rows.append(row)
                targets.append(position)
    row = torch.full((vocab,), 35.0, dtype=torch.float64)
    row[0] = row[-1] = 40.0
    rows.extend((row, row.clone()))
    targets.extend((vocab - 1, 1))
    return torch.stack(rows).unsqueeze(0).to(device=device, dtype=dtype), torch.tensor([targets], device=device)


def decimal_reference(logits: torch.Tensor, targets: torch.Tensor) -> SoftmaxReference:
    values = []
    entropies = []
    gradients = []
    entropy_gradients = []
    with localcontext() as context:
        context.prec = 80
        for row, target in zip(
            logits.detach().double().cpu().reshape(-1, logits.shape[-1]), targets.cpu().reshape(-1), strict=True
        ):
            unique, inverse, counts = row.unique(return_inverse=True, return_counts=True)
            numbers = [Decimal.from_float(value) for value in unique.tolist()]
            maximum = max(numbers)
            weights = [(number - maximum).exp() for number in numbers]
            denominator = sum(weight * int(count) for weight, count in zip(weights, counts, strict=True))
            probabilities = [weight / denominator for weight in weights]
            logprobs = [number - maximum - denominator.ln() for number in numbers]
            entropy = -sum(
                int(count) * probability * logprob
                for count, probability, logprob in zip(counts, probabilities, logprobs, strict=True)
            )
            selected = int(inverse[target])
            values.append(float(logprobs[selected]))
            entropies.append(float(entropy))
            gradient = torch.tensor([-float(p) for p in probabilities], dtype=torch.float64)[inverse]
            gradient[target] = float(1 - probabilities[selected])
            gradients.append(gradient)
            entropy_gradients.append(
                torch.tensor(
                    [-float(p * (lp + entropy)) for p, lp in zip(probabilities, logprobs, strict=True)],
                    dtype=torch.float64,
                )[inverse]
            )
    shape = logits.shape[:-1]
    return SoftmaxReference(
        torch.tensor(values, dtype=torch.float64).reshape(shape),
        torch.tensor(entropies, dtype=torch.float64).reshape(shape),
        torch.stack(gradients).reshape(logits.shape),
        torch.stack(entropy_gradients).reshape(logits.shape),
    )


def assert_softmax_numerics(
    data: torch.Tensor,
    targets: torch.Tensor,
    logprobs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    entropy: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    expected_lp, expected_h, expected_glp, expected_gh = decimal_reference(data, targets)
    weights = torch.linspace(-0.7, 1.3, targets.numel(), device=data.device).reshape(targets.shape)
    grad_tolerance = 5e-3 if data.dtype == torch.bfloat16 else 5e-5
    for lp_weight, h_weight in ((1.0, 0.0), (0.0, 1.0), (0.7, -0.3)):
        logits = data.detach().clone().requires_grad_()
        lp, h = logprobs(logits, targets), entropy(logits)
        torch.testing.assert_close(lp.double().cpu(), expected_lp, rtol=5e-6, atol=0)
        torch.testing.assert_close(h.double().cpu(), expected_h, rtol=5e-6, atol=0)
        loss = ((lp_weight * lp + h_weight * h) * weights).sum()
        actual = torch.autograd.grad(loss, logits)[0].double().cpu()
        lp_gradient = lp_weight * expected_glp * weights.double().cpu().unsqueeze(-1)
        h_gradient = h_weight * expected_gh * weights.double().cpu().unsqueeze(-1)
        expected = lp_gradient + h_gradient
        if data.dtype == torch.bfloat16 and lp_weight and h_weight:
            expected = (lp_gradient.to(data.dtype) + h_gradient.to(data.dtype)).double()
        error = actual - expected
        relative = error.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-300)
        assert relative.max() < grad_tolerance
        index = targets.cpu().unsqueeze(-1)
        torch.testing.assert_close(actual.gather(-1, index), expected.gather(-1, index), rtol=grad_tolerance, atol=0)
        torch.testing.assert_close(logits.detach(), data, rtol=0, atol=0)
        print(
            f"softmax {data.device} {data.dtype} weights={lp_weight,h_weight}: gradient max_abs={error.abs().max().item():.3g} mean_abs={error.abs().mean().item():.3g}"
        )
    with torch.no_grad():
        full_lp, full_h = logprobs(data, targets), entropy(data)
        torch.testing.assert_close(full_lp.double().cpu(), expected_lp, rtol=5e-6, atol=0)
        torch.testing.assert_close(full_h.double().cpu(), expected_h, rtol=5e-6, atol=0)
        one = data[:1, :1].expand(2, -1, -1)
        one_targets = targets[:1, :1].expand(2, -1)
        torch.testing.assert_close(logprobs(one, one_targets), full_lp[:1, :1].expand(2, -1), rtol=0, atol=0)
        torch.testing.assert_close(entropy(one), full_h[:1, :1].expand(2, -1), rtol=0, atol=0)
