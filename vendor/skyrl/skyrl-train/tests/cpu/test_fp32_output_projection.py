import torch
from torch import nn
from torch.nn import functional as F

from skyrl_train.workers.megatron.output_projection_layout import use_fp32_output_projection


class _OutputProjection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.gradient_accumulation_fusion = True

    def forward(self, hidden: torch.Tensor, weight: torch.Tensor | None = None):
        return F.linear(hidden, self.weight if weight is None else weight), None


def test_fp32_output_projection_preserves_logits_and_backward_gradients():
    hidden = torch.tensor([[1.125, -0.703125, 0.421875], [0.5546875, 1.0625, -1.375]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [[0.84375, -1.125, 0.359375], [-0.4765625, 0.71875, 1.28125]], dtype=torch.bfloat16
    )
    input_under_test = hidden.clone().requires_grad_()
    projection = _OutputProjection(weight.clone())
    use_fp32_output_projection(projection)

    logits, _ = projection(input_under_test)
    assert logits.dtype == torch.float32
    assert not torch.equal(logits, F.linear(hidden, weight).float())
    logits.square().sum().backward()

    reference_input = hidden.clone().requires_grad_()
    reference_weight = weight.clone().requires_grad_()
    reference_logits = F.linear(reference_input.float(), reference_weight.float())
    reference_logits.square().sum().backward()

    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(input_under_test.grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(projection.weight.grad, reference_weight.grad, rtol=0, atol=0)
    assert projection.gradient_accumulation_fusion is False
