import pytest
import torch

from skyrl_train.distillation import compact_student_topk_logprobs


@pytest.mark.parametrize("lengths", [[(3, 2)], [(3, 2), (5, 6)], [(5, 8)]])
def test_compact_selected_scores_and_gradients_match_padded_reference(lengths):
    prompt_width, response_width, vocabulary, top_k = 5, 8, 17, 3
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(
        len(lengths), max(p + r for p, r in lengths), vocabulary, generator=generator, requires_grad=True
    )
    reference_logits = logits.detach().clone().requires_grad_()
    mask = torch.zeros(len(lengths), prompt_width + response_width, dtype=torch.bool)
    indices = torch.full((len(lengths), response_width, top_k), -1, dtype=torch.long)
    reference_rows = []
    for row, (prompt_length, response_length) in enumerate(lengths):
        mask[row, prompt_width - prompt_length : prompt_width + response_length] = True
        indices[row, :response_length] = torch.randint(
            vocabulary, (response_length, top_k), generator=generator
        )
        indices[row, 0, -1] = -1
        padded_logits = torch.zeros(prompt_width + response_width, vocabulary)
        padded_logits[mask[row].clone()] = reference_logits[row, : prompt_length + response_length]
        prediction_logits = padded_logits[prompt_width - 1 : -1]
        scores = prediction_logits.log_softmax(-1).gather(-1, indices[row].clamp_min(0))
        reference_rows.append(scores.masked_fill(indices[row] == -1, torch.nan))
    expected = torch.stack(reference_rows)
    actual = compact_student_topk_logprobs(logits, indices, mask)
    torch.testing.assert_close(actual, expected, equal_nan=True)
    weights = torch.randn(actual.shape, generator=generator)
    torch.nan_to_num(actual).mul(weights).sum().backward()
    torch.nan_to_num(expected).mul(weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
