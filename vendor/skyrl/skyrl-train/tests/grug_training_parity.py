"""Shared contract for PyTorch parity with the committed Levanter oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from skyrl_train.models.grug_query_bias import next_query_bias, query_bias_candidate_count

FP32_OUTPUT_ATOL = 2e-5
FP32_OUTPUT_RTOL = 2e-5
FP32_GRAD_ATOL = 5e-5
FP32_GRAD_RTOL = 5e-5
ROUTE_MARGIN_MULTIPLIER = 10
ORACLE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "grug_training_oracle"


@dataclass(frozen=True)
class GrugTrainingOracle:
    manifest: dict[str, Any]
    observations: dict[str, np.ndarray]


def load_grug_training_oracle() -> GrugTrainingOracle:
    """Load the small committed oracle into memory."""

    manifest = json.loads((ORACLE_FIXTURE_DIR / "manifest.json").read_text())
    with np.load(ORACLE_FIXTURE_DIR / "observations.npz") as archive:
        observations = {name: archive[name] for name in archive.files}
    return GrugTrainingOracle(manifest=manifest, observations=observations)


def assert_close(label: str, actual: torch.Tensor, expected: np.ndarray, *, gradient: bool = False) -> None:
    """Apply the committed FP32 output or gradient tolerance policy."""

    expected_tensor = torch.from_numpy(expected).to(device=actual.device, dtype=actual.dtype)
    difference = (actual - expected_tensor).abs().float()
    atol = FP32_GRAD_ATOL if gradient else FP32_OUTPUT_ATOL
    rtol = FP32_GRAD_RTOL if gradient else FP32_OUTPUT_RTOL
    print(f"Grug parity {label}: max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}")
    torch.testing.assert_close(
        actual,
        expected_tensor,
        atol=atol,
        rtol=rtol,
        msg=lambda message: (
            f"{label}: max_abs={difference.max().item():.8g}, mean_abs={difference.mean().item():.8g}\n{message}"
        ),
    )


def assert_exact_routes(label: str, actual: torch.Tensor, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual.detach().cpu().numpy(), expected, err_msg=label)


def assert_route_margin(label: str, route_margin: float) -> None:
    """Require the oracle route decision to be insensitive to output tolerance."""

    minimum = ROUTE_MARGIN_MULTIPLIER * FP32_OUTPUT_ATOL
    if route_margin <= minimum:
        raise AssertionError(f"{label}: route margin {route_margin:.8g} must exceed {minimum:.8g}")


def run_grug_training_parity() -> None:
    """Check the PyTorch implementation against the committed Levanter oracle."""

    if not torch.cuda.is_available():
        raise RuntimeError("the locked tolerances require an H100")

    oracle = load_grug_training_oracle()
    manifest = oracle.manifest
    assert manifest["schema_version"] == 1
    assert manifest["padding"] == "none"
    observations = oracle.observations
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        ORACLE_FIXTURE_DIR,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    if not isinstance(model, GrugMoeForCausalLM):
        raise TypeError(f"expected GrugMoeForCausalLM, got {type(model).__name__}")
    model.to(device)
    model.train()
    input_ids = torch.from_numpy(observations["input_ids"]).long().to(device)
    attention_mask = torch.ones_like(input_ids)
    candidate_count = query_bias_candidate_count(
        input_ids.numel(),
        model.config.num_experts_per_tok,
        model.config.num_local_experts,
    )
    model.begin_query_bias_capture(candidate_count, attention_mask)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        output_hidden_states=True,
    )
    query_bias_observation = model.take_query_bias_observation(candidate_count=candidate_count)

    assert_close("loss", output.loss.reshape(1), observations["loss"].reshape(1))
    assert_close("logits", output.logits, observations["logits"])
    assert_close("hidden.embed", output.hidden_states[0], observations["hidden.embed"])
    for layer_idx in range(model.config.num_hidden_layers):
        assert_route_margin(f"route.{layer_idx}", float(observations[f"route_margin.{layer_idx}"]))
        assert_close(
            f"hidden.layer.{layer_idx}",
            output.hidden_states[layer_idx + 1],
            observations[f"hidden.layer.{layer_idx}"],
        )
        layer_observation = query_bias_observation.layers[layer_idx]
        assert_exact_routes(
            f"route.{layer_idx}",
            layer_observation.selected_experts,
            observations[f"route.{layer_idx}"],
        )
        assert_close(
            f"weight.{layer_idx}",
            layer_observation.combine_weights,
            observations[f"weight.{layer_idx}"],
        )
        assert_close(
            f"beta.{layer_idx}",
            layer_observation.candidates[:, candidate_count - 1],
            observations[f"beta.{layer_idx}"],
        )
    assert_close("hidden.final", output.hidden_states[-1], observations["hidden.final"])

    output.loss.backward()
    parameters = dict(model.named_parameters())
    gradient_names = tuple(manifest["update_parameter_names"])
    for name in gradient_names:
        assert_close(
            f"gradient.{name}",
            parameters[name].grad,
            observations[f"gradient.{name}"],
            gradient=True,
        )

    learning_rate = float(manifest["sgd_learning_rate"])
    with torch.no_grad():
        for name in gradient_names:
            parameters[name].add_(parameters[name].grad, alpha=-learning_rate)
        betas = torch.stack(
            [layer.candidates[:, candidate_count - 1] for layer in query_bias_observation.layers],
            dim=0,
        )
        model.set_query_bias(next_query_bias(betas))
        next_observation_routes = []
        model.begin_query_bias_capture(candidate_count, attention_mask)
        model(input_ids, attention_mask=attention_mask)
        next_observation = model.take_query_bias_observation(candidate_count=candidate_count)
        next_observation_routes.extend(layer.selected_experts for layer in next_observation.layers)
    for layer_idx, routes in enumerate(next_observation_routes):
        assert_exact_routes(f"next_route.{layer_idx}", routes, observations[f"next_route.{layer_idx}"])


if __name__ == "__main__":
    run_grug_training_parity()
