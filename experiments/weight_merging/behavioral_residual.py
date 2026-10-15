# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select routed experts by the remaining donor-behavior residual."""

import gc
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.weight_merging.calibration_model import FrozenGrugBank, teacher_cache

logger = logging.getLogger(__name__)
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
TOTAL_ELEMENTS = 67_078_882_816
ELEMENTS_PER_EXPERT_ROW_SET = 9_830_400


@dataclass
class ResidualTotal:
    logit_kl: float = 0.0
    hidden_normalized_mse: float = 0.0
    count: int = 0


def residuals(bank: FrozenGrugBank, weights: dict, examples: list, cache: dict, hidden_weight: float) -> dict:
    domains = {example.domain for example in examples}
    totals = {domain: ResidualTotal() for domain in domains}
    with torch.no_grad():
        for example in examples:
            teacher_logits, teacher_hidden = cache[example.identity]
            output = bank.forward(weights, example.tokens, example.positions)
            device = output.logits.device
            student_log = F.log_softmax(output.logits.float(), dim=-1)
            teacher_log = F.log_softmax(teacher_logits.to(device).float(), dim=-1)
            kl = F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) / example.positions.numel()
            hidden = torch.zeros((), device=device)
            if hidden_weight:
                for student, teacher in zip(output.hidden_states, teacher_hidden, strict=True):
                    teacher = teacher.to(student.device).float()
                    layer_error = F.mse_loss(student.float(), teacher) / teacher.square().mean().clamp_min(1e-8)
                    hidden = hidden + layer_error.to(device)
                hidden = hidden / len(output.hidden_states)
            entry = totals[example.domain]
            entry.logit_kl += kl.item()
            entry.hidden_normalized_mse += hidden.item()
            entry.count += 1
            del output, student_log, teacher_log, kl, hidden
    return {
        domain: {
            "logit_kl": values.logit_kl / values.count,
            "hidden_normalized_mse": values.hidden_normalized_mse / values.count,
        }
        for domain, values in totals.items()
    }


def add_rows(
    bank: FrozenGrugBank,
    weights: dict,
    selected: list[set[int]],
    ordering: list[list[int]],
    rows_per_layer: int,
    step_size: float,
) -> list[list[int]]:
    added = []
    with torch.no_grad():
        for layer, ranking in enumerate(ordering):
            rows = [row for row in ranking if row not in selected[layer]][:rows_per_layer]
            if len(rows) != rows_per_layer:
                raise ValueError(f"Layer {layer} exhausted its candidate ranking")
            selected[layer].update(rows)
            added.append(rows)
            index_cpu = torch.tensor(rows)
            for projection in PROJECTIONS:
                name = f"model.layers.{layer}.mlp.experts.{projection}.weight"
                device = bank.device_for(name)
                anchor = bank.states[0][name].index_select(0, index_cpu).to(device=device, dtype=torch.float32)
                donor = bank.states[1][name].index_select(0, index_cpu).to(device=device, dtype=torch.float32)
                blended = (anchor + step_size * (donor - anchor)).to(weights[name].dtype)
                weights[name].index_copy_(0, index_cpu.to(device), blended)
    return added


def support(selected: list[set[int]]) -> list[dict]:
    result = []
    for layer, rows in enumerate(selected):
        for projection in PROJECTIONS:
            result.append({"name": f"model.layers.{layer}.mlp.experts.{projection}.weight", "rows": sorted(rows)})
    return result


def support_fraction(selected: list[set[int]]) -> float:
    return sum(map(len, selected)) * ELEMENTS_PER_EXPERT_ROW_SET / TOTAL_ELEMENTS


def select_behavioral_residuals(bank: FrozenGrugBank, examples: list, recipe: dict, write) -> None:
    domains = {example.domain for example in examples}
    expected = {"bfcl", "humanevalplus", "mbppplus"}
    if domains != expected:
        raise ValueError(f"Residual calibration domains differ: {domains}")
    cache = teacher_cache(bank, examples)
    rankings = recipe["rankings"]
    completed = []
    for config in recipe["experiments"]:
        selected = [set() for _ in range(bank.layer_count)]
        weights = bank.frozen_weights(0)
        history = []
        for round_index in range(config["rounds"]):
            measured = residuals(bank, weights, examples, cache, config["hidden_weight"])
            objectives = {
                domain: (
                    config["task_weights"][domain]
                    * (
                        config["logit_weight"] * values["logit_kl"]
                        + config["hidden_weight"] * values["hidden_normalized_mse"]
                    )
                )
                for domain, values in measured.items()
            }
            chosen = max(sorted(objectives), key=objectives.get)
            added = add_rows(
                bank,
                weights,
                selected,
                rankings[chosen],
                config["rows_per_round_per_layer"],
                config["step_size"],
            )
            fraction = support_fraction(selected)
            if fraction > recipe["support_cap"]:
                raise ValueError(f"{config['name']} exceeds support cap after round {round_index + 1}: {fraction}")
            entry = {
                "round": round_index + 1,
                "residuals_before_graft": measured,
                "weighted_objectives": objectives,
                "selected_task": chosen,
                "added_rows": added,
                "support_fraction": fraction,
            }
            history.append(entry)
            logger.info("%s", json.dumps({"experiment": config["name"], **entry}))
        final_residuals = residuals(bank, weights, examples, cache, config["hidden_weight"])
        payload = {
            "configuration": config,
            "history": history,
            "final_residuals": final_residuals,
            "support": support(selected),
            "support_fraction": support_fraction(selected),
            "complete": True,
        }
        write(f"{config['name']}.json", json.dumps(payload).encode())
        completed.append(config["name"])
        del weights
        gc.collect()
        torch.cuda.empty_cache()
    write("complete.json", json.dumps({"passed": True, "experiments": completed, "recipe": recipe}).encode())
