# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit convex recipient/donor coefficients on a fixed routed-expert support."""

import json
import logging
import random

import torch
import torch.nn.functional as F
from marin.merging.learned import differentiable_weight_blend

from experiments.weight_merging.calibration_model import (
    FrozenGrugBank,
    allocate_chunk_counts,
    coefficient_group,
    teacher_cache,
)

logger = logging.getLogger(__name__)
BLOCK_ELEMENTS = 4194304


def selected_states(bank: FrozenGrugBank, support: list[dict]) -> dict:
    return {
        item["name"]: tuple(state[item["name"]].index_select(0, torch.tensor(item["rows"])) for state in bank.states)
        for item in support
    }


def supported_weights(
    bank: FrozenGrugBank, support: list[dict], selected: dict, coefficients: torch.nn.ParameterDict
) -> dict:
    rows = {item["name"]: item["rows"] for item in support}
    result = {}
    for name, anchor in bank.states[0].items():
        device = bank.device_for(name)
        if name not in rows:
            result[name] = anchor.to(device)
            continue
        group = coefficient_group(name)
        alpha = coefficients[group].softmax(-1)[:, 1:]
        sources = selected[name]
        values = differentiable_weight_blend(sources[0], sources[1:], alpha, block_elements=BLOCK_ELEMENTS)
        result[name] = anchor.to(device=device, copy=True).index_copy_(
            0, torch.tensor(rows[name], device=device), values
        )
    return result


def allocate_chunks(selected: dict, coefficients: torch.nn.ParameterDict, budget: int) -> dict[str, int]:
    scores = dict.fromkeys(coefficients, 0.0)
    limits = dict.fromkeys(coefficients, 2**63 - 1)
    for name, sources in selected.items():
        group = coefficient_group(name)
        anchor = sources[0].flatten()
        limits[group] = min(limits[group], anchor.numel())
        alpha = coefficients[group].detach().softmax(-1).mean(0)[1:].cpu()
        for donor, value in zip(sources[1:], alpha, strict=True):
            donor = donor.flatten()
            for start in range(0, anchor.numel(), BLOCK_ELEMENTS):
                difference = (
                    donor[start : start + BLOCK_ELEMENTS].float() - anchor[start : start + BLOCK_ELEMENTS].float()
                )
                scores[group] += float(value) * difference.abs().sum().item()
    return allocate_chunk_counts({key: scores[key] for key in sorted(scores)}, limits, budget)


def train_supported_coefficients(bank: FrozenGrugBank, examples: list, recipe: dict, write) -> None:
    config = recipe["optimization"]
    torch.manual_seed(config["seed"])
    cache = teacher_cache(bank, examples)
    support = recipe["support"]
    selected = selected_states(bank, support)
    coefficients = torch.nn.ParameterDict()
    for item in support:
        name = item["name"]
        group = coefficient_group(name)
        if group not in coefficients:
            value = torch.tensor(config["initial"], device=bank.device_for(name)).log().unsqueeze(0)
            coefficients[group] = torch.nn.Parameter(value)
    domains = ("numeric", "tools", "retention")
    by_domain = {domain: [x for x in examples if x.domain == domain] for domain in domains}
    rng = random.Random(config["seed"])
    for values in by_domain.values():
        rng.shuffle(values)
    history = []
    for phase, steps in (("warmup", config["warmup_steps"]), ("chunks", config["train_steps"])):
        if phase == "chunks":
            chunks = allocate_chunks(selected, coefficients, config["chunk_budget"])
            coefficients = torch.nn.ParameterDict(
                {key: torch.nn.Parameter(value.detach().repeat(chunks[key], 1)) for key, value in coefficients.items()}
            )
        initial = {key: value.detach().softmax(-1).clone() for key, value in coefficients.items()}
        optimizer = torch.optim.Adam(coefficients.parameters(), lr=config["learning_rate"])
        for step in range(steps):
            domain = domains[step % len(domains)]
            pool = by_domain[domain]
            example = pool[(step // len(domains)) % len(pool)]
            target = cache[example.identity]
            optimizer.zero_grad(set_to_none=True)
            weights = supported_weights(bank, support, selected, coefficients)
            output = bank.forward(weights, example.tokens, example.positions)
            device = output.logits.device
            student_log = F.log_softmax(output.logits.float(), dim=-1)
            teacher_log = F.log_softmax(target.logits.to(device).float(), dim=-1)
            kl = F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) / example.positions.numel()
            hidden = torch.zeros((), device=device)
            if config["hidden_weight"]:
                for student, teacher in zip(output.hidden_states, target.hidden_states, strict=True):
                    teacher = teacher.to(student.device).float()
                    hidden = hidden + (
                        F.mse_loss(student.float(), teacher) / (teacher.square().mean() + config["hidden_epsilon"])
                    ).to(device)
                hidden = hidden / len(output.hidden_states)
            regularization = sum(
                (value.softmax(-1) - initial[key]).abs().mean().to(device) for key, value in coefficients.items()
            ) / len(coefficients)
            loss = (
                config["domain_weights"][domain] * (config["logit_weight"] * kl + config["hidden_weight"] * hidden)
                + config["regularization"] * regularization
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite coefficient objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(coefficients.parameters(), config["gradient_clip"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite coefficient gradient")
            optimizer.step()
            entry = {
                "phase": phase,
                "step": step,
                "example": example.identity,
                "loss": loss.item(),
                "kl": kl.item(),
                "hidden_normalized_mse": hidden.item(),
                "gradient_norm": norm.item(),
            }
            history.append(entry)
            logger.info("%s", json.dumps(entry))
            del output, weights, loss, kl, hidden, regularization, student_log, teacher_log
        payload = {
            "phase": phase,
            "configuration": config,
            "support": support,
            "history": history,
            "coefficients": {key: value.detach().softmax(-1).cpu().tolist() for key, value in coefficients.items()},
        }
        write(f"{phase}.json", json.dumps(payload).encode())
    write("complete.json", json.dumps(payload).encode())
