# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate frozen Grug merge coefficients or sequence-gradient second moments."""

import argparse
import gc
import hashlib
import json
import logging
import random
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F
from marin.merging.checkpoint import INDEX_NAME, METADATA_NAMES, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.weight_merging.calibration_model import FrozenGrugBank, coefficient_group

logger = logging.getLogger(__name__)
CPU_BLOCK_ELEMENTS = 2**20


@dataclass(frozen=True)
class TokenExample:
    identity: str
    domain: str
    teacher: int
    tokens: torch.Tensor
    positions: torch.Tensor
    targets: torch.Tensor


def token_examples(tokenizer, rows: list[dict], max_length: int, max_positions: int) -> list[TokenExample]:
    result = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        full = encoded["input_ids"]
        if len(full) > max_length:
            raise ValueError(f"Calibration example exceeds maximum length: {row['id']}: {len(full)}")
        response = torch.tensor(encoded["assistant_masks"], dtype=torch.bool).nonzero().flatten()
        if response.numel() == 0 or response[0] == 0:
            raise ValueError(f"Missing assistant token mask: {row['id']}")
        positions = response - 1
        if positions.numel() > max_positions:
            positions = positions[torch.linspace(0, positions.numel() - 1, max_positions).long()]
        tokens = torch.tensor(full, dtype=torch.long).unsqueeze(0)
        result.append(
            TokenExample(row["id"], row["domain"], row["teacher"], tokens, positions, tokens[0, positions + 1])
        )
    return result


def allocate_chunks(bank: FrozenGrugBank, coefficients: torch.nn.ParameterDict, budget: int) -> dict[str, int]:
    """Allocate extra contiguous chunks using learned absolute update importance."""
    scores = dict.fromkeys(coefficients, 0.0)
    limits = dict.fromkeys(coefficients, 2**63 - 1)
    for name, anchor in bank.states[0].items():
        group = coefficient_group(name)
        if group is None:
            continue
        limits[group] = min(limits[group], anchor.numel())
        alpha = coefficients[group].detach().abs().mean(0).cpu()
        for donor, value in zip(bank.states[1:], alpha, strict=True):
            # CPU blocks avoid an FP32 copy of an entire expert bank.
            total = 0.0
            for start in range(0, anchor.numel(), CPU_BLOCK_ELEMENTS):
                total += (
                    (
                        donor[name].flatten()[start : start + CPU_BLOCK_ELEMENTS].float()
                        - anchor.flatten()[start : start + CPU_BLOCK_ELEMENTS].float()
                    )
                    .abs()
                    .sum()
                    .item()
                )
            scores[group] += float(value) * total
    chunks = dict.fromkeys(scores, 1)
    if budget < len(chunks):
        raise ValueError("Chunk budget must cover every coefficient group")
    for _ in range(budget - len(chunks)):
        eligible = [key for key in chunks if chunks[key] < limits[key]]
        if not eligible:
            break
        selected = max(eligible, key=lambda key: scores[key] / chunks[key])
        chunks[selected] += 1
    return chunks


def train_coefficients(bank: FrozenGrugBank, examples: list[TokenExample], recipe: dict, write) -> None:
    cache = {}
    for teacher in range(len(bank.states)):
        weights = bank.frozen_weights(teacher)
        with torch.no_grad():
            for example in examples:
                if example.teacher != teacher:
                    continue
                output = bank.forward(weights, example.tokens, example.positions)
                cache[example.identity] = (output.logits.cpu(), tuple(state.cpu() for state in output.hidden_states))
                logger.info("Cached teacher %d: %s", teacher, example.identity)
                del output
        del weights
        gc.collect()
        torch.cuda.empty_cache()
    for config in recipe["experiments"]:
        torch.manual_seed(config["seed"])
        coefficients = bank.initial_coefficients({}, config["initial"])
        history = []
        for phase, steps in (("warmup", config["warmup_steps"]), ("chunks", config["train_steps"])):
            if phase == "chunks":
                chunks = allocate_chunks(bank, coefficients, config["chunk_budget"])
                coefficients = torch.nn.ParameterDict(
                    {
                        key: torch.nn.Parameter(value.detach().mean(0).repeat(chunks[key], 1))
                        for key, value in coefficients.items()
                    }
                )
            initial = {key: value.detach().clone() for key, value in coefficients.items()}
            optimizer = torch.optim.Adam(coefficients.parameters(), lr=config["learning_rate"])
            order = list(range(len(examples)))
            rng = random.Random(config["seed"])
            for step in range(steps):
                if step % len(order) == 0:
                    rng.shuffle(order)
                example = examples[order[step % len(order)]]
                teacher_logits, teacher_hidden = cache[example.identity]
                optimizer.zero_grad(set_to_none=True)
                weights = bank.learned_weights(coefficients, recipe["block_elements"])
                output = bank.forward(weights, example.tokens, example.positions)
                device = output.logits.device
                student_log = F.log_softmax(output.logits.float(), dim=-1)
                teacher_log = F.log_softmax(teacher_logits.to(device).float(), dim=-1)
                kl = F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) / example.positions.numel()
                hidden_loss = torch.zeros((), device=device)
                for student, teacher in zip(output.hidden_states, teacher_hidden, strict=True):
                    hidden_loss = hidden_loss + F.mse_loss(student.float(), teacher.to(student.device).float()).to(
                        device
                    )
                hidden_loss = hidden_loss / len(output.hidden_states)
                regularization = sum(
                    (value - initial[key]).abs().mean().to(device) for key, value in coefficients.items()
                ) / len(coefficients)
                loss = (
                    config["domain_weights"][example.domain]
                    * (config["logit_weight"] * kl + config["hidden_weight"] * hidden_loss)
                    + config["regularization"] * regularization
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite calibration objective")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(coefficients.parameters(), config["gradient_clip"])
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite coefficient gradient")
                optimizer.step()
                entry = {
                    "phase": phase,
                    "step": step,
                    "example": example.identity,
                    "loss": loss.item(),
                    "kl": kl.item(),
                    "hidden_mse": hidden_loss.item(),
                    "gradient_norm": grad_norm.item(),
                }
                history.append(entry)
                logger.info("%s %s", config["name"], json.dumps(entry))
                del output, weights, loss, kl, hidden_loss, regularization, student_log, teacher_log
            payload = {
                "configuration": config,
                "phase": phase,
                "history": history,
                "coefficients": {key: value.detach().cpu().tolist() for key, value in coefficients.items()},
            }
            write(f"{config['name']}/{phase}.json", json.dumps(payload).encode())
        write(f"{config['name']}/complete.json", json.dumps(payload).encode())


def collect_moments(bank: FrozenGrugBank, examples: list[TokenExample], write) -> None:
    """Average squared per-sequence mean-response-NLL gradients, not optimizer state."""
    weights = {key: value.requires_grad_(True) for key, value in bank.frozen_weights(0).items()}
    moments = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in bank.states[0].items()}
    disconnected = set(weights)
    losses = []
    for example in examples:
        output = bank.forward(weights, example.tokens, example.positions)
        loss = F.cross_entropy(output.logits[0].float(), example.targets.to(output.logits.device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite calibration NLL")
        loss.backward()
        for name, value in weights.items():
            if value.grad is not None:
                disconnected.discard(name)
                for start in range(0, value.numel(), CPU_BLOCK_ELEMENTS):
                    grad = value.grad.flatten()[start : start + CPU_BLOCK_ELEMENTS].float().cpu()
                    if not torch.isfinite(grad).all():
                        raise ValueError(f"Nonfinite gradient: {name}")
                    moments[name].flatten()[start : start + CPU_BLOCK_ELEMENTS].add_(grad.square() / len(examples))
                value.grad = None
        losses.append(loss.item())
        logger.info("Moment sample %d/%d %s: %.6f", len(losses), len(examples), example.identity, loss.item())
        del output, loss
    weight_map = {}
    total_size = 0
    for index, (name, moment) in enumerate(moments.items()):
        shard = f"moment-{index:05d}.safetensors"
        write(shard, safetensors.torch.save({name: moment}))
        weight_map[name] = shard
        total_size += moment.numel() * moment.element_size()
    write(INDEX_NAME, json.dumps({"weight_map": weight_map, "metadata": {"total_size": total_size}}).encode())
    write(
        "complete.json",
        json.dumps(
            {
                "estimator": "mean_squared_sequence_mean_response_nll_gradient",
                "examples": [example.identity for example in examples],
                "losses": losses,
                "disconnected_zero_moment_parameters": sorted(disconnected),
            }
        ).encode(),
    )


def gpu_probe() -> dict:
    """Check native grouped-expert forward and backward before loading full weights."""
    grug = import_module("skyrl_train.models.grug_moe")
    config = grug.GrugMoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = grug.GrugMoeForCausalLM(config).to(device="cuda:0", dtype=torch.bfloat16).eval()
    tokens = torch.arange(128, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        reference = model(tokens).logits.float()
    assert grug.enable_grug_grouped_mm(model) == 2
    output = model(tokens).logits.float()
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.002)
    loss = F.cross_entropy(output[0, :-1], tokens[0, 1:])
    loss.backward()
    gradients = [parameter.grad for name, parameter in model.named_parameters() if "experts." in name]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    result = {
        "max_logit_error": (output - reference).abs().max().item(),
        "loss": loss.item(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    del model, output, reference, loss, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    sources = [CheckpointSource(**source) for source in recipe["sources"]]
    fs, path = filesystem_for(sources[0].path)
    config = json.loads(fs.cat_file(prefix_join(path, "config.json")))
    data = Path(recipe["data"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == recipe["data_sha256"]
    rows = [json.loads(line) for line in data.splitlines()]
    if recipe["mode"] == "moments":
        rows = [row for row in rows if row["domain"] == recipe["domain"]]
    with tempfile.TemporaryDirectory() as directory:
        for name in METADATA_NAMES:
            if fs.exists(prefix_join(path, name)):
                Path(directory, name).write_bytes(fs.cat_file(prefix_join(path, name)))
        tokenizer = AutoTokenizer.from_pretrained(directory)
        examples = token_examples(tokenizer, rows, recipe["max_length"], recipe["max_positions"])
    devices = tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
    if len(devices) != recipe["gpus"]:
        raise ValueError(f"Expected {recipe['gpus']} GPUs, found {len(devices)}")
    output_fs, output_path = filesystem_for(recipe["output"])
    if output_fs.exists(output_path) and output_fs.ls(output_path):
        raise FileExistsError(recipe["output"])
    output_fs.makedirs(output_path, exist_ok=True)

    def write(name: str, payload: bytes) -> None:
        destination = prefix_join(output_path, name)
        output_fs.makedirs(str(StoragePath.parse(destination).parent), exist_ok=True)
        output_fs.pipe_file(destination, payload)

    write(
        "recipe.json",
        json.dumps(
            {
                **recipe,
                "code_revision": args.code_revision,
                "torch_version": torch.__version__,
                "token_lengths": {example.identity: example.tokens.numel() for example in examples},
            }
        ).encode(),
    )
    write("gpu-probe.json", json.dumps(gpu_probe()).encode())
    bank = FrozenGrugBank(sources, config, devices)
    bank.enable_grouped_experts()
    if recipe["mode"] == "coefficients":
        train_coefficients(bank, examples, recipe, write)
    elif recipe["mode"] == "moments":
        collect_moments(bank, examples, write)
    else:
        raise ValueError(f"Unknown calibration mode: {recipe['mode']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
