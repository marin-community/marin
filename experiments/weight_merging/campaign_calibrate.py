# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect independent routing/curvature evidence and fit constrained mixing coefficients."""

import argparse
import gc
import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F
from marin.merging.checkpoint import INDEX_NAME, METADATA_NAMES, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.weight_merging.calibrate import gpu_probe, token_examples
from experiments.weight_merging.calibration_model import FrozenGrugBank
from experiments.weight_merging.constrained_coefficients import train_supported_coefficients

logger = logging.getLogger(__name__)


def routing_statistics(bank: FrozenGrugBank, examples: list) -> dict:
    counts = {domain: torch.zeros(bank.layer_count, 256, dtype=torch.int64) for domain in ("numeric", "tools")}
    current = "numeric"
    handles = []

    def observer(layer):
        def collect(module, inputs, output):
            selected = output.selected_experts.detach().flatten().cpu()
            counts[current][layer] += torch.bincount(selected, minlength=256)

        return collect

    for layer in range(bank.layer_count):
        handles.append(
            bank.model.get_submodule(f"model.layers.{layer}.mlp.router").register_forward_hook(observer(layer))
        )
    weights = bank.frozen_weights(0)
    identities = []
    try:
        with torch.no_grad():
            for example in examples:
                if example.domain not in counts:
                    continue
                current = example.domain
                bank.forward(weights, example.tokens, example.positions)
                identities.append(example.identity)
                logger.info("Routing sample: %s", example.identity)
    finally:
        for handle in handles:
            handle.remove()
    return {"counts": {domain: value.tolist() for domain, value in counts.items()}, "examples": identities}


def supported_moments(bank: FrozenGrugBank, examples: list, support: list[dict], write) -> None:
    """Collect donor sequence-gradient second moments only at supported expert rows."""
    rows = {item["name"]: torch.tensor(item["rows"]) for item in support}
    weights = bank.frozen_weights(0)
    for name in rows:
        weights[name].requires_grad_(True)
    moments = {
        name: torch.zeros((len(index), *weights[name].shape[1:]), dtype=torch.float32) for name, index in rows.items()
    }
    connected = set()
    losses = []
    for example in examples:
        output = bank.forward(weights, example.tokens, example.positions)
        loss = F.cross_entropy(output.logits[0].float(), example.targets.to(output.logits.device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite response loss")
        loss.backward()
        for name, index in rows.items():
            gradient = weights[name].grad
            if gradient is None:
                continue
            connected.add(name)
            # Transfer one complete expert matrix at a time to bound staging memory.
            for position, row in enumerate(index.tolist()):
                selected = gradient[row].float().cpu()
                if not torch.isfinite(selected).all():
                    raise ValueError(f"Nonfinite gradient: {name}:{row}")
                moments[name][position].add_(selected.square() / len(examples))
            weights[name].grad = None
        losses.append(loss.item())
        logger.info("Moment sample %d/%d %s: %.6f", len(losses), len(examples), example.identity, loss.item())
        del output, loss
    weight_map = {}
    for i, (name, moment) in enumerate(moments.items()):
        shard = f"moment-{i:03d}.safetensors"
        write(shard, safetensors.torch.save({name: moment}))
        weight_map[name] = shard
    write(INDEX_NAME, json.dumps({"weight_map": weight_map}).encode())
    write(
        "complete.json",
        json.dumps(
            {
                "estimator": "mean_squared_sequence_mean_response_nll_gradient",
                "support": support,
                "examples": [example.identity for example in examples],
                "losses": losses,
                "disconnected_zero_moment_parameters": sorted(set(rows) - connected),
            }
        ).encode(),
    )


def foundation(recipe: dict, config: dict, examples: list, devices: tuple, write) -> None:
    for i, source in enumerate(recipe["sources"]):
        bank = FrozenGrugBank([CheckpointSource(**source)], config, devices)
        bank.enable_grouped_experts()
        label = ("base", "nupa", "bfcl")[i]
        routing = routing_statistics(bank, examples)
        routing.update(source=source, data_sha256=recipe["data_sha256"], complete=True)
        write(f"routing-{label}.json", json.dumps(routing).encode())
        if i:
            domain = "numeric" if i == 1 else "tools"
            supported_moments(
                bank,
                [x for x in examples if x.domain == domain],
                recipe["support"],
                lambda name, payload, label=label: write(f"moments-{label}/{name}", payload),
            )
        del bank
        gc.collect()
        torch.cuda.empty_cache()
    write("complete.json", json.dumps({"passed": True, "recipe": recipe}).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    recipe = json.loads(args.recipe.read_text())
    torch.set_num_threads(recipe["threads"])
    source = CheckpointSource(**recipe["sources"][0])
    fs, path = filesystem_for(source.path)
    config = json.loads(fs.cat_file(prefix_join(path, "config.json")))
    data = Path(recipe["data"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != recipe["data_sha256"]:
        raise ValueError("Calibration data hash mismatch")
    with tempfile.TemporaryDirectory() as directory:
        for name in METADATA_NAMES:
            if fs.exists(prefix_join(path, name)):
                Path(directory, name).write_bytes(fs.cat_file(prefix_join(path, name)))
        tokenizer = AutoTokenizer.from_pretrained(directory)
        examples = token_examples(
            tokenizer, [json.loads(line) for line in data.splitlines()], recipe["max_length"], recipe["max_positions"]
        )
    devices = tuple(torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()))
    if len(devices) != recipe["gpus"]:
        raise ValueError("GPU reservation mismatch")
    output_fs, output_path = filesystem_for(recipe["output"])
    if output_fs.exists(output_path) and output_fs.ls(output_path):
        raise FileExistsError(recipe["output"])

    def write(name, payload):
        destination = prefix_join(output_path, name)
        output_fs.makedirs(str(StoragePath.parse(destination).parent), exist_ok=True)
        output_fs.pipe_file(destination, payload)

    start = time.monotonic()
    write(
        "recipe.json",
        json.dumps(
            {
                **recipe,
                "code_revision": args.code_revision,
                "torch_version": torch.__version__,
                "token_lengths": {x.identity: x.tokens.numel() for x in examples},
            }
        ).encode(),
    )
    write("gpu-probe.json", json.dumps(gpu_probe()).encode())
    if recipe["mode"] == "foundation":
        foundation(recipe, config, examples, devices, write)
    elif recipe["mode"] == "coefficients":
        bank = FrozenGrugBank([CheckpointSource(**source) for source in recipe["sources"]], config, devices)
        bank.enable_grouped_experts()
        train_supported_coefficients(bank, examples, recipe, write)
    else:
        raise ValueError(f"Unknown mode: {recipe['mode']}")
    write("timing.json", json.dumps({"elapsed_seconds": time.monotonic() - start}).encode())


if __name__ == "__main__":
    main()
