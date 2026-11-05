# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dedicated four-GPU drafter receiving hero snapshots over a private endpoint."""

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from iris.client.client import iris_ctx
from jax.sharding import AxisType, Mesh
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import build_train_loader
from experiments.grug.noise_belay.candidate11 import REFERENCE, ROOT, load_model_config
from experiments.grug.noise_belay.candidate11_data import native_train_dataset
from experiments.grug.tied_drafter.calibration import MODEL_SOURCE_SHA256
from experiments.grug.tied_drafter.coordinates import DraftRecipe, draft_config
from experiments.grug.tied_drafter.exchange import decode_packet, encode_packet, exchange_session
from experiments.grug.tied_drafter.runtime import draft_training_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hero-endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    protocol_bytes = args.protocol.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    for name, digest in protocol["implementation_sha256"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen experiment source changed: {name}")
    record = json.loads(REFERENCE.read_text())["config"]
    for key, value in record["trainer"]["trainer"]["jax_config"].items():
        jax.config.update(key, value)
    hero_config = load_model_config(source_sha256=MODEL_SOURCE_SHA256)
    cfg = draft_config(hero_config, DraftRecipe(protocol["recipe"]))
    devices = jax.devices("gpu")
    if len(devices) != 4:
        raise ValueError("The initial drafter requires four H100s")
    mesh = Mesh(
        np.asarray(devices).reshape(1, 1, 4, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with jax.set_mesh(mesh), exchange_session(os.environ["TIED_DRAFTER_TOKEN"]) as client:
        dataset, schedule, _ = native_train_dataset(ROOT / "tokenizers/hero-bpe-v16384")
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh, max_buffered_batches=2)
        warm_iterator = iter(loader.iter_from_step(0))
        first = next(warm_iterator)
        del warm_iterator
        model = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.PRNGKey(protocol["seed"]))
        model_leaves, model_tree = jax.tree.flatten(model)
        shapes = [x.shape for x in model_leaves]
        shardings = [x.sharding for x in model_leaves]
        leaves, structure, update, _ = draft_training_state(
            model, learning_rate=protocol["draft_learning_rate"], z_loss_weight=record["trainer"]["z_loss_weight"]
        )
        began = time.perf_counter()
        executable = update.lower(leaves, first, loop_active=None, router_tie_active=None).compile()
        warmed = executable(leaves, first)
        jax.block_until_ready(warmed)
        compile_and_warmup = time.perf_counter() - began
        del model, model_leaves, leaves, warmed
        context = iris_ctx()
        deadline = time.monotonic() + 180
        while True:
            try:
                address = context.client.resolve_endpoint(args.hero_endpoint)
                break
            except ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                threading.Event().wait(1)
        client.post(address + "/ready", timeout=30).raise_for_status()
        with (args.output / "proposals.jsonl").open("w") as stream:
            while True:
                response = client.get(address + "/snapshot", timeout=60)
                if response.status_code == 410:
                    client.post(address + "/done", timeout=30).raise_for_status()
                    break
                if response.status_code == 204:
                    continue
                response.raise_for_status()
                began = time.perf_counter()
                metadata, arrays = decode_packet(response.content)
                if metadata["protocol_sha256"] != protocol_hash:
                    raise ValueError("Hero and drafter protocols differ")
                if [a.shape for a in arrays] != shapes or any(a.dtype != np.float32 for a in arrays):
                    raise ValueError("Snapshot parameter schema differs from the draft")
                model = jax.tree.unflatten(
                    model_tree, [jax.device_put(a, s) for a, s in zip(arrays, shardings, strict=True)]
                )
                leaves, episode_structure, _, _ = draft_training_state(
                    model,
                    learning_rate=protocol["draft_learning_rate"],
                    z_loss_weight=record["trainer"]["z_loss_weight"],
                )
                if episode_structure != structure:
                    raise ValueError("Draft optimizer structure changed on refresh")
                jax.block_until_ready(leaves)
                upload_seconds = time.perf_counter() - began
                iterator = iter(loader.iter_from_step(metadata["snapshot_step"]))
                episode = {**metadata, "upload_and_optimizer_reset_seconds": upload_seconds, "updates": []}
                episode["compile_and_warmup_seconds"] = compile_and_warmup
                episode["gpu_allocator"] = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")
                for step in range(metadata["horizon"]):
                    began = time.perf_counter()
                    batch = next(iterator)
                    loaded = time.perf_counter()
                    leaves, metrics, watches = executable(leaves, batch)
                    jax.block_until_ready((leaves, metrics, watches))
                    loss = float(metrics["train/loss"])
                    if not np.isfinite(loss):
                        raise FloatingPointError("Nonfinite draft loss")
                    episode["updates"].append(
                        {
                            "step": step,
                            "loss": loss,
                            "data_seconds": loaded - began,
                            "update_seconds": time.perf_counter() - loaded,
                        }
                    )
                del iterator
                state = jax.tree.unflatten(structure, leaves)
                began = time.perf_counter()
                displacement = [
                    np.asarray(end) - start for start, end in zip(arrays, jax.tree.leaves(state.params), strict=True)
                ]
                if not all(np.all(np.isfinite(x)) for x in displacement):
                    raise FloatingPointError("Nonfinite draft displacement")
                episode["download_and_subtract_seconds"] = time.perf_counter() - began
                del state, leaves, model, arrays
                began = time.perf_counter()
                payload = encode_packet(episode, displacement)
                serialization_seconds = time.perf_counter() - began
                began = time.perf_counter()
                client.post(
                    address + "/proposal",
                    data=payload,
                    headers={"X-Generation": str(metadata["generation"])},
                    timeout=60,
                ).raise_for_status()
                episode["return_transfer_seconds"] = time.perf_counter() - began
                episode["serialization_seconds"] = serialization_seconds
                episode["payload_bytes"] = len(payload)
                del payload, displacement
                stream.write(json.dumps(episode, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            "event": "proposal",
                            "generation": metadata["generation"],
                            "snapshot_step": metadata["snapshot_step"],
                            "horizon": metadata["horizon"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
