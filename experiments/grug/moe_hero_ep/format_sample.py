# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run pinned format panels with raw special tokens and interface-specific stopping."""

import argparse
import dataclasses
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from pydantic import Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Panel, Serialization, grade_panel, render_panel
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, SampleRequest, SampleResult
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import generate
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import next_logits
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights

NATIVE_BACKEND = "native-hero-bf16"


def native_evaluation_model(config: GrugModelConfig) -> GrugModelConfig:
    """Use the existing Hero evaluation routing convention with source forward settings."""
    return dataclasses.replace(config, moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION, expert_chunks=1)


class NativePanelRequest(Record):
    panel: Panel
    checkpoint: Checkpoint
    model: dict
    serialization: Serialization
    target_cluster: str
    expert_axis_size: int = Field(ge=1)
    replica_axis_size: int = Field(ge=1)


def sample_panel(
    request: NativePanelRequest,
    tokenizer: MarinTokenizer,
    logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    batch_size: int,
) -> tuple[SampleResult, dict]:
    """Sample through the native-logits boundary without hiding serialization tokens."""
    if request.panel.backend != NATIVE_BACKEND:
        raise ValueError(f"This runner implements only {NATIVE_BACKEND}")
    spec = render_panel(request.panel, tokenizer, request.serialization)
    stop = EOT_ID if request.serialization == Serialization.MARIN else tokenizer.eos_token_id
    if stop is None:
        raise ValueError("The legacy tokenizer has no EOS token")
    prompt_ids = [tokenizer.encode(prompt.text, add_special_tokens=False) for prompt in spec.prompts]
    completions = generate(
        spec,
        prompt_ids,
        batch_size=batch_size,
        eos_token_id=stop,
        logits=logits,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    )
    result = SampleResult(
        request=SampleRequest(
            checkpoint=request.checkpoint,
            spec=spec,
            model=request.model,
            source_revision=request.panel.source_revision,
            target_cluster=request.target_cluster,
        ),
        completions=completions,
        completed_at=datetime.now(UTC).isoformat(),
        eos_token_id=stop,
    )
    return result, grade_panel(request.panel, result, request.serialization)


def native_logits(model: Transformer, mesh: Mesh) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Bind the existing sharded native sampler to one restored model."""
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert")))

    def logits(tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        ids = jax.make_array_from_callback(tokens.shape, sharding, lambda index: tokens[index])
        last = jax.make_array_from_callback(positions.shape, sharding, lambda index: positions[index])
        return np.asarray(next_logits(model, ids, last))

    return logits


def sample_native(request: NativePanelRequest, destination: str) -> None:
    """Restore one immutable checkpoint, generate the fixed panel, and retain results."""
    if request.panel.backend != NATIVE_BACKEND:
        raise ValueError(f"This runner implements only {NATIVE_BACKEND}")
    config = draccus.decode(GrugModelConfig, request.model)
    if config.max_seq_len < request.panel.context_length:
        raise ValueError("Panel context exceeds the pinned model configuration")
    if config.moe_implementation != DEFAULT_DROPLESS_MOE_IMPLEMENTATION:
        raise ValueError("Native format evaluation requires dropless routing independent of filler rows")
    root = StoragePath(destination)
    if root.exists() and root.ls():
        raise FileExistsError("Choose a fresh destination for this immutable panel result")
    tokenizer = load_tokenizer(f"{request.panel.tokenizer}@{request.panel.tokenizer_revision}")
    # Validate actual rendered lengths and token identities before weight I/O.
    render_panel(request.panel, tokenizer, request.serialization)
    mesh = compact_grug_mesh(expert_axis_size=request.expert_axis_size, replica_axis_size=request.replica_axis_size)
    with jax.set_mesh(mesh):
        model = restore_weights(request.checkpoint.uri, request.checkpoint.metadata_digest, config, mesh)
        model = cast_weights(model, jnp.bfloat16)
        jax.block_until_ready(model)
        result, report = sample_panel(request, tokenizer, native_logits(model, mesh), batch_size=jax.device_count())
        if jax.process_index() == 0:
            root.mkdirs()
            (root / "request.json").write_text(request.model_dump_json(indent=2))
            (root / "samples.json").write_text(result.model_dump_json(indent=2))
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
            (root / "runtime.json").write_text(
                json.dumps(
                    {
                        "backend": NATIVE_BACKEND,
                        "processes": jax.process_count(),
                        "devices": jax.device_count(),
                        "compute_dtype": str(jnp.bfloat16),
                        "evaluation_scope": (
                            "native format generations; held-out loss, content and serving parity are separate"
                        ),
                    },
                    indent=2,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    sample_native(NativePanelRequest.model_validate_json(args.request.read_bytes()), args.destination)


if __name__ == "__main__":
    main()
