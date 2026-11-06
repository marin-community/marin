# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a small native Hero checkpoint and its BF16 vLLM export."""

import argparse
import json
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.ops.export_vllm import ExportConfig, export
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def prepare(output: Path, source_revision: str) -> None:
    """Keep Hero's head geometry and ShortConv with small projections and experts."""
    output.mkdir(parents=True, exist_ok=False)
    config = GrugModelConfig(
        vocab_size=128,
        hidden_dim=128,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=64,
        num_layers=4,
        num_heads=48,
        num_kv_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        head_dim=128,
        max_seq_len=1024,
        sliding_window=16,
        global_every=2,
        initializer_std=0.1,
        sconv=True,
        rope_fused=True,
        sconv_kernel=4,
        attention_implementation="reference",
        moe_implementation="sonic",
    )
    checkpoint = str(output / "native")
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(42))
        pending = jnp.broadcast_to(jnp.asarray([0.2, -0.3, 0.1, 0.0]), (4, 4))
        save_checkpoint(
            {"params": model, "pending_qb_betas": pending}, step=0, checkpoint_path=checkpoint, is_temporary=False
        )
        metadata = json.loads((output / "native/metadata.json").read_text())
        request = ExportConfig(checkpoint, digest(metadata), config, str(output / "hf"), source_revision)
        export(request)
    reference = {
        "request": {
            "checkpoint": {"uri": checkpoint, "metadata_digest": request.metadata_digest},
            "source_revision": source_revision,
            "spec": {"model": draccus.encode(config)},
        }
    }
    (output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    cases = {
        "cases": [
            {
                "id": "small-hero",
                "token_ids": [1 + i % 90 for i in range(64)],
                "positions": [2, 3, 4, 14, 15, 16, 17, 31, 32, 63],
            },
            {
                "id": "shared-prefix-branch",
                "token_ids": [1 + i % 90 for i in range(32)] + [100 + i % 20 for i in range(32)],
                "positions": [31, 32, 33, 63],
            },
            {"id": "other-session", "token_ids": [70 - i % 55 for i in range(64)], "positions": [3, 4, 17, 63]},
        ]
    }
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    prepare(args.output, args.source_revision)


if __name__ == "__main__":
    main()
