# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load-balanced placement of routed experts on the ragged all-to-all's capacity blocks.

The ragged expert-parallel transport clips each receiver's load to ``capacity_factor`` times the even
share of one capacity block: a contiguous run of global expert ids within one shard
(`capacity_block_experts`). With a frozen router bias, SFT data concentrates load on a few persistent
hot experts, and the blocks that hold them overflow. Renumbering the experts of each layer so every
block carries a similar load removes most of those drops without changing the model's function: the
router columns, router bias, expert weights, and their optimizer state move together.

`balanced_expert_order` assigns experts to blocks by greedy longest-processing-time: heaviest expert
first, each into the lightest block that still has a free slot. The loads come from a run's per-expert
routing histograms (`train/router/layer_{i}/routing_hist/histogram`).
"""

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click
import numpy as np
import wandb
from levanter.grug._moe.ep_ragged_all_to_all import capacity_block_experts

logger = logging.getLogger(__name__)

ROUTING_HISTOGRAM_KEY = "train/router/layer_{layer}/routing_hist/histogram"
# Key of the placement in the metadata of each checkpoint a placed run saves.
CHECKPOINT_METADATA_KEY = "expert_placement"


@dataclasses.dataclass(frozen=True)
class ExpertPlacement:
    """A per-layer renumbering of routed experts, relative to one checkpoint's expert order.

    ``order[layer][slot]`` is the expert of ``source_checkpoint`` that moves to global expert id ``slot``.
    The order balances capacity blocks only for the mesh it was built for, so it records
    ``expert_axis_size`` and the block width.
    """

    source_checkpoint: str
    expert_axis_size: int
    block_experts: int
    order: tuple[tuple[int, ...], ...]
    loads_from: str

    def __post_init__(self):
        if not self.order:
            raise ValueError("an expert placement needs at least one layer")
        num_experts = len(self.order[0])
        if num_experts % (self.expert_axis_size * self.block_experts):
            raise ValueError(
                f"{num_experts} experts do not split into {self.expert_axis_size} shards of "
                f"{self.block_experts}-expert blocks"
            )
        for layer, layer_order in enumerate(self.order):
            if sorted(layer_order) != list(range(num_experts)):
                raise ValueError(f"layer {layer} order is not a permutation of {num_experts} experts")

    @property
    def num_layers(self) -> int:
        return len(self.order)

    @property
    def num_experts(self) -> int:
        return len(self.order[0])

    @property
    def fingerprint(self) -> str:
        """Short hash of the order: names the checkpoints a placed run writes."""
        return hashlib.sha256(json.dumps(self.order).encode()).hexdigest()[:12]

    def inverse(self) -> "ExpertPlacement":
        """The placement that restores the source checkpoint's expert order.

        Tools that pair experts across checkpoints, such as weight merging against the pretraining
        checkpoint, must first return a placed checkpoint to the source order.
        """
        return dataclasses.replace(
            self,
            order=tuple(tuple(int(e) for e in np.argsort(layer)) for layer in self.order),
            loads_from=f"inverse of {self.loads_from}",
        )

    def permutes(self, checkpoint: str) -> bool:
        """Whether ``checkpoint`` is the one whose expert order this placement renumbers."""
        return self.source_checkpoint.rstrip("/") == checkpoint.rstrip("/")

    def validate_for(self, *, num_layers: int, num_experts: int, expert_axis_size: int) -> None:
        """Fail unless this placement was built for the model and mesh it is about to be applied to."""
        if (self.num_layers, self.num_experts) != (num_layers, num_experts):
            raise ValueError(
                f"placement covers {self.num_layers} layers x {self.num_experts} experts, "
                f"the model has {num_layers} x {num_experts}"
            )
        expected_block = capacity_block_experts(num_experts // expert_axis_size)
        if (self.expert_axis_size, self.block_experts) != (expert_axis_size, expected_block):
            raise ValueError(
                f"placement balances {self.block_experts}-expert blocks on {self.expert_axis_size} expert shards; "
                f"this mesh has {expected_block}-expert blocks on {expert_axis_size}"
            )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "ExpertPlacement":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpertPlacement":
        return cls(**{**raw, "order": tuple(tuple(int(e) for e in layer) for layer in raw["order"])})

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Metadata that records this placement in each checkpoint of a placed run."""
        return {CHECKPOINT_METADATA_KEY: dataclasses.asdict(self)}

    @classmethod
    def recorded_in(cls, metadata: Mapping[str, Any]) -> "ExpertPlacement | None":
        """The placement a checkpoint's metadata records, or None for a checkpoint in its source expert order."""
        raw = metadata.get(CHECKPOINT_METADATA_KEY)
        return None if raw is None else cls.from_dict(raw)


def balanced_expert_order(loads: np.ndarray, *, block_experts: int) -> np.ndarray:
    """Per-layer expert order whose consecutive ``block_experts``-wide blocks carry balanced load.

    Args:
        loads: ``[layers, experts]`` routed assignments per expert.
        block_experts: experts per capacity block.

    Returns:
        ``[layers, experts]`` order; ``order[layer, slot]`` is the expert placed at global id ``slot``.
    """
    num_layers, num_experts = loads.shape
    if num_experts % block_experts:
        raise ValueError(f"{num_experts} experts do not split into {block_experts}-expert blocks")
    num_blocks = num_experts // block_experts
    order = np.empty((num_layers, num_experts), dtype=np.int64)
    for layer in range(num_layers):
        block_load = np.zeros(num_blocks)
        blocks: list[list[int]] = [[] for _ in range(num_blocks)]
        # A stable sort keeps ties in expert order, so equal loads give a deterministic placement.
        for expert in np.argsort(-loads[layer], kind="stable"):
            open_blocks = [b for b in range(num_blocks) if len(blocks[b]) < block_experts]
            target = min(open_blocks, key=lambda b: block_load[b])
            blocks[target].append(int(expert))
            block_load[target] += loads[layer, expert]
        order[layer] = np.concatenate([np.asarray(block) for block in blocks])
    return order


def projected_drop_fraction(
    loads: np.ndarray, order: np.ndarray, *, block_experts: int, capacity_factor: float
) -> float:
    """Share of routed assignments that capacity clipping drops when ``order`` places the experts.

    Each block accepts ``capacity_factor`` times its even share of the layer's assignments.
    """
    placed = np.take_along_axis(loads, order, axis=1)
    block_loads = placed.reshape(placed.shape[0], -1, block_experts).sum(axis=2)
    capacity = capacity_factor * block_loads.mean(axis=1, keepdims=True)
    return float(np.clip(block_loads - capacity, 0, None).sum() / loads.sum())


def routing_loads_from_wandb(run_path: str, *, num_layers: int, first_step: int, last_step: int) -> np.ndarray:
    """Sum each layer's per-expert routing histogram over ``[first_step, last_step]`` of a W&B run."""
    keys = [ROUTING_HISTOGRAM_KEY.format(layer=layer) for layer in range(num_layers)]
    run = wandb.Api(timeout=120).run(run_path)
    rows = list(run.scan_history(keys=keys, min_step=first_step, max_step=last_step + 1))
    if not rows:
        raise ValueError(f"{run_path} logged no routing histograms in steps {first_step}..{last_step}")
    loads = np.stack([np.sum([row[key]["values"] for row in rows], axis=0) for key in keys]).astype(np.float64)
    logger.info("Summed %d steps of routing histograms from %s", len(rows), run_path)
    return loads


def build_placement(
    loads: np.ndarray, *, source_checkpoint: str, expert_axis_size: int, loads_from: str
) -> ExpertPlacement:
    block_experts = capacity_block_experts(loads.shape[1] // expert_axis_size)
    order = balanced_expert_order(loads, block_experts=block_experts)
    return ExpertPlacement(
        source_checkpoint=source_checkpoint,
        expert_axis_size=expert_axis_size,
        block_experts=block_experts,
        order=tuple(tuple(int(e) for e in layer) for layer in order),
        loads_from=loads_from,
    )


@click.command()
@click.option("--wandb-run", required=True, help="W&B run path (entity/project/run_id) with routing histograms.")
@click.option("--first-step", type=int, required=True, help="First logged step whose loads count.")
@click.option("--last-step", type=int, required=True, help="Last logged step whose loads count.")
@click.option("--num-layers", type=int, required=True)
@click.option("--expert-axis-size", type=int, required=True, help="Expert shards of the mesh the SFT runs on.")
@click.option("--source-checkpoint", required=True, help="Checkpoint whose expert order the placement permutes.")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option("--capacity-factor", "capacity_factors", type=float, multiple=True, default=(1.5, 2.0))
def main(
    wandb_run: str,
    first_step: int,
    last_step: int,
    num_layers: int,
    expert_axis_size: int,
    source_checkpoint: str,
    out_path: Path,
    capacity_factors: Sequence[float],
) -> None:
    logging.basicConfig(level=logging.INFO)
    loads = routing_loads_from_wandb(wandb_run, num_layers=num_layers, first_step=first_step, last_step=last_step)
    placement = build_placement(
        loads,
        source_checkpoint=source_checkpoint,
        expert_axis_size=expert_axis_size,
        loads_from=f"{wandb_run} steps {first_step}..{last_step}",
    )
    identity = np.broadcast_to(np.arange(loads.shape[1]), loads.shape)
    balanced = np.asarray(placement.order)
    for capacity_factor in capacity_factors:
        logger.info(
            "capacity factor %g: projected drops %.4f in expert-id order, %.4f placed (in-sample)",
            capacity_factor,
            projected_drop_fraction(
                loads, identity, block_experts=placement.block_experts, capacity_factor=capacity_factor
            ),
            projected_drop_fraction(
                loads, balanced, block_experts=placement.block_experts, capacity_factor=capacity_factor
            ),
        )
    out_path.write_text(placement.to_json() + "\n")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
