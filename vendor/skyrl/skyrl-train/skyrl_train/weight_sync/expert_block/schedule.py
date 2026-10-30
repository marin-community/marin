"""The schedule of broadcasts for expert-block weight sync.

Trainer ranks are Megatron ranks. Receivers are vLLM workers, one per expert-parallel rank,
pipeline stage and replica. Both sides run TP=1. Each expert matrix is broadcast once, from a
trainer rank that holds it to the receivers that serve that expert. There is one NCCL group
per (root, receiver stage, receiver expert block). Trainer and receiver EP sizes may differ.

Dense weights use the same groups. Each slice goes from a root of its stage to one receiver
per replica, which broadcasts it to the other workers of that stage over a node-local group.

Participants are numbered trainers first, by rank, then receivers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from typing import Any, get_args, get_origin

from skyrl_train.json_serialization import to_jsonable

BF16 = "bfloat16"
FP32 = "float32"
WIRE_DTYPE_BYTES = {BF16: 2, FP32: 4}
# Expert groups are named "expert-{root}-{stage}-{block}". A replica stage's own group is named
# "local-{replica}-{stage}"; the stream finds it by this prefix.
LOCAL_GROUP_PREFIX = "local-"


@dataclass(frozen=True)
class TrainerRank:
    """A Megatron rank's coordinates. They are informational: the inventory says what the rank holds."""

    rank: int
    dp: int
    pp: int
    ep: int


@dataclass(frozen=True)
class ReceiverRank:
    """A vLLM worker. ``pp`` is its pipeline stage."""

    rank: int
    replica: int
    ep: int
    pp: int = 0


@dataclass(frozen=True)
class ExpertEntry:
    """One expert matrix on a trainer rank. ``fc1`` is ``[gate;up]`` and ``fc2`` is ``down``."""

    name: str
    layer: int
    pp: int
    expert: int
    projection: str
    nbytes: int


@dataclass(frozen=True)
class DenseSlice:
    """A run of ``numel`` elements of a dense HF tensor, starting at ``hf_offset``, stored in one trainer parameter."""

    hf_name: str
    hf_offset: int
    numel: int
    wire_dtype: str
    source_key: str
    source_offset: int
    pp: int

    @property
    def nbytes(self) -> int:
        return self.numel * WIRE_DTYPE_BYTES[self.wire_dtype]

    def identity(self) -> tuple:
        """The fields that identify a transfer, whichever rank sends it."""
        return (self.hf_name, self.hf_offset, self.numel, self.wire_dtype, self.pp)


@dataclass(frozen=True)
class Group:
    name: str
    members: tuple[int, ...]


@dataclass(frozen=True)
class ExpertBroadcast:
    entry: ExpertEntry
    group: str
    root: int
    destinations: tuple[int, ...]


@dataclass(frozen=True)
class DenseBroadcast:
    source: DenseSlice
    group: str
    root: int
    # One receiver per replica gets the slice from the root, then broadcasts it on its local group.
    landings: tuple[int, ...]
    local_groups: tuple[str, ...]


@dataclass(frozen=True)
class Schedule:
    trainer_count: int
    groups: tuple[Group, ...]
    experts: tuple[ExpertBroadcast, ...]
    dense: tuple[DenseBroadcast, ...]
    # Per receiver: the bytes and the expert matrices it must receive in each sync.
    receiver_bytes: tuple[tuple[int, int], ...]
    receiver_experts: tuple[tuple[int, int], ...]


def receiver_participant(trainer_count: int, receiver: ReceiverRank) -> int:
    return trainer_count + receiver.rank


def build_schedule(
    trainers: Sequence[TrainerRank],
    receivers: Sequence[ReceiverRank],
    expert_inventories: Mapping[int, Sequence[ExpertEntry]],
    dense_inventories: Mapping[int, Sequence[DenseSlice]],
    *,
    receiver_ep: int,
    num_experts: int,
    receiver_layers_by_pp: Sequence[Sequence[int]],
    dense_holders: Mapping[str, Sequence[int]],
    dense_numel: Mapping[str, int],
) -> Schedule:
    """Build the schedule from the inventories. It touches no tensors and does no communication.

    ``expert_inventories`` and ``dense_inventories`` map each trainer rank to what it holds.
    ``receiver_layers_by_pp`` lists the layers of each receiver stage. ``dense_holders`` maps a
    dense HF tensor to the receiver stages that hold it, and ``dense_numel`` to its size there.
    """
    trainer_count = len(trainers)
    if sorted(row.rank for row in trainers) != list(range(trainer_count)):
        raise ValueError("Trainer native ranks must be 0..N-1")
    if sorted(row.rank for row in receivers) != list(range(len(receivers))):
        raise ValueError("Receiver native ranks must be 0..N-1")
    if set(expert_inventories) != set(range(trainer_count)) or set(dense_inventories) != set(range(trainer_count)):
        raise ValueError("Every trainer rank must report its inventory")
    if num_experts % receiver_ep:
        raise ValueError(f"{num_experts} experts do not split evenly across receiver EP {receiver_ep}")
    replica_count = len({row.replica for row in receivers})
    receiver_pp_count = len(receiver_layers_by_pp)
    receiver_at = {(row.replica, row.pp, row.ep): row for row in receivers}
    if len(receiver_at) != len(receivers) or set(receiver_at) != set(
        product(range(replica_count), range(receiver_pp_count), range(receiver_ep))
    ):
        raise ValueError("Incomplete or duplicate receiver topology")
    receiver_stage = _layer_stages(receiver_layers_by_pp, "receiver")

    def receivers_of(stage: int, block: int) -> tuple[int, ...]:
        return tuple(
            receiver_participant(trainer_count, receiver_at[replica, stage, block]) for replica in range(replica_count)
        )

    groups: dict[str, Group] = {}

    def group_for(root: int, stage: int, block: int) -> str:
        name = f"expert-{root}-{stage}-{block}"
        if name not in groups:
            groups[name] = Group(name, (root, *receivers_of(stage, block)))
        return name

    # --- Expert transfers: each (layer, expert, projection) once, from one rank that holds it ---
    holders: dict[tuple, list[int]] = {}
    entries: dict[tuple, ExpertEntry] = {}
    for rank in sorted(expert_inventories):
        for entry in expert_inventories[rank]:
            key = (entry.layer, entry.expert, entry.projection)
            if key in entries and entries[key] != entry:
                raise ValueError(f"Trainer ranks disagree on expert transfer {entry.name}")
            entries.setdefault(key, entry)
            holders.setdefault(key, []).append(rank)
    layers = {entry.layer for entry in entries.values()}
    if layers != set(receiver_stage):
        raise ValueError("Receiver stages do not hold exactly the trainer's layers")
    if set(entries) != set(product(sorted(layers), range(num_experts), ("fc1", "fc2"))):
        raise ValueError("Expert entries do not cover every layer, expert and projection exactly once")
    per_receiver_block = num_experts // receiver_ep
    experts = []
    expert_roots: dict[tuple[int, int], set[int]] = {}
    for key in sorted(entries, key=lambda item: entries[item].name):
        entry = entries[key]
        # --- Unequal EP sizes: the group is keyed by the receiver's expert block ---
        block = entry.expert // per_receiver_block
        stage = receiver_stage[entry.layer]
        ranks = holders[key]
        # The ranks holding one matrix are data-parallel copies. Rotate the root by stage and
        # block to spread the sending across them.
        root = ranks[(entry.pp + block) % len(ranks)]
        experts.append(ExpertBroadcast(entry, group_for(root, stage, block), root, receivers_of(stage, block)))
        expert_roots.setdefault((entry.pp, stage), set()).add(root)

    # --- Receiver pipeline stages: one local group per replica stage ---
    local_groups = {}
    for replica, stage in product(range(replica_count), range(receiver_pp_count)):
        name = f"{LOCAL_GROUP_PREFIX}{replica}-{stage}"
        local_groups[name] = Group(name, tuple(receivers_of(stage, block)[replica] for block in range(receiver_ep)))

    # --- Dense transfers: each slice once, from a rank that is already a root where possible ---
    dense_holder_ranks: dict[tuple, list[int]] = {}
    slices: dict[tuple, DenseSlice] = {}
    for rank in sorted(dense_inventories):
        for item in dense_inventories[rank]:
            key = item.identity()
            slices.setdefault(key, item)
            dense_holder_ranks.setdefault(key, []).append(rank)
    by_name: dict[str, list[DenseSlice]] = {}
    for item in slices.values():
        by_name.setdefault(item.hf_name, []).append(item)
    if set(by_name) != set(dense_holders) or set(by_name) != set(dense_numel):
        raise ValueError(
            f"Dense weights differ: trainer only {sorted(set(by_name) - set(dense_holders))}, "
            f"receiver only {sorted(set(dense_holders) - set(by_name))}"
        )
    dense_broadcasts = []
    counter = 0
    for name in sorted(by_name):
        _check_slices_cover(name, by_name[name], dense_numel[name])
        for item in sorted(by_name[name], key=lambda item: item.hf_offset):
            if item.wire_dtype not in WIRE_DTYPE_BYTES:
                raise ValueError(f"Dense slice of {name} has unsupported dtype {item.wire_dtype}")
            for stage in dense_holders[name]:
                candidates = sorted(
                    set(dense_holder_ranks[item.identity()]) & expert_roots.get((item.pp, stage), set())
                )
                if not candidates:
                    candidates = dense_holder_ranks[item.identity()]
                root = candidates[counter % len(candidates)]
                counter += 1
                # Reuse a group this root already has for the stage. Otherwise give it one for its own block.
                existing = [
                    group
                    for group in groups.values()
                    if group.members[0] == root and group.name.startswith(f"expert-{root}-{stage}-")
                ]
                if existing:
                    group_name = existing[0].name
                    block = int(group_name.rsplit("-", 1)[1])
                else:
                    block = root % receiver_ep
                    group_name = group_for(root, stage, block)
                dense_broadcasts.append(
                    DenseBroadcast(
                        item,
                        group_name,
                        root,
                        receivers_of(stage, block),
                        tuple(f"{LOCAL_GROUP_PREFIX}{replica}-{stage}" for replica in range(replica_count)),
                    )
                )

    receiver_bytes = {receiver_participant(trainer_count, row): 0 for row in receivers}
    receiver_experts = dict.fromkeys(receiver_bytes, 0)
    for broadcast in experts:
        for destination in broadcast.destinations:
            receiver_bytes[destination] += broadcast.entry.nbytes
            receiver_experts[destination] += 1
    for item in dense_broadcasts:
        stage = int(item.local_groups[0].rsplit("-", 1)[1])
        for block in range(receiver_ep):
            for participant in receivers_of(stage, block):
                receiver_bytes[participant] += item.source.nbytes
    return Schedule(
        trainer_count,
        tuple(groups.values()) + tuple(local_groups.values()),
        tuple(experts),
        tuple(dense_broadcasts),
        tuple(sorted(receiver_bytes.items())),
        tuple(sorted(receiver_experts.items())),
    )


def _layer_stages(layers_by_pp: Sequence[Sequence[int]], side: str) -> dict[int, int]:
    owner = {layer: pp for pp, layers in enumerate(layers_by_pp) for layer in layers}
    if sorted(owner) != list(range(sum(map(len, layers_by_pp)))):
        raise ValueError(f"Missing, duplicate or noncontiguous {side} layer ownership")
    return owner


def _check_slices_cover(name: str, slices: Sequence[DenseSlice], numel: int) -> None:
    """Check that the slices cover ``[0, numel)`` with no gap or overlap."""
    cursor = 0
    for item in sorted(slices, key=lambda item: item.hf_offset):
        if item.hf_offset != cursor or item.numel <= 0:
            raise ValueError(f"Dense slices of {name} have a gap or overlap at element {cursor}")
        cursor += item.numel
    if cursor != numel:
        raise ValueError(f"Trainer slices cover {cursor} of {numel} elements of {name}")


def to_wire(value: Any) -> Any:
    """Convert to JSON-compatible data: dataclasses become dicts and tuples become lists."""
    return to_jsonable(value)


def from_wire(kind: type, value: Any) -> Any:
    """Rebuild a ``kind`` from ``to_wire`` output, using the dataclass field types."""
    if is_dataclass(kind):
        hints = {field.name: field.type for field in fields(kind)}
        return kind(**{name: from_wire(hint, value[name]) for name, hint in hints.items()})
    origin = get_origin(kind)
    if origin is tuple:
        args = get_args(kind)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_wire(args[0], item) for item in value)
        return tuple(from_wire(arg, item) for arg, item in zip(args, value, strict=True))
    if not isinstance(value, kind):
        raise TypeError(f"Expected {kind.__name__}, got {type(value).__name__}")
    return value
