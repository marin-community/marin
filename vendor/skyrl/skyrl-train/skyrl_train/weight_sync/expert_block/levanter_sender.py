# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch transport for a model-free parent of a Levanter learner process.

The JAX child streams canonical source slices through its existing subprocess
channel. This parent stages one slice on its source GPU and uses the unchanged
expert-block schedule, communicators, receiver, and full receiver-byte replay.
The JAX and rollout environments therefore keep their own CUDA/NCCL libraries.
"""

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import WIRE_DTYPE_BYTES, DenseBroadcast, ExpertBroadcast, Schedule, from_wire
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from skyrl_train.weight_sync.expert_block.verify_weights import ReplayReport, ReplicaReport

MAX_PUBLICATION_BYTES = 64 << 20


def _expert_payload_key(name: str) -> str:
    return f"expert:{name}"


def _dense_payload_key(name: str, offset: int, numel: int, dtype: str) -> str:
    return f"dense:{name}:{offset}:{numel}:{dtype}"


def _payload_key(item: ExpertBroadcast | DenseBroadcast) -> str:
    if isinstance(item, ExpertBroadcast):
        return _expert_payload_key(item.entry.name)
    source = item.source
    return _dense_payload_key(source.hf_name, source.hf_offset, source.numel, source.wire_dtype)


def _wire_bytes(item: ExpertBroadcast | DenseBroadcast) -> int:
    return item.entry.nbytes if isinstance(item, ExpertBroadcast) else item.source.nbytes


class LevanterExpertBlockSender:
    """Expose the existing sender RPCs for all local logical trainer ranks.

    ``publication`` is the actor's synchronous child request callback. It accepts
    ``inventory``, ``begin``, ``next`` and ``end`` with a payload dictionary. Each
    public RPC returns a tuple ordered by global rank.
    """

    def __init__(self, publication: Callable[[str, dict], dict | None]):
        self.publication = publication
        self.devices: dict[int, torch.device] = {}
        self.inventories: dict[int, dict] = {}
        self.payload_order: tuple[tuple[int, str], ...] = ()
        self.schedule: Schedule | None = None
        self.groups: dict[int, dict[str, dist.ProcessGroup]] = {}
        self.group_order: list[tuple[int, str]] = []

    def inventory(self) -> tuple[dict, ...]:
        metadata = self.publication("inventory", {})
        if metadata is None:
            raise RuntimeError("The Levanter child did not report its source inventory")
        inventories = {int(row["trainer"]["rank"]): row for row in metadata["inventories"]}
        device_type = metadata["device_type"]
        if device_type not in ("cpu", "cuda"):
            raise ValueError("Levanter sources must be on CPU or CUDA devices")
        devices = {int(rank): torch.device(device_type, int(local)) for rank, local in metadata["devices"].items()}
        if not inventories or set(inventories) != set(devices) or len(set(devices.values())) != len(devices):
            raise ValueError("Levanter source ranks must identify distinct local devices")
        payload_order = tuple(metadata["payload_order"])
        if self.schedule is not None and (
            inventories != self.inventories or devices != self.devices or payload_order != self.payload_order
        ):
            raise RuntimeError("Levanter publication topology changed after communicator setup")
        self.inventories = inventories
        self.devices = devices
        self.payload_order = payload_order
        return tuple(inventories[rank] for rank in sorted(inventories))

    def trainer_init(self, init_info: dict) -> tuple[dict, ...]:
        if not self.devices or self.schedule is not None:
            raise RuntimeError("Levanter sender needs an inventory and exactly one communicator setup")
        schedule = from_wire(Schedule, init_info["schedule"])
        self._check_schedule(schedule)
        rendezvous = Rendezvous(**init_info["rendezvous"])
        self.groups = {rank: {} for rank in self.devices}
        warmed = {rank: {} for rank in self.devices}
        try:
            # A host represents several roots. Both creation and warmup must
            # follow the receiver's global group order, rather than completing
            # one root before entering the next root's groups on the same host.
            # Every transfer group contains exactly one trainer participant.
            for group in schedule.groups:
                rank = group.members[0]
                if rank not in self.devices:
                    continue
                if any(member < schedule.trainer_count for member in group.members[1:]):
                    raise ValueError("A Levanter transfer group must contain exactly one trainer")
                device = self.devices[rank]
                with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                    self.groups[rank].update(
                        create_groups(rank, (group,), rendezvous, backend=init_info["backend"])
                    )
                    self.group_order.append((rank, group.name))
            for group in schedule.groups:
                rank = group.members[0]
                if rank in self.devices:
                    device = self.devices[rank]
                    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                        warmed[rank].update(
                            warm_groups(rank, schedule.groups, {group.name: self.groups[rank][group.name]}, device)
                        )
        except BaseException:
            self.shutdown()
            raise
        self.schedule = schedule
        return tuple({"participant": rank, "warmup_seconds": warmed[rank]} for rank in sorted(self.devices))

    def _check_schedule(self, schedule: Schedule) -> None:
        expected = {}
        for rank, row in self.inventories.items():
            for entry in row["experts"]:
                expected[(rank, _expert_payload_key(entry["name"]))] = entry["nbytes"]
            for source in row["dense"]:
                dtype = source["wire_dtype"]
                key = _dense_payload_key(source["hf_name"], source["hf_offset"], source["numel"], dtype)
                expected[(rank, key)] = source["numel"] * WIRE_DTYPE_BYTES[dtype]
        actual = {}
        payload_order = []
        for item in (*schedule.experts, *schedule.dense):
            if item.root in self.devices:
                key = (item.root, _payload_key(item))
                actual[key] = _wire_bytes(item)
                if not payload_order or payload_order[-1] != key:
                    payload_order.append(key)
        if actual != expected:
            raise ValueError("The expert-block schedule differs from the Levanter child source inventory")
        if tuple(payload_order) != self.payload_order:
            raise ValueError("The expert-block schedule differs from the Levanter child payload order")
        if any(size > MAX_PUBLICATION_BYTES for size in actual.values()):
            raise ValueError("A scheduled Levanter source exceeds the staging budget")

    def send_weights(self, update_info: dict) -> tuple[dict, ...]:
        return tuple(asdict(report) for report in self._send(update_info["version"]))

    def verify(self, update_info: dict) -> tuple[dict, ...]:
        """Replay source bytes for receiver comparison.

        The trainer reports zero receive bytes and zero replica bytes because
        each shard has exactly one trainer owner.
        """
        version = update_info["version"]
        reports = self._send(version)
        return tuple(
            {
                "replay": asdict(ReplayReport(report.participant, version, 0, 0, 0)),
                "replicas": asdict(ReplicaReport(report.participant, version, 0, 0)),
                "source_replication": "partitioned",
            }
            for report in reports
        )

    @torch.no_grad()
    def _send(self, version: int) -> tuple[InstallReport, ...]:
        if self.schedule is None:
            raise RuntimeError("Levanter sender communicators have not been prepared")
        started = time.perf_counter()
        begun = self.publication("begin", {"version": version})
        if begun is None or begun["version"] != version:
            raise RuntimeError("The Levanter child did not freeze the requested policy version")
        counts = {rank: [0, 0] for rank in self.devices}
        dense_started = None
        previous_key = None
        wire = None
        try:
            for item in (*self.schedule.experts, *self.schedule.dense):
                if dense_started is None and isinstance(item, DenseBroadcast):
                    dense_started = time.perf_counter()
                if item.root not in self.devices:
                    continue
                key = (item.root, _payload_key(item))
                if key != previous_key:
                    # Drop the previous GPU staging buffer before allocating another.
                    del wire
                    payload = self.publication("next", {})
                    wire = self._stage(item, payload, version)
                    del payload
                    previous_key = key
                device = self.devices[item.root]
                with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                    dist.broadcast(wire, src=0, group=self.groups[item.root][item.group])
                    # NCCL may still be reading the source when broadcast returns.
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                counts[item.root][0] += isinstance(item, ExpertBroadcast)
                counts[item.root][1] += _wire_bytes(item)
            del wire
            if self.publication("next", {}) is not None:
                raise RuntimeError("Levanter child emitted sources outside the expert-block schedule")
        except BaseException:
            self.publication("end", {})
            raise
        ended = self.publication("end", {})
        if ended is None or ended["version"] != version or not ended["complete"]:
            raise RuntimeError("Levanter child did not finish the scheduled policy publication")
        finished = time.perf_counter()
        expert_end = dense_started if dense_started is not None else finished
        return tuple(
            InstallReport(
                rank,
                version,
                counts[rank][0],
                counts[rank][1],
                finished - started,
                expert_seconds=expert_end - started,
                dense_seconds=finished - expert_end,
            )
            for rank in sorted(self.devices)
        )

    def _stage(self, item: ExpertBroadcast | DenseBroadcast, payload: dict | None, version: int) -> torch.Tensor:
        if payload is None:
            raise RuntimeError("The Levanter child stopped before the source schedule was complete")
        dtype = "bfloat16" if isinstance(item, ExpertBroadcast) else item.source.wire_dtype
        data = payload["data"]
        if (
            payload["version"] != version
            or payload["root"] != item.root
            or payload["key"] != _payload_key(item)
            or payload["dtype"] != dtype
            or len(data) != _wire_bytes(item)
            or len(data) > MAX_PUBLICATION_BYTES
        ):
            raise RuntimeError("Levanter source payload differs from the frozen version or transfer schedule")
        if isinstance(item, ExpertBroadcast):
            model = self.inventories[item.root]["model"]
            shape = (
                (2 * model["intermediate_size"], model["expert_hidden_size"])
                if item.entry.projection == "fc1"
                else (model["expert_hidden_size"], model["intermediate_size"])
            )
        else:
            shape = (item.source.numel,)
        if tuple(payload["shape"]) != shape:
            raise RuntimeError("Levanter source shape differs from the receiver matrix or dense slice")
        # The owned mutable CPU buffer avoids aliasing an immutable IPC frame.
        host = torch.frombuffer(bytearray(data), dtype=getattr(torch, dtype)).reshape(shape)
        return host.to(device=self.devices[item.root], non_blocking=False)

    def shutdown(self) -> tuple[dict, ...]:
        for rank, name in reversed(self.group_order):
            device = self.devices[rank]
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                destroy_groups({name: self.groups[rank].pop(name)})
        self.group_order.clear()
        self.groups.clear()
        self.schedule = None
        return tuple({"participant": rank} for rank in sorted(self.devices))
