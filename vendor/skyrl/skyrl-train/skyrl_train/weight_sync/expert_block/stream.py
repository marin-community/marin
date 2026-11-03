"""Run the broadcasts of one sync on one participant.

Every participant walks the same schedule and runs only the broadcasts of groups it belongs
to, so all members of a group issue them in the same order. A broadcast writes directly into
the live parameter. The router weight is the exception: it is BF16 on the wire and FP32 in
vLLM, so it is received into scratch and copied.
"""

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import LOCAL_GROUP_PREFIX, DenseBroadcast, ExpertBroadcast, Schedule
from skyrl_train.weight_sync.expert_block.source_views import (
    ExpertSource,
    dense_installed_view,
    dense_source_view,
    expert_slot_view,
    expert_source_view,
)


def storage_identity(tensors: dict[str, torch.Tensor]) -> dict[str, tuple]:
    """Each tensor's storage address, shape, stride, dtype and device. A sync refuses to run if these changed."""
    return {
        name: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
        for name, value in tensors.items()
    }


@dataclass(frozen=True)
class InstallReport:
    """What one participant sent or received in a sync, and how long the expert and dense phases took."""

    participant: int
    version: int
    expert_matrices: int
    wire_bytes: int
    seconds: float
    expert_seconds: float = 0.0
    dense_seconds: float = 0.0


@dataclass(frozen=True)
class Landing:
    """Where a receiver writes a transfer, and the dtype it has on the wire."""

    installed: torch.Tensor
    wire_dtype: torch.dtype

    @property
    def direct(self) -> bool:
        return self.installed.dtype == self.wire_dtype

    @property
    def nbytes(self) -> int:
        return self.installed.numel() * torch.empty(0, dtype=self.wire_dtype).element_size()


class Stream:
    """One participant's groups, tensors and schedule, ready to run syncs."""

    def __init__(
        self,
        participant: int,
        schedule: Schedule,
        groups: dict[str, dist.ProcessGroup],
        *,
        sources: dict[str, torch.Tensor] | None = None,
        expert_sources: dict[str, ExpertSource] | None = None,
        parameters: dict[str, torch.Tensor] | None = None,
        expert_maps: dict[str, tuple[int, ...]] | None = None,
        device: torch.device,
    ):
        self.participant = participant
        self.schedule = schedule
        self.groups = groups
        self.trainer = participant < schedule.trainer_count
        self.sources = sources if self.trainer else None
        self.expert_sources = expert_sources if self.trainer else None
        self.parameters = None if self.trainer else parameters
        self.expert_maps = None if self.trainer else expert_maps
        self.device = device
        self.local = self.local_group()
        # Resolve every view now, so a bad layout fails at startup and not during a sync.
        scratch_bytes = 0
        for item, landing in self.transfers():
            if landing is None:
                self.source_view(item)
            elif not landing.direct:
                scratch_bytes = max(scratch_bytes, landing.nbytes)
        self.scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)

    def lands(self, item: DenseBroadcast) -> bool:
        """Whether this receiver's stage holds the dense tensor."""
        return not self.trainer and self.local is not None and self.local[0] in item.local_groups

    def expert_landing(self, broadcast: ExpertBroadcast) -> Landing:
        return Landing(expert_slot_view(broadcast.entry, self.parameters, self.expert_maps), torch.bfloat16)

    def dense_landing(self, item: DenseBroadcast) -> Landing:
        return Landing(dense_installed_view(item.source, self.parameters), getattr(torch, item.source.wire_dtype))

    def source_view(self, item) -> torch.Tensor:
        """The trainer tensor for an item this participant sends."""
        if isinstance(item, ExpertBroadcast):
            return expert_source_view(self.expert_sources[item.entry.name], self.sources)
        return dense_source_view(item.source, self.sources)

    def wire_tensor(self, landing: Landing) -> torch.Tensor:
        """The tensor the broadcast writes: the destination itself, or scratch of the same shape."""
        if landing.direct:
            return landing.installed
        return self.scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)

    def run(self, version: int) -> InstallReport:
        with torch.no_grad():
            return self._run(version)

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def transfers(self):
        """This participant's items, in schedule order.

        Yields ``(item, landing)``. ``landing`` is None when this participant sends the item.
        """
        for broadcast in self.schedule.experts:
            if broadcast.root == self.participant:
                yield broadcast, None
            elif self.participant in broadcast.destinations:
                yield broadcast, self.expert_landing(broadcast)
        for item in self.schedule.dense:
            if item.root == self.participant:
                yield item, None
            elif self.lands(item):
                yield item, self.dense_landing(item)

    def send(self, item) -> int:
        """Broadcast one item from this participant. Returns the bytes sent."""
        dist.broadcast(self.source_view(item), src=0, group=self.groups[item.group])
        return item.entry.nbytes if isinstance(item, ExpertBroadcast) else item.source.nbytes

    def receive(self, item, tensor: torch.Tensor) -> int:
        """Receive one item into ``tensor``. Returns the bytes received."""
        if isinstance(item, ExpertBroadcast):
            dist.broadcast(tensor, src=0, group=self.groups[item.group])
            return item.entry.nbytes
        # One receiver per replica gets the slice from the root. It then broadcasts it to the
        # rest of its stage.
        if self.participant in item.landings:
            dist.broadcast(tensor, src=0, group=self.groups[item.group])
        local_name, members = self.local
        origin = next(rank for rank in item.landings if rank in members)
        dist.broadcast(tensor, src=members.index(origin), group=self.groups[local_name])
        return item.source.nbytes

    def _run(self, version: int) -> InstallReport:
        self._sync_device()
        started = time.perf_counter()
        matrices = wire_bytes = 0
        experts_done = None
        for item, landing in self.transfers():
            if experts_done is None and isinstance(item, DenseBroadcast):
                self._sync_device()
                experts_done = time.perf_counter()
            if landing is None:
                wire_bytes += self.send(item)
            else:
                tensor = self.wire_tensor(landing)
                wire_bytes += self.receive(item, tensor)
                if not landing.direct:
                    landing.installed.copy_(tensor)
            if isinstance(item, ExpertBroadcast):
                matrices += 1
        self._sync_device()
        finished = time.perf_counter()
        if experts_done is None:
            experts_done = finished
        return InstallReport(
            self.participant,
            version,
            matrices,
            wire_bytes,
            finished - started,
            expert_seconds=experts_done - started,
            dense_seconds=finished - experts_done,
        )

    def local_group(self) -> tuple[str, tuple[int, ...]] | None:
        for group in self.schedule.groups:
            if group.name.startswith(LOCAL_GROUP_PREFIX) and self.participant in group.members:
                return group.name, group.members
        return None


def bind(participant: int, plan: Schedule, rendezvous: Rendezvous, device: torch.device, **tensors) -> tuple:
    """Create and warm this participant's groups and build its ``Stream``.

    ``tensors`` are this side's ``Stream`` keyword arguments. If anything fails, the groups already
    created are destroyed. Returns ``(groups, stream, warm-up seconds)``.
    """
    groups = create_groups(participant, plan.groups, rendezvous)
    try:
        warm = warm_groups(participant, plan.groups, groups, device)
        stream = Stream(participant, plan, groups, device=device, **tensors)
    except BaseException:
        destroy_groups(groups)
        raise
    return groups, stream, warm
