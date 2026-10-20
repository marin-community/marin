"""Driver side of expert-block weight sync.

``prepare`` runs once. It collects an inventory from every trainer rank and receiver,
builds the schedule, hosts the rendezvous store and tells every participant to create
its groups. ``sync`` runs after each update while generation is paused. It fails if a
participant raised, a receiver did not report, or a report has the wrong version or
byte count.
"""

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import time

from marinskyrl.inference_placement import InferenceReplicaPlacement
from skyrl_train.weight_sync.expert_block.groups import RendezvousStore
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    Schedule,
    TrainerRank,
    build_schedule,
    from_wire,
    receiver_participant,
    to_wire,
)
from skyrl_train.weight_sync.expert_block.source_views import is_widened_router
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from skyrl_train.weight_sync.expert_block.verify_weights import ReplayReport, ReplicaReport


@dataclass(frozen=True)
class SyncTimings:
    """Timings of one sync, in seconds.

    ``install_seconds`` is the driver's wall time. The others are the maximum over the
    participants' reports.
    """

    install_seconds: float
    policy_seconds: float
    receiver_seconds: float
    expert_seconds: float
    dense_seconds: float

    def as_metrics(self) -> dict[str, float]:
        return {f"expert_block_sync/{name}": value for name, value in asdict(self).items()}


def plan_from_inventories(
    policy: list[dict], receivers: list[tuple[dict, InferenceReplicaPlacement]]
) -> tuple[Schedule, dict[str, int]]:
    """Build the schedule from the trainer and receiver inventories.

    ``receivers`` pairs each worker's inventory with its verified placement. Returns the schedule
    and a map from GPU UUID to participant number.
    """
    trainers = sorted((from_wire(TrainerRank, row["trainer"]) for row in policy), key=lambda row: row.rank)
    rows = {from_wire(TrainerRank, row["trainer"]).rank: row for row in policy}
    models = {tuple(sorted(row["model"].items())) for row in policy}
    if len(models) != 1:
        raise ValueError("Trainer ranks disagree on model dimensions")
    model = dict(models.pop())
    expert_inventories = {rank: [from_wire(ExpertEntry, item) for item in row["experts"]] for rank, row in rows.items()}
    dense_inventories = {rank: [from_wire(DenseSlice, item) for item in row["dense"]] for rank, row in rows.items()}
    trainer_layers = {entry.layer for entries in expert_inventories.values() for entry in entries}

    receiver_ranks, participants_by_gpu = [], {}
    receiver_eps, receiver_models = set(), set()
    receiver_stages: dict[int, dict] = {}
    for report, placement in receivers:
        worker = placement.worker
        if report["gpu_uuid"] != worker.gpu_uuid or report["ep_rank"] != worker.ep_rank:
            raise ValueError(f"Receiver on {report['gpu_uuid']} does not match its verified placement {placement}")
        receiver_eps.add(report["expert_parallel_size"])
        receiver_models.add(tuple(sorted(report["model"].items())))
        stage = report["pp_rank"]
        reference = receiver_stages.setdefault(stage, report)
        if (reference["layers"], reference["dense"], reference["pp_size"]) != (
            report["layers"],
            report["dense"],
            report["pp_size"],
        ):
            raise ValueError(f"Receivers of stage {stage} disagree on their layers or dense parameters")
        receiver = ReceiverRank(
            (placement.replica * report["pp_size"] + stage) * report["expert_parallel_size"] + worker.ep_rank,
            placement.replica,
            worker.ep_rank,
            stage,
        )
        receiver_ranks.append(receiver)
        participants_by_gpu[worker.gpu_uuid] = receiver_participant(len(trainers), receiver)
    if len(receiver_eps) != 1 or len(receiver_models) != 1:
        raise ValueError("Receivers disagree on expert-parallel size or model dimensions")
    receiver_ep = receiver_eps.pop()
    receiver_model = dict(receiver_models.pop())
    if any(receiver_model[key] != model[key] for key in model) or receiver_model["num_hidden_layers"] != len(
        trainer_layers
    ):
        raise ValueError(f"Receiver model {receiver_model} differs from trainer model {model}")
    if sorted(receiver_stages) != list(range(next(iter(receiver_stages.values()))["pp_size"])):
        raise ValueError("Receiver pipeline stages are incomplete")
    receiver_layers_by_pp = tuple(tuple(receiver_stages[stage]["layers"]) for stage in sorted(receiver_stages))
    dense_holders: dict[str, list[int]] = {}
    dense_numel: dict[str, int] = {}
    wire_dtypes = {item.hf_name: item.wire_dtype for items in dense_inventories.values() for item in items}
    for stage in sorted(receiver_stages):
        for name, (shape, dtype) in receiver_stages[stage]["dense"].items():
            if name not in wire_dtypes:
                raise ValueError(f"Dense weights differ: the receiver holds {name}, which no trainer rank exports")
            if wire_dtypes[name] != dtype and not is_widened_router(name, wire_dtypes[name], dtype):
                raise ValueError(f"{name} is {wire_dtypes[name]} on the trainer and {dtype} on the receiver")
            numel = 1
            for dimension in shape:
                numel *= dimension
            if dense_numel.setdefault(name, numel) != numel:
                raise ValueError(f"Receiver stages disagree on the shape of {name}")
            dense_holders.setdefault(name, []).append(stage)
    schedule = build_schedule(
        trainers,
        sorted(receiver_ranks, key=lambda row: row.rank),
        expert_inventories,
        dense_inventories,
        receiver_ep=receiver_ep,
        num_experts=model["num_experts"],
        receiver_layers_by_pp=receiver_layers_by_pp,
        dense_holders=dense_holders,
        dense_numel=dense_numel,
    )
    return schedule, participants_by_gpu


class ExpertBlockSync:
    def __init__(self, *, policy_model, inference_engine_client, timeout_seconds: int):
        self.policy_model = policy_model
        self.client = inference_engine_client
        self.timeout_seconds = timeout_seconds
        self.store: RendezvousStore | None = None
        self.schedule: Schedule | None = None
        self.root_bytes: dict[int, int] = {}

    async def _policy(self, method: str, *args) -> list:
        refs = self.policy_model.async_run_ray_method("pass_through", "expert_block_rpc", method, *args)
        return list(await asyncio.gather(*refs))

    async def _receivers(self, method: str, *args) -> list:
        """Call a receiver method on every worker. Returns one reply per worker, in engine then worker order."""
        return [row for rows in await self.client.expert_block_rpc(method, *args) for row in rows]

    async def prepare(self) -> dict[str, float]:
        """Build the schedule and have every participant create its groups. Returns the seconds each phase took."""
        if self.schedule is not None:
            raise RuntimeError("Expert-block sync is already prepared")
        engines = self.client.engines
        if any(engine.worker_placements is None for engine in engines):
            raise ValueError(
                "Expert-block sync requires checked worker placements, which the engine factory records for "
                "non-colocated TP=1 vLLM engines using expert-block transport"
            )
        started = time.perf_counter()
        policy_rows, engine_rows = await asyncio.gather(
            self._policy("inventory"), self.client.expert_block_rpc("inventory")
        )
        if len(engine_rows) != len(engines):
            raise RuntimeError(f"{len(engines) - len(engine_rows)} inference engines did not report an inventory")
        receivers = []
        for engine, rows in zip(engines, engine_rows, strict=True):
            if len(rows) != len(engine.worker_placements):
                raise RuntimeError(
                    f"An engine reported {len(rows)} workers but {len(engine.worker_placements)} were placed"
                )
            receivers.extend(zip(rows, engine.worker_placements, strict=True))
        schedule, participants = plan_from_inventories(policy_rows, receivers)
        planned = time.perf_counter()
        self.store = RendezvousStore(f"expert-block-{id(self):x}", timeout_seconds=self.timeout_seconds)
        init_info = {"schedule": to_wire(schedule), "rendezvous": asdict(self.store.rendezvous)}
        try:
            bound = await asyncio.gather(
                self._policy("trainer_init", init_info),
                self._receivers("init_transfer_engine", {**init_info, "participants": participants}),
            )
        except BaseException:
            await self.close()
            raise
        reported = sorted(row["participant"] for rows in bound for row in rows)
        expected = sorted(set(range(schedule.trainer_count)) | set(participants.values()))
        if reported != expected:
            raise RuntimeError(f"Bound participants {reported} differ from the planned {expected}")
        self.schedule = schedule
        self.root_bytes = Counter()
        for item in schedule.experts:
            self.root_bytes[item.root] += item.entry.nbytes
        for item in schedule.dense:
            self.root_bytes[item.root] += item.source.nbytes
        return {"plan": planned - started, "bind": time.perf_counter() - planned}

    async def sync(self, version: int) -> SyncTimings:
        """Run one sync for ``version`` and check that every receiver got the bytes planned for it."""
        if self.schedule is None:
            raise RuntimeError("Expert-block sync is not prepared")
        if not self.client.generation_paused_event.is_set():
            raise RuntimeError("Expert-block sync requires paused generation")
        started = time.perf_counter()
        update_info = {"version": version}
        policy_rows, receiver_rows = await asyncio.gather(
            self._policy("send_weights", update_info), self._receivers("receive_weights", update_info)
        )
        policy = [InstallReport(**row) for row in policy_rows]
        receivers = [InstallReport(**row) for row in receiver_rows]
        expected_bytes = dict(self.schedule.receiver_bytes)
        expected_experts = dict(self.schedule.receiver_experts)
        if sorted(report.participant for report in receivers) != sorted(expected_bytes):
            raise RuntimeError("Not every planned receiver reported this sync")
        for report in receivers:
            if report.version != version:
                raise RuntimeError(
                    f"Receiver {report.participant} installed version {report.version}, expected {version}"
                )
            expected = (expected_experts[report.participant], expected_bytes[report.participant])
            if (report.expert_matrices, report.wire_bytes) != expected:
                raise RuntimeError(
                    f"Receiver {report.participant} landed {report.expert_matrices} expert matrices and "
                    f"{report.wire_bytes} bytes, expected {expected[0]} and {expected[1]}"
                )
        for report in policy:
            if report.version != version or report.wire_bytes != self.root_bytes.get(report.participant, 0):
                raise RuntimeError(
                    f"Trainer rank {report.participant} sent {report.wire_bytes} bytes for version {report.version}"
                )
        return SyncTimings(
            install_seconds=time.perf_counter() - started,
            policy_seconds=max(report.seconds for report in policy),
            receiver_seconds=max(report.seconds for report in receivers),
            expert_seconds=max(report.expert_seconds for report in (*policy, *receivers)),
            dense_seconds=max(report.dense_seconds for report in (*policy, *receivers)),
        )

    async def verify(self, version: int) -> dict[str, float]:
        """Replay the sync and fail if any byte differs, a parameter was not covered, or data-parallel peers disagree."""
        if self.schedule is None:
            raise RuntimeError("Expert-block sync is not prepared")
        if not self.client.generation_paused_event.is_set():
            raise RuntimeError("Expert-block verification requires paused generation")
        started = time.perf_counter()
        update_info = {"version": version}
        policy_rows, receiver_rows = await asyncio.gather(
            self._policy("verify", update_info), self._receivers("verify", update_info)
        )
        expected_bytes = dict(self.schedule.receiver_bytes)
        receivers = [ReplayReport(**row) for row in receiver_rows]
        if sorted(report.participant for report in receivers) != sorted(expected_bytes):
            raise RuntimeError("Not every planned receiver reported the replay")
        for report in receivers:
            if report.version != version or report.mismatched_bytes != 0:
                raise RuntimeError(
                    f"Receiver {report.participant}: {report.mismatched_bytes} of {report.compared_bytes} replayed "
                    f"bytes differ from what it installed for version {version}"
                )
            if (
                report.compared_bytes != expected_bytes[report.participant]
                or report.compared_bytes != report.parameter_bytes
            ):
                raise RuntimeError(
                    f"Receiver {report.participant} compared {report.compared_bytes} bytes; the schedule assigns "
                    f"{expected_bytes[report.participant]} and it holds {report.parameter_bytes} parameter bytes"
                )
        for row in policy_rows:
            replicas = ReplicaReport(**row["replicas"])
            if replicas.version != version or replicas.mismatched_bytes != 0:
                raise RuntimeError(
                    f"Trainer rank {replicas.participant} differs from its data-parallel peers on "
                    f"{replicas.mismatched_bytes} of {replicas.compared_bytes} bytes"
                )
        return {"verify_seconds": time.perf_counter() - started}

    async def close(self) -> None:
        try:
            await asyncio.gather(self._policy("shutdown"), self._receivers("shutdown"))
        finally:
            if self.store is not None:
                self.store.close()
                self.store = None
            self.schedule = None
