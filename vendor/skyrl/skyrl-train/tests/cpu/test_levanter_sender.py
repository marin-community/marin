# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the parent IPC contract against real expert-block receiver streams."""

import multiprocessing
import traceback
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, destroy_groups
from skyrl_train.weight_sync.expert_block.levanter_sender import LevanterExpertBlockSender
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    TrainerRank,
    build_schedule,
    to_wire,
)
from skyrl_train.weight_sync.expert_block.stream import bind
from skyrl_train.weight_sync.expert_block.verify_weights import replay


def _sources(version):
    matrices = {}
    for expert in range(16):
        for projection, shape in (("fc1", (6, 2)), ("fc2", (2, 3))):
            name = f"model.layers.0.mlp.experts.{projection}.expert{expert}"
            value = torch.arange(shape[0] * shape[1]).reshape(shape) + expert * 32 + version * 512
            matrices[name] = value.to(torch.bfloat16)
    dense = {
        "model.embed_tokens.weight": (torch.arange(16).reshape(8, 2) + version * 128).to(torch.bfloat16),
        "model.layers.0.mlp.router.bias": torch.arange(16, dtype=torch.float32) * 0.100123 + version,
    }
    return matrices, dense


class _ChildChannel:
    """An in-memory child IPC boundary; no transport or receiver operations are faked."""

    def __init__(self):
        self.version = 0
        self.position = 0
        self.payloads = []
        self.rows = []
        self.experts = {}
        self.dense = {}
        matrices, dense = _sources(0)
        for rank in range(4):
            experts = [
                ExpertEntry(name, 0, 0, int(name.rsplit("expert", 1)[1]), name.split(".")[-2], value.numel() * 2)
                for name, value in matrices.items()
                if int(name.rsplit("expert", 1)[1]) // 4 == rank
            ]
            slices = [DenseSlice("model.embed_tokens.weight", rank * 4, 4, "bfloat16", "embedding", 0, 0)]
            if rank == 0:
                slices.append(DenseSlice("model.layers.0.mlp.router.bias", 0, 16, "float32", "bias", 0, 0))
            self.experts[rank] = experts
            self.dense[rank] = slices
            self.rows.append(
                {
                    "trainer": asdict(TrainerRank(rank, 0, 0, rank)),
                    "expert_parallel_size": 4,
                    "layers": [0],
                    "model": {"num_experts": 16, "hidden_size": 2, "expert_hidden_size": 2, "intermediate_size": 3},
                    "experts": [asdict(entry) for entry in experts],
                    "dense": [asdict(source) for source in slices],
                }
            )
        self.dense_shapes = {name: tuple(value.shape) for name, value in dense.items()}

    def __call__(self, method, payload):
        if method == "inventory":
            self("begin", {"version": self.version})
            return {
                "inventories": self.rows,
                "devices": dict(enumerate(range(4))),
                "device_type": "cpu",
                "payload_order": tuple((item["root"], item["key"]) for item in self.payloads),
            }
        if method == "begin":
            assert payload["version"] == self.version
            self.position = 0
            self.payloads = []
            matrices, dense = _sources(self.version)
            for name, value in sorted(matrices.items()):
                root = int(name.rsplit("expert", 1)[1]) // 4
                self.payloads.append(self._payload(root, f"expert:{name}", value))
            for name in sorted(dense):
                for rank, slices in self.dense.items():
                    for source in slices:
                        if source.hf_name == name:
                            value = dense[name].reshape(-1)[source.hf_offset : source.hf_offset + source.numel]
                            key = f"dense:{name}:{source.hf_offset}:{source.numel}:{source.wire_dtype}"
                            self.payloads.append(self._payload(rank, key, value))
            return {"version": self.version}
        if method == "next":
            if self.position == len(self.payloads):
                return None
            result = self.payloads[self.position]
            self.position += 1
            return result
        if method == "end":
            return {"version": self.version, "complete": self.position == len(self.payloads)}
        raise AssertionError(method)

    def _payload(self, root, key, value):
        return {
            "root": root,
            "key": key,
            "version": self.version,
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": tuple(value.shape),
            "data": value.contiguous().view(torch.uint8).numpy().tobytes(),
        }


def _receiver(ep, schedule, rendezvous, result):
    torch.set_num_threads(1)
    groups = {}
    try:
        _, dense = _sources(0)
        parameters = {name: torch.full_like(value, torch.nan) for name, value in dense.items()}
        prefix = "model.layers.0.mlp.experts.routed_experts"
        parameters[f"{prefix}.w13_weight"] = torch.full((8, 6, 2), torch.nan, dtype=torch.bfloat16)
        parameters[f"{prefix}.w2_weight"] = torch.full((8, 2, 3), torch.nan, dtype=torch.bfloat16)
        expert_maps = {prefix: tuple(index - 8 * ep if 8 * ep <= index < 8 * (ep + 1) else -1 for index in range(16))}
        groups, stream, _ = bind(
            schedule.trainer_count + ep,
            schedule,
            rendezvous,
            torch.device("cpu"),
            backend="gloo",
            parameters=parameters,
            expert_maps=expert_maps,
        )
        receipts = []
        for version in (0, 1):
            installed = stream.run(version)
            verified = replay(stream, version)
            matrices, dense = _sources(version)
            for name, expected in dense.items():
                torch.testing.assert_close(parameters[name], expected, rtol=0, atol=0)
            for expert in range(8 * ep, 8 * (ep + 1)):
                for projection, target in (("fc1", "w13_weight"), ("fc2", "w2_weight")):
                    name = f"model.layers.0.mlp.experts.{projection}.expert{expert}"
                    torch.testing.assert_close(parameters[f"{prefix}.{target}"][expert - 8 * ep], matrices[name], rtol=0, atol=0)
            receipts.append({"installed": asdict(installed), "verified": asdict(verified)})
        result.put({"ep": ep, "receipts": receipts})
    except BaseException:
        result.put({"ep": ep, "error": traceback.format_exc()})
    finally:
        destroy_groups(groups)


def _schedule(child, rows):
    return build_schedule(
        [TrainerRank(**row["trainer"]) for row in rows],
        [ReceiverRank(0, 0, 0), ReceiverRank(1, 0, 1)],
        child.experts,
        child.dense,
        receiver_ep=2,
        num_experts=16,
        receiver_layers_by_pp=((0,),),
        dense_holders={name: (0,) for name in child.dense_shapes},
        dense_numel={name: torch.Size(shape).numel() for name, shape in child.dense_shapes.items()},
    )


def test_levanter_sender_rejects_out_of_order_schedule_before_creating_groups():
    child = _ChildChannel()
    sender = LevanterExpertBlockSender(child)
    schedule = _schedule(child, sender.inventory())
    reordered = replace(schedule, experts=tuple(reversed(schedule.experts)))
    with pytest.raises(ValueError, match="payload order"):
        sender.trainer_init({"schedule": to_wire(reordered)})


@pytest.mark.timeout(60)
def test_levanter_sender_two_versions_install_and_replay_all_receiver_bytes():
    child = _ChildChannel()
    sender = LevanterExpertBlockSender(child)
    schedule = _schedule(child, sender.inventory())
    store = dist.TCPStore("127.0.0.1", 0, world_size=None, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=25))
    rendezvous = Rendezvous("127.0.0.1", store.port, "levanter-publication-test", 25)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    processes = [context.Process(target=_receiver, args=(ep, schedule, rendezvous, result)) for ep in range(2)]
    for process in processes:
        process.start()
    try:
        bound = sender.trainer_init({"schedule": to_wire(schedule), "rendezvous": asdict(rendezvous), "backend": "gloo"})
        assert [row["participant"] for row in bound] == list(range(4))
        root_bytes = Counter()
        for item in schedule.experts:
            root_bytes[item.root] += item.entry.nbytes
        for item in schedule.dense:
            root_bytes[item.root] += item.source.nbytes
        for version in (0, 1):
            child.version = version
            installed = sender.send_weights({"version": version})
            assert {row["participant"]: row["wire_bytes"] for row in installed} == dict(root_bytes)
            verified = sender.verify({"version": version})
            assert all(row["source_replication"] == "partitioned" and row["replicas"]["compared_bytes"] == 0 for row in verified)
        for _ in processes:
            receipt = result.get(timeout=25)
            assert "error" not in receipt, receipt.get("error")
            for row in receipt["receipts"]:
                installed, verified = row["installed"], row["verified"]
                participant = installed["participant"]
                assert installed["wire_bytes"] == dict(schedule.receiver_bytes)[participant]
                assert installed["expert_matrices"] == dict(schedule.receiver_experts)[participant]
                assert verified["compared_bytes"] == verified["parameter_bytes"] == dict(schedule.receiver_bytes)[participant]
                assert verified["mismatched_bytes"] == 0
    finally:
        sender.shutdown()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result.close()
    assert all(process.exitcode == 0 for process in processes)
