# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving parameter receipts and replica-group comparisons."""

import hashlib
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ParameterReceipt:
    rank: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    group: tuple[int, ...]
    hf_names: tuple[str, ...]
    nbytes: int
    sha256: str


def parameter_sha256(parameter: torch.Tensor, chunk_bytes: int = 16 << 20) -> str:
    """Hash stored bytes without a floating-point conversion or a full host copy."""
    raw = parameter.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), chunk_bytes):
        digest.update(raw[start : start + chunk_bytes].cpu().numpy().tobytes())
    return digest.hexdigest()


def comparator_snapshot(sources: dict[str, torch.Tensor], groups: dict[str, dist.ProcessGroup]) -> dict:
    """Capture the live sender storage and the exact groups passed to comparison."""
    bindings = {}
    for group in groups.values():
        identity = str(id(group))
        if identity in bindings:
            continue
        bindings[identity] = {
            "local_object_id": id(group),
            "group_name": group.group_name,
            "group_description": group.group_desc,
            "members": dist.get_process_group_ranks(group),
            "backend": dist.get_backend(group),
            "group_rank": dist.get_rank(group),
        }
    return {
        "rank": dist.get_rank(),
        "groups": bindings,
        "parameters": [
            {
                "name": name,
                "group_object_id": str(id(groups[name])),
                "tensor_object_id": id(source),
                "data_ptr": source.data_ptr(),
                "storage_ptr": source.untyped_storage().data_ptr(),
                "storage_offset": source.storage_offset(),
                "stride": list(source.stride()),
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "device": str(source.device),
                "nbytes": source.numel() * source.element_size(),
                "sha256": parameter_sha256(source),
            }
            for name, source in sources.items()
        ],
    }


def known_group_minmax(groups: dict[str, dist.ProcessGroup], device: torch.device) -> list[dict]:
    """Check known int32 inputs on the comparison's live process-group handles."""
    unique = {id(group): group for group in groups.values()}
    rank = dist.get_rank()
    reports = []
    for identity, group in unique.items():
        members = dist.get_process_group_ranks(group)
        for count in [1024, 4194305, 16777216]:
            base = torch.arange(count, dtype=torch.int32, device=device).remainder_(128)
            source = base + rank
            before = parameter_sha256(source)
            low, high = source.clone(), source.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
            low_errors = int(low.ne(base + min(members)).sum().item())
            high_errors = int(high.ne(base + max(members)).sum().item())
            after = parameter_sha256(source)
            reports.append(
                {
                    "group_object_id": str(identity),
                    "group_name": group.group_name,
                    "members": members,
                    "elements": count,
                    "input_sha256": before,
                    "input_after_sha256": after,
                    "min_sha256": parameter_sha256(low),
                    "max_sha256": parameter_sha256(high),
                    "min_errors": low_errors,
                    "max_errors": high_errors,
                    "passed": low_errors == high_errors == 0 and before == after,
                }
            )
    return reports


def replica_differences(receipts: list[ParameterReceipt]) -> list[dict]:
    """Report missing members, layout differences and byte differences within each group."""
    parameters: dict[tuple[str, tuple[int, ...]], list[ParameterReceipt]] = {}
    for receipt in receipts:
        parameters.setdefault((receipt.name, receipt.group), []).append(receipt)
    differences = []
    for (name, group), members in sorted(parameters.items()):
        layouts = {(row.shape, row.dtype, row.hf_names, row.nbytes) for row in members}
        if (
            sorted(row.rank for row in members) == list(group)
            and len(layouts) == 1
            and len({row.sha256 for row in members}) == 1
        ):
            continue
        differences.append({"name": name, "group": group, "members": [asdict(row) for row in members]})
    return differences


def replica_iteration_differences(receipts: list[ParameterReceipt], source_orders: dict[int, list[str]]) -> list[dict]:
    """Find group members that would compare different tensors at the same collective."""
    memberships = {(row.rank, row.name): row.group for row in receipts}
    groups: dict[tuple[int, ...], dict[int, list[str]]] = {}
    for rank, names in source_orders.items():
        for name in names:
            group = memberships[rank, name]
            groups.setdefault(group, {}).setdefault(rank, []).append(name)
    return [
        {"group": group, "source_orders": orders}
        for group, orders in sorted(groups.items())
        if sorted(orders) != list(group) or len({tuple(order) for order in orders.values()}) != 1
    ]
