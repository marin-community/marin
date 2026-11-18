# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving parameter receipts and replica-group comparisons."""

import hashlib
from dataclasses import asdict, dataclass

import torch


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
