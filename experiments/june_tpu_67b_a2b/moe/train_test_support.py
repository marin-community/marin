# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model config, meshes, batches and HLO helpers shared by the June train-step tests."""

import collections
import re

import jax
import numpy as np
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

BASE = GrugModelConfig(
    vocab_size=32,
    hidden_dim=16,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=3,
    num_heads=2,
    num_kv_heads=1,
    head_dim=8,
    max_seq_len=8,
    sliding_window=4,
    capacity_factor=8.0,
    attention_implementation="reference",
    moe_implementation="ring_filtered",
    pooled_transport_capacity_factor=4.0,
    use_array_stacked_blocks=True,
    disable_pko=True,
    replicated_grad_reduction="after_scan",
    router_stats_reduction="after_scan",
)


def _meshes():
    return {
        "replica4_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=4),
        "replica2_data2_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=2),
    }


# 32 rows: four microbatches of one row per device.
TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (32, 8), 0, BASE.vocab_size))
LOSS_WEIGHT = np.ones((32, 8), np.float32)
LOSS_WEIGHT[:, -1] = 0
LOSS_WEIGHT[::3, :4] = 0  # Microbatches with unequal loss weight.
Batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")
BATCH = Batch(TOKENS, LOSS_WEIGHT, None)
# Rows 0, 3, 6, ... end in three padding positions, so the microbatches route unequal numbers of tokens.
SEGMENT_IDS = np.zeros((32, 8), np.int32)
SEGMENT_IDS[::3, -3:] = -1
PADDED_BATCH = Batch(
    TOKENS,
    np.where(SEGMENT_IDS >= 0, LOSS_WEIGHT, 0),
    AttentionMask(is_causal=True, segment_ids=(SEGMENT_IDS, SEGMENT_IDS)),
)
# Rows 0, 4, 8, ... carry no loss weight: microbatch 0 of four has none.
UNWEIGHTED_ROWS_WEIGHT = LOSS_WEIGHT.copy()
UNWEIGHTED_ROWS_WEIGHT[::4] = 0
UNWEIGHTED_MICROBATCH = Batch(TOKENS, UNWEIGHTED_ROWS_WEIGHT, None)


Reduction = collections.namedtuple("Reduction", "in_loop elements scalar op_name")
_REDUCTIONS = ("all-reduce", "all-reduce-start", "reduce-scatter")


def replica_reductions(hlo, mesh, ops=_REDUCTIONS):
    # The all-reduces and reduce-scatters (or other collectives ``ops``) of a compiled CPU module that cross
    # replica_dcn: whether each runs inside a while loop (the microbatch loop, the layer scans), how many elements it
    # reduces (its operands', so a reduce-scatter counts its output times its group size; an all-gather its output),
    # whether every operand is a scalar, and its source op.
    computations, current = {}, None
    for line in hlo.splitlines():
        header = re.match(r"(ENTRY )?%?([\w.\-]+) .*\{$", line)
        if header and not line.startswith(" "):
            current = header.group(2)
            computations[current] = (bool(header.group(1)), [])
        elif current and line.startswith(" "):
            computations[current][1].append(line.strip())
    # A computation runs in a loop if it is a while body or is called from one.
    in_loop, stack = {}, [(name, False) for name, (entry, _) in computations.items() if entry]
    while stack:
        name, looped = stack.pop()
        if in_loop.get(name, False) >= looped and name in in_loop:
            continue
        in_loop[name] = looped
        for line in computations[name][1]:
            body = re.search(r"\bbody=%?([\w.\-]+)", line)
            for callee in re.findall(r"%([\w.\-]+)", line):
                if callee in computations:
                    stack.append((callee, looped or (body is not None and callee == body.group(1))))
    coords = np.stack(np.unravel_index(np.arange(mesh.devices.size), mesh.devices.shape), axis=1)
    replica = mesh.axis_names.index("replica_dcn")
    # The step's own mesh as the HLO prints it.
    mesh_text = ",".join(f"'{axis}'={size}" for axis, size in mesh.shape.items())
    out = []
    for name, (_, lines) in computations.items():
        for line in lines:
            m = re.match(r"(?:ROOT )?%?[\w.\-]+ = (.+?) ([\w\-]+)\((.*)$", line)
            if not m or m.group(2) not in ops:
                continue
            shape, _, attrs = m.groups()
            named = re.search(
                r"replica_groups=mesh\[([^\]]*)\]" r"(?:, device_ids=\(\[([\d,]+)\](?:T\(([\d,]+)\))?\))?\s*\{([^}]*)\}",
                attrs,
            )
            # Iota groups: [groups, size]<=[dims], optionally T(permutation) of the dims.
            iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", attrs)
            if named and named.group(1) == mesh_text:
                sizes = {axis: int(size) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))}
                axes = re.findall(r"'(\w+)'", named.group(4))
                crosses = "replica_dcn" in axes
                group_size = int(np.prod([sizes[axis] for axis in axes]))
            elif named:
                # A mesh of XLA's own (axis_0, ...), with the device order in device_ids when it is not the iota.
                sizes = [(axis, int(size)) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))]
                names = [axis for axis, _ in sizes]
                axes = re.findall(r"'(\w+)'", named.group(4))
                group_size = int(np.prod([size for axis, size in sizes if axis in axes]))
                ids = np.arange(mesh.devices.size)
                if named.group(2):
                    ids = ids.reshape([int(d) for d in named.group(2).split(",")])
                    if named.group(3):
                        ids = ids.transpose([int(d) for d in named.group(3).split(",")])
                ids = ids.reshape([size for _, size in sizes])
                ids = np.moveaxis(ids, [names.index(axis) for axis in axes], list(range(-len(axes), 0)))
                groups = ids.reshape(-1, group_size).tolist()
                crosses = any(len({coords[int(d)][replica] for d in group}) > 1 for group in groups)
            else:
                if iota:
                    count, size, dims, permutation = iota.groups()
                    ids = np.arange(mesh.devices.size).reshape([int(d) for d in dims.split(",")])
                    if permutation:
                        ids = ids.transpose([int(d) for d in permutation.split(",")])
                    groups = ids.reshape(int(count), int(size)).tolist()
                else:
                    groups = re.search(r"replica_groups=\{(.*?)\}\}", attrs)
                    assert groups, line
                    groups = [group.split(",") for group in re.findall(r"\{([\d,]+)", groups.group(0))]
                crosses = any(len({coords[int(d)][replica] for d in group}) > 1 for group in groups)
                group_size = len(groups[0])
            if crosses:
                dims = [[int(d) for d in each.split(",") if d] for each in re.findall(r"\[([\d,]*)\]", shape)]
                elements = sum(int(np.prod(each)) for each in dims)
                if m.group(2) == "reduce-scatter":
                    elements *= group_size
                op_name = re.search(r'op_name="([^"]*)"', attrs)
                scalar = all(not each for each in dims)
                out.append(Reduction(in_loop[name], elements, scalar, op_name.group(1) if op_name else ""))
    return out


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]
