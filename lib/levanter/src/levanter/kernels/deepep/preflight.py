# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI preflight check for Levanter's DeepEP JAX FFI integration."""

from __future__ import annotations

import argparse
import sys

from levanter.kernels.deepep.availability import (
    INTERNODE_TRANSPORT_REQUIRED_FILES,
    LAYOUT_REQUIRED_FILES,
    TRANSPORT_REQUIRED_FILES,
    deepep_install_help,
    deepep_preflight_status,
)
from levanter.kernels.deepep.transport_ffi import TransportBuildMode, build_transport_library
from levanter.kernels.deepep.transport_ffi import preflight_internode_process_topology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("layout", "transport", "transport-internode"),
        default="transport",
        help="DeepEP FFI component to check. Transport includes layout dependencies.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Compile the selected transport shared library after environment checks pass.",
    )
    parser.add_argument(
        "--check-topology",
        action="store_true",
        help="For transport-internode, validate the current JAX process/GPU topology before building.",
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        default=None,
        help="GPU ranks per RDMA node for --check-topology, e.g. 8 for H100x8 EP16 on two nodes.",
    )
    args = parser.parse_args(argv)

    if args.component == "layout":
        required_files = LAYOUT_REQUIRED_FILES
        cache_component = "deepep_layout_ffi"
        requires_nvshmem = False
        requires_rdma = False
    elif args.component == "transport-internode":
        required_files = INTERNODE_TRANSPORT_REQUIRED_FILES
        cache_component = "deepep_transport_ffi"
        requires_nvshmem = True
        requires_rdma = True
    else:
        required_files = TRANSPORT_REQUIRED_FILES
        cache_component = "deepep_transport_ffi"
        requires_nvshmem = False
        requires_rdma = False
    status = deepep_preflight_status(
        required_files=required_files,
        component=cache_component,
        requires_nvshmem=requires_nvshmem,
        requires_rdma=requires_rdma,
    )

    print("DeepEP preflight")
    print(f"  component: {args.component}")
    print(f"  source_root: {status.source_root or '<unset>'}")
    print(f"  source_revision: {status.source_revision or '<unverified>'}")
    print(f"  cache_root: {status.cache_root}")
    print(f"  cuda_arch: {status.cuda_arch}")
    print(f"  nvcc: {status.nvcc_path or '<missing>'}")
    print(f"  nvshmem_dir: {status.nvshmem_dir or '<missing>'}")
    print(f"  nvshmem_host_lib: {status.nvshmem_host_lib or '<missing>'}")
    print(f"  nvshmem_device_lib: {status.nvshmem_device_lib or '<missing>'}")
    print(
        "  rdma_include_dirs: "
        + (", ".join(str(path) for path in status.rdma_include_dirs) if status.rdma_include_dirs else "<missing>")
    )
    print(
        "  missing_rdma_headers: "
        + (", ".join(status.missing_rdma_headers) if status.missing_rdma_headers else "<none>")
    )
    for warning in status.warnings:
        print(f"  warning: {warning}")
    for error in status.errors:
        print(f"  error: {error}")

    if not status.ok:
        print()
        print(deepep_install_help())
        return 1
    if args.check_topology:
        if args.component != "transport-internode":
            print("  error: --check-topology is only supported with --component transport-internode")
            return 1
        try:
            topology = preflight_internode_process_topology(ranks_per_node=args.ranks_per_node)
        except RuntimeError as exc:
            print(f"  topology_error: {exc}")
            return 1
        print(f"  process_model: {topology.process_model}")
        print(f"  process_index: {topology.process_index}")
        print(f"  process_count: {topology.process_count}")
        print(f"  node_rank: {topology.node_rank}")
        print(f"  node_count: {topology.node_count}")
        print(f"  local_rank: {topology.local_rank if topology.local_rank is not None else '<all-local-ranks>'}")
        print(f"  ranks_per_node: {topology.ranks_per_node}")
        print(f"  visible_local_gpus: {topology.visible_local_gpus}")
    if args.build:
        if args.component == "layout":
            print("  error: --build is only supported for transport components")
            return 1
        build_mode = (
            TransportBuildMode.INTERNODE if args.component == "transport-internode" else TransportBuildMode.INTRANODE
        )
        artifact = build_transport_library(build_mode)
        print(f"  built_library: {artifact.library_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
