# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded model-free reproduction of Snowball's native TPU attention shapes."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.cluster.worker.env_probe import _probe_tpu_metadata
from iris.jax.init import initialize_jax
from jax.experimental import multihost_utils
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import kernel as native_kernel
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import _reshard_for_init
from levanter.layers import attention
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration


def cases(family):
    if family == "serial":
        return [
            dict(name=f"serial-{width}-{end}", width=width, lengths=[min(width, 128)], ends=[end])
            for width, end in ((256, 128), (256, 2048), (2, 2048), (1, 2048), (256, 4099))
        ] + [dict(name="serial-2-one-valid", width=2, lengths=[1], ends=[2048])]
    if family == "packed":
        return [
            dict(name="packed-first", width=256, lengths=[128, 128], ends=[128, 128]),
            *[
                dict(name=f"packed-boundary-{end}", width=2, lengths=[1, 1], ends=[end, end])
                for end in range(2047, 2053)
            ],
            dict(name="packed-short-tail", width=256, lengths=[128, 127], ends=[4099, 4098]),
        ]
    if family == "exhausted":
        return [
            dict(name="exhausted-second", width=256, lengths=[128, 0], ends=[4227, 4098]),
            dict(name="exhausted-first", width=256, lengths=[0, 128], ends=[4098, 4227], page_counts=[33, 42]),
            dict(name="one-active-capacity-two", width=256, lengths=[128, 0], ends=[4227, 4098], active=1),
        ]
    raise ValueError(family)


def _invoke(q_arg, kv_arg, lens_arg, indices_arg, cu_arg, num_arg, *, window, conditional):
    def call(q, win):
        return attention._do_tpu_ragged_paged_attention(
            hax.named(q, q_arg.axes),
            kv_arg,
            lens_arg,
            indices_arg,
            cu_arg,
            num_arg,
            sm_scale=1.0,
            sliding_window=win,
        ).array

    if conditional:

        def layer(q, use_window):
            result = jax.lax.cond(use_window, lambda _: call(q, window), lambda _: call(q, None), None)
            return result, result

        _, layers = jax.lax.scan(layer, q_arg.array, jnp.array([True, False]))
        return layers
    return call(q_arg.array, window)[None]


_compiled_invoke = jax.jit(_invoke, static_argnames=("window", "conditional"))


def run(family, output, q_tile, source_commit):
    initialize_jax()
    slice_id, variant, _, worker_id, _ = _probe_tpu_metadata()
    if variant != "v4-32" or jax.process_count() != 4 or jax.device_count() != 16:
        raise ValueError("Expected one four-host v4-32")
    identity = np.frombuffer(hashlib.sha256(slice_id.encode()).digest(), np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(identity))
    if not np.all(gathered == gathered[0]):
        raise ValueError("Probe gang spans multiple slices")
    root = StoragePath(output)
    kernel = attention.tpu_ragged_paged_attention
    if q_tile:

        def tiled(*args, **kwargs):
            kwargs["num_queries_per_block"] = q_tile
            return kernel(*args, **kwargs)

        attention.tpu_ragged_paged_attention = tiled
    trainer = TrainerConfig(
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(
            axes=dict(data=1, context=1, expert=16, model=1),
            dcn_axes=dict(replica_dcn=1),
            batch_axis_name=None,
            param_mapping={},
        ),
    )
    reports = []
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):

        def arr(value):
            return _reshard_for_init(jnp.asarray(value), P())

        for case in cases(family):
            n = len(case["lengths"])
            counts = case.get("page_counts", [42, 33][:n])
            offsets = np.cumsum([0, *counts[:-1]])
            pages = np.full((n, 128), -1, np.int32)
            for i, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
                pages[i, :count] = offset + np.arange(count)
            cu = np.cumsum([0, *case["lengths"]], dtype=np.int32)
            dest = np.full(case["width"], -1, np.int32)
            starts = np.asarray(case["ends"]) - case["lengths"]
            for i in range(n):
                dest[cu[i] : cu[i + 1]] = offsets[i] * 128 + np.arange(starts[i], case["ends"][i])
            for window in (None, 2048):
                for conditional in (False, True):
                    # Fresh buffers and an explicit readiness barrier isolate each launch.
                    value = 0.01 + len(reports) * 0.00001
                    q = hax.named(
                        arr(np.full((case["width"], 5, 4, 128), value, np.float32)).astype(jnp.bfloat16),
                        ("position", "kv_head", "q_heads_per_group", "head_size"),
                    )
                    kv = hax.named(
                        arr(np.full((260, 128, 10, 128), value, np.float32)).astype(jnp.bfloat16),
                        ("page", "page_size", "kv_head", "head_size"),
                    )
                    lens = hax.named(arr(np.asarray(case["ends"], np.int32)), "seq")
                    indices = hax.named(arr(pages), ("seq", "page"))
                    cu_q = hax.named(arr(cu), "seq")
                    num = arr(np.asarray(case.get("active", n), np.int32))
                    jax.block_until_ready((q, kv, lens, indices, cu_q, num))
                    name = f"{case['name']}-window{window}-conditional{conditional}"
                    report = dict(
                        case=case,
                        starts=starts.tolist(),
                        page_indices=pages.tolist(),
                        new_token_dests=dest.tolist(),
                        cu_q_lens=cu.tolist(),
                        num_seqs=int(np.asarray(num)),
                        kv_shape=list(kv.array.shape),
                        query_shape=list(q.array.shape),
                        sliding_window=window,
                        conditional=conditional,
                        conditional_path_expected=conditional and window is not None,
                        global_only_control=window is None,
                        consumes_all_layer_outputs=True,
                        numerical_acceptance="Finiteness only; constant delta is reported, not a numerical gate",
                        num_kv_pages_per_block=4,
                        head_tile=list(native_kernel.get_min_heads_per_blk(32, 16, jnp.bfloat16, jnp.bfloat16)),
                        num_queries_per_block=q_tile or min(32, case["width"]),
                        source_commit=source_commit,
                        jax_version=jax.__version__,
                        slice_id=slice_id,
                        worker_id=str(worker_id),
                        kernel_sha256=hashlib.sha256(Path(native_kernel.__file__).read_bytes()).hexdigest(),
                    )
                    if jax.process_index() == 0:
                        (root / (name + "-start.json")).write_text(json.dumps(report, indent=2))
                        print("KERNEL_CASE_START", name, json.dumps(report), flush=True)

                    start = time.monotonic()
                    lowered = _compiled_invoke.lower(
                        q, kv, lens, indices, cu_q, num, window=window, conditional=conditional
                    )
                    lowered_text = lowered.compiler_ir("hlo").as_hlo_text()
                    report["lowered_has_conditional"] = "conditional(" in lowered_text
                    report["lowered_has_while"] = "while(" in lowered_text
                    if jax.process_index() == 0:
                        (root / (name + "-lowered.hlo.txt")).write_text(lowered_text)
                    compiled = lowered.compile()
                    optimized = compiled.as_text()
                    if jax.process_index() == 0:
                        (root / (name + "-optimized.hlo.txt")).write_text(optimized)
                    report["optimized_has_conditional"] = "conditional(" in optimized
                    report["optimized_has_while"] = "while(" in optimized
                    report["conditional_scan_survives"] = (
                        report["conditional_path_expected"]
                        and report["optimized_has_conditional"]
                        and report["optimized_has_while"]
                    )
                    result = _compiled_invoke(q, kv, lens, indices, cu_q, num, window=window, conditional=conditional)
                    jax.block_until_ready(result)
                    valid = np.asarray(result[:, : int(cu[-1])])
                    if not np.isfinite(valid).all():
                        raise ValueError("Nonfinite valid attention output")
                    report.update(
                        wall_seconds=time.monotonic() - start,
                        max_delta_constant=float(np.max(np.abs(valid - float(jnp.bfloat16(value))))),
                    )
                    reports.append(report)
                    if jax.process_index() == 0:
                        (root / (name + "-pass.json")).write_text(json.dumps(report, indent=2))
                        print("KERNEL_CASE_PASS", name, flush=True)
    if jax.process_index() == 0:
        (root / "complete.json").write_text(json.dumps(dict(passed=len(reports), family=family), indent=2))
        print("KERNEL_PROBE_COMPLETE", family, len(reports), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--family", choices=("serial", "packed", "exhausted"), required=True)
    parser.add_argument("--q-tile", type=int, choices=(0, 1, 2, 16, 32), default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.output.startswith("gs://marin-us-central2/"):
        parser.error("Use central2 artifacts")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.submit and subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode:
        parser.error("Commit source first")
    resources = ResourceConfig.with_tpu(
        "v4-32", cpu=8, ram="32g", disk="40g", regions=["us-central2"], zone="us-central2-b", preemptible=False
    )
    print(
        json.dumps(
            dict(source=source, family=args.family, q_tile=args.q_tile, cases=cases(args.family), runtime_minutes=20)
        )
    )
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(run, args.family, args.output, args.q_tile, source),
                resources=convert_resources(resources),
                constraints=convert_constraints(resources),
                replicas=4,
                coscheduling=resolve_coscheduling(resources, 4),
                ports=["jax"],
                environment=EnvironmentSpec(extras=("tpu",)),
                timeout=Duration.from_seconds(1200),
                scheduling_timeout=Duration.from_seconds(14400),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job, flush=True)


if __name__ == "__main__":
    main()
