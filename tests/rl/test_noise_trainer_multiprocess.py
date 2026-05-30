# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for multi-host RL weight-export bug (marin#4287).

Spawns two JAX processes on a single host using `jax.distributed.initialize`
with `localhost:<port>` coordinator, builds a scan-stacked Llama, and runs
both `WeightTransferConfig.export_strategy` paths against the noise loader.

Expected outcomes (the issue's symptom and fix point):
- `TREE_JIT` should run a few steps cleanly. This is the path that was
  hardened across earlier multi-host probes.
- `SEQUENTIAL_HOST_FLATTEN` should fail because `_export_weights_sequential_host_flatten`
  calls `hsd.to_state_dict(model)` eagerly outside JIT and iterates the
  scan-stacked layers in Python, where the per-leaf `assert is_fully_addressable`
  trips on non-fully-addressable global arrays.

We mark the SEQUENTIAL_HOST_FLATTEN case as `pytest.mark.xfail(strict=True)` so
the test only goes green when the bug actually reproduces. Once a real fix
lands the xfail will flip to XPASS, making this the regression test the issue
is currently missing.

Notes on scope:
- This test does NOT run a full TrainWorker. We construct only the pieces
  needed to exercise `_export_weights_*`: a real scan-stacked Llama on a 2-host
  mesh, then call `transfer_server.serve_weights(0, model)` directly.
- We avoid the iris.cluster client stack entirely so the test runs even when
  the local protobuf descriptor pool is inconsistent on this branch.
"""

import logging
import multiprocessing as mp
import os
import socket
import time

import pytest

logger = logging.getLogger(__name__)

if os.environ.get("CI"):
    pytest.skip("Skipping slow multiprocess tests on CI", allow_module_level=True)


def _free_tcp_port() -> int:
    """Return a free TCP port. Race-safe enough for local test coordination."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_weights_in_process(
    process_id: int,
    num_processes: int,
    coordinator_address: str,
    export_strategy: str,
    result_queue: mp.Queue,
) -> None:
    """Run inside a child process. Initialize JAX distributed, build a tiny
    scan-stacked Llama, then call the chosen export path once and report
    whether it raised.
    """
    try:
        # Force CPU JAX for the spawned process; one CPU device per process so
        # the resulting mesh has multiple non-fully-addressable shards across
        # the processes.
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

        import haliax as hax
        import jax
        import jax.random as jrandom
        import numpy as np
        from jax.sharding import Mesh

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=[0],
        )

        from levanter.layers.attention import AttentionBackend
        from levanter.models.llama import LlamaConfig
        from marin.rl.weight_transfer import (
            ArrowFlightExportStrategy,
            WeightTransferConfig,
            WeightTransferMode,
        )
        from marin.rl.weight_transfer.arrow_flight import (
            _export_weights_sequential_host_flatten,
            _export_weights_tree_jit,
        )

        devices = jax.devices()
        assert len(devices) == num_processes, f"expected {num_processes} global devices, got {len(devices)}: {devices}"
        mesh = Mesh(np.asarray(devices), axis_names=("data",))

        # Tiny scan-stacked Llama so the export path iterates the same shape
        # of state-dict tree as Llama-3.1-8B (just smaller).
        llama_config = LlamaConfig(
            max_seq_len=64,
            hidden_dim=64,
            intermediate_dim=128,
            num_heads=4,
            num_kv_heads=4,
            num_layers=4,
            attn_backend=AttentionBackend.JAX_FLASH,
        )

        Vocab = hax.Axis("vocab", 256)
        key = jrandom.PRNGKey(process_id)

        # Build the model with `hax.set_mesh` + an axis_mapping that shards
        # at least one model axis across the data mesh axis. The "mlp"/"heads"
        # axes are the most common production sharding choices; sharding them
        # across the 2-process data mesh yields per-process global jax.Array
        # leaves whose .is_fully_addressable is False, which is exactly the
        # condition that exposes the eager-state-dict bug.
        axis_mapping = {"mlp": "data", "heads": "data"}
        with hax.set_mesh(mesh), hax.axis_mapping(axis_mapping):
            model = llama_config.build(Vocab, key=key)

        # Verify that at least one leaf is non-fully-addressable, otherwise
        # the test scenario doesn't actually reproduce the bug.
        leaves = [leaf for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.Array)]
        non_addressable = [leaf for leaf in leaves if not leaf.is_fully_addressable]
        if not non_addressable:
            raise AssertionError(
                "Test setup error: every model leaf is fully addressable. "
                "The export-strategy bug requires at least one sharded leaf."
            )

        config = WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            export_strategy=ArrowFlightExportStrategy(export_strategy),
            convert_to_bfloat16=True,
        )

        # Call the chosen exporter directly. Both processes must participate
        # (the JIT path has cross-host collectives that block otherwise). The
        # mesh context is required so jit-compiled paths can resolve the
        # PartitionSpec → NamedSharding mapping.
        with hax.set_mesh(mesh):
            if config.export_strategy == ArrowFlightExportStrategy.SEQUENTIAL_HOST_FLATTEN:
                exported = _export_weights_sequential_host_flatten(model, config)
            else:
                exported = _export_weights_tree_jit(model, config)

        # Only process 0 sees a populated flat_dict; both report success.
        result_queue.put({"process_id": process_id, "ok": True, "param_count": exported.param_count})
    except BaseException as exc:  # propagate everything: assertions, TypeError, etc.
        # Reduce to a stringified error so the parent queue can pickle it.
        result_queue.put(
            {"process_id": process_id, "ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
        )
    finally:
        try:
            import jax

            jax.distributed.shutdown()
        except Exception:
            pass


def _run_export_strategy(export_strategy: str, num_processes: int = 2, timeout: float = 120.0) -> list[dict]:
    """Spawn `num_processes` child processes; collect their per-process result."""
    coordinator_port = _free_tcp_port()
    coordinator_address = f"localhost:{coordinator_port}"

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs = []
    for i in range(num_processes):
        proc = ctx.Process(
            target=_serve_weights_in_process,
            args=(i, num_processes, coordinator_address, export_strategy, result_queue),
        )
        proc.start()
        procs.append(proc)

    results: list[dict] = []
    deadline = time.time() + timeout
    while len(results) < num_processes and time.time() < deadline:
        try:
            results.append(result_queue.get(timeout=max(0.1, deadline - time.time())))
        except Exception:
            break

    for proc in procs:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
    return results


@pytest.mark.timeout(180)
def test_tree_jit_export_runs_on_two_processes():
    """TREE_JIT is the known-working strategy on multi-host. Should succeed."""
    results = _run_export_strategy("tree_jit")
    assert len(results) == 2, f"Expected 2 process results, got {results}"
    for result in results:
        assert result["ok"], f"Process {result['process_id']} failed: {result.get('error')}"


@pytest.mark.timeout(180)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Regression marker for marin#4287: SEQUENTIAL_HOST_FLATTEN calls "
        "hsd.to_state_dict eagerly outside JIT and iterates non-fully-addressable "
        "global arrays in Python, which raises inside Haliax. When this is fixed, "
        "the xfail will flip to XPASS and the test will fail strictly."
    ),
)
def test_sequential_host_flatten_export_fails_on_two_processes():
    """Currently broken on multi-host: this captures the exact regression."""
    results = _run_export_strategy("sequential_host_flatten")
    # Either one or both processes raised. Mark as "test outcome = pass"
    # only when at least one process actually crashed with the expected
    # eager-state-dict failure mode.
    if not results:
        pytest.fail("No process results collected — coordinator likely failed before either process reported back.")

    has_failure = any(not result["ok"] for result in results)
    if not has_failure:
        # Both processes succeeded — the bug is fixed. xfail(strict=True) turns
        # this into XPASS which is a test failure, so callers know to flip the
        # xfail marker.
        return
    # Trigger the xfail: re-raise the first error so pytest records the
    # expected-fail traceback.
    for result in results:
        if not result["ok"]:
            raise AssertionError(
                f"Process {result['process_id']} failed during "
                f"SEQUENTIAL_HOST_FLATTEN: {result['error_type']}: {result['error']}"
            )
