# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 10-probe suite for the agent-docs experiment, encoded as data.

Each probe is a concrete script-writing task handed to a weak coder model that
has the relevant doc bundle and NO execution privileges. The judge scores the
emitted script against the probe's source-verified ``anchors`` (the ground
truth) plus a holistic quality rubric.

Anchors are grounded in real source. Paths are repo-relative. Where the plan's
one-line description named a symbol that does not exist in source (notably the
``maybe_proxy`` / ``iris://`` framing of P10), the probe is grounded against the
mechanism that actually exists and the non-existent terms are listed in
``forbidden`` so the judge flags them as hallucinations. P10 was mined from a
rigging-flavored history question, but that client reachability code actually
lives in finelog's client, so it is tagged ``FINELOG``.

finelog (P5, P6) lives at ``lib/finelog/`` on the ``main`` branch; it is not
checked out in every worktree but the paths are stable. A correct answer for P6
must NOT claim a private Artifact Registry / pypi mirror exists — the wheel
(``marin-finelog``) is published to public PyPI.
"""

from dataclasses import dataclass, field
from enum import StrEnum

# Appended to every probe prompt so the coder cannot route around bad docs.
DOC_ONLY_CONSTRAINT = "Use ONLY the provided documentation. Do not invent function names, parameters, or import paths."


class Intent(StrEnum):
    USE = "use"
    UNDERSTAND = "understand"


class Subproject(StrEnum):
    IRIS = "iris"
    MARIN = "marin"
    LEVANTER = "levanter"
    HALIAX = "haliax"
    FRAY = "fray"
    RIGGING = "rigging"
    ZEPHYR = "zephyr"
    FINELOG = "finelog"
    DUPEKIT = "dupekit"


@dataclass(frozen=True)
class Anchor:
    """A concrete, source-verified thing a correct answer must use."""

    symbol: str  # e.g. "dedup_fuzzy_document" or "ExecutorStep"
    path: str  # real repo-relative file path
    note: str  # what it is / why it matters (one line)


@dataclass(frozen=True)
class Probe:
    """One probe: a script-writing task plus its ground truth."""

    id: str  # "P1".."P10"
    subproject: Subproject
    intent: Intent
    prompt: str  # the task given to the coder (a concrete scripting task)
    doc_bundle: tuple[str, ...]  # docs answerable from; MAP.md is always implicit
    anchors: tuple[Anchor, ...]  # source-verified ground truth (3-6 per probe)
    forbidden: tuple[str, ...] = field(default_factory=tuple)  # hallucination markers


def _prompt(body: str) -> str:
    """Attach the doc-only constraint to a probe body."""
    return f"{body.rstrip()}\n\n{DOC_ONLY_CONSTRAINT}"


PROBES: tuple[Probe, ...] = (
    Probe(
        id="P1",
        subproject=Subproject.IRIS,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a Python snippet that demonstrates how an Iris worker discovers "
            "and connects to its controller. Read the controller address from the "
            "environment exactly as the worker runtime does, normalize it to an "
            "http URL, and construct the controller client the worker uses to "
            "register. Add a comment tracing the resolution path (env var -> "
            "worker config -> client)."
        ),
        doc_bundle=("iris/architecture.md",),
        anchors=(
            Anchor(
                "IRIS_CONTROLLER_ADDRESS",
                "lib/iris/src/iris/cluster/runtime/env.py",
                "Env var (set by build_common_iris_env) the worker reads to find the controller.",
            ),
            Anchor(
                "build_common_iris_env",
                "lib/iris/src/iris/cluster/runtime/env.py",
                "Builds the worker's system env, including IRIS_CONTROLLER_ADDRESS.",
            ),
            Anchor(
                "worker_config_from_proto",
                "lib/iris/src/iris/cluster/worker/worker.py",
                "Normalizes controller_address, prefixing http:// when missing.",
            ),
            Anchor(
                "ControllerServiceClientSync",
                "lib/iris/src/iris/cluster/worker/worker.py",
                "gRPC/Connect client the worker builds to talk to the controller.",
            ),
            Anchor(
                "Worker.start",
                "lib/iris/src/iris/cluster/worker/worker.py",
                "Where the worker instantiates the controller client and registers.",
            ),
        ),
    ),
    Probe(
        id="P2",
        subproject=Subproject.IRIS,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a Python snippet that builds an Iris job request for a 64x8 "
            "gang-scheduled run (64 hosts, 8 chips each) and explain, in comments, "
            "how the controller schedules it all-or-nothing. Express the device "
            "shape, set the coscheduling configuration that binds tasks to one "
            "slice, and name the scheduler routine that finds the atomic "
            "assignment."
        ),
        doc_bundle=("iris/architecture.md",),
        anchors=(
            Anchor(
                "CoschedulingConfig",
                "lib/iris/src/iris/cluster/types.py",
                "Config (group_by) that requests gang/atomic scheduling.",
            ),
            Anchor(
                "tpu_device",
                "lib/iris/src/iris/cluster/types.py",
                "Helper that builds a DeviceConfig (e.g. a TPU variant + count) for the slice shape.",
            ),
            Anchor(
                "_find_coscheduled_assignments",
                "lib/iris/src/iris/cluster/controller/scheduling/scheduler.py",
                "Scheduler routine that finds an all-or-nothing assignment for a coscheduled group.",
            ),
            Anchor(
                "JobRequirements.is_coscheduled",
                "lib/iris/src/iris/cluster/controller/scheduling/scheduler.py",
                "Flag the scheduler keys gang admission on.",
            ),
            Anchor(
                "K8sTaskProvider",
                "lib/iris/src/iris/cluster/backends/k8s/tasks.py",
                "The direct/k8s task provider that launches the placed tasks.",
            ),
        ),
    ),
    Probe(
        id="P3",
        subproject=Subproject.IRIS,
        intent=Intent.USE,
        prompt=_prompt(
            "Write a complete, runnable Python script that submits an Iris job which "
            "stands up an HTTP endpoint backed by an actor over a multi-host JAX "
            "mesh (happy path). Use the public client to submit with a multi-host "
            "device spec and coscheduling, allocate a named port, and have the "
            "entrypoint register an actor on the actor server and serve it. Include "
            '`if __name__ == "__main__"`.'
        ),
        doc_bundle=("iris/ops.md",),
        anchors=(
            Anchor(
                "IrisClient.submit",
                "lib/iris/src/iris/client/client.py",
                "Public entry point to submit a job (entrypoint, resources, ports, coscheduling, replicas).",
            ),
            Anchor(
                "ResourceSpec",
                "lib/iris/src/iris/cluster/types.py",
                "Resource spec with the device field for the multi-host mesh.",
            ),
            Anchor(
                "tpu_device",
                "lib/iris/src/iris/cluster/types.py",
                "Builds the multi-host DeviceConfig passed to ResourceSpec.",
            ),
            Anchor(
                "ActorServer",
                "lib/iris/src/iris/actor/server.py",
                "Hosts the actor and serves it (serve_background) over HTTP.",
            ),
            Anchor(
                "iris_ctx",
                "lib/iris/src/iris/client/client.py",
                "In-job context; get_port(name) resolves the allocated port for the endpoint.",
            ),
        ),
    ),
    Probe(
        id="P4",
        subproject=Subproject.IRIS,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a Python snippet that sketches the Iris controller's "
            "reconciliation loop: a periodic poll that syncs every execution unit "
            "by observing provider/worker state (no status RPC pushed from the "
            "worker), then reaps stale workers. Name the real loop and sync "
            "methods and the worker/task state model in comments, and explain why "
            "poll-only reconciliation is the source of truth."
        ),
        doc_bundle=("iris/architecture.md",),
        anchors=(
            Anchor(
                "_run_polling_loop",
                "lib/iris/src/iris/cluster/controller/controller.py",
                "Periodic poll loop; calls _reconcile_tick each tick on the heartbeat interval.",
            ),
            Anchor(
                "_reconcile_tick",
                "lib/iris/src/iris/cluster/controller/controller.py",
                "Reconciliation step: snapshot -> reconcile RPCs -> apply (build_reconcile_plans).",
            ),
            Anchor(
                "_run_ping_loop",
                "lib/iris/src/iris/cluster/controller/controller.py",
                "Reaps stale workers via WorkerHealthTracker.workers_over_threshold -> _terminate_workers.",
            ),
            Anchor(
                "WorkerStatus",
                "lib/iris/src/iris/cluster/types.py",
                "Worker state model derived from polling (worker_id, running_task_ids).",
            ),
        ),
    ),
    Probe(
        id="P5",
        subproject=Subproject.FINELOG,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a commented Python/pseudocode walkthrough that traces a finelog "
            "record from the in-RAM write buffer through the storage tiers: flush "
            "to an L0 segment, tier-based compaction (L0->L1->...), and offload of "
            "higher tiers to remote object storage. Name the real store routines "
            "for flush, the compaction planner/executor, and the eviction/offload "
            "step, and describe what the integer segment `level` means."
        ),
        doc_bundle=("finelog/architecture.md",),
        anchors=(
            Anchor(
                "seg_filename",
                "lib/finelog/rust/src/store/types.rs",
                "Names a segment seg_L{level}_... ; integer level IS the tier (0=L0 flush output).",
            ),
            Anchor(
                "CompactionConfig.level_targets",
                "lib/finelog/rust/src/store/compaction/config.rs",
                "Per-tier byte thresholds that trigger promotion (default 64/256/256 MiB).",
            ),
            Anchor(
                "RamBuffers",
                "lib/finelog/rust/src/store/ram_buffer.rs",
                "In-RAM LSM write buffer; seal() moves chunks into an in-flight flush buffer.",
            ),
            Anchor(
                "Namespace::flush_once",
                "lib/finelog/rust/src/store/namespace.rs",
                "Drains the RAM buffer to a new level-0 segment (the 'promote to L0' / flush path).",
            ),
            Anchor(
                "planner::plan",
                "lib/finelog/rust/src/store/compaction/planner.rs",
                "Pure compaction planner: picks the first promotable contiguous run per tier.",
            ),
            Anchor(
                "Namespace::evict_segment",
                "lib/finelog/rust/src/store/namespace.rs",
                "Offload step: flips a remote-backed segment Both->Remote and unlinks the local copy.",
            ),
        ),
    ),
    Probe(
        id="P6",
        subproject=Subproject.FINELOG,
        intent=Intent.USE,
        prompt=_prompt(
            "Write a script (or annotated shell-in-python) that publishes the "
            "finelog native Rust wheel and shows how a downstream worker pulls it. "
            "Drive the real build entry point that runs maturin, use the correct "
            "distribution name, reference the release workflow, and pin the "
            "dependency the way consumers resolve it. State accurately where the "
            "wheel is published and pulled from."
        ),
        doc_bundle=("finelog/ops.md",),
        anchors=(
            Anchor(
                "build_linux_wheels",
                "lib/finelog/build_package.py",
                "Build driver that invokes maturin to produce the manylinux wheels.",
            ),
            Anchor(
                "_maturin",
                "lib/finelog/build_package.py",
                "Wraps the maturin invocation that builds the native package.",
            ),
            Anchor(
                "marin-finelog",
                "lib/finelog/pyproject.toml",
                "The distribution name (module finelog._native); published to public PyPI.",
            ),
            Anchor(
                "finelog-release-wheels.yaml",
                ".github/workflows/finelog-release-wheels.yaml",
                "Release workflow; publishes to PyPI via trusted publishing on tag/nightly.",
            ),
        ),
        # Claiming finelog ships to a private Artifact Registry is wrong — it is
        # published to public PyPI via trusted publishing. (The pull side may
        # legitimately mention a pip mirror, so those terms are not forbidden.)
        forbidden=(
            "artifact registry",
            "private index",
        ),
    ),
    Probe(
        id="P7",
        subproject=Subproject.MARIN,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a commented Python snippet that constructs two ExecutorSteps "
            "(one depending on the other) and explains how the Marin executor makes "
            "a step's output path a deterministic function of its inputs. Name the "
            "version-hashing routine and the output-path construction, show how an "
            "InputName wires one step's output into the next, and explain in "
            "comments which config fields affect the version hash (and which do "
            "not)."
        ),
        doc_bundle=("marin/architecture.md",),
        anchors=(
            Anchor(
                "ExecutorStep",
                "lib/marin/src/marin/execution/types.py",
                "Frozen step (name, fn, config); name + versioned config fields determine identity.",
            ),
            Anchor(
                "Executor.compute_version",
                "lib/marin/src/marin/execution/executor.py",
                "Builds the canonical version dict and md5-hashes it (deterministic step id).",
            ),
            Anchor(
                "collect_dependencies_and_version",
                "lib/marin/src/marin/execution/executor.py",
                "Collects only VersionedValue config fields + dependency versions into the hash.",
            ),
            Anchor(
                "InputName",
                "lib/marin/src/marin/execution/types.py",
                "Wires an upstream step's output path into a downstream step's config.",
            ),
        ),
    ),
    Probe(
        id="P8",
        subproject=Subproject.MARIN,
        intent=Intent.USE,
        prompt=_prompt(
            "Write a complete, runnable Python script that runs Marin's two-stage "
            "fuzzy document deduplication pipeline. Stage 1: call the MinHash "
            "attribute step with the correct default MinHash parameters. Stage 2: "
            "feed its output into the fuzzy-duplicates step, providing the required "
            "max_parallelism. Pass all arguments as keyword arguments, wire stage 1's "
            'output into stage 2, and include `if __name__ == "__main__"` with argparse. '
            "Output the script contents only."
        ),
        doc_bundle=("marin/ops.md",),
        anchors=(
            Anchor(
                "compute_minhash_attrs",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py",
                "Stage 1: read NormalizedData parquet, run dupekit MinHash+LSH, write MinHashAttrData; keyword-only.",
            ),
            Anchor(
                "num_perms",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py",
                "MinHash permutations on compute_minhash_attrs; default 286.",
            ),
            Anchor(
                "num_bands",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py",
                "LSH bands on compute_minhash_attrs; default 26.",
            ),
            Anchor(
                "ngram_size",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py",
                "Shingle n-gram size on compute_minhash_attrs; default 5.",
            ),
            Anchor(
                "seed",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py",
                "MinHash seed on compute_minhash_attrs; default 42.",
            ),
            Anchor(
                "compute_fuzzy_dups_attrs",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py",
                "Stage 2: consume MinHashAttrData inputs, run connected components, emit dup markers.",
            ),
            Anchor(
                "max_parallelism",
                "lib/marin/src/marin/processing/classification/deduplication/fuzzy_dups.py",
                "REQUIRED keyword arg (no default) on compute_fuzzy_dups_attrs.",
            ),
        ),
        forbidden=(
            "datasketch",
            "dedup_fuzzy_document",
        ),
    ),
    Probe(
        id="P9",
        subproject=Subproject.LEVANTER,
        intent=Intent.USE,
        prompt=_prompt(
            "Write a Python snippet that configures jax.distributed for a "
            "multi-host Levanter training run. Build the Levanter distributed "
            "config with the coordinator address, process count, this process's "
            "id, and local device ids, attach it to the trainer config, and call "
            "the public Levanter initialize entry point that ultimately invokes "
            "jax.distributed.initialize. Name the field that toggles the init."
        ),
        doc_bundle=("levanter/ops.md",),
        anchors=(
            Anchor(
                "DistributedConfig",
                "lib/levanter/src/levanter/distributed.py",
                "Config with coordinator_address, num_processes, process_id, local_device_ids.",
            ),
            Anchor(
                "DistributedConfig.initialize",
                "lib/levanter/src/levanter/distributed.py",
                "Method that actually calls jax.distributed.initialize.",
            ),
            Anchor(
                "initialize",
                "lib/levanter/src/levanter/trainer.py",
                "Public levanter.initialize(config) entry point users call to bring up distributed.",
            ),
            Anchor(
                "initialize_jax_distributed",
                "lib/levanter/src/levanter/distributed.py",
                "DistributedConfig bool field (default True); when False, initialize() skips jax.distributed init.",
            ),
        ),
    ),
    Probe(
        id="P10",
        subproject=Subproject.FINELOG,
        intent=Intent.UNDERSTAND,
        prompt=_prompt(
            "Write a Python snippet that connects a log client to a remote log "
            "server and explains, in comments, how the client reaches the server "
            "and recovers when the address goes stale. Use the real client connect "
            "API, supply the pluggable resolver callable that maps a logical "
            "server url to a concrete address, and describe how re-resolution on "
            "failure (invalidation) provides reachability. Name the proxy adapter "
            "that forwards the RPCs."
        ),
        doc_bundle=("finelog/architecture.md",),
        anchors=(
            Anchor(
                "LogClient.connect",
                "lib/finelog/src/finelog/client/log_client.py",
                "Client connect API; accepts an endpoint and a pluggable resolver callable.",
            ),
            Anchor(
                "resolver",
                "lib/finelog/src/finelog/client/log_client.py",
                "Callable[[str], str] mapping a logical server url to a concrete address (default identity).",
            ),
            Anchor(
                "LogClient._invalidate",
                "lib/finelog/src/finelog/client/log_client.py",
                "Forces re-resolution on transient failure — the actual reachability mechanism.",
            ),
            Anchor(
                "LogServiceProxy",
                "lib/finelog/src/finelog/client/proxy.py",
                "Async adapter that forwards log RPCs to the resolved remote address.",
            ),
        ),
        # The plan's framing ("maybe_proxy", "iris://") does not exist in source.
        # Reachability is a pluggable resolver callable, not a maybe_proxy function,
        # and endpoints are http URLs / (host, port) tuples, not an iris:// scheme.
        forbidden=(
            "maybe_proxy",
            "iris://",
        ),
    ),
)
