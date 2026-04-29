# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
The `Executor` framework provides a way to specify a DAG of `ExecutorStep`s that
are executed in a topological order using Fray.  Beyond that:

1. The key distinguishing feature of the framework is allowing the user to
   flexibly control what steps are "new".

2. A secondary feature of the framework is that it creates sensible output paths
   for each step to free the user from having to come up with interpretable
   names that don't clash.

As an example, suppose you have a two-step pipeline:

    transform(method) -> tokenize(method)

which can be instantiated as:

    [A] transform(trafilatura) -> tokenize(llama2)
    [B] transform(resiliparse) -> tokenize(llama2)
    [C] transform(trafilatura) -> tokenize(llama3)
    [D] transform(resiliparse) -> tokenize(llama3)

If you have already run a particular instantiation, running it again
should be a no-op (assume idempotence).  If you run [A], then running [C] should
reuse `transform(trafilatura)`.

## Versioning

But the big question is: when is a step `transform(trafilatura)` "new"?
In the extreme, you have to hash the code of `transform` and the precise
configuration passed into it, but this is too strict: Semantics-preserving
changes to the code or config (e.g., adding logging) should not trigger a rerun.

We want to compute a *version* for each step.  Here's what the user supplies:
1. a `name` (that characterizes the code and also is useful for interpretability).
2. which fields of a `config` should be included in the version (things like the
   "method", not default thresholds that don't change).

The version of a step is identified by the name, versioned fields, and the
versions of all the dependencies. This version is represented as a hash (e.g.,
8ce902).

## Output paths

Having established the version, the question is what the output path should be.
One extreme is to let the framework automatically specify all the paths, but
then the paths are opaque and you can't easily find where things are stored.

Solution: based on the name and version, the output path of a step is computed.
For example, if name is "documents/fineweb-resiliparse", then the full path
might be:

    gs://marin-us-central2/documents/fineweb-resiliparse-8c2f3a

## Final remarks

- If you prefer to manage the output paths yourself, you can not use `versioned`
  fields and specify everything you want in the name.  Note the version will
  still depend on upstream dependencies and "pseudo-dependencies."

- The pipeline might get too big and unwieldy, in which case we can cut it up by
  specifying a hard-coded path as the input to a step.  Or perhaps we can have
  our cake and eat it to by putting in an "assert" statement to ensure the input
  path that's computed from upstream dependencies is what we expect.

- If we decide to rename fields, we can extend `versioned` to take a string of
  the old field name to preserve backward compatibility.

- "Pseudo-dependencies" are dependencies that do not block the execution of
  the step, but are still included in the version.  This is useful for depending
   on checkpoints of in-progress training runs, for example. When you run a step
  that has a pseudo-dependency, it will not wait for the pseudo-dependency to
  finish executing (or even check if it is executing or failed) before running.
"""

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import draccus
import levanter.utils.fsspec_utils as fsspec_utils
from fray.types import TpuConfig
from iris.cluster.constraints import WellKnownAttribute
from rigging.filesystem import (
    collect_gcs_paths,
    get_bucket_location,
    marin_prefix,
    open_url,
    region_from_prefix,
    split_gcs_path,
)
from rigging.log_setup import configure_logging

from marin.execution.dag import (
    ExecutorStep,
    InputName,
    InputNameEvent,
    MirroredValue,
    OutputName,
    THIS_OUTPUT_PATH,
    VersionedEvent,
    VersionedValue,
    _make_prefix_absolute_path,
    ensure_versioned,
    get_executor_step,
    instantiate_config,
    mirrored,
    output_path_of,
    resolve_local_placeholders,
    this_output_path,
    unwrap_versioned_value,
    upstream_steps,
    versioned,
    walk_config,
)
from marin.execution.step_spec import StepSpec
from marin.execution.step_runner import StepRunner, worker_id
from marin.execution.remote import RemoteCallable
from marin.execution.executor_step_status import (
    STATUS_SUCCESS,
    StatusFile,
)
from marin.execution.remote import RemoteCallable
from marin.execution.step_runner import StepRunner, worker_id
from marin.execution.step_spec import StepSpec
from marin.utilities.json_encoder import CustomJsonEncoder

# Re-exports kept so the ~100 importers of `from marin.execution.executor
# import …` continue to work after the placeholder dataclasses moved to
# dag.py. The cycle is gone (dag.py does not import from this module), so
# these re-exports add no risk.
__all__ = [
    "THIS_OUTPUT_PATH",
    "Executor",
    "ExecutorInfo",
    "ExecutorMainConfig",
    "ExecutorStep",
    "ExecutorStepInfo",
    "InputName",
    "InputNameEvent",
    "MirroredValue",
    "OutputName",
    "VersionedEvent",
    "VersionedValue",
    "ensure_versioned",
    "executor_main",
    "get_executor_step",
    "instantiate_config",
    "materialize",
    "mirrored",
    "output_path_of",
    "resolve_local_placeholders",
    "this_output_path",
    "unwrap_versioned_value",
    "upstream_steps",
    "versioned",
    "walk_config",
]

logger = logging.getLogger(__name__)

_LOCAL_DATA_BROWSER_PORT_RE = re.compile(r"^\s*port\s*:\s*(\d+)\s*(?:#.*)?$")
_LOCAL_DATA_BROWSER_CONFIG_REL = Path("data_browser") / "conf" / "local.conf"


def _find_data_browser_local_conf(max_parents: int = 6) -> Path | None:
    here = Path.cwd().resolve()
    for _ in range(max_parents + 1):
        candidate = here / _LOCAL_DATA_BROWSER_CONFIG_REL
        if candidate.exists():
            return candidate
        parent = here.parent
        if parent == here:
            break
        here = parent
    return None


def _get_local_data_browser_port(default: int = 5000) -> int:
    # looks for the port in the local data browser config file
    config_path = _find_data_browser_local_conf()
    if config_path is None:
        return default

    try:
        with config_path.open() as fp:
            for line in fp:
                match = _LOCAL_DATA_BROWSER_PORT_RE.match(line)
                if match:
                    return int(match.group(1))
    except OSError:
        return default

    return default


ConfigT = TypeVar("ConfigT", covariant=True, bound=dataclass)

ExecutorFunction = Callable | None


_NON_REGIONAL_BUCKET_LOCATIONS = {"us", "eu", "asia", "nam4", "eur4", "asia1"}
_GCP_REGION_PATTERN = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")


def _normalize_region(region: str, *, step_name: str, path: str) -> str:
    normalized = region.lower()
    if normalized in _NON_REGIONAL_BUCKET_LOCATIONS or "+" in normalized or not _GCP_REGION_PATTERN.match(normalized):
        raise ValueError(
            f"Executor step {step_name!r} references {path!r} in a non-regional bucket location "
            f"({normalized!r}); cannot infer a single region pin."
        )
    return normalized


def _is_bucket_location_permission_error(exc: Exception) -> bool:
    return isinstance(exc, PermissionError) or exc.__class__.__name__ in {"Forbidden", "PermissionDenied"}


def _region_for_gcs_path(path: str, *, step_name: str, bucket_region_cache: dict[str, str]) -> str | None:
    region = region_from_prefix(path)
    if region is not None:
        return _normalize_region(region, step_name=step_name, path=path)

    bucket, _ = split_gcs_path(path)
    if bucket not in bucket_region_cache:
        try:
            bucket_region_cache[bucket] = get_bucket_location(path)
        except Exception as e:
            if _is_bucket_location_permission_error(e):
                logger.warning(
                    "Could not infer bucket location for %s due to permission error; "
                    "skipping this path for region inference.",
                    path,
                    exc_info=True,
                )
                return None
            raise
    return _normalize_region(bucket_region_cache[bucket], step_name=step_name, path=path)


def _infer_gcs_regions(
    *,
    step_name: str,
    config: Any,
    output_path: str,
    deps: list[StepSpec] | None,
    dag_tpu_regions: list[str] | None = None,
) -> list[str] | None:
    """Return inferred GCS regions referenced by config/deps/output, or None if no GCS paths."""
    # label -> path evidence for useful error messages
    path_to_labels: dict[str, list[str]] = {}

    def add_path(label: str, path: str):
        path_to_labels.setdefault(path, []).append(label)

    for label, path in collect_gcs_paths(config, path_prefix="config"):
        add_path(label, path)

    for i, dep in enumerate(deps or []):
        dep_path = dep.output_path
        if dep_path.startswith("gs://"):
            add_path(f"dependency[{i}]", dep_path)

    if output_path.startswith("gs://"):
        add_path("output_path", output_path)

    gcs_regions: set[str] | None = None
    region_to_evidence: dict[str, list[str]] = {}
    if path_to_labels:
        bucket_region_cache: dict[str, str] = {}
        for path, labels in path_to_labels.items():
            region = _region_for_gcs_path(path, step_name=step_name, bucket_region_cache=bucket_region_cache)
            if region is None:
                continue
            region_to_evidence.setdefault(region, []).extend(f"{label}={path}" for label in labels)
        if region_to_evidence:
            gcs_regions = set(region_to_evidence)

        if gcs_regions is not None and len(gcs_regions) > 1:
            detail = "; ".join(
                f"{region}: {', '.join(sorted(evidence)[:3])}" for region, evidence in sorted(region_to_evidence.items())
            )
            raise ValueError(
                f"Executor step {step_name!r} has cross-region GCS dependencies. "
                f"Found regions {{{', '.join(sorted(region_to_evidence))}}}. {detail}"
            )

    if dag_tpu_regions:
        tpu_region_set = {r.lower() for r in dag_tpu_regions}
        if gcs_regions is None:
            gcs_regions = tpu_region_set
        else:
            intersection = gcs_regions & tpu_region_set
            if intersection:
                gcs_regions = intersection
            else:
                raise ValueError(
                    f"Executor step {step_name!r} has no overlap between GCS regions {sorted(gcs_regions)} "
                    f"and TPU-capable DAG regions {sorted(tpu_region_set)}."
                )

    if gcs_regions is None:
        return None
    return sorted(gcs_regions)


def _allowed_regions_for_step(
    *,
    step_name: str,
    remote_fn: RemoteCallable | None,
    config: Any,
    output_path: str,
    deps: list[StepSpec] | None,
    dag_tpu_regions: list[str] | None = None,
) -> set[str] | None:
    """Return the allowed regional placements for a step after combining all constraints."""
    allowed_regions = _infer_gcs_regions(
        step_name=step_name,
        config=config,
        output_path=output_path,
        deps=deps,
        dag_tpu_regions=dag_tpu_regions,
    )
    allowed = set(allowed_regions) if allowed_regions is not None else None

    if remote_fn is None or remote_fn.resources.regions is None:
        return allowed

    explicit_regions = {region.lower() for region in remote_fn.resources.regions}
    if not explicit_regions:
        return allowed

    if allowed is None:
        return explicit_regions

    intersection = allowed & explicit_regions
    if intersection:
        return intersection

    raise ValueError(
        f"Executor step {step_name!r} has no overlap between explicit regions {sorted(explicit_regions)} "
        f"and inferred regions {sorted(allowed)}."
    )


def _regions_for_tpu_variant_from_iris(variant: str) -> set[str] | None:
    from fray.client import current_client
    from fray.iris_backend import FrayIrisClient
    from iris.rpc import config_pb2

    try:
        client = current_client()
    except Exception:
        return None
    if not isinstance(client, FrayIrisClient):
        return None

    variant = variant.lower()
    try:
        # TODO: expose autoscaler status through a public Fray API.
        autoscaler_status = client._iris._cluster_client.get_autoscaler_status()
    except Exception:
        logger.warning("Could not query Iris autoscaler status for TPU region inference", exc_info=True)
        return None

    regions: set[str] = set()
    for group in autoscaler_status.status.groups:
        resources = group.config.resources
        if resources.device_type != config_pb2.ACCELERATOR_TYPE_TPU:
            continue
        group_variant = resources.device_variant.lower().strip()
        if group_variant and group_variant != variant:
            continue

        attrs = group.config.worker.attributes
        region = attrs.get(WellKnownAttribute.REGION, "").strip().lower()
        if region:
            regions.add(region)
            continue

        zone = attrs.get(WellKnownAttribute.ZONE, "").strip().lower()
        if zone and "-" in zone:
            regions.add(zone.rsplit("-", 1)[0])

    return regions or None


def _regions_for_tpu_variants_from_iris(
    variants: list[str],
    *,
    variant_region_cache: dict[str, set[str] | None],
) -> set[str] | None:
    inferred_regions: set[str] = set()
    for variant in variants:
        normalized_variant = variant.lower()
        if normalized_variant not in variant_region_cache:
            variant_region_cache[normalized_variant] = _regions_for_tpu_variant_from_iris(normalized_variant)
        cached = variant_region_cache[normalized_variant]
        if cached is None:
            return None
        inferred_regions |= cached
    return inferred_regions


def infer_tpu_variant_regions_from_iris(variants: Sequence[str]) -> list[str] | None:
    """Return sorted TPU-capable regions for the requested variants, if known."""
    inferred_regions = _regions_for_tpu_variants_from_iris(
        list(variants),
        variant_region_cache={},
    )
    if not inferred_regions:
        return None
    return sorted(inferred_regions)


def _tpu_regions_for_remote_callable(
    remote_fn: RemoteCallable,
    *,
    variant_region_cache: dict[str, set[str] | None],
) -> set[str] | None:
    if not isinstance(remote_fn.resources.device, TpuConfig):
        return None
    if remote_fn.resources.regions:
        return {r.lower() for r in remote_fn.resources.regions}

    variants = [remote_fn.resources.device.variant]
    if remote_fn.resources.device_alternatives:
        variants.extend(remote_fn.resources.device_alternatives)
    return _regions_for_tpu_variants_from_iris(variants, variant_region_cache=variant_region_cache)


def _dag_tpu_regions(steps: list["ExecutorStep"]) -> list[str] | None:
    """Infer allowed regions for TPU steps in this DAG, if any."""
    tpu_region_intersection: set[str] | None = None
    tpu_variant_region_cache: dict[str, set[str] | None] = {}

    for step in steps:
        step_fn = step.fn
        if not isinstance(step_fn, RemoteCallable):
            continue
        step_regions = _tpu_regions_for_remote_callable(step_fn, variant_region_cache=tpu_variant_region_cache)
        if not step_regions:
            continue

        if tpu_region_intersection is None:
            tpu_region_intersection = set(step_regions)
        else:
            tpu_region_intersection &= step_regions

        if not tpu_region_intersection:
            raise ValueError("No common region satisfies all TPU steps in this DAG.")

    return sorted(tpu_region_intersection) if tpu_region_intersection else None


def _step_dag_tpu_regions(
    steps: list["ExecutorStep"],
    dependencies: dict["ExecutorStep", list["ExecutorStep"]],
) -> dict["ExecutorStep", list[str] | None]:
    """Infer TPU-capable regions per step from downstream TPU consumers in the same component."""
    dependents: dict[ExecutorStep, list[ExecutorStep]] = {step: [] for step in steps}
    for step in steps:
        for dep in dependencies.get(step, []):
            if dep in dependents:
                dependents[dep].append(step)

    reachable_tpu_regions: dict[ExecutorStep, set[str] | None] = {}
    variant_region_cache: dict[str, set[str] | None] = {}

    def regions_for_step(step: ExecutorStep) -> set[str] | None:
        if step in reachable_tpu_regions:
            return reachable_tpu_regions[step]

        step_regions: set[str] | None = None
        step_fn = step.fn
        if isinstance(step_fn, RemoteCallable):
            step_regions = _tpu_regions_for_remote_callable(step_fn, variant_region_cache=variant_region_cache)

        downstream_tpu_regions: set[str] | None = step_regions
        for dependent in dependents[step]:
            dependent_regions = regions_for_step(dependent)
            if dependent_regions is None:
                continue
            if downstream_tpu_regions is None:
                downstream_tpu_regions = set(dependent_regions)
            else:
                downstream_tpu_regions &= dependent_regions
            if not downstream_tpu_regions:
                raise ValueError(f"No common region satisfies TPU consumers downstream of executor step {step.name!r}.")

        reachable_tpu_regions[step] = downstream_tpu_regions
        return downstream_tpu_regions

    return {
        step: sorted(regions) if regions else None
        for step, regions in ((step, regions_for_step(step)) for step in steps)
    }


def _component_tpu_pins(
    steps: list["ExecutorStep"],
    dependencies: dict["ExecutorStep", list["ExecutorStep"]],
    *,
    configs: dict["ExecutorStep", Any],
    output_paths: dict["ExecutorStep", str],
    dep_stubs_by_step: dict["ExecutorStep", list[StepSpec]],
    dag_tpu_regions_by_step: dict["ExecutorStep", list[str] | None],
) -> dict["ExecutorStep", str | None]:
    relevant_steps = {step for step in steps if dag_tpu_regions_by_step[step] is not None}
    if not relevant_steps:
        return {step: None for step in steps}

    adjacency: dict[ExecutorStep, set[ExecutorStep]] = {step: set() for step in relevant_steps}
    for step in relevant_steps:
        for dep in dependencies.get(step, []):
            if dep in adjacency:
                adjacency[step].add(dep)
                adjacency[dep].add(step)

    chosen_region_by_step: dict[ExecutorStep, str | None] = {step: None for step in steps}
    visited: set[ExecutorStep] = set()

    for step in relevant_steps:
        if step in visited:
            continue

        stack = [step]
        component: list[ExecutorStep] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)

        component_regions: set[str] | None = None
        for component_step in component:
            remote_fn = component_step.fn if isinstance(component_step.fn, RemoteCallable) else None
            step_regions = _allowed_regions_for_step(
                step_name=component_step.name,
                remote_fn=remote_fn,
                config=configs[component_step],
                output_path=output_paths[component_step],
                deps=dep_stubs_by_step[component_step],
                dag_tpu_regions=dag_tpu_regions_by_step[component_step],
            )
            if step_regions is None:
                continue
            if component_regions is None:
                component_regions = set(step_regions)
            else:
                component_regions &= step_regions
            if not component_regions:
                component_step_names = ", ".join(sorted(s.name for s in component))
                raise ValueError(
                    f"No common concrete region satisfies TPU-connected executor steps: {component_step_names}."
                )

        if not component_regions:
            continue

        chosen_region = sorted(component_regions)[0]

        for component_step in component:
            chosen_region_by_step[component_step] = chosen_region

    return chosen_region_by_step


def _iris_backend_is_active() -> bool:
    from fray.client import current_client
    from fray.iris_backend import FrayIrisClient

    try:
        client = current_client()
    except Exception:
        return False
    return isinstance(client, FrayIrisClient)


def _maybe_attach_inferred_region_constraint(
    *,
    step_name: str,
    remote_fn: RemoteCallable,
    config: Any,
    output_path: str,
    deps: list[StepSpec] | None,
    dag_tpu_regions: list[str] | None = None,
    forced_region: str | None = None,
) -> RemoteCallable:
    if not _iris_backend_is_active():
        return remote_fn

    allowed_regions = _allowed_regions_for_step(
        step_name=step_name,
        remote_fn=remote_fn,
        config=config,
        output_path=output_path,
        deps=deps,
        dag_tpu_regions=dag_tpu_regions,
    )
    if forced_region is not None:
        pinned_region = forced_region.lower()
        if allowed_regions is not None and pinned_region not in allowed_regions:
            raise ValueError(
                f"Executor step {step_name!r} cannot be pinned to {pinned_region!r}; "
                f"allowed regions are {sorted(allowed_regions)}."
            )
        return dataclasses.replace(
            remote_fn,
            resources=dataclasses.replace(remote_fn.resources, regions=[pinned_region]),
        )

    if remote_fn.resources.regions is not None:
        return remote_fn

    if allowed_regions is None:
        return remote_fn

    logger.info(
        "Inferred Iris region constraints %s for executor step %s from GCS path dependencies",
        allowed_regions,
        step_name,
    )
    return dataclasses.replace(
        remote_fn,
        resources=dataclasses.replace(remote_fn.resources, regions=sorted(allowed_regions)),
    )


def asdict_without_description(obj: dataclass) -> dict[str, Any]:
    """Return the dict form of a dataclass, but remove the `description` field."""

    def recurse(value: Any):
        if is_dataclass(value):
            return {f.name: recurse(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(recurse(v) for v in value))
        if isinstance(value, (list, tuple)):
            return type(value)(recurse(v) for v in value)
        if isinstance(value, dict):
            # RuntimeEnv (and other dict subclasses) require keyword-only init,
            # so we normalize to a plain dict to avoid construction errors.
            return {recurse(k): recurse(v) for k, v in value.items()}
        return copy.deepcopy(value)

    d = recurse(obj)
    assert isinstance(d, dict)
    d.pop("description", None)
    assert isinstance(d, dict)
    return d


def resolve_executor_step(
    step: "ExecutorStep",
    config: Any,
    output_path: str,
    deps: list[StepSpec] | None = None,
    dag_tpu_regions: list[str] | None = None,
    forced_region: str | None = None,
    mirror_budget_gb: float | None = None,
) -> StepSpec:
    """Convert an ExecutorStep into a StepSpec.

    ``config`` should already be instantiated (no InputName / OutputName /
    VersionedValue markers). The ExecutorStep convention is ``fn(config)``;
    we wrap that into a ``fn(output_path)`` closure expected by ``StepRunner``.

    If *step* was created by :meth:`StepSpec.as_executor_step`, the original
    ``StepSpec`` is returned directly (with deps replaced by the resolved
    versions), preserving round-trip identity.
    """
    original: StepSpec | None = getattr(step, "_original_step_spec", None)
    if original is not None:
        # Pin override_output_path so replacing deps with executor-built stubs
        # (which lack the originals' hash_attrs) doesn't shift name_with_hash
        # and silently change output_path.
        return dataclasses.replace(
            original,
            deps=deps or list(original.deps),
            override_output_path=original.output_path,
        )

    remote_callable = step.fn if isinstance(step.fn, RemoteCallable) else None
    if remote_callable is not None:
        remote_callable = _maybe_attach_inferred_region_constraint(
            step_name=step.name,
            remote_fn=remote_callable,
            config=config,
            output_path=output_path,
            deps=deps,
            dag_tpu_regions=dag_tpu_regions,
            forced_region=forced_region,
        )

    step_fn = remote_callable.fn if remote_callable is not None else step.fn
    assert step_fn is not None, f"Step {step.name} has no callable"

    # ExecutorStep functions take the resolved config as their only argument.
    # The output_path is already baked into the config, so the StepRunner-style
    # output_path argument is ignored here.
    captured_fn = step_fn
    captured_config = config
    captured_budget = mirror_budget_gb

    def resolved_fn(output_path):
        if captured_budget is not None:
            from rigging.filesystem import mirror_budget

            with mirror_budget(captured_budget):
                return captured_fn(captured_config)
        return captured_fn(captured_config)

    # Preserve the RemoteCallable wrapper for @remote-decorated fns so Fray
    # dispatch is retained; plain functions run inline in the runner thread.
    final_fn: Callable = resolved_fn
    if remote_callable is not None:
        final_fn = dataclasses.replace(remote_callable, fn=resolved_fn)

    return StepSpec(
        name=step.name,
        deps=deps or [],
        override_output_path=output_path,
        fn=final_fn,
        resources=step.resources,
    )


############################################################


@dataclass(frozen=True)
class ExecutorStepInfo:
    """
    Contains the information about an `ExecutorStep` that can be serialized into JSON.
    Note that this conversion is not reversible.
    """

    name: str
    """`step.name`."""

    fn_name: str
    """Rendered string of `step.fn`."""

    config: dataclass
    """`step.config`, but concretized (no more `InputName`, `OutputName`, or `VersionedValue`)."""

    description: str | None
    """`step.description`."""

    override_output_path: str | None
    """`step.override_output_path`."""

    version: dict[str, Any]
    """`executor.versions[step]`."""

    dependencies: list[str]
    """Fully realized output_paths of the dependencies."""

    output_path: str
    """`executor.output_paths[step]`."""


@dataclass(frozen=True)
class ExecutorInfo:
    """Contains information about an execution."""

    # Metadata related to the launch
    worker_id: str
    git_commit: str | None
    caller_path: str
    created_date: str
    user: str | None

    # Information taken from `Executor`
    prefix: str
    description: str | None
    steps: list[ExecutorStepInfo]


def _get_info_path(output_path: str) -> str:
    """Return the `path` of the info file associated with `output_path`."""
    return os.path.join(output_path, ".executor_info")


############################################################


def dependency_index_str(i: int) -> str:
    return f"DEP[{i}]"


@dataclass(frozen=True)
class _Dependencies:
    """
    Contains the dependencies of a step, the pseudo-dependencies, and the version of the dependencies.
    Internal use.
    """

    dependencies: list[ExecutorStep]
    """List of dependencies."""
    pseudo_dependencies: list[ExecutorStep]
    """List of pseudo-dependencies."""
    version: dict[str, Any]
    """Version of the dependencies."""


def collect_dependencies_and_version(obj: Any) -> _Dependencies:
    """Recurse through `obj` to find all the versioned values, and return them
    as a dict where the key is the sequence of fields identifying where the
    value resides in obj.  Example:

        get_version(Foo(a=versioned(1), b=Bar(c=versioned(2)))

           should return

        {"a": 1, "b.c": 2}

    Along the way, compute the list of dependencies.

    Each ``InputName`` with ``step is not None`` gets a ``DEP[i]`` slot in
    the version string, where ``i`` is the index in the combined
    ``dependencies + pseudo_dependencies`` list at the moment the dep is
    discovered.

    Returns:
        - dependencies: list of `ExecutorStep`s that are dependencies of the
          current step.
        - version: dict of versioned values, where the key is the sequence of
          fields identifying where the value resides in obj.
        - pseudo_dependencies: list of `ExecutorStep`s that are dependencies of the step but that we won't
            actually block on
    """
    pseudo_dependencies: list[ExecutorStep] = []
    dependencies: list[ExecutorStep] = []
    version: dict[str, Any] = {}

    for event in walk_config(obj):
        if isinstance(event, VersionedEvent):
            version[event.prefix] = event.value
            continue
        assert isinstance(event, InputNameEvent)
        input_name = event.input_name
        if input_name.step is None:
            version[event.prefix] = input_name.name
            continue
        index = len(dependencies) + len(pseudo_dependencies)
        if not input_name.block_on_step:
            pseudo_dependencies.append(input_name.step)
        else:
            dependencies.append(input_name.step)
        version[event.prefix] = dependency_index_str(index) + ("/" + input_name.name if input_name.name else "")

    return _Dependencies(dependencies, pseudo_dependencies, version)


def _max_mirror_budget(config: Any) -> float | None:
    """Extract the maximum mirror budget from MirroredValue entries in a raw config."""
    max_budget: float | None = None

    def recurse(obj: Any) -> None:
        nonlocal max_budget
        if obj is None:
            return
        if isinstance(obj, MirroredValue):
            if max_budget is None or obj.budget_gb > max_budget:
                max_budget = obj.budget_gb
            return
        if isinstance(obj, VersionedValue):
            recurse(obj.value)
            return
        if isinstance(obj, InputName | ExecutorStep):
            return
        if is_dataclass(obj):
            for field in fields(obj):
                recurse(getattr(obj, field.name))
        elif isinstance(obj, list):
            for x in obj:
                recurse(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                recurse(x)

    recurse(config)
    return max_budget


class Executor:
    """
    Performs the execution of a pipeline of `ExecutorStep`s.
    1. Instantiate all the `output_path`s for each `ExecutorStep` based on `prefix`, names, and versions of everything.
    2. Run each `ExecutorStep` in a proper topological sort order.
    """

    def __init__(
        self,
        prefix: str,
        executor_info_base_path: str,
        description: str | None = None,
    ):
        self.prefix = prefix
        self.executor_info_base_path = executor_info_base_path
        self.description = description

        self.configs: dict[ExecutorStep, dataclass] = {}
        self.dependencies: dict[ExecutorStep, list[ExecutorStep]] = {}
        self.versions: dict[ExecutorStep, dict[str, Any]] = {}
        # pseudo-dependencies only impact version but don't block execution of descendants
        # this dict contains is True for steps that are only used as pseudo-dependencies
        self.is_pseudo_dep: dict[ExecutorStep, bool] = {}
        self.version_strs: dict[ExecutorStep, str] = {}
        self.version_str_to_step: dict[str, ExecutorStep] = {}
        self.hashed_versions: dict[ExecutorStep, str] = {}
        self.output_paths: dict[ExecutorStep, str] = {}
        self.steps: list[ExecutorStep] = []
        self.step_infos: list[ExecutorStepInfo] = []
        self.executor_info: ExecutorInfo | None = None
        self._depth_cache: dict[ExecutorStep, int] = {}

    def run(
        self,
        steps: list[ExecutorStep | InputName],
        *,
        dry_run: bool = False,
        run_only: list[str] | None = None,
        force_run_failed: bool = True,
        max_concurrent: int | None = None,
    ) -> dict["ExecutorStep", str]:
        """
        Run the pipeline of `ExecutorStep`s.

        Args:
            steps: The steps to run.
            dry_run: If True, only print out what needs to be done. Reads existing
                statuses to report which steps would actually be executed.
            run_only: If not None, only run the steps in the list and their dependencies. Matches steps' names as regex
            force_run_failed: If True, run steps even if they have already been run (including if they failed)
            max_concurrent: Maximum number of steps to run concurrently. If None, run all ready steps in parallel.

        Returns:
            Mapping from every known `ExecutorStep` (including transitive
            dependencies discovered while walking `steps`) to its concrete
            output path. The returned dict is the same `self.output_paths`
            reference, so its keys may include steps that were not in the
            originally-passed `steps` list.
        """
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be a positive integer, got {max_concurrent}")

        # Gather all the steps, compute versions and output paths for all of them.
        logger.info(f"### Inspecting the {len(steps)} provided steps ###")
        for step in steps:
            if isinstance(step, InputName):  # Interpret InputName as the underlying step
                step = step.step
            if step is not None:
                self.compute_version(step, is_pseudo_dep=False)

        self.get_infos()
        logger.info(f"### Reading {len(self.steps)} statuses ###")

        if run_only is not None:
            steps_to_run = self._compute_transitive_deps(self.steps, run_only)
        else:
            steps_to_run = [step for step in self.steps if not self.is_pseudo_dep[step]]

        if steps_to_run != self.steps:
            logger.info(f"### Running {len(steps_to_run)} steps out of {len(self.steps)} ###")

        logger.info("### Writing metadata ###")
        self.write_infos()

        logger.info(f"### Launching {len(steps_to_run)} steps ###")
        if max_concurrent is not None:
            logger.info(f"### Max concurrent steps: {max_concurrent} ###")

        resolved_steps = self._resolve_steps(steps_to_run)
        StepRunner().run(
            resolved_steps,
            dry_run=dry_run,
            force_run_failed=force_run_failed,
            max_concurrent=max_concurrent,
        )
        return self.output_paths

    def _resolve_steps(self, steps: list[ExecutorStep]) -> list[StepSpec]:
        """Convert computed ExecutorStep state into a flat list of StepSpec."""
        dag_tpu_regions_by_step = _step_dag_tpu_regions(steps, self.dependencies)
        dep_stubs_by_step = {
            step: [
                StepSpec(name=dep.name, override_output_path=self.output_paths[dep])
                for dep in self.dependencies[step]
                if dep in self.output_paths
            ]
            for step in steps
        }
        forced_region_by_step = _component_tpu_pins(
            steps,
            self.dependencies,
            configs=self.configs,
            output_paths=self.output_paths,
            dep_stubs_by_step=dep_stubs_by_step,
            dag_tpu_regions_by_step=dag_tpu_regions_by_step,
        )
        # First pass: create StepSpecs without deps so we have a mapping
        spec_by_step: dict[ExecutorStep, StepSpec] = {}
        for step in steps:
            spec_by_step[step] = resolve_executor_step(
                step=step,
                config=self.configs[step],
                output_path=self.output_paths[step],
                deps=dep_stubs_by_step[step],
                dag_tpu_regions=dag_tpu_regions_by_step[step],
                forced_region=forced_region_by_step[step],
                mirror_budget_gb=_max_mirror_budget(step.config),
            )
        # Second pass: rebuild with deps pointing to resolved StepSpecs
        result = []
        for step in steps:
            dep_specs = [spec_by_step[dep] for dep in self.dependencies[step] if dep in spec_by_step]
            if dep_specs:
                result.append(dataclasses.replace(spec_by_step[step], deps=dep_specs))
            else:
                result.append(spec_by_step[step])
        return result

    def _compute_transitive_deps(self, steps: list[ExecutorStep], run_steps: list[str]) -> list[ExecutorStep]:
        """
        Compute the transitive dependencies of the steps that match the run_steps list.

        Returns steps in topological order.

        Args:
            steps: The list of all steps.
            run_steps: The list of step names to run. The names are matched as regex.
        """
        regexes = [re.compile(run_step) for run_step in run_steps]
        used_regexes: set[int] = set()

        def matches(step: ExecutorStep) -> bool:
            # track which regexes have been used
            for i, regex in enumerate(regexes):
                if regex.search(step.name):
                    used_regexes.add(i)
                    return True

            return False

        # Compute the transitive dependencies of the steps that match the run_steps list
        to_run: list[ExecutorStep] = []
        visited: set[ExecutorStep] = set()
        in_stack: set[ExecutorStep] = set()  # cycle detection

        def dfs(step: ExecutorStep):
            if step in in_stack:
                raise ValueError(f"Cycle detected in {step.name}")

            if step in visited:
                return

            visited.add(step)
            in_stack.add(step)

            info = self.step_infos[self.steps.index(step)]

            # only run if the step hasn't already been run
            status_file = StatusFile(info.output_path, worker_id="check")
            if status_file.status != STATUS_SUCCESS:
                for dep in self.dependencies[step]:
                    dfs(dep)
                to_run.append(step)
            else:
                logger.info(f"Skipping {step.name}'s dependencies as it has already been run")
            in_stack.remove(step)

        for step in steps:
            if matches(step):
                dfs(step)

        if used_regexes != set(range(len(regexes))):
            unused_regexes = [regexes[i].pattern for i in set(range(len(regexes))) - used_regexes]
            logger.warning(f"Regexes {unused_regexes} did not match any steps")

        return to_run

    def compute_version(self, step: ExecutorStep, is_pseudo_dep: bool):
        if step in self.versions:
            if not is_pseudo_dep and self.is_pseudo_dep[step]:
                logger.info(f"Step {step.name} was previously marked as skippable, but is not anymore.")
                self.is_pseudo_dep[step] = False

            return

        # Collect dependencies and the config version
        computed_deps = collect_dependencies_and_version(obj=step.config)
        # Recurse on dependencies
        for dep in computed_deps.dependencies:
            self.compute_version(dep, is_pseudo_dep=is_pseudo_dep)

        for dep in computed_deps.pseudo_dependencies:
            self.compute_version(dep, is_pseudo_dep=True)

        # The version specifies precisely all the information that uniquely
        # identifies this step.  Note that the fn name is not part of the
        # version.
        #
        # For deep dependency chains (depth > 4), we use output_paths (which
        # already encode the version hash) instead of the full nested version
        # dicts to avoid exponential blowup of the version structure.
        version = {
            "name": step.name,
            "config": computed_deps.version,
            "dependencies": [self._dep_version(dep) for dep in computed_deps.dependencies],
        }

        if computed_deps.pseudo_dependencies:
            # don't put this in the literal to avoid changing the hash for runs without pseudo-deps
            version["pseudo_dependencies"] = [self._dep_version(dep) for dep in computed_deps.pseudo_dependencies]

        # Compute output path
        version_str = json.dumps(version, sort_keys=True, cls=CustomJsonEncoder)
        hashed_version = hashlib.md5(version_str.encode()).hexdigest()[:6]
        output_path = os.path.join(self.prefix, step.name + "-" + hashed_version)

        # Override output path if specified
        override_path = step.override_output_path
        if override_path is not None:
            override_path = _make_prefix_absolute_path(self.prefix, override_path)

            if output_path != override_path:
                logger.warning(
                    f"Output path {output_path} doesn't match given "
                    f"override {step.override_output_path}, using the latter."
                )
                output_path = override_path

        # Record everything
        # Multiple `ExecutorStep`s can have the same version, so only keep one
        # of them.  Note that some `ExecutorStep`s might have depenedencies that
        # are not part of `self.steps`, but there will be some step with the
        # same version.
        if version_str not in self.version_str_to_step:
            self.steps.append(step)
            self.version_str_to_step[version_str] = step
        else:
            logger.warning(
                f"Multiple `ExecutorStep`s (named {step.name}) have the same version; try to instantiate only once."
            )

        self.configs[step] = instantiate_config(
            config=step.config,
            output_path=output_path,
            output_paths=self.output_paths,
            prefix=self.prefix,
        )
        self.dependencies[step] = list(map(self.canonicalize, computed_deps.dependencies))
        self.versions[step] = version
        self.version_strs[step] = version_str
        self.hashed_versions[step] = hashed_version
        self.output_paths[step] = output_path
        self.is_pseudo_dep[step] = is_pseudo_dep

    _MAX_INLINE_DEPTH = 4

    def _dep_depth(self, step: ExecutorStep) -> int:
        """Return the maximum dependency chain depth for a step (cached)."""
        if step in self._depth_cache:
            return self._depth_cache[step]
        deps = self.dependencies.get(step, [])
        if not deps:
            depth = 0
        else:
            depth = 1 + max(self._dep_depth(dep) for dep in deps)
        self._depth_cache[step] = depth
        return depth

    def _dep_version(self, dep: ExecutorStep) -> dict[str, Any] | str:
        """Full version dict for shallow deps, region-stable name+hash for deep ones.

        Avoids ``output_paths[dep]`` because that would bake the bucket prefix
        (e.g. ``gs://marin-us-central1``) into the hashed version, making the
        same logical pipeline hash differently under a different
        ``MARIN_PREFIX``. ``{name}-{hashed_version}`` is the region-independent
        suffix that already encodes the dep's full transitive version.
        """
        if self._dep_depth(dep) <= self._MAX_INLINE_DEPTH:
            return self.versions[dep]
        return f"{dep.name}-{self.hashed_versions[dep]}"

    def canonicalize(self, step: ExecutorStep) -> ExecutorStep:
        """Multiple instances of `ExecutorStep` might have the same version."""
        return self.version_str_to_step[self.version_strs[step]]

    def get_infos(self):
        """Calculates info files for each step and also entire execution"""
        # Compute info for each step
        for step in self.steps:
            self.step_infos.append(
                ExecutorStepInfo(
                    name=step.name,
                    fn_name=get_fn_name(step.fn),
                    config=self.configs[step],
                    description=step.description,
                    override_output_path=step.override_output_path,
                    version=self.versions[step],
                    dependencies=[self.output_paths[dep] for dep in self.dependencies[step]],
                    output_path=self.output_paths[step],
                )
            )

        # Compute info for the entire execution
        path = get_caller_path()
        self.executor_info = ExecutorInfo(
            git_commit=get_git_commit(),
            caller_path=path,
            created_date=datetime.now().isoformat(),
            user=get_user(),
            worker_id=worker_id(),
            prefix=self.prefix,
            description=self.description,
            steps=self.step_infos,
        )

    def get_experiment_url(self) -> str:
        """Return the URL where the experiment can be viewed."""
        if self.prefix.startswith("gs://"):
            host = "https://marin.community/data-browser"
        else:
            host = f"http://localhost:{_get_local_data_browser_port()}"

        return host + "/experiment?path=" + urllib.parse.quote(self.executor_info_path)

    def write_infos(self):
        """Output JSON files (one for the entire execution, one for each step)."""

        # Set executor_info_path based on hash and caller path name (e.g., 72_baselines-8c2f3a.json)
        # we pre-compute the asdict as it can be expensive.
        executor_info_dict = asdict_without_description(self.executor_info)
        step_infos = executor_info_dict["steps"]
        for s in step_infos:
            s.pop("description", None)

        executor_version_str = json.dumps(step_infos, sort_keys=True, cls=CustomJsonEncoder)
        executor_version_hash = hashlib.md5(executor_version_str.encode()).hexdigest()[:6]
        name = os.path.basename(self.executor_info.caller_path).replace(".py", "")
        self.executor_info_path = os.path.join(
            self.executor_info_base_path,
            f"{name}-{executor_version_hash}.json",
        )

        # Print where to find the executor info (experiments JSON)
        logger.info(f"Writing executor info to {self.executor_info_path}")
        if not self.prefix.startswith("gs://"):
            logger.info("Start data browser: cd data_browser && uv run python run-dev.py --config conf/local.conf")
        logger.info("To view the experiment page, go to:")
        logger.info("")
        logger.info(self.get_experiment_url())
        logger.info("")
        # Write out info for each step
        for step, info in zip(self.steps, executor_info_dict["steps"], strict=True):
            info_path = _get_info_path(self.output_paths[step])
            fsspec_utils.mkdirs(os.path.dirname(info_path))
            with open_url(info_path, "w") as f:
                print(json.dumps(info, indent=2, cls=CustomJsonEncoder), file=f)

        # Write out info for the entire execution
        fsspec_utils.mkdirs(os.path.dirname(self.executor_info_path))
        with open_url(self.executor_info_path, "w") as f:
            print(json.dumps(executor_info_dict, indent=2, cls=CustomJsonEncoder), file=f)


def get_fn_name(fn: ExecutorFunction, short: bool = False):
    """Just for debugging: get the name of the function."""
    if fn is None:
        return "None"
    if isinstance(fn, RemoteCallable):
        return fn.fn.__name__
    if short:
        return f"{fn.__name__}"
    else:
        return str(fn)


def get_git_commit() -> str | None:
    """Return the git commit of the current branch (if it can be found)"""
    if os.path.exists(".git"):
        return os.popen("git rev-parse HEAD").read().strip()
    else:
        return None


def get_caller_path() -> str:
    """Return the path of the file that called this function.

    Walks the stack from the outermost frame inward, returning the first
    frame that corresponds to a real file (skips ``<frozen runpy>`` and
    similar synthetic frames produced by ``python -m`` invocation).
    """
    for frame_info in reversed(inspect.stack()):
        if not frame_info.filename.startswith("<"):
            return frame_info.filename
    # All frames are synthetic (shouldn't happen in practice) — fall back to argv.
    return sys.argv[0]


def get_user() -> str | None:
    return subprocess.check_output("whoami", shell=True).strip().decode("utf-8")


############################################################


@dataclass(frozen=True)
class ExecutorMainConfig:
    prefix: str | None = None
    """Attached to every output path that's constructed (e.g., the GCS bucket)."""

    executor_info_base_path: str | None = None
    """Where the executor info should be stored under a file determined by a hash."""

    dry_run: bool = False
    force_run_failed: bool = True  # Force run failed steps
    run_only: list[str] | None = None
    """Run these steps (matched by regex.search) and their dependencies only. If None, run all steps."""

    max_concurrent: int | None = None
    """Maximum number of steps to run concurrently. If None, run all ready steps in parallel (default)."""


@draccus.wrap()
def executor_main(config: ExecutorMainConfig, steps: list[ExecutorStep], description: str | None = None):
    """Main entry point for experiments (to standardize)"""

    configure_logging(level=logging.INFO)
    time_in = time.time()

    prefix = config.prefix or marin_prefix()

    executor_info_base_path = config.executor_info_base_path
    if executor_info_base_path is None:
        # infer from prefix
        executor_info_base_path = os.path.join(prefix, "experiments")

    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
        description=description,
    )

    executor.run(
        steps=steps,
        dry_run=config.dry_run,
        run_only=config.run_only,
        force_run_failed=config.force_run_failed,
        max_concurrent=config.max_concurrent,
    )
    time_out = time.time()
    logger.info(f"Executor run took {time_out - time_in:.2f}s")
    # print json path again so it's easy to copy
    logger.info(f"Executor info written to {executor.executor_info_path}")
    if not executor.prefix.startswith("gs://"):
        logger.info("Start data browser: cd data_browser && uv run python run-dev.py --config conf/local.conf")
    logger.info(f"View the experiment at {executor.get_experiment_url()}")


############################################################
# Materialize: helpers that drive an Executor instance to resolve
# placeholder configs at runtime. Live here (not in dag.py) because they
# instantiate `Executor`, and the dependency direction is executor → dag.
############################################################


def compute_output_path(
    name: str,
    config: Any,
    *,
    override_output_path: str | None = None,
    prefix: str | None = None,
) -> str:
    """Compute the concrete output path a step with this name+config will produce.

    Drives ``Executor.compute_version`` (which walks the config's dependency
    graph and hashes versioned values — no GCS I/O, no job submission) far
    enough to populate the resulting output path. Honors ``override_output_path``
    if provided. Otherwise resolves ``prefix`` from ``marin_prefix()`` and
    derives the path from ``name`` + a hash of the config's versioned values,
    matching ``Executor``'s scheme so a step run via ``Executor.run`` and a
    path computed here agree on the same value.
    """
    resolved_prefix = prefix if prefix is not None else marin_prefix()
    executor_info_base_path = os.path.join(resolved_prefix, "experiments")
    executor = Executor(
        prefix=resolved_prefix,
        executor_info_base_path=executor_info_base_path,
    )
    # Build a transient ExecutorStep purely as input to compute_version. The
    # step is never submitted; only its name + config + override_output_path
    # are read to derive the path. Keeping this private to compute_output_path
    # means callers don't have to construct a fake step they never run.
    step = ExecutorStep(
        name=name,
        fn=_noop_step_fn,
        config=config,
        override_output_path=override_output_path,
    )
    executor.compute_version(step, is_pseudo_dep=False)
    return executor.output_paths[step]


def _noop_step_fn(config: Any) -> None:
    """Placeholder fn for the transient step built inside ``compute_output_path``.

    The step is discarded after path computation; this fn is never called.
    """
    return None


def materialize(
    config: ConfigT,
    *,
    prefix: str | None = None,
    output_path: str | None = None,
) -> ConfigT:
    """Run any ``ExecutorStep``s embedded in ``config``, then return a copy of
    ``config`` with all placeholder paths substituted.

    Composes three pieces:

      1. ``upstream_steps(config)`` — find embedded ``ExecutorStep``s.
      2. ``Executor(prefix=...).run(steps)`` — submit them as sub-jobs and
         block on completion. The distributed ``step_lock`` keeps concurrent
         callers (sweep members, multi-VM TPU tasks) coordinated; only one
         actually runs each step, the rest read ``STATUS_SUCCESS`` and skip.
      3. ``instantiate_config(config, output_path=<resolved>,
         output_paths=executor.output_paths, prefix=prefix)`` — substitute
         ``InputName`` / ``OutputName`` / ``VersionedValue`` / ``ExecutorStep``
         placeholders using the just-computed paths.

    Args:
        config: A launcher config dataclass that may embed ``ExecutorStep``s
            and placeholder values.
        prefix: Storage prefix for newly-submitted sub-jobs. Defaults to
            ``marin_prefix()`` (the worker's regional ``gs://marin-{R}``
            bucket), so upstream data is co-located with training.
        output_path: Concrete output path for the current step, used to
            resolve ``OutputName(name=...)`` placeholders inside ``config``.
            If ``None``, ``materialize`` reads ``config.output_path``. For
            callers whose config type does not expose ``output_path``, pass
            it explicitly.

    Returns:
        A copy of ``config`` with all placeholders substituted to concrete
        paths. A config containing no placeholders round-trips unchanged
        (idempotent — no sub-jobs submitted).
    """
    resolved_prefix = prefix if prefix is not None else marin_prefix()

    steps = upstream_steps(config)

    # Idempotence guard: if no sub-steps reference the config, skip the
    # `Executor.run` path entirely. `Executor.run([])` would otherwise still
    # write out an executor-info JSON to GCS (`write_infos` runs unconditionally
    # in `executor.run`), which is both pointless and an unwanted I/O side
    # effect for a placeholder-free config.
    if steps:
        executor_info_base_path = os.path.join(resolved_prefix, "experiments")
        executor = Executor(
            prefix=resolved_prefix,
            executor_info_base_path=executor_info_base_path,
        )
        output_paths: dict[ExecutorStep, str] = executor.run(steps=steps)
    else:
        output_paths = {}

    if output_path is None:
        current_output_path = config.output_path  # type: ignore[attr-defined]
    else:
        current_output_path = output_path

    if isinstance(current_output_path, OutputName):
        raise TypeError(
            "materialize(config): output_path is still an OutputName "
            "placeholder. The launcher / job-submission layer must resolve the "
            "current step's output_path to a concrete string before calling "
            "the worker function. Got: "
            f"{current_output_path!r}"
        )

    return instantiate_config(
        config=config,
        output_path=current_output_path,
        output_paths=output_paths,
        prefix=resolved_prefix,
    )
