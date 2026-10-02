# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the host-offload mesh-rebinding fix to the pinned JAXPP diagnostic build."""

import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("jaxpp")
path = Path(next(iter(spec.submodule_search_locations))) / "core.py"
source = path.read_text()
old = """"devices": updated_named_sharding_mesh(eqn.params["devices"], new_mesh)"""
new = """"devices": jax.tree.map(
                    lambda target: target if isinstance(target, jcore.MemorySpace)
                    else updated_named_sharding_mesh(target, new_mesh),
                    eqn.params["devices"],
                )"""
if source.count(new) == 1 and old not in source:
    print("JAXPP offload patch already present in cached environment")
else:
    assert source.count(old) == 1, "Unexpected pinned JAXPP device_put rebinding code"
    path.write_text(source.replace(old, new))
print("Patched JAXPP MemorySpace transfer rebinding:", path)

source = path.read_text()
forward_offload_dependencies = """def _schedule_pipeline_dependencies(eqns, tgt_eqn_idx):
    boundary = eqns[tgt_eqn_idx]
    if boundary.params["task_type"] is not TaskType.FWD:
        return schedule_dependencies(eqns, tgt_eqn_idx)
    # Checkpoint host copies have no forward data dependency. Keep copies that
    # precede this forward boundary here, rather than delaying them until backward.
    targets = list(boundary.outvars)
    for eqn in eqns[:tgt_eqn_idx]:
        if eqn.primitive is jax.lax.device_put_p and any(
            target is jcore.MemorySpace.Host for target in eqn.params["devices"]
        ):
            targets.extend(eqn.outvars)
    dependencies, deferred, _ = partition_eqns(eqns[:tgt_eqn_idx + 1], targets)
    return dependencies, deferred + eqns[tgt_eqn_idx + 1:]


"""
anchor = "def cluster_by_yield_eqns("
if "def _schedule_pipeline_dependencies(" not in source:
    assert source.count(anchor) == 1
    source = source.replace(anchor, forward_offload_dependencies + anchor)
for stage in ("stage_0", "stage_i"):
    old_call = f"{stage}, eqns = schedule_dependencies(eqns, pp_eqn_idx)"
    new_call = f"{stage}, eqns = _schedule_pipeline_dependencies(eqns, pp_eqn_idx)"
    assert source.count(old_call) + source.count(new_call) == 1
    source = source.replace(old_call, new_call)
path.write_text(source)
print("Patched JAXPP forward host-residual placement:", path)

sharding_path = path.with_name("sharding_inference.py")
source = sharding_path.read_text()
old_memory = """        if requested is None:
            return sharding
        return update_named_sharding(sharding, memory_kind=requested.memory_kind)"""
new_memory = """        if requested is None:
            if outvar.aval.memory_space is jcore.MemorySpace.Host:
                return update_named_sharding(sharding, memory_kind="pinned_host")
            return sharding
        return update_named_sharding(sharding, memory_kind=requested.memory_kind)"""
assert source.count(old_memory) + source.count(new_memory) == 1
sharding_path.write_text(source.replace(old_memory, new_memory))
print("Patched JAXPP intermediate host-residual shardings:", sharding_path)

path = path.with_name("jax_primitives.py")
source = path.read_text()
marker = "_precompiled_pipeline_tasks: dict"
registry = '''def _pipeline_warmup_input(aval, sharding):
    if aval.weak_type:
        assert aval.shape == (), f"Unexpected weak nonscalar input: {aval}"
        value = 1.0 if np.issubdtype(aval.dtype, np.inexact) else 0
        array = jax.device_put(value, sharding)
        assert array.dtype == aval.dtype, (array.dtype, aval.dtype)
        return array
    shard_shape = sharding.shard_shape(aval.shape)
    fill = 1 if aval.shape == () and np.issubdtype(aval.dtype, np.inexact) else 0
    return jax.make_array_from_callback(
        aval.shape, sharding,
        # BF16 np.full uses a custom per-element fill loop; zero initialization
        # avoids that loop for the large synthetic parameter shards.
        lambda _: np.zeros(shard_shape, dtype=aval.dtype) if fill == 0 else np.full(shard_shape, fill, dtype=aval.dtype),
    )


def _warm_pipeline_local_jaxpr(closed_jaxpr, mpmd_mesh):
    """Replay local dataflow with synthetic external inputs and no pipeline I/O."""
    jaxpr = closed_jaxpr.jaxpr
    if any(eqn.primitive is dax_pscan_p for eqn in jaxpr.eqns):
        raise NotImplementedError("Pipeline warmup requires an unrolled local JAXPR")
    local_mesh = mpmd_mesh.unstack[mpmd_mesh.my_mpmd_axis_index]
    shardings = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive is task_p:
            for variables, specs in (
                (eqn.invars, eqn.params["in_shardings"]),
                (eqn.outvars, eqn.params["out_shardings"]),
            ):
                for var, sharding in zip(variables, specs, strict=True):
                    if isinstance(var, jcore.Var):
                        shardings[var] = sharding
    def input_sharding(var):
        if var in shardings:
            return shardings[var]
        aval = var.aval
        spec = aval.sharding.spec
        memory_kind = "pinned_host" if aval.memory_space == jcore.MemorySpace.Host else "device"
        return jax.sharding.NamedSharding(local_mesh, spec, memory_kind=memory_kind)

    # Only external inputs are synthesized. Produced values must come from their
    # actual local equations, including forward residuals and intervening aliases.
    env = {var: _pipeline_warmup_input(var.aval, input_sharding(var)) for var in jaxpr.invars}
    for var, value in zip(jaxpr.constvars, closed_jaxpr.consts, strict=True):
        env[var] = jax.device_put(value, input_sharding(var), may_alias=False)
    forward_values = set()
    last_use = {}
    for index, eqn in enumerate(jaxpr.eqns):
        for var in eqn.invars:
            if isinstance(var, jcore.Var):
                last_use[var] = index
    for var in jaxpr.outvars:
        if isinstance(var, jcore.Var):
            last_use[var] = len(jaxpr.eqns)

    for index, eqn in enumerate(jaxpr.eqns):
        inputs = [var.val if isinstance(var, jcore.Literal) else env[var] for var in eqn.invars]
        reused = sum(isinstance(var, jcore.Var) and var in forward_values for var in eqn.invars)
        if eqn.primitive is transfer_start_p:
            # Real receives are external to this rank. Use fresh zero buffers;
            # a Python token is sufficient because recv_done is handled below.
            recv_avals = [var.aval for var in eqn.outvars[1:]]
            outputs = [None] + [
                _pipeline_warmup_input(aval, sharding)
                for aval, sharding in zip(recv_avals, eqn.params["recv_local_shardings"], strict=True)
            ]
        elif eqn.primitive is recv_done_p or eqn.primitive is transfer_done_p:
            outputs = inputs[1:]
        elif eqn.primitive is delete_p:
            # The primitive returns its input handles. Keep aliases valid during
            # replay; last-use reference release below owns disposable storage.
            outputs = inputs
        elif eqn.primitive is reuse_fence_p:
            outputs = inputs
        else:
            if communication_effect in eqn.effects:
                raise NotImplementedError(f"Unsupported pipeline warmup communication: {eqn.primitive}")
            if eqn.primitive is task_p:
                name = eqn.params["task_name"]
                if name.startswith("bwd") and not reused:
                    raise RuntimeError(f"Backward warmup has no forward-produced residual inputs: {name}")
                p = eqn.params
                _, task_mesh = _resolve_placement(mpmd_mesh, p["mpmd_idx"], name="task mpmd_idx")
                task_params = PjitKwargs(
                    jaxpr=p["call_jaxpr"],
                    in_shardings=p["in_shardings"],
                    out_shardings=p["out_shardings"],
                    in_layouts=(None,) * len(inputs),
                    out_layouts=(None,) * len(p["out_shardings"]),
                    donated_invars=tuple(p["donate_invars"]),
                    ctx_mesh=task_mesh,
                    name=name,
                )
                executable = _precompiled_pipeline_tasks[(jc.jit_p, task_params)]
                expected_inputs = jax.tree.leaves(executable.input_shardings)
                actual_kinds, expected_kinds, output_kinds = {}, {}, {}
                mismatches = []
                for input_index, (value, expected) in enumerate(zip(inputs, expected_inputs, strict=True)):
                    actual = value.sharding if isinstance(value, jax.Array) else None
                    kind = actual.memory_kind if actual is not None else "python_scalar"
                    actual_kinds[kind] = actual_kinds.get(kind, 0) + 1
                    expected_kinds[expected.memory_kind] = expected_kinds.get(expected.memory_kind, 0) + 1
                    if actual is not None and not actual.is_equivalent_to(expected, value.ndim):
                        mismatches.append({
                            "input_index": input_index, "shape": value.shape,
                            "actual": str(actual), "expected": str(expected),
                        })
                for output_sharding in jax.tree.leaves(executable.output_shardings):
                    kind = output_sharding.memory_kind
                    output_kinds[kind] = output_kinds.get(kind, 0) + 1
                print("HERO_TASK_MEMORY " + json.dumps({
                    "event": "pipeline_task_abi", "process_index": jax.process_index(),
                    "task": name, "input_count": len(inputs),
                    "actual_input_memory_kinds": actual_kinds,
                    "compiled_input_memory_kinds": expected_kinds,
                    "compiled_output_memory_kinds": output_kinds,
                    "donated_input_indices": [i for i, donated in enumerate(p["donate_invars"]) if donated],
                    "mismatch_count": len(mismatches), "mismatches": mismatches[:8],
                }), flush=True)
                del executable, expected_inputs
                if inputs:
                    del value
                print("HERO_TASK_MEMORY " + json.dumps({
                    "event": "pipeline_task_warmup", "process_index": jax.process_index(),
                    "task": name, "reused_forward_inputs": reused,
                    "input_count": len(inputs),
                    "local_input_bytes": sum(
                        shard.data.nbytes for value in inputs if isinstance(value, jax.Array)
                        for shard in value.addressable_shards
                    ),
                    "device_memory_before_execute": {
                        str(device.id): device.memory_stats() for device in jax.local_devices()
                    },
                }), flush=True)
            with eqn.ctx.manager:
                result = eqn.primitive.bind(*inputs, **eqn.primitive.get_bind_params(eqn.params))
            outputs = list(result) if eqn.primitive.multiple_results else [result]
            del result
            jax.block_until_ready(outputs)
            if eqn.primitive is task_p:
                logging.info("Warmed pipeline task %s", eqn.params["task_name"])
        assert len(outputs) == len(eqn.outvars), eqn.primitive
        from_forward = eqn.primitive is not transfer_start_p and (
            reused or (eqn.primitive is task_p and eqn.params["task_name"].startswith("fwd"))
        )
        for var, value in zip(eqn.outvars, outputs, strict=True):
            if isinstance(var, jcore.Var) and var in last_use:
                env[var] = value
                if from_forward:
                    forward_values.add(var)
        for var in eqn.invars:
            if isinstance(var, jcore.Var) and last_use[var] == index:
                env.pop(var, None)
                forward_values.discard(var)
        del inputs, outputs
    jax.block_until_ready(list(env.values()))
    env.clear()


_precompiled_pipeline_tasks: dict[tuple[jcore.Primitive, PjitKwargs], jax.stages.Compiled] = {}
_require_precompiled_pipeline_tasks = False


def precompile_pipeline_tasks(closed_jaxpr, mpmd_mesh):
    """Compile and execute disposable inputs before pipeline communication."""
    global _require_precompiled_pipeline_tasks
    _require_precompiled_pipeline_tasks = False
    _precompiled_pipeline_tasks.clear()

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive is task_p:
                p = eqn.params
                call_jaxpr = p["call_jaxpr"]
                _, mesh = _resolve_placement(mpmd_mesh, p["mpmd_idx"], name="task mpmd_idx")
                params = PjitKwargs(
                    jaxpr=call_jaxpr,
                    in_shardings=p["in_shardings"],
                    out_shardings=p["out_shardings"],
                    in_layouts=(None,) * len(call_jaxpr.in_avals),
                    out_layouts=(None,) * len(p["out_shardings"]),
                    donated_invars=tuple(p["donate_invars"]),
                    ctx_mesh=mesh,
                    name=p["task_name"],
                )
                key = (jc.jit_p, params)
                if key not in _precompiled_pipeline_tasks:
                    logging.info("Precompiling pipeline task %s", params.name)
                    with jax.set_mesh(mesh):
                        executable = callable_task(jc.jit_p, params).lower(*call_jaxpr.in_avals).compile()
                    _precompiled_pipeline_tasks[key] = executable
                    memory = executable.memory_analysis()
                    assert memory is not None, "GPU executable did not expose memory analysis"
                    memory_fields = (
                        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                        "alias_size_in_bytes", "generated_code_size_in_bytes",
                        "host_argument_size_in_bytes", "host_output_size_in_bytes",
                        "host_temp_size_in_bytes", "host_alias_size_in_bytes",
                        "host_generated_code_size_in_bytes",
                    )
                    print("HERO_TASK_MEMORY " + json.dumps({
                        "event": "pipeline_task_memory", "process_index": jax.process_index(),
                        "task": params.name,
                        "compiled": {field: getattr(memory, field) for field in memory_fields},
                        "device_memory_before_dummy_inputs": {
                            str(device.id): device.memory_stats() for device in jax.local_devices()
                        },
                    }), flush=True)
            else:
                for value in jax.tree.leaves(eqn.params):
                    if isinstance(value, (jcore.Jaxpr, jcore.ClosedJaxpr)):
                        visit(value)

    with mpmd_mesh:
        visit(closed_jaxpr)
    _require_precompiled_pipeline_tasks = True
    with mpmd_mesh:
        _warm_pipeline_local_jaxpr(closed_jaxpr, mpmd_mesh)
    return len(_precompiled_pipeline_tasks)


'''
old = "def apply_task(prim: jcore.Primitive, *args, params: PjitKwargs):\n"
new = (
    old
    + """    if _require_precompiled_pipeline_tasks:
        key = (prim, params)
        if key not in _precompiled_pipeline_tasks:
            raise RuntimeError(f"Pipeline task was not precompiled: {params.name}")
        with jax.set_mesh(params.ctx_mesh):
            return _precompiled_pipeline_tasks[key](*args)
"""
)
if marker in source:
    start_marker = "def _pipeline_warmup_input(" if "def _pipeline_warmup_input(" in source else marker
    start = source.index(start_marker)
    end = source.index(old, start)
    source = source[:start] + registry + source[end:]
    # Reset the explicitly optional completion wrapper when refreshing its helper region.
    # A subsequent opt-in overlay can enable it again; cached environments must not.
    source = source.replace(
        "            return _complete_pipeline_task(_precompiled_pipeline_tasks[key], args, params.name)",
        "            return _precompiled_pipeline_tasks[key](*args)",
    )
else:
    assert source.count(old) == 1, "Unexpected pinned JAXPP apply_task definition"
    source = source.replace(old, registry + new)
if "import json\n" not in source:
    source = source.replace("import logging\n", "import logging\nimport json\n")
if "import numpy as np\n" not in source:
    assert source.count("import jax\n") == 1
    source = source.replace("import jax\n", "import jax\nimport numpy as np\n")
path.write_text(source)
print("Patched JAXPP explicit pipeline task compilation:", path)
