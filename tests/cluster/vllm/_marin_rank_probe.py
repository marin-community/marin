# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-extension probe for the vLLM DP-rank variance investigation (plan G1/G2).

Shipped by ``_experiment_c_rank_variance.py``: the entrypoint writes this file to the
job node as ``marin_rank_probe.py``, prepends its directory to ``PYTHONPATH``, and
passes ``--worker-extension-cls marin_rank_probe.MarinRankProbe``. vLLM dynamically
mixes the class into the Worker (``WorkerWrapperBase.init_worker``), so every method
here is callable on all 8 DP workers via ``POST /collective_rpc`` (requires
``VLLM_SERVER_DEV_MODE=1``).

Data path: the DP client broadcasts the RPC to every engine but returns only the
first engine's result, so methods write per-rank JSON/tensor files into ``out_dir``
(all DP workers are local processes) and return a small ack dict. The harness reads
the files directly; nothing bulk travels through HTTP or job logs.

vLLM imports are deliberately deferred to method bodies: the module must import on a
CPU workspace (local dry-run test) and is imported inside the worker before device
init, where eager vllm imports would be premature. Exception: the optional NCCL-env
contrast hook below must patch ``override_envs_for_invariance`` at import time —
worker init imports this module *before* ``init_batch_invariance`` runs, which is the
only window where an NCCL env contrast can land (G3 combine branch).
"""

import json
import os
import traceback
import zlib
from typing import Any

import torch

_PROBE_VERSION = 3
_SIZES_LOG_MAX_LINES = 2000
_TRACE_MAX_CALLS = 64
# Deterministic operand seeds: cross-launch micro comparisons rely on these being fixed.
_MICRO_SEED_BASE = 1000

# G3 combine-branch contrast hook: overlay NCCL env *after* the invariance pinning so a
# contrast arm can observe e.g. NCCL_ALGO changes. Inert unless the env var is set.
_NCCL_OVERLAY = os.environ.get("MARIN_PROBE_NCCL_ENV_JSON")
if _NCCL_OVERLAY:
    from vllm.model_executor.layers import batch_invariant as _bi

    _original_override = _bi.override_envs_for_invariance

    def _overlaid_override() -> None:
        _original_override()
        os.environ.update(json.loads(_NCCL_OVERLAY))

    _bi.override_envs_for_invariance = _overlaid_override


def _text(value) -> str:
    """pynvml returns str on some versions and bytes on others."""
    return (value.decode() if isinstance(value, bytes) else str(value)).strip("\x00")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().contiguous().view(-1)
    if flat.dtype == torch.bfloat16:
        flat = flat.view(torch.uint16)
    elif flat.dtype == torch.float32:
        flat = flat.view(torch.int32)
    return flat.cpu().numpy().tobytes()


def _checksum(tensor: torch.Tensor) -> str:
    return f"{zlib.crc32(_tensor_bytes(tensor)):08x}"


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    as_f32 = tensor.detach().float()
    return {
        "max_abs": float(as_f32.abs().max()),
        "mean_abs": float(as_f32.abs().mean()),
        "l2": float(as_f32.norm()),
    }


def _write_json(out_dir: str, name: str, payload: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path + ".tmp", "w") as handle:
        json.dump(payload, handle)
    os.replace(path + ".tmp", path)


def _dp_group():
    from vllm.distributed.parallel_state import get_dp_group  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run

    return get_dp_group()


def _find_all2all_manager() -> tuple[Any, str]:
    """Return (manager, group_name).

    EP first, and it is the one that matters: the AgRs prepare/finalize path calls
    ``get_ep_group().dispatch/.combine`` (``prepare_finalize/naive_dp_ep.py:158,208``),
    which delegates to ``CudaCommunicator.{dispatch,combine}`` and then to this manager
    instance — so patching the instance's attributes intercepts the production path.
    """
    from vllm.distributed.parallel_state import (  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run
        get_dp_group,
        get_ep_group,
    )

    for name, group in (("ep", get_ep_group()), ("dp", get_dp_group())):
        manager = getattr(group.device_communicator, "all2all_manager", None)
        if manager is not None:
            return manager, name
    raise RuntimeError("no all2all_manager found on ep or dp device communicator")


def _reported(method):
    """Return probe failures as data instead of raising.

    An exception escaping a ``collective_rpc`` method kills the WorkerProc and takes
    the whole server (and the job's model load) with it — see multiproc_executor's
    worker_busy_loop. An instrument must not be able to destroy the measurement it is
    attached to, so every probe reports its own failure and lets the battery continue.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            return {"ok": False, "method": method.__name__, "error": traceback.format_exc()}

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


def _rank_order_reference(partials: list[torch.Tensor], order: tuple[int, ...]) -> torch.Tensor:
    """Fixed-order fp32 sum of bf16 partials, cast back to bf16 (the deterministic
    combine H1 predicts would collapse the spread)."""
    accumulator = torch.zeros_like(partials[0], dtype=torch.float32)
    for source in order:
        accumulator += partials[source].float()
    return accumulator.to(partials[0].dtype)


class MarinRankProbe:
    """Methods are attached to the vLLM Worker; every name must be collision-free
    against the Worker class (``marin_probe_`` prefix) or init_worker asserts."""

    @_reported
    def marin_probe_env(self, out_dir: str) -> dict:
        """Effective-configuration record: env, GPU identity, resolved backends."""
        import pynvml  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run
        import vllm  # noqa: PLC0415
        from vllm.distributed.parallel_state import get_dp_group, get_ep_group  # noqa: PLC0415

        dp_group = get_dp_group()
        rank = dp_group.rank_in_group
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        # Match by UUID across all NVML devices: worker CUDA indices are
        # CUDA_VISIBLE_DEVICES-relative while NVML indices are absolute, and the
        # by-UUID lookup needs a "GPU-" prefix torch does not include. pynvml returns
        # str or bytes depending on version, hence _text.
        pci_bus_id = None
        pynvml.nvmlInit()
        for nvml_index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            if _text(pynvml.nvmlDeviceGetUUID(handle)).removeprefix("GPU-") == str(properties.uuid):
                pci_bus_id = _text(pynvml.nvmlDeviceGetPciInfo(handle).busId)
                break

        vllm_config = self.vllm_config
        # Record the resolved MoE kernel stack per layer class. Matching on class name
        # alone found nothing on the first run, so key off the modules that actually
        # carry a quant_method and report the whole resolved chain — this is what a
        # launch-to-launch kernel-selection difference would show up in.
        moe_layers: dict[str, list[str]] = {}
        model = self.model_runner.model
        for module_name, module in model.named_modules():
            kernel = getattr(module, "quant_method", None)
            if kernel is None or "moe" not in type(module).__name__.lower():
                continue
            experts = getattr(kernel, "fused_experts", None)
            signature = "/".join(type(part).__name__ for part in (module, kernel, experts) if part is not None)
            moe_layers.setdefault(signature, []).append(module_name)

        record = {
            "probe_version": _PROBE_VERSION,
            "dp_rank": rank,
            "dp_world": dp_group.world_size,
            "ep_world": get_ep_group().world_size,
            "device_index": device,
            "gpu_name": properties.name,
            "gpu_uuid": str(properties.uuid),
            "pci_bus_id": pci_bus_id,
            "vllm_version": vllm.__version__,
            "vllm_commit": getattr(vllm, "__commit__", None),
            "torch_version": torch.__version__,
            "nccl_version": ".".join(str(part) for part in torch.cuda.nccl.version()),
            "env": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("NCCL_", "VLLM_", "CUBLAS_", "CUDA_VISIBLE", "TRITON_"))
            },
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": vllm_config.model_config.enforce_eager,
            "cudagraph_mode": str(vllm_config.compilation_config.cudagraph_mode),
            "max_num_batched_tokens": vllm_config.scheduler_config.max_num_batched_tokens,
            "expert_placement": vllm_config.parallel_config.expert_placement_strategy,
            "all2all_backend": vllm_config.parallel_config.all2all_backend,
            "moe_kernel_by_layer_group": {
                key: [len(value), value[0] if value else None] for key, value in moe_layers.items()
            },
        }
        _write_json(out_dir, f"env_rank{rank}.json", record)
        return {"ok": True, "rank": rank}

    @_reported
    def marin_probe_install_fixed_combine(self) -> dict:
        """G3: replace the EP combine with a destination-independent one.

        The production path reduce-scatters bf16 partials, so each destination's chunk
        is accumulated in a ring order rotated by destination — the microreproducer
        measured eight different results for one mathematical sum. This instead
        all-gathers every rank's partial and sums them in rank order in fp32, so every
        destination performs the identical arithmetic and only then slices its chunk.

        Deliberately not efficient: it moves world_size times the data. At the gate's
        --max-num-seqs 1 that is affordable, and the question here is whether removing
        the destination dependence removes the rank spread.
        """
        manager, group_name = _find_all2all_manager()
        rank = _dp_group().rank_in_group
        original_combine = manager.combine

        def fixed_order_combine(hidden_states, is_sequence_parallel=False, *args, **kwargs):
            from vllm.distributed.parallel_state import (  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run
                get_dp_group,
                get_ep_group,
            )
            from vllm.forward_context import get_forward_context  # noqa: PLC0415

            group = get_ep_group() if is_sequence_parallel else get_dp_group()
            sizes = get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()
            world = group.world_size
            gathered = group.all_gather(hidden_states.contiguous(), dim=0)
            stacked = gathered.view(world, hidden_states.shape[0], *hidden_states.shape[1:])
            accumulator = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
            for source in range(world):
                accumulator += stacked[source].float()
            start = sum(sizes[: group.rank_in_group])
            chunk = accumulator[start : start + sizes[group.rank_in_group]]
            return chunk.to(hidden_states.dtype).contiguous()

        manager.combine = fixed_order_combine
        self._marin_fixed_combine_original = (manager, original_combine)
        return {"ok": True, "rank": rank, "group": group_name}

    @_reported
    def marin_probe_uninstall_fixed_combine(self) -> dict:
        manager, original_combine = self._marin_fixed_combine_original
        manager.combine = original_combine
        return {"ok": True}

    @_reported
    def marin_probe_install_sizes_log(self, out_dir: str) -> dict:
        """Wrap the AgRs combine to record per-call DP chunk sizes — logs which
        collective variant fired (equal sizes: single ncclReduceScatter; uneven:
        grouped per-root ncclReduce).

        The per-call budget is per *tag*: a 15k-token prefill alone spends ~780 calls
        (30 chunks x 26 layers), so a single global budget is exhausted by the warmup
        and later modes get no coverage. Call marin_probe_rotate_sizes_log between
        modes to open a new file with a fresh budget.
        """
        manager, group_name = _find_all2all_manager()
        rank = _dp_group().rank_in_group
        os.makedirs(out_dir, exist_ok=True)
        state = {"calls": 0, "tag": "init", "out_dir": out_dir, "rank": rank}
        original_combine = manager.combine

        def logged_combine(hidden_states, *args, **kwargs):
            from vllm.forward_context import get_forward_context  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run

            state["calls"] += 1
            if state["calls"] <= _SIZES_LOG_MAX_LINES:
                metadata = get_forward_context().dp_metadata
                sizes = metadata.get_chunk_sizes_across_dp_rank() if metadata else None
                path = os.path.join(state["out_dir"], f"sizes_rank{rank}_{state['tag']}.jsonl")
                with open(path, "a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "call": state["calls"],
                                "sizes": sizes,
                                "equal": sizes is not None and len(set(sizes)) == 1,
                                "rows": int(hidden_states.shape[0]),
                            }
                        )
                        + "\n"
                    )
            return original_combine(hidden_states, *args, **kwargs)

        manager.combine = logged_combine
        self._marin_sizes_state = state
        return {"ok": True, "rank": rank, "group": group_name}

    @_reported
    def marin_probe_rotate_sizes_log(self, tag: str) -> dict:
        """Start a new sizes-log file with a fresh per-call budget."""
        state = self._marin_sizes_state
        state["tag"], state["calls"] = tag, 0
        return {"ok": True, "rank": state["rank"], "tag": tag}

    @_reported
    def marin_probe_microreproducer(
        self,
        out_dir: str,
        tokens_per_chunk: int = 512,
        hidden_dim: int = 2560,
        iters: int = 16,
    ) -> dict:
        """H1 primitive test through the production reduce-scatter path.

        Runs inside the live worker on the same communicator the MoE combine uses.
        Modes cover both production collective variants (equal sizes -> single
        ncclReduceScatter; isolated-shaped uneven sizes -> grouped per-root
        ncclReduce), dense and 4-of-8-sparse partials, at three magnitudes. Every
        rank regenerates all ranks' operands from fixed seeds, so fixed-order fp32
        references and permutation-order bounds are computed locally; operands are
        replicated across destination chunks in dense modes, so destinations that
        disagree do so purely through reduction order/rounding.
        """
        from vllm.distributed.parallel_state import get_ep_group  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run

        # The EP group is the one the MoE combine reduces over (naive_dp_ep.py:208), so
        # the primitive test must use it, not DP — same communicator, env, and stream.
        group = get_ep_group()
        rank, world = group.rank_in_group, group.world_size
        device = torch.device("cuda", torch.cuda.current_device())

        def block(source: int, scale: float, rows: int) -> torch.Tensor:
            generator = torch.Generator(device="cpu").manual_seed(_MICRO_SEED_BASE + source)
            values = torch.randn(rows, hidden_dim, generator=generator) * scale
            return values.to(torch.bfloat16).to(device)

        equal_sizes = [tokens_per_chunk] * world
        uneven_sizes = [tokens_per_chunk] + [1] * (world - 1)
        orderings: dict[str, tuple[int, ...]] = {
            "rank_order": tuple(range(world)),
            "reversed": tuple(reversed(range(world))),
            "ring_from_dest": tuple((rank + offset) % world for offset in range(world)),
        }
        results = []
        for mode, sizes, sparse in (
            ("equal_dense", equal_sizes, False),
            ("equal_sparse4", equal_sizes, True),
            ("uneven_dense", uneven_sizes, False),
        ):
            for scale in (1.0, 8.0, 32.0):
                # Partial from `source` for destination chunk d: same block per dest
                # (dense) or zeroed unless source is one of dest's 4 active ranks.
                # Loop vars bound as defaults: the closure is called in-iteration only.
                def partial(source: int, dest: int, sizes=sizes, scale=scale, sparse=sparse) -> torch.Tensor:
                    rows = sizes[dest]
                    values = block(source, scale, rows)
                    if sparse and source not in {(dest + step) % world for step in range(4)}:
                        return torch.zeros_like(values)
                    return values

                my_input = torch.cat([partial(rank, dest) for dest in range(world)], dim=0)
                my_partials = [partial(source, rank) for source in range(world)]
                references = {name: _rank_order_reference(my_partials, order) for name, order in orderings.items()}
                checksums, outputs = [], None
                for _ in range(iters):
                    outputs = group.reduce_scatterv(my_input, dim=0, sizes=list(sizes))
                    torch.cuda.synchronize(device)
                    checksums.append(_checksum(outputs))
                assert outputs is not None
                reference_spread = max(
                    float((references[a] != references[b]).sum()) for a in orderings for b in orderings
                )
                results.append(
                    {
                        "mode": mode,
                        "scale": scale,
                        "sizes": sizes,
                        "iters": iters,
                        "bitwise_stable_across_iters": len(set(checksums)) == 1,
                        "output_checksum": checksums[-1],
                        "output_stats": _stats(outputs),
                        "vs_reference_max_abs": {
                            name: float((outputs.float() - reference.float()).abs().max())
                            for name, reference in references.items()
                        },
                        "vs_reference_mismatch_elements": {
                            name: int((outputs != reference).sum()) for name, reference in references.items()
                        },
                        "reference_order_spread_elements": reference_spread,
                        "max_ulp_scale_bf16": float((references["rank_order"].float().abs().max()) * 2**-8),
                    }
                )
        record = {
            "probe_version": _PROBE_VERSION,
            "dp_rank": rank,
            "group": "ep",
            "tokens_per_chunk": tokens_per_chunk,
            "hidden_dim": hidden_dim,
            "results": results,
        }
        _write_json(out_dir, f"micro_rank{rank}.json", record)
        return {"ok": True, "rank": rank, "modes": len(results)}

    @_reported
    def marin_probe_arm_trace(self, out_dir: str, tag: str = "t", max_calls: int = _TRACE_MAX_CALLS) -> dict:
        """G2 boundary trace: capture dispatch/combine inputs and outputs (boundaries
        1/2/4/5) for the next ``max_calls`` MoE collective calls on this rank.

        ``tag`` names the capture so one server can trace several prompt lengths. Each
        entry records the DP size vector, which is what lets the analysis verify that
        call N means the same layer on every rank before comparing checksums.
        """
        manager, group_name = _find_all2all_manager()
        rank = _dp_group().rank_in_group
        os.makedirs(out_dir, exist_ok=True)
        state = {"dispatch": 0, "combine": 0}
        original_dispatch, original_combine = manager.dispatch, manager.combine

        def dp_sizes():
            from vllm.forward_context import get_forward_context  # noqa: PLC0415 -- lazy: no vllm on CPU dry-run

            metadata = get_forward_context().dp_metadata
            return metadata.get_chunk_sizes_across_dp_rank() if metadata else None

        def snap(boundary: str, call: int, tensor: torch.Tensor) -> dict:
            entry = {
                "tag": boundary,
                "call": call,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "checksum": _checksum(tensor),
                "sizes": dp_sizes(),
                **_stats(tensor),
            }
            torch.save(
                tensor[-1].detach().float().cpu(),
                os.path.join(out_dir, f"trace_rank{rank}_{tag}_{boundary}_{call}_lastrow.pt"),
            )
            return entry

        def append(entry: dict) -> None:
            with open(os.path.join(out_dir, f"trace_rank{rank}_{tag}.jsonl"), "a") as handle:
                handle.write(json.dumps(entry) + "\n")

        def traced_dispatch(hidden_states, topk_weights, topk_ids, *args, **kwargs):
            state["dispatch"] += 1
            call = state["dispatch"]
            if call <= max_calls:
                append(snap("b1_predispatch_hidden", call, hidden_states))
                # The selected expert IDs are the amplification channel: a ~1 ulp
                # perturbation only becomes a 0.1-0.3 probability move by flipping one
                # of these, so the final token's top-4 must be comparable across ranks.
                append(
                    {
                        "tag": "b1_topk_ids",
                        "call": call,
                        "checksum": _checksum(topk_ids),
                        "shape": list(topk_ids.shape),
                        "last_token_expert_ids": topk_ids[-1].detach().cpu().tolist(),
                        "last_token_weights": topk_weights[-1].detach().float().cpu().tolist(),
                    }
                )
            outputs = original_dispatch(hidden_states, topk_weights, topk_ids, *args, **kwargs)
            if call <= max_calls:
                append(snap("b2_gathered_hidden", call, outputs[0]))
            return outputs

        def traced_combine(hidden_states, *args, **kwargs):
            state["combine"] += 1
            call = state["combine"]
            if call <= max_calls:
                append(snap("b4_precombine_partials", call, hidden_states))
            outputs = original_combine(hidden_states, *args, **kwargs)
            if call <= max_calls:
                append(snap("b5_postcombine", call, outputs))
            return outputs

        manager.dispatch, manager.combine = traced_dispatch, traced_combine
        # The router-logits variant is the entry point for the other AgRs prepare/finalize
        # class (prepare_finalize/naive_dp_ep.py:258); wrap it when present so the trace
        # does not silently miss boundary 1/2 depending on which one this model builds.
        original_dispatch_router = getattr(manager, "dispatch_router_logits", None)
        if original_dispatch_router is not None:

            def traced_dispatch_router(hidden_states, router_logits, *args, **kwargs):
                state["dispatch"] += 1
                call = state["dispatch"]
                if call <= max_calls:
                    append(snap("b1_predispatch_hidden", call, hidden_states))
                    append(snap("b1_router_logits", call, router_logits))
                outputs = original_dispatch_router(hidden_states, router_logits, *args, **kwargs)
                if call <= max_calls:
                    append(snap("b2_gathered_hidden", call, outputs[0]))
                return outputs

            manager.dispatch_router_logits = traced_dispatch_router
        self._marin_trace_originals = (manager, original_dispatch, original_combine, original_dispatch_router)
        return {"ok": True, "rank": rank, "group": group_name}

    @_reported
    def marin_probe_disarm_trace(self) -> dict:
        manager, original_dispatch, original_combine, original_dispatch_router = self._marin_trace_originals
        manager.dispatch, manager.combine = original_dispatch, original_combine
        if original_dispatch_router is not None:
            manager.dispatch_router_logits = original_dispatch_router
        return {"ok": True}
