# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from functools import lru_cache
import hashlib
import logging
import time
from typing import Literal, Optional, TypeAlias, cast, overload
import warnings

import jax
from jax import core as jax_core
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Float, Int

from levanter.kernels.pallas import autotune_cache_utils

from .config import BlockSizes
from .tuned_block_sizes import (
    infer_block_sizes,
    infer_block_sizes_with_tuned_match,
    shape_bucket_name,
    widest_dtype_name,
)
from .reference import linear_softmax_cross_entropy_loss_reference
from .xla import linear_softmax_cross_entropy_loss_xla


Implementation: TypeAlias = Literal[
    "pallas_tpu",
    "pallas_gpu",
    "xla",
    "reference",
]
Reduction: TypeAlias = Literal["sum", "mean"] | None


KernelOutput: TypeAlias = tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]
ArrayImpl = Callable[..., KernelOutput]


IMPLEMENTATIONS: dict[str, ArrayImpl] = {
    "reference": linear_softmax_cross_entropy_loss_reference,
    "xla": linear_softmax_cross_entropy_loss_xla,
}
_DEFAULT_IMPLEMENTATION: tuple[Implementation, ...] = ("xla",)
_PALLAS_FALLBACK_WARNINGS_EMITTED: set[str] = set()
_SELECTED_IMPL_LOGGED: set[str] = set()
_AUTOTUNE_ON_MISS_ENV_VAR = "LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS"
_AUTOTUNE_KERNEL_NAME = "fused_cross_entropy_loss"
_AUTOTUNE_CACHE_FILENAME = "block_sizes_v1.json"
_AUTOTUNE_BLOCK_SIZE_CACHE: dict[str, BlockSizes] = {}
_AUTOTUNE_CACHE_LOADED = False
_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20
_VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED: set[str] = set()

logger = logging.getLogger(__name__)
_CANONICAL_PALLAS_IMPLEMENTATIONS: dict[str, ArrayImpl] = {}

try:
    from .pallas_tpu import (
        PallasUnsupportedError,
        linear_softmax_cross_entropy_loss_pallas,
    )

    IMPLEMENTATIONS["pallas_tpu"] = linear_softmax_cross_entropy_loss_pallas
    _CANONICAL_PALLAS_IMPLEMENTATIONS["pallas_tpu"] = linear_softmax_cross_entropy_loss_pallas
except ImportError:
    PallasUnsupportedError = NotImplementedError  # type: ignore[assignment]

try:
    from .pallas_gpu import PallasUnsupportedError, linear_softmax_cross_entropy_loss_pallas_gpu

    IMPLEMENTATIONS["pallas_gpu"] = linear_softmax_cross_entropy_loss_pallas_gpu
    _CANONICAL_PALLAS_IMPLEMENTATIONS["pallas_gpu"] = linear_softmax_cross_entropy_loss_pallas_gpu
except ImportError:
    pass


@lru_cache(maxsize=1)
def _default_implementations() -> tuple[Implementation, ...]:
    implementations = _DEFAULT_IMPLEMENTATION
    backend = jax.default_backend()

    if backend == "gpu" and "pallas_gpu" in IMPLEMENTATIONS:
        devices = jax.devices()
        device_kind = devices[0].device_kind.lower() if devices else ""
        if "gb10" in device_kind:
            return cast(tuple[Implementation, ...], implementations + ("pallas_gpu",))
        return cast(tuple[Implementation, ...], ("pallas_gpu",) + implementations)
    if backend == "tpu":
        # Keep TPU default stable and robust unless Pallas is explicitly requested.
        return implementations
    return implementations


def _warn_pallas_fallback_once(exc: Exception) -> None:
    message = str(exc)
    if "requires TPU backend" in message:
        return
    if message in _PALLAS_FALLBACK_WARNINGS_EMITTED:
        return
    _PALLAS_FALLBACK_WARNINGS_EMITTED.add(message)
    warnings.warn(
        f"Pallas fused cross-entropy unavailable, falling back to XLA: {message}",
        RuntimeWarning,
    )


def _is_tpu_vmem_compile_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "resource_exhausted" in message and "vmem" in message


def _sharding_of(value: jax.Array):
    sharding = None
    try:
        sharding = value.sharding  # type: ignore[attr-defined]
    except Exception:
        sharding = None
    if sharding is not None:
        return sharding

    aval = getattr(value, "aval", None)
    if aval is None:
        return None
    return getattr(aval, "sharding", None)


def _named_sharding_of(value: jax.Array) -> NamedSharding | None:
    sharding = _sharding_of(value)
    if isinstance(sharding, NamedSharding):
        return sharding
    return None


def _shape_dtype_struct_with_sharding(value: jax.Array) -> jax.ShapeDtypeStruct:
    sharding = _sharding_of(value)
    if sharding is None:
        return jax.ShapeDtypeStruct(value.shape, value.dtype)
    return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)


def _maybe_wrap_loss_in_shard_map_for_benchmark(
    fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    x_sharding = _named_sharding_of(x)
    labels_sharding = _named_sharding_of(labels)
    w_sharding = _named_sharding_of(w)
    if x_sharding is None or labels_sharding is None or w_sharding is None:
        return fn
    if x_sharding.mesh != labels_sharding.mesh or x_sharding.mesh != w_sharding.mesh:
        return fn

    return jax.shard_map(
        fn,
        mesh=x_sharding.mesh,
        in_specs=(x_sharding.spec, labels_sharding.spec, w_sharding.spec),
        out_specs=labels_sharding.spec,
        check_vma=False,
    )


def _warn_vmem_compile_fallback_once(exc: Exception, *, impl_name: str) -> None:
    message = str(exc)
    key = f"{impl_name}|{message}"
    if key in _VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED:
        return
    _VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED.add(key)
    warnings.warn(
        f"Pallas fused cross-entropy hit TPU vmem compile OOM in {impl_name}; "
        f"trying the next implementation. Error: {message}",
        RuntimeWarning,
    )


def _pallas_impl_matches_current_backend(impl_name: str, *, fn: ArrayImpl | None = None) -> bool:
    canonical_impl = _CANONICAL_PALLAS_IMPLEMENTATIONS.get(impl_name)
    if canonical_impl is None:
        return True
    if fn is None:
        fn = IMPLEMENTATIONS.get(impl_name)
    if fn is not canonical_impl:
        return True

    backend = jax.default_backend()
    return (impl_name == "pallas_tpu" and backend == "tpu") or (impl_name == "pallas_gpu" and backend == "gpu")


def _autotune_enabled() -> bool:
    return autotune_cache_utils.is_enabled_from_env(_AUTOTUNE_ON_MISS_ENV_VAR, default=True)


def _kernel_autotune_cache_url() -> str | None:
    return autotune_cache_utils.kernel_autotune_cache_url(
        kernel_name=_AUTOTUNE_KERNEL_NAME,
        filename=_AUTOTUNE_CACHE_FILENAME,
    )


def _ensure_autotune_cache_loaded() -> None:
    global _AUTOTUNE_CACHE_LOADED
    if _AUTOTUNE_CACHE_LOADED:
        return
    _AUTOTUNE_CACHE_LOADED = True
    cache_url = _kernel_autotune_cache_url()
    if cache_url is None:
        return
    try:
        payload = autotune_cache_utils.load_json(cache_url)
        for key, entry in payload.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            b = entry.get("b_block_size")
            h = entry.get("h_block_size")
            v = entry.get("v_block_size")
            if all(isinstance(val, int) for val in (b, h, v)):
                _AUTOTUNE_BLOCK_SIZE_CACHE[key] = BlockSizes(b_block_size=b, h_block_size=h, v_block_size=v)
        logger.debug("Loaded %d fused CE autotune entries from %s.", len(_AUTOTUNE_BLOCK_SIZE_CACHE), cache_url)
    except Exception as exc:
        logger.debug("Unable to load fused CE autotune cache from %s: %s", cache_url, exc)
        return


def _persist_autotune_cache() -> None:
    cache_url = _kernel_autotune_cache_url()
    if cache_url is None:
        return
    try:
        payload = {
            key: {
                "b_block_size": value.b_block_size,
                "h_block_size": value.h_block_size,
                "v_block_size": value.v_block_size,
            }
            for key, value in _AUTOTUNE_BLOCK_SIZE_CACHE.items()
        }
        autotune_cache_utils.write_json(cache_url, payload)
    except Exception as exc:
        logger.debug("Unable to persist fused CE autotune cache to %s: %s", cache_url, exc)
        return


def _autotune_jaxpr_hash(
    *,
    fn: ArrayImpl,
    inferred: BlockSizes,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    return_argmax: bool,
) -> str | None:
    try:
        kwargs = dict(
            block_sizes=inferred,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
        if return_argmax:
            kwargs["return_argmax"] = True

        def _loss_only(x_value: jax.Array, labels_value: jax.Array, w_value: jax.Array) -> jax.Array:
            out = fn(x_value, labels_value, w_value, **kwargs)
            return out[0]

        traced = jax.make_jaxpr(_loss_only)(x, labels, w)
        return hashlib.sha256(str(traced.jaxpr).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _autotune_cache_key(
    *,
    impl_name: str,
    fn: ArrayImpl,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    inferred: BlockSizes,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    return_argmax: bool,
) -> str:
    devices = jax.devices()
    device_kind = devices[0].device_kind.lower() if devices else ""
    compute_dtype = jnp.dtype(dtype).name if dtype is not None else "none"
    jaxpr_hash = _autotune_jaxpr_hash(
        fn=fn,
        inferred=inferred,
        x=x,
        labels=labels,
        w=w,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
        return_argmax=return_argmax,
    )
    return "|".join(
        (
            impl_name,
            jax.default_backend(),
            device_kind,
            str(x.shape[0]),
            str(x.shape[1]),
            str(w.shape[1]),
            jnp.dtype(x.dtype).name,
            jnp.dtype(w.dtype).name,
            compute_dtype,
            str(logit_soft_cap),
            str(precision),
            str(return_argmax),
            f"jaxpr={jaxpr_hash}" if jaxpr_hash is not None else "jaxpr=unavailable",
        )
    )


def _candidate_block_sizes(
    impl_name: str,
    inferred: BlockSizes,
    *,
    x: jax.Array,
    w: jax.Array,
    dtype: Optional[jnp.dtype],
) -> list[BlockSizes]:
    candidates: list[BlockSizes] = [inferred]
    if impl_name == "pallas_tpu":
        bucket = shape_bucket_name(x.shape[0], x.shape[1], w.shape[1])
        if bucket == "large-batch-medium-h":
            for h_block in (256, 512, 1024, 2048):
                if x.shape[1] % h_block != 0:
                    continue
                for v_block in (128, 256, 512, 768, 1024):
                    candidates.append(
                        BlockSizes(
                            b_block_size=1024,
                            h_block_size=h_block,
                            v_block_size=v_block,
                        )
                    )
        else:
            widest_dtype = widest_dtype_name(dtype=dtype, x_dtype=x.dtype, w_dtype=w.dtype)
            if widest_dtype == jnp.dtype(jnp.float32).name:
                v_blocks = (256, 512, 768, 1024)
            else:
                v_blocks = (256, 512, 1024, 2048, 4096)
            for v_block in v_blocks:
                candidates.append(
                    BlockSizes(
                        b_block_size=inferred.b_block_size,
                        h_block_size=inferred.h_block_size,
                        v_block_size=v_block,
                    )
                )
    elif impl_name == "pallas_gpu":
        for v_block in (64, 128, 256, 512, 1024, 2048, 4096):
            candidates.append(
                BlockSizes(
                    b_block_size=inferred.b_block_size,
                    h_block_size=inferred.h_block_size,
                    v_block_size=v_block,
                )
            )
    deduped: list[BlockSizes] = []
    seen: set[tuple[int, int, int]] = set()
    for entry in candidates:
        key = (entry.b_block_size, entry.h_block_size, entry.v_block_size)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _is_tracer(x: jax.Array) -> bool:
    return isinstance(x, jax_core.Tracer)


def _benchmark_block_sizes_candidate(
    *,
    fn: ArrayImpl,
    candidate: BlockSizes,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    return_argmax: bool,
) -> float:
    def _loss_only(x_value: jax.Array, labels_value: jax.Array, w_value: jax.Array) -> jax.Array:
        kwargs = dict(
            block_sizes=candidate,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
        if return_argmax:
            kwargs["return_argmax"] = True
        out = fn(x_value, labels_value, w_value, **kwargs)
        return out[0]

    benchmark_fn = _maybe_wrap_loss_in_shard_map_for_benchmark(
        _loss_only,
        x=x,
        labels=labels,
        w=w,
    )
    jitted = jax.jit(benchmark_fn)

    abstract_args = (
        _shape_dtype_struct_with_sharding(x),
        _shape_dtype_struct_with_sharding(labels),
        _shape_dtype_struct_with_sharding(w),
    )
    start = time.perf_counter()
    lowered = jitted.lower(*abstract_args)
    lowered.compile()
    compile_time = time.perf_counter() - start
    if compile_time <= _AUTOTUNE_COMPILE_HIT_THRESHOLD_S:
        logger.info(
            "Fused CE autotune candidate %s likely hit JAX compilation cache (compile %.3fs).",
            candidate,
            compile_time,
        )

    if _is_tracer(x) or _is_tracer(labels) or _is_tracer(w):
        return compile_time

    start = time.perf_counter()
    out = jitted(x, labels, w)
    jax.block_until_ready(out)
    run_time = time.perf_counter() - start
    return run_time


def _autotune_block_sizes_on_miss(
    *,
    impl_name: str,
    fn: ArrayImpl,
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    inferred: BlockSizes,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    return_argmax: bool,
) -> BlockSizes:
    if not _autotune_enabled():
        return inferred
    _ensure_autotune_cache_loaded()
    cache_key = _autotune_cache_key(
        impl_name=impl_name,
        fn=fn,
        x=x,
        labels=labels,
        w=w,
        inferred=inferred,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
        return_argmax=return_argmax,
    )
    cached = _AUTOTUNE_BLOCK_SIZE_CACHE.get(cache_key)
    if cached is not None:
        logger.info("Fused CE autotune cache hit for %s. Using cached block sizes %s.", impl_name, cached)
        return cached

    candidates = _candidate_block_sizes(impl_name, inferred, x=x, w=w, dtype=dtype)
    logger.info(
        "Fused CE autotune miss for %s. Sweeping %d block-size candidates.",
        impl_name,
        len(candidates),
    )
    best: BlockSizes | None = None
    best_score = float("inf")
    errors: list[Exception] = []
    for candidate in candidates:
        try:
            score = _benchmark_block_sizes_candidate(
                fn=fn,
                candidate=candidate,
                x=x,
                labels=labels,
                w=w,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
                return_argmax=return_argmax,
            )
        except Exception as exc:
            errors.append(exc)
            continue
        if score < best_score:
            best_score = score
            best = candidate

    if best is None:
        raise ExceptionGroup(
            f"Fused CE autotune found no viable block-size candidates for {impl_name}",
            errors or [RuntimeError(f"No candidates generated for {impl_name}.")],
        )

    _AUTOTUNE_BLOCK_SIZE_CACHE[cache_key] = best
    _persist_autotune_cache()
    logger.info("Fused CE autotune selected block sizes %s for %s.", best, impl_name)
    return best


def _validate_inputs(x: jax.Array, labels: jax.Array, w: jax.Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [B, H], got shape {x.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be rank-1 [B], got shape {labels.shape}.")
    if w.ndim != 2:
        raise ValueError(f"w must be rank-2 [H, V], got shape {w.shape}.")
    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"Batch mismatch: x has B={x.shape[0]}, labels has B={labels.shape[0]}.")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"Hidden mismatch: x has H={x.shape[1]}, w has H={w.shape[0]}.")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"labels must be integer dtype, got {labels.dtype}.")


def _resolve_block_sizes(
    block_size: Optional[int],
    block_sizes: Optional[BlockSizes],
    *,
    x: jax.Array,
    w: jax.Array,
    dtype: Optional[jnp.dtype],
) -> BlockSizes:
    if block_sizes is None:
        if block_size is None:
            return infer_block_sizes(x.shape[0], x.shape[1], w.shape[1], dtype=dtype, x_dtype=x.dtype, w_dtype=w.dtype)
        return BlockSizes(v_block_size=block_size)
    if block_size is not None and block_size != block_sizes.v_block_size:
        raise ValueError(
            "block_size and block_sizes.v_block_size disagree: "
            f"block_size={block_size}, block_sizes.v_block_size={block_sizes.v_block_size}."
        )
    return block_sizes


def _apply_reduction(loss: jax.Array, reduction: Reduction, weight: Optional[jax.Array]) -> jax.Array:
    if weight is not None:
        weight = weight.astype(loss.dtype)
        loss = loss * weight

    if reduction is None:
        return loss
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        if weight is None:
            return jnp.mean(loss)
        denom = jnp.sum(weight)
        return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
    raise ValueError(f"Unsupported reduction: {reduction}")


@overload
def fused_cross_entropy_loss_and_logsumexp_penalty(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    reduction: Reduction = "mean",
    weight: Optional[Float[Array, "B"]] = None,
    logsumexp_weight: Optional[float] = 0.0,
    block_size: Optional[int] = None,
    block_sizes: Optional[BlockSizes] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
    return_argmax: Literal[False] = False,
) -> jax.Array: ...


@overload
def fused_cross_entropy_loss_and_logsumexp_penalty(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    reduction: Reduction = "mean",
    weight: Optional[Float[Array, "B"]] = None,
    logsumexp_weight: Optional[float] = 0.0,
    block_size: Optional[int] = None,
    block_sizes: Optional[BlockSizes] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
    return_argmax: Literal[True] = True,
) -> tuple[jax.Array, jax.Array]: ...


def fused_cross_entropy_loss_and_logsumexp_penalty(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    reduction: Reduction = "mean",
    weight: Optional[Float[Array, "B"]] = None,
    logsumexp_weight: Optional[float] = 0.0,
    block_size: Optional[int] = None,
    block_sizes: Optional[BlockSizes] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | Sequence[Implementation | ArrayImpl] | None = None,
    return_argmax: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Fused cross-entropy + logsumexp penalty on raw arrays.

    Args:
        x: [B, H] input activations.
        labels: [B] integer labels.
        w: [H, V] projection weights.
        reduction: "sum", "mean", or None to return per-example loss.
        weight: Optional per-example weights/mask, broadcastable to [B].
        logsumexp_weight: Weight for the logsumexp (z-loss) penalty.
        block_size: Optional convenience for setting block_sizes.v_block_size.
        block_sizes: Block size configuration for the kernel.
        dtype: Optional dtype for logits/softmax computations.
        logit_soft_cap: Optional tanh soft cap for logits.
        precision: Optional matmul precision override for XLA/reference paths.
        implementation: Backend selector or override implementation list.
        return_argmax: Whether to additionally return per-example argmax ids.

    Returns:
        If return_argmax=False: reduced loss (scalar) or per-example loss [B] if reduction is None.
        If return_argmax=True: tuple of (loss, argmax_ids[B]).
    """
    _validate_inputs(x, labels, w)
    explicit_block_sizes = block_size is not None or block_sizes is not None
    resolved_block_sizes = (
        _resolve_block_sizes(block_size, block_sizes, x=x, w=w, dtype=dtype) if explicit_block_sizes else None
    )

    if implementation is None:
        impls = cast(Sequence[Implementation | ArrayImpl], _default_implementations())
        explicit = False
        user_requested_impls = False
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation | ArrayImpl], implementation)
        explicit = len(impls) == 1
        user_requested_impls = True
    else:
        impls = (cast(Implementation, implementation),)
        explicit = True
        user_requested_impls = True

    errors: list[Exception] = []
    for impl in impls:
        impl_for_call = impl
        if explicit_block_sizes:
            block_sizes_for_impl = resolved_block_sizes
        elif impl_for_call in ("xla", "reference"):
            block_sizes_for_impl = None
        elif isinstance(impl_for_call, str) and impl_for_call in ("pallas_tpu", "pallas_gpu"):
            inferred, has_tuned_match = infer_block_sizes_with_tuned_match(
                x.shape[0],
                x.shape[1],
                w.shape[1],
                dtype=dtype,
                x_dtype=x.dtype,
                w_dtype=w.dtype,
            )
            fn = IMPLEMENTATIONS.get(impl_for_call)
            if fn is None or not _pallas_impl_matches_current_backend(impl_for_call, fn=fn):
                block_sizes_for_impl = inferred
            elif has_tuned_match:
                block_sizes_for_impl = inferred
            else:
                try:
                    block_sizes_for_impl = _autotune_block_sizes_on_miss(
                        impl_name=impl_for_call,
                        fn=fn,
                        x=x,
                        labels=labels,
                        w=w,
                        inferred=inferred,
                        dtype=dtype,
                        logit_soft_cap=logit_soft_cap,
                        precision=precision,
                        return_argmax=return_argmax,
                    )
                except Exception as exc:
                    if explicit:
                        raise
                    _warn_pallas_fallback_once(exc)
                    errors.append(exc)
                    continue
        else:
            block_sizes_for_impl = infer_block_sizes(
                x.shape[0],
                x.shape[1],
                w.shape[1],
                dtype=dtype,
                x_dtype=x.dtype,
                w_dtype=w.dtype,
            )
        if callable(impl_for_call):
            try:
                kwargs = dict(
                    block_sizes=block_sizes_for_impl,
                    dtype=dtype,
                    logit_soft_cap=logit_soft_cap,
                    precision=precision,
                )
                if return_argmax:
                    kwargs["return_argmax"] = True
                result = impl_for_call(x, labels, w, **kwargs)
            except PallasUnsupportedError as e:
                if explicit:
                    raise
                _warn_pallas_fallback_once(e)
                errors.append(e)
                continue
            except NotImplementedError as e:
                if explicit:
                    raise
                _warn_pallas_fallback_once(e)
                errors.append(e)
                continue
        else:
            fn = IMPLEMENTATIONS.get(impl_for_call)
            if fn is None:
                raise ValueError(f"Unsupported implementation: {impl_for_call}")
            try:
                kwargs = dict(
                    block_sizes=block_sizes_for_impl,
                    dtype=dtype,
                    logit_soft_cap=logit_soft_cap,
                    precision=precision,
                )
                if return_argmax:
                    kwargs["return_argmax"] = True
                result = fn(x, labels, w, **kwargs)
            except PallasUnsupportedError as e:
                if explicit:
                    raise
                _warn_pallas_fallback_once(e)
                errors.append(e)
                continue
            except NotImplementedError as e:
                if explicit:
                    raise
                _warn_pallas_fallback_once(e)
                errors.append(e)
                continue
            except Exception as e:
                should_try_next_impl = (
                    not explicit
                    and isinstance(impl_for_call, str)
                    and impl_for_call in ("pallas_tpu", "pallas_gpu")
                    and _is_tpu_vmem_compile_error(e)
                )
                if should_try_next_impl:
                    _warn_vmem_compile_fallback_once(e, impl_name=impl_for_call)
                    errors.append(e)
                    continue
                if explicit or user_requested_impls:
                    raise
                errors.append(e)
                continue

        selected = str(impl_for_call)
        if selected not in _SELECTED_IMPL_LOGGED:
            _SELECTED_IMPL_LOGGED.add(selected)
            logger.info("Fused cross-entropy selected implementation: %s", selected)

        if len(result) == 2:
            loss, lse = result
            argmax = None
        elif len(result) == 3:
            loss, lse, argmax = result
        else:
            raise ValueError(f"Implementation returned unexpected output tuple length: {len(result)}")

        if return_argmax and argmax is None:
            raise ValueError("Implementation does not support return_argmax=True")

        if logsumexp_weight is not None and logsumexp_weight != 0.0:
            loss = loss + logsumexp_weight * (lse**2)
        reduced_loss = _apply_reduction(loss, reduction, weight)
        if return_argmax:
            return reduced_loss, argmax
        return reduced_loss

    raise ExceptionGroup("all implementations failed", errors)


__all__ = [
    "BlockSizes",
    "Implementation",
    "IMPLEMENTATIONS",
    "Reduction",
    "fused_cross_entropy_loss_and_logsumexp_penalty",
]
