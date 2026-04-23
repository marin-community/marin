# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""LoRA-debug instrumentation.

Instrumentation for the Bug-1 DPO LoRA topology investigation (see
`.agents/logbooks/bug_1_dpo_lora_physical_topology.md`, section "L0: Common
instrumentation upgrade"). When enabled, this callback publishes to W&B
per-LoRA-module gradient / parameter / update / delta_W statistics, Adam first-
and second-moment norms, and fixed-index sentinel gradient values. It also
logs a one-time topology summary (mesh, devices, XLA flags, every MARIN_DEBUG_*
env var) so cross-run diffs (e.g. canonical vs reverse permutations) can be
reconstructed from the W&B run config alone.

Flag surface:
    TrainerConfig.lora_debug.enabled=True      # primary, structured config
    WandbConfig.lora_debug=True                 # convenience: flips the above
    MARIN_DEBUG_LORA_DEBUG=1                    # env override

All three paths converge to installing :class:`LoraDebugCallback` at the
interval in ``LoraDebugConfig.interval``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

import levanter.tracker
from levanter.callbacks._core import JitCallback
from levanter.lora import LowRankLinear, is_lora_param
from levanter.trainer_state import InsideJitInfo, TrainerState


M = TypeVar("M", bound=PyTree)
S = TypeVar("S", bound=TrainerState)


# --- Default sentinel module names -------------------------------------------

DEFAULT_SENTINEL_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTENTION_MODULES: frozenset[str] = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_MODULES: frozenset[str] = frozenset({"gate_proj", "up_proj", "down_proj"})


# --- Config ------------------------------------------------------------------

_ENV_FLAG = "MARIN_DEBUG_LORA_DEBUG"


def _env_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "0").strip() not in ("", "0", "false", "False")


@dataclass(frozen=True)
class LoraDebugConfig:
    """LoRA debug instrumentation.

    Logs per-module grad/param/update norms, delta_W Frobenius norms,
    Adam moments, and fixed-index sentinel grad values to W&B.
    """

    enabled: bool = False
    """Install the callback. Also set via ``MARIN_DEBUG_LORA_DEBUG=1`` or
    ``WandbConfig.lora_debug=True`` without editing the config."""

    interval: int = 1
    """Log every N steps. 1 = every step, which is what the Bug-1 probes want."""

    include_grads: bool = True
    include_params: bool = True
    include_updates: bool = True
    include_delta_w: bool = True
    """Log ``||B @ A|| * scale`` (Frobenius) per LoRA module. Cheap via the
    ``trace(B^T B · A A^T)`` identity — no materialization of delta_W."""

    include_opt_state: bool = True
    """Walk ``state.opt_state`` and log L2 norm of every array leaf whose path
    contains ``lora_A`` / ``lora_B`` (captures Adam m/v for LoRA params)."""

    include_effective_adam_update: bool = True
    """Log ``|m / (sqrt(v) + eps)|_2`` per LoRA param (Adam's un-LR-scaled update
    direction, useful for H5 optimizer-amplification hypothesis)."""

    include_sentinels: bool = True
    """Log fixed-index scalar values from sentinel LoRA_B gradients. Pairs with
    the existing stderr ``DEBUGJ GRAD_VAL`` dump; this routes the same signals
    to W&B so canonical vs reverse can be diffed from run history."""

    sentinel_modules: Sequence[str] = field(default_factory=lambda: DEFAULT_SENTINEL_MODULES)
    sentinel_fractions: Sequence[float] = field(default_factory=lambda: (0.0, 0.25, 0.5, 0.75, 1.0))
    """Fractional positions in the flattened tensor (0 = first element,
    1 = last). Matches the BL/Exp-Z1 pattern."""

    include_sentinel_grad_a: bool = False
    """Also dump fixed-index values for ``lora_A`` grads. Off by default since
    the main cross-run diff target is ``lora_B`` (zero-init + tiny first
    update)."""

    include_sentinel_adam_m_a: bool = False
    include_sentinel_adam_m_b: bool = True
    """Also dump fixed-index values for Adam first moments. B-side is on by
    default for Bug-1 because the norm-only D1 rerun showed nearly identical
    ``|m|`` across canonical/reverse, so we now care about sign/direction."""

    include_sentinel_effective_update_a: bool = False
    include_sentinel_effective_update_b: bool = True
    """Also dump fixed-index values for ``m / (sqrt(v) + eps)``. This is the
    signed unscaled Adam update direction, which is the highest-value next
    probe after the D1 moment-norm reruns."""

    include_attn_mlp_rollups: bool = True
    """Emit ``lora_debug/agg/{attn,mlp}/*`` aggregates for H1 module-family
    ablations."""

    include_max_abs: bool = True
    """Log ``max_abs`` alongside ``l2`` for grads/params. Useful for catching
    single-element blowups."""

    adam_eps: float = 1e-8
    """Epsilon for the effective-Adam-update metric (``|m / (sqrt(v) + eps)|``)."""

    # --- Full-tensor GCS dump (P2) ------------------------------------------
    #
    # Scalars (norms / sentinels) collapse each tensor to a few numbers and
    # cannot distinguish a direction defect from a distribution match. The
    # Bug-1 investigation (D1c / D1d) now has evidence pointing at a
    # directional, not scalar, discrepancy between canonical and reverse, so
    # we need the full signed tensor side by side across variants. The knobs
    # below dump raw ``grad_B`` / Adam ``mu`` / ``nu`` / effective-update
    # tensors to a filesystem path (local or gs://) at a small handful of
    # configured steps, for offline cosine / sign-agreement analysis.

    verbose_grads: bool = False
    """Top-level guard. When ``False`` (default) the full-tensor dump
    machinery below is entirely skipped, regardless of the other knobs —
    the callback behaves exactly as if P2 did not exist. Turn on only for
    targeted debug runs. Typical production / long-horizon training should
    never have this set."""

    dump_tensors_at_steps: tuple[int, ...] = ()
    """Steps at which to write full tensors. Empty tuple = never. Usage
    example for the D1 post-warmup window: ``(11, 12, 13)``. Each step
    produces ``len(dump_tensor_modules) * 4`` ``.npy`` files. Ignored when
    ``verbose_grads=False``."""

    dump_tensor_modules: tuple[str, ...] = ()
    """Module name suffixes to dump (matched against the LoRA module
    path, same matching rule as ``sentinel_modules``). Empty = nothing
    is dumped even if ``dump_tensors_at_steps`` is set. Bug-1 default
    of interest: ``("o_proj", "down_proj")``."""

    dump_tensor_path: Optional[str] = None
    """Output directory prefix for the full-tensor dump. Supports fsspec
    schemes (``gs://...``, local absolute path, etc.). Per-tensor layout is
    ``{dump_tensor_path}/step_{N:04d}/{module_key}/{metric}.npy`` where
    ``metric`` is one of ``grad_B``, ``mu_B``, ``nu_B``,
    ``effective_update_B``. Must be set when ``verbose_grads=True``."""

    def resolve_enabled(self) -> bool:
        """True if config, WandbConfig convenience flag, or env var say on."""
        return self.enabled or _env_enabled()

    def build(self) -> "LoraDebugCallback":
        return LoraDebugCallback(self)


# --- Helpers -----------------------------------------------------------------


# Internal key prefix used to pass full tensors from inside_step to on_step
# without mixing them into the scalar tracker payload. Never emitted to W&B.
_TENSOR_DUMP_PREFIX = "__lora_tensor_dump__/"


def _fmt_path(path: Sequence[Any]) -> str:
    """Format a jax tree key path into a stable slash-joined string.

    Strips:
      - leading dot on GetAttrKey (``.q_proj`` -> ``q_proj``)
      - brackets / quotes on DictKey (``['q_proj']`` -> ``q_proj``)
      - FlattenedIndexKey fragments (``<flat index 0>``) — these are leaf-index
        artifacts from hax NamedArray flattening that add no information and
        produce noisy W&B keys.
    """
    parts = []
    for p in path:
        s = str(p)
        if s.startswith("."):
            s = s[1:]
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip("'\"")
        # Drop FlattenedIndexKey artifacts (``<flat index 0>`` etc.) — after
        # bracket-strip they leak through as bare tokens with no information.
        if s.startswith("<flat index") and s.endswith(">"):
            continue
        parts.append(s)
    return "/".join(p for p in parts if p)


def _module_family(path: str) -> Optional[str]:
    """Return 'attn' | 'mlp' | None for a path string."""
    for suffix in ATTENTION_MODULES:
        if suffix in path:
            return "attn"
    for suffix in MLP_MODULES:
        if suffix in path:
            return "mlp"
    return None


def _suffix_for_sentinel(path: str, sentinel: str) -> bool:
    return sentinel in path


def _as_f32(x: Any) -> jax.Array:
    arr = x.array if isinstance(x, hax.NamedArray) else x
    return arr.astype(jnp.float32)


def _l2(x: Any) -> jax.Array:
    arr = _as_f32(x)
    return jnp.sqrt(jnp.sum(jnp.square(arr)))


def _max_abs(x: Any) -> jax.Array:
    arr = _as_f32(x)
    return jnp.max(jnp.abs(arr))


def _delta_w_frobenius(mod: LowRankLinear) -> jax.Array:
    """Frobenius norm of ``delta_W = scale * B @ A`` over all axes.

    Uses the existing ``LowRankLinear.merge()`` helper, which does
    ``hax.dot(A, B, axis=LORA_R) * scale`` in named-axis space. This handles
    arbitrary Out / In axis topology automatically (e.g. Llama's multi-axis
    ``q_proj.lora_B`` with axes ``(LORA_R, Heads, HeadSize)``) and scan
    stacking. For an 8B Llama that's roughly ``embed × head_stuff × r ×
    layers`` ops per module — cheap vs a training step.
    """
    dw = mod.merge()
    dw_arr = _as_f32(dw)
    return jnp.sqrt(jnp.sum(jnp.square(dw_arr)))


def _collect_lora_modules(tree: PyTree) -> list[tuple[str, LowRankLinear]]:
    """Walk a model tree and return (path_string, LowRankLinear) pairs.

    The path is trimmed to the owning linear's attribute, e.g. ``q_proj``
    rather than ``q_proj/lora`` (``lora`` is the attribute on the LoraLinear
    wrapper and is constant for every module)."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_lora_param)
    out: list[tuple[str, LowRankLinear]] = []
    for path, leaf in flat:
        if isinstance(leaf, LowRankLinear):
            path_str = _fmt_path(path)
            if path_str.endswith("/lora"):
                path_str = path_str[: -len("/lora")]
            out.append((path_str, leaf))
    return out


def _collect_lora_arrays(tree: PyTree, factor: str) -> list[tuple[str, jax.Array]]:
    """Walk a gradient / update / param tree and return ``lora_{factor}`` arrays.

    We match on the flattened path string so this works for grads (where the
    leaves are plain arrays, not LowRankLinear objects).
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    out = []
    for path, leaf in flat:
        path_str = _fmt_path(path)
        if f"lora_{factor}" not in path_str:
            continue
        arr = leaf.array if isinstance(leaf, hax.NamedArray) else leaf
        if hasattr(arr, "shape"):
            out.append((path_str, arr))
    return out


def _module_key_from_arr_path(arr_path: str) -> str:
    """Turn ``...q_proj/lora/lora_A/weight`` into ``...q_proj``.

    Drops any ``/lora``, ``/lora_A``, ``/lora_B``, or ``/weight`` tail segments
    so grads, params, updates, and delta_W all land on the same per-module
    W&B key."""
    _drop = {"lora", "lora_A", "lora_B", "weight"}
    segments = arr_path.split("/")
    while segments and segments[-1] in _drop:
        segments.pop()
    return "/".join(segments)


def _strip_adam_slot_segments(path: str) -> str:
    """Remove Adam-slot path components (``mu`` / ``nu``) anywhere in the path.

    Why: optax flattens ``ScaleByAdamState.mu`` / ``nu`` as sibling subtrees, so
    the same parameter appears twice in the opt_state flatten — once under
    ``.mu.<tree>``, once under ``.nu.<tree>``. Without stripping, the per-module
    key for mu and nu differ by exactly that one segment, and the two halves
    never pair up for the ``m / (sqrt(v) + eps)`` computation.
    """
    return "/".join(s for s in path.split("/") if s not in ("mu", "nu"))


# --- Callback ----------------------------------------------------------------


class LoraDebugCallback(JitCallback[S, M, dict[str, jax.Array]]):
    """JitCallback that publishes LoRA-specific training diagnostics every step."""

    config: LoraDebugConfig

    def __init__(self, config: LoraDebugConfig):
        self.config = config

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]) -> dict[str, jax.Array]:
        cfg = self.config
        model = state.trainable_model
        grads = inside_info.grads
        updates = inside_info.updates

        out: dict[str, jax.Array] = {}

        # --- per-factor walks ------------------------------------------------
        g_A = _collect_lora_arrays(grads, "A") if cfg.include_grads else []
        g_B = _collect_lora_arrays(grads, "B") if cfg.include_grads else []
        p_A = _collect_lora_arrays(model, "A") if cfg.include_params else []
        p_B = _collect_lora_arrays(model, "B") if cfg.include_params else []
        u_A = _collect_lora_arrays(updates, "A") if cfg.include_updates else []
        u_B = _collect_lora_arrays(updates, "B") if cfg.include_updates else []

        _emit_factor_stats(out, "grad", "A", g_A, cfg)
        _emit_factor_stats(out, "grad", "B", g_B, cfg)
        _emit_factor_stats(out, "param", "A", p_A, cfg)
        _emit_factor_stats(out, "param", "B", p_B, cfg)
        _emit_factor_stats(out, "update", "A", u_A, cfg)
        _emit_factor_stats(out, "update", "B", u_B, cfg)

        # --- attn/mlp roll-ups ---------------------------------------------
        if cfg.include_attn_mlp_rollups:
            for family in ("attn", "mlp"):
                _emit_family_rollup(out, "grad", "A", g_A, family)
                _emit_family_rollup(out, "grad", "B", g_B, family)
                _emit_family_rollup(out, "param", "A", p_A, family)
                _emit_family_rollup(out, "param", "B", p_B, family)
                _emit_family_rollup(out, "update", "A", u_A, family)
                _emit_family_rollup(out, "update", "B", u_B, family)

        # --- delta_W per module --------------------------------------------
        if cfg.include_delta_w:
            modules = _collect_lora_modules(model)
            dw_norms: list[jax.Array] = []
            dw_attn: list[jax.Array] = []
            dw_mlp: list[jax.Array] = []
            for mod_path, mod in modules:
                dw = _delta_w_frobenius(mod)
                out[f"lora_debug/delta_W/{mod_path}/l2"] = dw
                dw_norms.append(dw * dw)
                fam = _module_family(mod_path)
                if fam == "attn":
                    dw_attn.append(dw * dw)
                elif fam == "mlp":
                    dw_mlp.append(dw * dw)
            if dw_norms:
                out["lora_debug/agg/delta_W/l2"] = jnp.sqrt(sum(dw_norms))
            if cfg.include_attn_mlp_rollups:
                if dw_attn:
                    out["lora_debug/agg/attn/delta_W/l2"] = jnp.sqrt(sum(dw_attn))
                if dw_mlp:
                    out["lora_debug/agg/mlp/delta_W/l2"] = jnp.sqrt(sum(dw_mlp))

        # --- fixed-index sentinel grads ------------------------------------
        if cfg.include_sentinels:
            _emit_sentinels(out, "grad_B", g_B, cfg)
            if cfg.include_sentinel_grad_a:
                _emit_sentinels(out, "grad_A", g_A, cfg)

        # --- Adam / optimizer state norms ----------------------------------
        if cfg.include_opt_state:
            _emit_opt_state_stats(out, state.opt_state, cfg)

        # --- Full-tensor dump (P2) — gated end-to-end on verbose_grads ------
        if (
            cfg.verbose_grads
            and cfg.dump_tensors_at_steps
            and cfg.dump_tensor_modules
            and cfg.dump_tensor_path
        ):
            _emit_full_tensor_dump(out, state, grads, cfg)

        return out

    def on_step(self, step_info, cb_info: dict[str, jax.Array]):
        if not cb_info:
            return

        # Split off any full-tensor dump payload so it never hits the scalar
        # tracker (otherwise we'd be streaming hundreds of MB to W&B per step).
        scalar_items: dict[str, jax.Array] = {}
        tensor_items: dict[str, jax.Array] = {}
        for k, v in cb_info.items():
            if k.startswith(_TENSOR_DUMP_PREFIX):
                tensor_items[k[len(_TENSOR_DUMP_PREFIX):]] = v
            else:
                scalar_items[k] = v

        if scalar_items:
            levanter.tracker.log(scalar_items, step=int(step_info.step))

        cfg = self.config
        if not (cfg.verbose_grads and tensor_items):
            return

        step = int(step_info.step)
        if step not in cfg.dump_tensors_at_steps:
            # Not a configured dump step — drop the zero-filled placeholders.
            return

        _write_tensor_dump(tensor_items, step, cfg.dump_tensor_path)


# --- Metric emission helpers -------------------------------------------------


def _emit_factor_stats(
    out: dict[str, jax.Array],
    target: str,  # "grad" | "param" | "update"
    factor: str,  # "A" | "B"
    arrs: list[tuple[str, jax.Array]],
    cfg: LoraDebugConfig,
) -> None:
    if not arrs:
        return
    # per-module
    l2_sq_sum: list[jax.Array] = []
    for path, arr in arrs:
        mod_key = _module_key_from_arr_path(path)
        l2 = _l2(arr)
        out[f"lora_debug/{target}/{factor}/{mod_key}/l2"] = l2
        if cfg.include_max_abs:
            out[f"lora_debug/{target}/{factor}/{mod_key}/max_abs"] = _max_abs(arr)
        l2_sq_sum.append(l2 * l2)
    if l2_sq_sum:
        out[f"lora_debug/agg/{target}_{factor}/l2"] = jnp.sqrt(sum(l2_sq_sum))


def _emit_family_rollup(
    out: dict[str, jax.Array],
    target: str,
    factor: str,
    arrs: list[tuple[str, jax.Array]],
    family: str,  # "attn" | "mlp"
) -> None:
    sq_sum: list[jax.Array] = []
    for path, arr in arrs:
        if _module_family(path) != family:
            continue
        l2 = _l2(arr)
        sq_sum.append(l2 * l2)
    if sq_sum:
        out[f"lora_debug/agg/{family}/{target}_{factor}/l2"] = jnp.sqrt(sum(sq_sum))


def _emit_sentinels(
    out: dict[str, jax.Array],
    tag: str,  # "grad_A" | "grad_B"
    arrs: list[tuple[str, jax.Array]],
    cfg: LoraDebugConfig,
) -> None:
    for sentinel in cfg.sentinel_modules:
        for path, arr in arrs:
            if not _suffix_for_sentinel(path, sentinel):
                continue
            mod_key = _module_key_from_arr_path(path)
            flat = arr.reshape(-1).astype(jnp.float32)
            n = flat.shape[0]
            # For scan-stacked tensors, flat has (layers * inner) elements. We
            # still pick indices over the flat tensor — the cross-run diff is
            # meaningful because the flat ordering is deterministic given the
            # same model structure.
            for frac in cfg.sentinel_fractions:
                idx = min(max(int(round(frac * (n - 1))), 0), n - 1) if n > 0 else 0
                if n == 0:
                    continue
                key = f"lora_debug/sentinel/{tag}/{mod_key}/idx_{frac:g}"
                out[key] = flat[idx]
            # Length is a compile-time constant; log it so the index reading is
            # interpretable.
            out[f"lora_debug/sentinel/{tag}/{mod_key}/n"] = jnp.asarray(n, dtype=jnp.int32)
            break  # only the first matching module (avoid firing on every
            # scan-layer-unpacked copy when we already aggregate).


def _emit_module_sentinels(
    out: dict[str, jax.Array],
    tag: str,
    arrs: list[tuple[str, jax.Array]],
    cfg: LoraDebugConfig,
) -> None:
    """Emit fixed-index values from arrays already keyed by module path.

    Unlike :func:`_emit_sentinels`, the incoming path is already the normalized
    module key (e.g. ``.../q_proj``), so there is no need to peel off
    ``lora_A`` / ``lora_B`` suffixes again.
    """
    for sentinel in cfg.sentinel_modules:
        for mod_key, arr in arrs:
            if not _suffix_for_sentinel(mod_key, sentinel):
                continue
            flat = arr.reshape(-1).astype(jnp.float32)
            n = flat.shape[0]
            for frac in cfg.sentinel_fractions:
                idx = min(max(int(round(frac * (n - 1))), 0), n - 1) if n > 0 else 0
                if n == 0:
                    continue
                out[f"lora_debug/sentinel/{tag}/{mod_key}/idx_{frac:g}"] = flat[idx]
            out[f"lora_debug/sentinel/{tag}/{mod_key}/n"] = jnp.asarray(n, dtype=jnp.int32)
            break


def _emit_opt_state_stats(
    out: dict[str, jax.Array],
    opt_state: PyTree,
    cfg: LoraDebugConfig,
) -> None:
    """Emit L2 norms for every array leaf in opt_state whose path mentions
    ``lora_A`` / ``lora_B``.

    For Adam (the canonical DPO optimizer), ``mu`` and ``nu`` are the paths we
    want; they live at something like ``opt_state[0].mu.<model_tree>`` in
    optax. We just walk every array and bucket on path substring."""
    # Find Adam-shaped (mu, nu) pairs so we can compute m / (sqrt(v) + eps).
    mu_by_key: dict[str, jax.Array] = {}
    nu_by_key: dict[str, jax.Array] = {}

    flat, _ = jax.tree_util.tree_flatten_with_path(opt_state)
    for path, leaf in flat:
        arr = leaf.array if isinstance(leaf, hax.NamedArray) else leaf
        if not hasattr(arr, "shape"):
            continue
        path_str = _fmt_path(path)
        if "lora_A" not in path_str and "lora_B" not in path_str:
            continue

        factor = "A" if "lora_A" in path_str else "B"
        arr_f32 = arr.astype(jnp.float32)

        # Bucket Adam moments by inspecting the path for ``mu`` / ``nu``. Do
        # this *before* stripping so the slot marker is still present.
        if "/mu/" in f"/{path_str}/" or path_str.endswith("/mu"):
            slot = "m"
        elif "/nu/" in f"/{path_str}/" or path_str.endswith("/nu"):
            slot = "v"
        elif "/count" in path_str:
            continue  # scalar counter, not a tensor we want
        else:
            slot = "other"

        # Module key must be slot-agnostic so mu and nu of the same parameter
        # collapse to one pairing key; otherwise effective_update never emits.
        mod_key = _module_key_from_arr_path(_strip_adam_slot_segments(path_str))

        key_prefix = f"lora_debug/adam/{slot}/{factor}/{mod_key}"
        out[f"{key_prefix}/l2"] = jnp.sqrt(jnp.sum(jnp.square(arr_f32)))

        if slot == "m":
            mu_by_key[f"{factor}/{mod_key}"] = arr_f32
        elif slot == "v":
            nu_by_key[f"{factor}/{mod_key}"] = arr_f32

    if cfg.include_sentinels:
        adam_m_sentinels: list[tuple[str, jax.Array]] = []
        for key, mu in mu_by_key.items():
            factor, mod_key = key.split("/", 1)
            if factor == "A" and not cfg.include_sentinel_adam_m_a:
                continue
            if factor == "B" and not cfg.include_sentinel_adam_m_b:
                continue
            adam_m_sentinels.append((mod_key, mu))
        if adam_m_sentinels:
            _emit_module_sentinels(out, "adam_m", adam_m_sentinels, cfg)

    if cfg.include_effective_adam_update:
        eps = cfg.adam_eps
        effective_update_sentinels: list[tuple[str, jax.Array]] = []
        for key, mu in mu_by_key.items():
            nu = nu_by_key.get(key)
            if nu is None or nu.shape != mu.shape:
                continue
            eff = mu / (jnp.sqrt(nu) + eps)
            out[f"lora_debug/adam/effective_update/{key}/l2"] = jnp.sqrt(jnp.sum(jnp.square(eff)))
            factor, mod_key = key.split("/", 1)
            if factor == "A" and cfg.include_sentinel_effective_update_a:
                effective_update_sentinels.append((mod_key, eff))
            elif factor == "B" and cfg.include_sentinel_effective_update_b:
                effective_update_sentinels.append((mod_key, eff))
        if cfg.include_sentinels and effective_update_sentinels:
            _emit_module_sentinels(out, "adam_effective_update", effective_update_sentinels, cfg)


# --- Full-tensor GCS dump (P2) ----------------------------------------------


def _first_matching_suffix(path: str, suffixes: Sequence[str]) -> Optional[str]:
    """Return the first suffix name in ``suffixes`` that is a substring of
    ``path``, or None if no match."""
    for s in suffixes:
        if s in path:
            return s
    return None


def _collect_dump_tensors(
    state: TrainerState,
    grads: PyTree,
    cfg: LoraDebugConfig,
) -> dict[str, jax.Array]:
    """Gather the four per-module tensors (grad_B, mu_B, nu_B,
    effective_update_B) for modules whose path suffix matches
    ``cfg.dump_tensor_modules``. Returns a dict keyed by
    ``{metric}/{module_name}`` where ``module_name`` is the matched suffix
    (e.g. ``o_proj``), not the full scan-stacked path — grads and opt_state
    use different outer paths (opt_state has chain-index prefixes) and the
    suffix is unique enough in a Llama block to serve as the join key.

    Assumes the caller has already verified ``cfg.verbose_grads`` is on.
    Does NOT gate on the current step — the caller is responsible for
    zero-masking on non-dump steps via ``lax.cond``.
    """
    module_suffixes = cfg.dump_tensor_modules
    eps = cfg.adam_eps

    # grad_B: walk the grads tree, bucket by matched suffix
    grad_by_name: dict[str, jax.Array] = {}
    for path, arr in _collect_lora_arrays(grads, "B"):
        name = _first_matching_suffix(path, module_suffixes)
        if name is None:
            continue
        grad_by_name.setdefault(name, arr.astype(jnp.float32))

    # Adam mu / nu for LoRA-B params: walk opt_state with the slot-aware logic
    mu_by_name: dict[str, jax.Array] = {}
    nu_by_name: dict[str, jax.Array] = {}
    flat, _ = jax.tree_util.tree_flatten_with_path(state.opt_state)
    for path, leaf in flat:
        arr = leaf.array if isinstance(leaf, hax.NamedArray) else leaf
        if not hasattr(arr, "shape"):
            continue
        path_str = _fmt_path(path)
        if "lora_B" not in path_str:
            continue
        name = _first_matching_suffix(path_str, module_suffixes)
        if name is None:
            continue
        if "/mu/" in f"/{path_str}/" or path_str.endswith("/mu"):
            mu_by_name.setdefault(name, arr.astype(jnp.float32))
        elif "/nu/" in f"/{path_str}/" or path_str.endswith("/nu"):
            nu_by_name.setdefault(name, arr.astype(jnp.float32))
        # else: other opt-state leaves (count, schedule state, etc.) — skip

    out: dict[str, jax.Array] = {}
    # Keep only modules where we have grad, mu, and nu — otherwise the
    # effective_update computation can't pair up and the dump is incomplete.
    for name in sorted(set(grad_by_name) & set(mu_by_name) & set(nu_by_name)):
        g = grad_by_name[name]
        mu = mu_by_name[name]
        nu = nu_by_name[name]
        if mu.shape != nu.shape:
            continue
        eff = mu / (jnp.sqrt(nu) + eps)
        out[f"grad_B/{name}"] = g
        out[f"mu_B/{name}"] = mu
        out[f"nu_B/{name}"] = nu
        out[f"effective_update_B/{name}"] = eff
    return out


def _emit_full_tensor_dump(
    out: dict[str, jax.Array],
    state: TrainerState,
    grads: PyTree,
    cfg: LoraDebugConfig,
) -> None:
    """Stash full tensors under ``__lora_tensor_dump__/`` keys in cb_info.

    Inside JIT we cannot vary the output *shape* by step, but we can zero
    out the values on non-dump steps via ``lax.cond`` so XLA can elide the
    real compute / transfer on those steps. The host-side ``on_step``
    writer checks the Python-level step before actually materializing to
    disk, so stray zero tensors never leak to the filesystem.
    """
    payload = _collect_dump_tensors(state, grads, cfg)
    if not payload:
        return

    # should_dump is a scalar bool: True iff state.step matches any
    # configured dump step.
    targets = jnp.asarray(cfg.dump_tensors_at_steps, dtype=state.step.dtype)
    should_dump = jnp.any(targets == state.step)

    payload = jax.lax.cond(
        should_dump,
        lambda p: p,
        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p),
        payload,
    )
    for sub_key, arr in payload.items():
        out[f"{_TENSOR_DUMP_PREFIX}{sub_key}"] = arr


def _write_tensor_dump(
    tensor_items: dict[str, jax.Array],
    step: int,
    root: str,
) -> None:
    """Host-side writer. Called once, only at configured dump steps.

    Each ``tensor_items`` key is ``{metric}/{module_key}``; the on-disk
    layout is ``{root}/step_{step:04d}/{module_key}/{metric}.npy``.
    Supports fsspec paths (``gs://...``, local filesystem, etc.).
    """
    # Lazy imports so the main code path stays import-cheap.
    import fsspec
    import numpy as np

    for key, arr in tensor_items.items():
        metric, _, module_key = key.partition("/")
        if not module_key:
            continue
        np_arr = np.asarray(arr)
        # Haliax NamedArrays should be unwrapped by inside_step already, but
        # be defensive in case a future caller passes one directly.
        if hasattr(np_arr, "array"):
            np_arr = np.asarray(np_arr.array)
        out_path = f"{root.rstrip('/')}/step_{step:04d}/{module_key}/{metric}.npy"
        with fsspec.open(out_path, "wb") as f:
            np.save(f, np_arr)


# --- One-time topology summary ----------------------------------------------


def _safe_mesh_shape(mesh) -> dict[str, int]:
    shape = getattr(mesh, "shape", None)
    if shape is None:
        return {}
    return {str(k): int(v) for k, v in shape.items()}


def _describe_devices(mesh) -> list[str]:
    devices = getattr(mesh, "devices", None)
    if devices is None:
        return []
    # mesh.devices is a numpy object-dtype ndarray of Device instances. Iterate
    # via its .flat attribute, which works on any numpy dtype including object;
    # jax.numpy.asarray rejects object dtypes.
    if hasattr(devices, "flat"):
        return [str(d) for d in devices.flat]
    return [str(d) for d in devices]


def collect_topology_summary(
    trainer,
    model: PyTree,
) -> dict[str, Any]:
    """Build the one-time hparam dict.

    Includes mesh shape, physical device ordering, param/compute axis mappings,
    the contents of ``XLA_FLAGS``, every ``MARIN_DEBUG_*`` env var, and the
    number / paths of LoRA modules so cross-run diffs can map module paths to
    metric keys.
    """
    mesh = getattr(trainer, "device_mesh", None) or getattr(trainer, "_mesh", None) or getattr(trainer, "mesh", None)

    env_dump = {k: v for k, v in os.environ.items() if k.startswith("MARIN_DEBUG_")}
    xla_flags = os.environ.get("XLA_FLAGS", "")
    experiment_tag = os.environ.get("MARIN_DEBUG_RUN_TAG", "")
    permutation = os.environ.get("EXPERIMENT_BL_ORDER") or os.environ.get("EXPERIMENT_B1_ORDER") or ""

    modules = _collect_lora_modules(model)
    module_paths = [p for p, _ in modules]

    summary: dict[str, Any] = {
        "lora_debug/mesh/shape": _safe_mesh_shape(mesh),
        "lora_debug/mesh/device_count": int(getattr(mesh, "size", 0)) if mesh is not None else 0,
        "lora_debug/mesh/devices": _describe_devices(mesh),
        "lora_debug/mesh/param_mapping": dict(getattr(trainer, "parameter_axis_mapping", {}) or {}),
        "lora_debug/mesh/compute_mapping": dict(getattr(trainer, "compute_axis_mapping", {}) or {}),
        "lora_debug/xla_flags": xla_flags,
        "lora_debug/debug_envs": env_dump,
        "lora_debug/experiment_tag": experiment_tag,
        "lora_debug/permutation": permutation,
        "lora_debug/num_lora_modules": len(module_paths),
        "lora_debug/lora_module_paths": module_paths,
    }

    return summary


def log_topology_summary(trainer, model: PyTree) -> None:
    """Push the topology dict into the current tracker as hyperparameters."""
    payload = collect_topology_summary(trainer, model)
    levanter.tracker.log_hyperparameters(payload)
