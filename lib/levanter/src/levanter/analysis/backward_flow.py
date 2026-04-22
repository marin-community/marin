# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import contextvars
from collections import defaultdict, deque
from dataclasses import dataclass
import functools
import html
from itertools import pairwise
import math
import re
from typing import Any, Iterable, Literal, Mapping, TypeAlias

import jax
from jax._src.core import Literal as JaxprLiteral
from jax._src import source_info_util
import jax.numpy as jnp

import levanter.tracker


_TRANSFORM_WRAPPERS = frozenset(
    {
        "checkpoint",
        "cond",
        "jit",
        "jvp",
        "linear",
        "partial_eval",
        "pjit",
        "remat",
        "scan",
        "shard_map",
        "transpose",
        "vmap",
        "while",
    }
)
_NAME_STACK_PART_RE = re.compile(r"^(?P<wrapper>[A-Za-z_][A-Za-z0-9_]*)\((?P<inner>.*)\)$")
_STAT_NAMES = ("norm", "rms", "rms_scaled", "mean_abs", "max_abs", "max_abs_scaled", "finite_fraction")
BackwardFlowSite: TypeAlias = Literal["in", "out"]
BackwardFlowTensorKind: TypeAlias = Literal["activation", "gradient"]
BACKWARD_FLOW_SITE_IN: BackwardFlowSite = "in"
BACKWARD_FLOW_SITE_OUT: BackwardFlowSite = "out"
BACKWARD_FLOW_KIND_ACTIVATION: BackwardFlowTensorKind = "activation"
BACKWARD_FLOW_KIND_GRADIENT: BackwardFlowTensorKind = "gradient"
_FLOW_SITES = (BACKWARD_FLOW_SITE_IN, BACKWARD_FLOW_SITE_OUT)
_TENSOR_KINDS = (BACKWARD_FLOW_KIND_ACTIVATION, BACKWARD_FLOW_KIND_GRADIENT)
_FLOW_DIRECTIONS = ("tb", "lr")
_DEFAULT_PREFIX = "backward_flow"
_DEFAULT_RESIDUAL_GAIN_HORIZON = 50


@dataclass(frozen=True)
class BackwardFlowConfig:
    """Configuration for sampled backward-flow logging."""

    interval: int = 0

    def __post_init__(self) -> None:
        if self.interval < 0:
            raise ValueError(f"interval must be non-negative, got {self.interval}")

    @property
    def is_enabled(self) -> bool:
        return self.interval > 0


@dataclass(frozen=True)
class BackwardFlowEdge:
    """A directed edge between two named runtime scopes."""

    source: str
    target: str
    count: int = 1


@dataclass(frozen=True)
class BackwardFlowGraph:
    """A compressed dataflow graph over named runtime scopes."""

    nodes: tuple[str, ...]
    edges: tuple[BackwardFlowEdge, ...]


@dataclass(frozen=True)
class BackwardFlowPlate:
    """A visual grouping of related backward-flow nodes."""

    label: str
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class BackwardFlowRenderHints:
    """Optional display hints inferred from backward-flow node names."""

    node_lanes: Mapping[str, int] | None = None
    display_edges: tuple[BackwardFlowEdge, ...] | None = None
    plates: tuple[BackwardFlowPlate, ...] = ()


@dataclass(frozen=True)
class SummaryStats:
    """Scalar summaries for a tensor logged by backward-flow probes."""

    norm: jax.Array
    rms: jax.Array
    mean_abs: jax.Array
    max_abs: jax.Array
    finite_fraction: jax.Array

    @classmethod
    def from_tensor(cls, tensor: jax.Array) -> "SummaryStats":
        tensor = tensor.astype(jnp.float32)
        abs_tensor = jnp.abs(tensor)
        return cls(
            norm=jnp.linalg.norm(jnp.ravel(tensor)),
            rms=jnp.sqrt(jnp.mean(jnp.square(tensor))),
            mean_abs=jnp.mean(abs_tensor),
            max_abs=jnp.max(abs_tensor),
            finite_fraction=jnp.mean(jnp.isfinite(tensor).astype(jnp.float32)),
        )

    def to_metrics(self, prefix: str) -> dict[str, jax.Array]:
        return {
            f"{prefix}_norm": self.norm,
            f"{prefix}_rms": self.rms,
            f"{prefix}_mean_abs": self.mean_abs,
            f"{prefix}_max_abs": self.max_abs,
            f"{prefix}_finite_fraction": self.finite_fraction,
        }


@dataclass(frozen=True)
class _PlateRegion:
    label: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class _BackwardFlowContext:
    prefix: str
    gradient_scale: Any | None = None


_ACTIVE_CONTEXT: contextvars.ContextVar[_BackwardFlowContext | None] = contextvars.ContextVar(
    "levanter_backward_flow_context",
    default=None,
)


@contextlib.contextmanager
def capture_backward_flow(
    config: BackwardFlowConfig | None,
    *,
    prefix: str = _DEFAULT_PREFIX,
    gradient_scale: Any | None = None,
):
    """Enable backward-flow instrumentation while tracing a function.

    Args:
        config: Sampling configuration. Disabled or missing configs leave values unchanged.
        prefix: Metric key prefix.
        gradient_scale: Optional multiplier for logged gradient RMS. For mean-reduced losses,
            pass the effective loss denominator to view cotangents on the unreduced loss scale.
    """
    if config is None or not config.is_enabled:
        yield
        return

    token = _ACTIVE_CONTEXT.set(_BackwardFlowContext(prefix=prefix, gradient_scale=gradient_scale))
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.reset(token)


def is_backward_flow_active() -> bool:
    """Return whether backward-flow probes are active for the current trace."""
    return _ACTIVE_CONTEXT.get() is not None


def normalize_name_stack(name_stack: str) -> str:
    """Remove JAX transform wrappers from a runtime name stack."""
    if not name_stack:
        return ""

    parts: list[str] = []
    for raw_part in name_stack.split("/"):
        part = _strip_transform_wrappers(raw_part)
        if not part:
            continue
        if not parts or parts[-1] != part:
            parts.append(part)
    return "/".join(parts)


def log_backward_activation(x: jax.Array, *, site: BackwardFlowSite = BACKWARD_FLOW_SITE_OUT) -> jax.Array:
    """Return ``x`` unchanged while logging activation and backward-gradient scale when enabled."""
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        return x
    if site not in _FLOW_SITES:
        raise ValueError(f"site must be one of {_FLOW_SITES}, got {site!r}")

    name_stack = normalize_name_stack(str(source_info_util.current_name_stack()))
    if not name_stack:
        return x

    if context.gradient_scale is None:
        return _tagged_identity(f"{context.prefix}/{name_stack}", site, x)
    return _tagged_identity_with_scale(f"{context.prefix}/{name_stack}", site, context.gradient_scale, x)


def trace_backward_activation(
    x: jax.Array, name: str, *, site: BackwardFlowSite = BACKWARD_FLOW_SITE_OUT
) -> jax.Array:
    """Return ``x`` unchanged while logging under an extra JAX named scope."""
    if not name:
        raise ValueError("name must be non-empty")

    with jax.named_scope(name):
        return log_backward_activation(x, site=site)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _tagged_identity(metric_prefix: str, site: BackwardFlowSite, x: jax.Array) -> jax.Array:
    return x


def _tagged_identity_fwd(metric_prefix: str, site: BackwardFlowSite, x: jax.Array) -> tuple[jax.Array, None]:
    levanter.tracker.jit_log(
        _tensor_metrics(metric_prefix, x, site=site, kind=BACKWARD_FLOW_KIND_ACTIVATION), step=None
    )
    return x, None


def _tagged_identity_bwd(
    metric_prefix: str, site: BackwardFlowSite, _residual: None, cotangent: jax.Array
) -> tuple[jax.Array]:
    levanter.tracker.jit_log(
        _tensor_metrics(metric_prefix, cotangent, site=site, kind=BACKWARD_FLOW_KIND_GRADIENT), step=None
    )
    return (cotangent,)


_tagged_identity.defvjp(_tagged_identity_fwd, _tagged_identity_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _tagged_identity_with_scale(
    metric_prefix: str, site: BackwardFlowSite, gradient_scale: jax.Array, x: jax.Array
) -> jax.Array:
    return x


def _tagged_identity_with_scale_fwd(
    metric_prefix: str, site: BackwardFlowSite, gradient_scale: jax.Array, x: jax.Array
) -> tuple[jax.Array, jax.Array]:
    levanter.tracker.jit_log(
        _tensor_metrics(metric_prefix, x, site=site, kind=BACKWARD_FLOW_KIND_ACTIVATION), step=None
    )
    return x, gradient_scale


def _tagged_identity_with_scale_bwd(
    metric_prefix: str,
    site: BackwardFlowSite,
    gradient_scale: jax.Array,
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    levanter.tracker.jit_log(
        _tensor_metrics(
            metric_prefix,
            cotangent,
            site=site,
            kind=BACKWARD_FLOW_KIND_GRADIENT,
            gradient_scale=gradient_scale,
        ),
        step=None,
    )
    return jnp.zeros_like(gradient_scale), cotangent


_tagged_identity_with_scale.defvjp(_tagged_identity_with_scale_fwd, _tagged_identity_with_scale_bwd)


def backward_flow_graph_from_jaxpr(closed_jaxpr: Any) -> BackwardFlowGraph:
    """Build a scope-level dataflow graph from a grad JAXpr."""
    jaxpr = _unwrap_jaxpr(closed_jaxpr)
    node_counts: dict[str, int] = defaultdict(int)
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    _process_jaxpr(jaxpr, [None] * len(jaxpr.invars), node_counts=node_counts, edge_counts=edge_counts)

    edges = tuple(
        BackwardFlowEdge(source=source, target=target, count=count)
        for (source, target), count in sorted(edge_counts.items())
    )
    return BackwardFlowGraph(nodes=tuple(sorted(node_counts)), edges=edges)


def collapse_backward_flow_graph(graph: BackwardFlowGraph, tracked_nodes: Iterable[str]) -> BackwardFlowGraph:
    """Collapse a dense graph down to paths between tracked nodes."""
    tracked = tuple(sorted(set(tracked_nodes) & set(graph.nodes)))
    if not tracked:
        return BackwardFlowGraph(nodes=(), edges=())

    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)

    collapsed_edges: dict[tuple[str, str], int] = defaultdict(int)
    tracked_set = set(tracked)

    for source in tracked:
        queue: deque[str] = deque(adjacency[source])
        seen: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            if node in tracked_set:
                if node != source:
                    collapsed_edges[(source, node)] += 1
                continue
            queue.extend(adjacency[node])

    edges = tuple(
        BackwardFlowEdge(source=source, target=target, count=count)
        for (source, target), count in sorted(collapsed_edges.items())
    )
    return BackwardFlowGraph(nodes=tracked, edges=edges)


def backward_flow_node_stats(
    metrics: Mapping[str, Any],
    *,
    prefix: str = _DEFAULT_PREFIX,
) -> dict[str, dict[str, float]]:
    """Parse node statistics back out of flattened metric keys."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    metric_prefix = f"{prefix}/"
    for key, value in metrics.items():
        if not key.startswith(metric_prefix):
            continue
        tail = key[len(metric_prefix) :]
        if "/" not in tail:
            continue
        node_name, metric_name = tail.rsplit("/", maxsplit=1)
        if not _is_supported_metric_name(metric_name):
            continue
        out[node_name][metric_name] = float(jax.device_get(jnp.asarray(value)))
    return dict(out)


def infer_backward_flow_render_hints(graph: BackwardFlowGraph) -> BackwardFlowRenderHints:
    """Infer generic render hints from common residual-stream node names.

    This intentionally uses only names already present in the runtime scope stack. Models
    that name residual stream probes `resid_in`, `resid_post_*`, and `resid_out` get a
    readable display graph without wiring model-specific layout code into the renderer.
    """
    residual_groups = _residual_groups(graph.nodes)
    if not residual_groups:
        return BackwardFlowRenderHints()

    return BackwardFlowRenderHints(
        node_lanes=_residual_node_lanes(graph.nodes, residual_groups),
        display_edges=_residual_display_edges(graph, residual_groups),
        plates=_residual_plates(graph.nodes, residual_groups),
    )


def render_backward_flow_html(
    graph: BackwardFlowGraph,
    node_stats: Mapping[str, Mapping[str, float]] | None = None,
    *,
    node_grid_positions: Mapping[str, tuple[int, int]] | None = None,
    node_lanes: Mapping[str, int] | None = None,
    display_edges: Iterable[BackwardFlowEdge] | None = None,
    plates: Iterable[BackwardFlowPlate] = (),
    flow_direction: Literal["tb", "lr"] = "tb",
    title: str = "Backward Flow",
    residual_gain_horizon: int = _DEFAULT_RESIDUAL_GAIN_HORIZON,
) -> str:
    """Render a small standalone HTML visualization for a backward-flow graph."""
    node_stats = node_stats or {}
    if flow_direction not in _FLOW_DIRECTIONS:
        raise ValueError(f"flow_direction must be one of {_FLOW_DIRECTIONS}, got {flow_direction!r}")
    if residual_gain_horizon <= 0:
        raise ValueError(f"residual_gain_horizon must be positive, got {residual_gain_horizon}")

    node_width = 260
    node_height = 92
    layer_gap = 110
    row_gap = 28
    margin_x = 40
    margin_y = 40
    rendered_edges = tuple(graph.edges if display_edges is None else display_edges)
    residual_back_gains = _residual_back_gains(rendered_edges, node_stats)
    layout_graph = graph if display_edges is None else BackwardFlowGraph(nodes=graph.nodes, edges=rendered_edges)
    positions, svg_width, svg_height = _layout_positions(
        layout_graph,
        node_width=node_width,
        node_height=node_height,
        layer_gap=layer_gap,
        row_gap=row_gap,
        margin_x=margin_x,
        margin_y=margin_y,
        node_grid_positions=node_grid_positions,
        node_lanes=node_lanes,
        flow_direction=flow_direction,
    )

    svg_lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 {svg_width} {svg_height}' width='{svg_width}' height='{svg_height}'>",
        "<defs>",
        "<marker id='arrow' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'>",
        "<path d='M0,0 L10,4 L0,8 z' fill='#64748b' />",
        "</marker>",
        "</defs>",
    ]

    for plate in _plate_regions(plates, positions, node_width=node_width, node_height=node_height):
        svg_lines.append(
            f"<rect class='flow-plate' x='{plate.x}' y='{plate.y}' width='{plate.width}' height='{plate.height}' "
            "rx='6' fill='#f8fafc' stroke='#cbd5e1' stroke-width='1.5' stroke-dasharray='6 4' />"
        )
        svg_lines.append(
            f"<text class='flow-plate-label' x='{plate.x + 10}' y='{plate.y + 18}' "
            "fill='#475569' font-family='monospace' font-size='13' font-weight='700'>"
            f"{html.escape(plate.label)}</text>"
        )

    for edge in rendered_edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        start_x, start_y = positions[edge.source]
        end_x, end_y = positions[edge.target]
        path = _edge_path(
            start_x,
            start_y,
            end_x,
            end_y,
            node_width=node_width,
            node_height=node_height,
            flow_direction=flow_direction,
        )
        svg_lines.append(
            "<path "
            f"data-source='{html.escape(edge.source, quote=True)}' "
            f"data-target='{html.escape(edge.target, quote=True)}' "
            f"d='{path}' stroke='#94a3b8' stroke-width='2' fill='none' marker-end='url(#arrow)' opacity='0.9' />"
        )

    for node in graph.nodes:
        if node not in positions:
            continue
        x, y = positions[node]
        stats = node_stats.get(node, {})
        residual_gain = residual_back_gains.get(node)
        fill = _node_fill(
            _preferred_gradient_rms(stats),
            residual_gain=residual_gain,
            residual_gain_horizon=residual_gain_horizon,
        )
        label_lines = _node_label_lines(node)
        metric_lines = _metric_lines(
            stats,
            residual_gain=residual_gain,
            residual_gain_horizon=residual_gain_horizon,
        )

        svg_lines.append(
            f"<rect x='{x}' y='{y}' width='{node_width}' height='{node_height}' rx='6' "
            f"fill='{fill}' stroke='#1f2937' stroke-width='1.5' />"
        )
        svg_lines.append(f"<title>{html.escape(node)}</title>")
        svg_lines.append(f"<text x='{x + 12}' y='{y + 22}' fill='#111827' font-family='monospace' font-size='12'>")
        first = True
        for line in label_lines + metric_lines:
            dy = 0 if first else 14
            first = False
            svg_lines.append(f"<tspan x='{x + 12}' dy='{dy}'>{html.escape(line)}</tspan>")
        svg_lines.append("</text>")

    svg_lines.append("</svg>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body { font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }"
        "h1 { margin: 0 0 16px; font-size: 24px; }"
        "p { margin: 16px 0 0; max-width: 960px; }"
        "svg { background: white; border: 1px solid #cbd5e1; border-radius: 8px; }"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"{''.join(svg_lines)}"
        "<p>Residual-stream nodes show local backward gain and are colored by that gain projected over "
        f"{residual_gain_horizon} residual steps; other nodes use scaled backward gradient RMS when available, "
        "otherwise raw gradient RMS. Green is near 1, blue is vanishing, and red is exploding. Node boxes also "
        "report activation RMS and the max absolute gradient component.</p>"
        "</body></html>"
    )


def _tensor_metrics(
    metric_prefix: str,
    tensor: jax.Array,
    *,
    site: BackwardFlowSite,
    kind: BackwardFlowTensorKind,
    gradient_scale: jax.Array | None = None,
) -> dict[str, jax.Array]:
    summary = SummaryStats.from_tensor(tensor)
    metrics = summary.to_metrics(f"{metric_prefix}/{site}_{kind}")
    if kind == BACKWARD_FLOW_KIND_GRADIENT and gradient_scale is not None:
        gradient_scale = jnp.asarray(gradient_scale, dtype=jnp.float32)
        metrics[f"{metric_prefix}/{site}_{kind}_rms_scaled"] = summary.rms * gradient_scale
        metrics[f"{metric_prefix}/{site}_{kind}_max_abs_scaled"] = summary.max_abs * gradient_scale
    return metrics


def _strip_transform_wrappers(part: str) -> str:
    current = part
    while True:
        match = _NAME_STACK_PART_RE.fullmatch(current)
        if match is None or match.group("wrapper") not in _TRANSFORM_WRAPPERS:
            return current
        current = match.group("inner")


def _process_jaxpr(
    jaxpr: Any,
    input_producers: list[str | None],
    *,
    node_counts: dict[str, int],
    edge_counts: dict[tuple[str, str], int],
) -> list[str | None]:
    producer_by_var: dict[Any, str | None] = {}
    for var, producer in zip(jaxpr.invars, input_producers, strict=False):
        producer_by_var[var] = producer

    for eqn in jaxpr.eqns:
        nested = _single_child_jaxpr(eqn)
        if nested is not None:
            nested_inputs = [None if _is_literal(invar) else producer_by_var.get(invar) for invar in eqn.invars]
            nested_outputs = _process_jaxpr(
                nested,
                nested_inputs,
                node_counts=node_counts,
                edge_counts=edge_counts,
            )
            if any(output is not None for output in nested_outputs):
                for outvar, producer in zip(eqn.outvars, nested_outputs, strict=False):
                    producer_by_var[outvar] = producer
                continue

        node_name = normalize_name_stack(str(eqn.source_info.name_stack))
        if node_name:
            node_counts[node_name] += 1
            for invar in eqn.invars:
                if _is_literal(invar):
                    continue
                source = producer_by_var.get(invar)
                if source is not None and source != node_name:
                    edge_counts[(source, node_name)] += 1

        for outvar in eqn.outvars:
            producer_by_var[outvar] = node_name or None

    return [None if _is_literal(outvar) else producer_by_var.get(outvar) for outvar in jaxpr.outvars]


def _single_child_jaxpr(eqn: Any) -> Any | None:
    children: list[Any] = []

    def _visit(value: Any) -> None:
        if hasattr(value, "jaxpr") and hasattr(value, "consts"):
            children.append(value.jaxpr)
        elif hasattr(value, "eqns") and hasattr(value, "invars") and hasattr(value, "outvars"):
            children.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                _visit(item)

    for value in eqn.params.values():
        _visit(value)

    if len(children) == 1:
        return children[0]
    return None


def _node_layers(graph: BackwardFlowGraph) -> dict[str, int]:
    indegree = {node: 0 for node in graph.nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    depths = {node: 0 for node in graph.nodes}
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for target in sorted(adjacency[node]):
            depths[target] = max(depths[target], depths[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    for node in graph.nodes:
        if node not in visited:
            depths[node] = 0
    return depths


def _auto_grid_positions_by_layer(
    nodes: Iterable[str],
    layers: Mapping[str, int],
    layer_order: Iterable[int],
    *,
    node_lanes: Mapping[str, int] | None,
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    layer_index_by_depth = {layer: index for index, layer in enumerate(layer_order)}
    for layer in layer_index_by_depth:
        used_lanes: set[int] = set()
        next_lane = 0
        layer_nodes = sorted((node for node in nodes if layers.get(node, 0) == layer), key=lambda node: node)
        for node in layer_nodes:
            preferred_lane = None if node_lanes is None else node_lanes.get(node)
            if preferred_lane is None or preferred_lane in used_lanes:
                while next_lane in used_lanes:
                    next_lane += 1
                lane = next_lane
            else:
                lane = preferred_lane
            used_lanes.add(lane)
            positions[node] = (layer_index_by_depth[layer], lane)
    return positions


def _layout_positions(
    graph: BackwardFlowGraph,
    *,
    node_width: int,
    node_height: int,
    layer_gap: int,
    row_gap: int,
    margin_x: int,
    margin_y: int,
    node_grid_positions: Mapping[str, tuple[int, int]] | None,
    node_lanes: Mapping[str, int] | None,
    flow_direction: Literal["tb", "lr"],
) -> tuple[dict[str, tuple[int, int]], int, int]:
    if node_grid_positions is None:
        layers = _node_layers(graph)
        layer_order = sorted({layers.get(node, 0) for node in graph.nodes})
        grid_positions = _auto_grid_positions_by_layer(graph.nodes, layers, layer_order, node_lanes=node_lanes)
        max_lane = max((lane for _, lane in grid_positions.values()), default=0)
        if flow_direction == "lr":
            svg_width = margin_x * 2 + max(1, len(layer_order)) * node_width + max(0, len(layer_order) - 1) * layer_gap
            svg_height = margin_y * 2 + (max_lane + 1) * node_height + max(0, max_lane) * row_gap
        else:
            svg_width = margin_x * 2 + (max_lane + 1) * node_width + max(0, max_lane) * layer_gap
            svg_height = margin_y * 2 + max(1, len(layer_order)) * node_height + max(0, len(layer_order) - 1) * row_gap

        positions: dict[str, tuple[int, int]] = {}
        for node, (depth, lane) in grid_positions.items():
            positions[node] = _grid_position(
                depth=depth,
                lane=lane,
                node_width=node_width,
                node_height=node_height,
                layer_gap=layer_gap,
                row_gap=row_gap,
                margin_x=margin_x,
                margin_y=margin_y,
                flow_direction=flow_direction,
            )
        return positions, svg_width, svg_height

    missing = sorted(node for node in graph.nodes if node not in node_grid_positions)
    if missing:
        raise ValueError(f"Missing grid positions for nodes: {missing}")

    max_depth = max((depth for depth, _ in node_grid_positions.values()), default=0)
    max_lane = max((lane for _, lane in node_grid_positions.values()), default=0)
    if flow_direction == "lr":
        svg_width = margin_x * 2 + (max_depth + 1) * node_width + max(0, max_depth) * layer_gap
        svg_height = margin_y * 2 + (max_lane + 1) * node_height + max(0, max_lane) * row_gap
    else:
        svg_width = margin_x * 2 + (max_lane + 1) * node_width + max(0, max_lane) * layer_gap
        svg_height = margin_y * 2 + (max_depth + 1) * node_height + max(0, max_depth) * row_gap
    positions = {
        node: _grid_position(
            depth=depth,
            lane=lane,
            node_width=node_width,
            node_height=node_height,
            layer_gap=layer_gap,
            row_gap=row_gap,
            margin_x=margin_x,
            margin_y=margin_y,
            flow_direction=flow_direction,
        )
        for node, (depth, lane) in node_grid_positions.items()
    }
    return positions, svg_width, svg_height


def _plate_regions(
    plates: Iterable[BackwardFlowPlate],
    positions: Mapping[str, tuple[int, int]],
    *,
    node_width: int,
    node_height: int,
) -> tuple[_PlateRegion, ...]:
    plate_regions: list[_PlateRegion] = []
    for plate in plates:
        plate_nodes = tuple(node for node in plate.nodes if node in positions)
        if len(plate_nodes) < 2:
            continue
        left = min(positions[node][0] for node in plate_nodes)
        top = min(positions[node][1] for node in plate_nodes)
        right = max(positions[node][0] + node_width for node in plate_nodes)
        bottom = max(positions[node][1] + node_height for node in plate_nodes)
        pad_x = 18
        pad_top = 30
        pad_bottom = 18
        plate_regions.append(
            _PlateRegion(
                label=plate.label,
                x=left - pad_x,
                y=top - pad_top,
                width=right - left + 2 * pad_x,
                height=bottom - top + pad_top + pad_bottom,
            )
        )
    return tuple(plate_regions)


def _residual_groups(nodes: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        plate = _residual_plate_prefix(node)
        if plate is None or _residual_stage_index(node) is None:
            continue
        groups[plate].append(node)

    return {
        plate: sorted(residual_nodes, key=lambda node: (_residual_stage_index(node) or 0, _natural_sort_key(node)))
        for plate, residual_nodes in groups.items()
        if len(residual_nodes) >= 2
    }


def _residual_stage_index(node: str) -> int | None:
    leaf = node.rsplit("/", maxsplit=1)[-1]
    if leaf == "resid_in":
        return 0
    if leaf.startswith("resid_post_"):
        return 1
    if leaf == "resid_out":
        return 2
    return None


def _residual_plates(
    nodes: Iterable[str],
    residual_groups: Mapping[str, list[str]],
) -> tuple[BackwardFlowPlate, ...]:
    nodes_by_plate: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        plate = _residual_plate_prefix(node)
        if plate is None or plate not in residual_groups:
            continue
        nodes_by_plate[plate].append(node)
    return tuple(
        BackwardFlowPlate(label=plate, nodes=tuple(sorted(plate_nodes, key=_natural_sort_key)))
        for plate, plate_nodes in sorted(nodes_by_plate.items(), key=lambda item: _natural_sort_key(item[0]))
    )


def _residual_node_lanes(
    nodes: Iterable[str],
    residual_groups: Mapping[str, list[str]],
) -> dict[str, int]:
    lanes = {node: 1 for node in nodes}
    for node in nodes:
        plate = _residual_plate_prefix(node)
        if plate is None or plate not in residual_groups:
            continue
        if _residual_stage_index(node) is not None:
            lanes[node] = 1
            continue
        segment_index = _fallback_branch_segment_index(node, len(residual_groups[plate]))
        if segment_index is None:
            continue
        lanes[node] = 0 if segment_index % 2 == 0 else 2
    return lanes


def _residual_display_edges(
    graph: BackwardFlowGraph,
    residual_groups: Mapping[str, list[str]],
) -> tuple[BackwardFlowEdge, ...]:
    edge_by_pair: dict[tuple[str, str], BackwardFlowEdge] = {}
    for edge in _residual_stream_edges(residual_groups):
        edge_by_pair[(edge.source, edge.target)] = edge
    for edge in _residual_branch_edges(graph, residual_groups):
        edge_by_pair[(edge.source, edge.target)] = edge
    for edge in _residual_inter_group_edges(graph, residual_groups):
        edge_by_pair[(edge.source, edge.target)] = edge
    return tuple(sorted(edge_by_pair.values(), key=lambda edge: (edge.source, edge.target)))


def _residual_stream_edges(residual_groups: Mapping[str, list[str]]) -> tuple[BackwardFlowEdge, ...]:
    edges: list[BackwardFlowEdge] = []
    for residual_nodes in residual_groups.values():
        edges.extend(BackwardFlowEdge(source=source, target=target) for source, target in pairwise(residual_nodes))
    return tuple(edges)


def _residual_branch_edges(
    graph: BackwardFlowGraph,
    residual_groups: Mapping[str, list[str]],
) -> tuple[BackwardFlowEdge, ...]:
    edges: list[BackwardFlowEdge] = []
    graph_edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
    residual_nodes = {node for group in residual_groups.values() for node in group}
    for node in graph.nodes:
        if node in residual_nodes:
            continue
        plate = _residual_plate_prefix(node)
        if plate is None or plate not in residual_groups:
            continue
        segment_index = _branch_segment_index(node, residual_groups[plate], graph_edge_pairs)
        if segment_index is None:
            continue
        source = residual_groups[plate][segment_index]
        target = residual_groups[plate][segment_index + 1]
        edges.append(BackwardFlowEdge(source=source, target=node))
        edges.append(BackwardFlowEdge(source=node, target=target))
    return tuple(edges)


def _branch_segment_index(
    node: str,
    residual_nodes: list[str],
    graph_edge_pairs: set[tuple[str, str]],
) -> int | None:
    candidates: list[int] = []
    for index, (source, target) in enumerate(pairwise(residual_nodes)):
        if (source, node) in graph_edge_pairs and (node, target) in graph_edge_pairs:
            candidates.append(index)
    if len(candidates) == 1:
        return candidates[0]
    return _fallback_branch_segment_index(node, len(residual_nodes))


def _fallback_branch_segment_index(node: str, num_residual_nodes: int) -> int | None:
    if num_residual_nodes < 2:
        return None
    leaf = node.rsplit("/", maxsplit=1)[-1].lower()
    if "attention" in leaf or "attn" in leaf:
        return 0
    if "mlp" in leaf or "ffn" in leaf or "feedforward" in leaf:
        return num_residual_nodes - 2
    return None


def _residual_inter_group_edges(
    graph: BackwardFlowGraph,
    residual_groups: Mapping[str, list[str]],
) -> tuple[BackwardFlowEdge, ...]:
    ordered_groups = [
        group for _, group in sorted(residual_groups.items(), key=lambda item: _natural_sort_key(item[0]))
    ]
    edges: list[BackwardFlowEdge] = [
        BackwardFlowEdge(source=source_group[-1], target=target_group[0])
        for source_group, target_group in pairwise(ordered_groups)
    ]

    first_residual = ordered_groups[0][0]
    last_residual = ordered_groups[-1][-1]
    incoming_edges: list[BackwardFlowEdge] = []
    outgoing_edges: list[BackwardFlowEdge] = []
    for edge in graph.edges:
        if edge.target == first_residual and _residual_plate_prefix(edge.source) not in residual_groups:
            incoming_edges.append(edge)
        if edge.source == last_residual and _residual_plate_prefix(edge.target) not in residual_groups:
            outgoing_edges.append(edge)

    preferred_incoming = [edge for edge in incoming_edges if _is_preferred_graph_input(edge.source)]
    preferred_outgoing = [edge for edge in outgoing_edges if _is_preferred_graph_output(edge.target)]
    edges.extend(preferred_incoming or incoming_edges)
    edges.extend(preferred_outgoing or outgoing_edges)
    if not incoming_edges:
        edges.extend(
            BackwardFlowEdge(source=node, target=first_residual)
            for node in _external_residual_nodes(graph.nodes, residual_groups)
            if _is_preferred_graph_input(node)
        )
    if not outgoing_edges:
        edges.extend(
            BackwardFlowEdge(source=last_residual, target=node)
            for node in _external_residual_nodes(graph.nodes, residual_groups)
            if _is_preferred_graph_output(node)
        )
    return tuple(edges)


def _residual_plate_prefix(node: str) -> str | None:
    parts = node.split("/")
    for index, part in enumerate(parts):
        if re.fullmatch(r".*block[_-]?\d+", part, flags=re.IGNORECASE):
            return "/".join(parts[: index + 1])
    if "/" not in node:
        return None
    return node.rsplit("/", maxsplit=1)[0]


def _external_residual_nodes(nodes: Iterable[str], residual_groups: Mapping[str, list[str]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (node for node in nodes if _residual_plate_prefix(node) not in residual_groups),
            key=_natural_sort_key,
        )
    )


def _is_preferred_graph_input(node: str) -> bool:
    leaf = node.rsplit("/", maxsplit=1)[-1].lower()
    return "embed" in leaf or leaf in {"input", "inputs", "x"}


def _is_preferred_graph_output(node: str) -> bool:
    leaf = node.rsplit("/", maxsplit=1)[-1].lower()
    return "/" not in node or leaf in {"output", "outputs", "logits", "final", "final_norm"}


def _natural_sort_key(value: str) -> tuple[int | str, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _grid_position(
    *,
    depth: int,
    lane: int,
    node_width: int,
    node_height: int,
    layer_gap: int,
    row_gap: int,
    margin_x: int,
    margin_y: int,
    flow_direction: Literal["tb", "lr"],
) -> tuple[int, int]:
    if flow_direction == "lr":
        return (
            margin_x + depth * (node_width + layer_gap),
            margin_y + lane * (node_height + row_gap),
        )

    return (
        margin_x + lane * (node_width + layer_gap),
        margin_y + depth * (node_height + row_gap),
    )


def _edge_path(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    *,
    node_width: int,
    node_height: int,
    flow_direction: Literal["tb", "lr"],
) -> str:
    if flow_direction == "lr":
        x1 = start_x + node_width
        y1 = start_y + node_height / 2
        x2 = end_x
        y2 = end_y + node_height / 2
        ctrl_dx = max(50, (x2 - x1) / 2)
        return f"M{x1:.1f},{y1:.1f} " f"C{x1 + ctrl_dx:.1f},{y1:.1f} {x2 - ctrl_dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"

    x1 = start_x + node_width / 2
    y1 = start_y + node_height
    x2 = end_x + node_width / 2
    y2 = end_y
    ctrl_dy = max(12.0, abs(y2 - y1) / 2)
    return f"M{x1:.1f},{y1:.1f} " f"C{x1:.1f},{y1 + ctrl_dy:.1f} {x2:.1f},{y2 - ctrl_dy:.1f} {x2:.1f},{y2:.1f}"


def _residual_back_gains(
    rendered_edges: Iterable[BackwardFlowEdge],
    node_stats: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    gains: dict[str, float] = {}
    severities: dict[str, float] = {}
    for edge in rendered_edges:
        if _residual_stage_index(edge.source) is None or _residual_stage_index(edge.target) is None:
            continue

        source_rms = _preferred_gradient_rms(node_stats.get(edge.source, {}))
        target_rms = _preferred_gradient_rms(node_stats.get(edge.target, {}))
        if source_rms is None or target_rms is None:
            continue
        if not math.isfinite(source_rms) or not math.isfinite(target_rms) or source_rms <= 0 or target_rms <= 0:
            continue

        gain = source_rms / target_rms
        severity = _rms_severity(gain)
        if severity > severities.get(edge.source, float("-inf")):
            gains[edge.source] = gain
            severities[edge.source] = severity
    return gains


def _projected_residual_gain(gain: float | None, residual_gain_horizon: int) -> float | None:
    if gain is None or not math.isfinite(gain) or gain <= 0:
        return None
    log_gain = math.log(gain) * residual_gain_horizon
    return math.exp(max(-700.0, min(700.0, log_gain)))


def _node_fill(
    rms: float | None,
    *,
    residual_gain: float | None = None,
    residual_gain_horizon: int = _DEFAULT_RESIDUAL_GAIN_HORIZON,
) -> str:
    value = _projected_residual_gain(residual_gain, residual_gain_horizon)
    if value is None:
        value = rms
    if value is None or not math.isfinite(value) or value <= 0:
        return "#e2e8f0"
    log2_scale = math.log2(value)
    severity = min(1.0, abs(log2_scale) / 4.0)
    if abs(log2_scale) < 0.1:
        return "#dcfce7"
    if log2_scale < 0:
        return _mix_hex("#dcfce7", "#93c5fd", severity)
    return _mix_hex("#dcfce7", "#f87171", severity)


def _node_label_lines(node: str) -> list[str]:
    parts = node.split("/")
    if len(parts) <= 3:
        return ["/".join(parts)]
    return ["/".join(parts[-3:])]


def _metric_lines(
    stats: Mapping[str, float],
    *,
    residual_gain: float | None = None,
    residual_gain_horizon: int = _DEFAULT_RESIDUAL_GAIN_HORIZON,
) -> list[str]:
    if not stats:
        return ["grad RMS: n/a", "max abs grad: n/a", "act RMS: n/a", "finite: n/a"]
    finite_fraction = _preferred_finite_fraction(stats)
    finite_pct = None if finite_fraction is None else 100.0 * finite_fraction
    gradient_label = "scaled grad RMS" if _has_scaled_gradient_rms(stats) else "grad RMS"
    max_gradient_label = "scaled max abs grad" if _has_scaled_gradient_max_abs(stats) else "max abs grad"
    fourth_line = f"finite: {_format_stat(finite_pct)}%"
    if residual_gain is not None:
        fourth_line = f"local gain: {_format_gain(residual_gain)}"
    return [
        f"{gradient_label}: {_format_stat(_preferred_gradient_rms(stats))}",
        f"{max_gradient_label}: {_format_stat(_preferred_gradient_max_abs(stats))}",
        f"act RMS: {_format_stat(_preferred_activation_rms(stats))}",
        fourth_line,
    ]


def _format_stat(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2e}"


def _format_gain(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if 0.01 <= abs(value) < 100:
        return f"{value:.2f}x"
    return f"{value:.2e}x"


def _is_supported_metric_name(metric_name: str) -> bool:
    for site in _FLOW_SITES:
        for kind in _TENSOR_KINDS:
            prefix = f"{site}_{kind}_"
            if metric_name.startswith(prefix) and metric_name[len(prefix) :] in _STAT_NAMES:
                return True
    return False


def _metric_value(
    stats: Mapping[str, float], site: BackwardFlowSite, kind: BackwardFlowTensorKind, metric: str
) -> float | None:
    return stats.get(f"{site}_{kind}_{metric}")


def _preferred_gradient_rms(stats: Mapping[str, float]) -> float | None:
    scaled = _preferred_metric(stats, BACKWARD_FLOW_KIND_GRADIENT, "rms_scaled", preferred_site=BACKWARD_FLOW_SITE_IN)
    if scaled is not None:
        return scaled
    return _preferred_metric(stats, BACKWARD_FLOW_KIND_GRADIENT, "rms", preferred_site=BACKWARD_FLOW_SITE_IN)


def _has_scaled_gradient_rms(stats: Mapping[str, float]) -> bool:
    return any(
        _metric_value(stats, site, BACKWARD_FLOW_KIND_GRADIENT, "rms_scaled") is not None for site in _FLOW_SITES
    )


def _preferred_gradient_max_abs(stats: Mapping[str, float]) -> float | None:
    scaled = _preferred_metric(
        stats, BACKWARD_FLOW_KIND_GRADIENT, "max_abs_scaled", preferred_site=BACKWARD_FLOW_SITE_IN
    )
    if scaled is not None:
        return scaled
    return _preferred_metric(stats, BACKWARD_FLOW_KIND_GRADIENT, "max_abs", preferred_site=BACKWARD_FLOW_SITE_IN)


def _has_scaled_gradient_max_abs(stats: Mapping[str, float]) -> bool:
    return any(
        _metric_value(stats, site, BACKWARD_FLOW_KIND_GRADIENT, "max_abs_scaled") is not None for site in _FLOW_SITES
    )


def _preferred_activation_rms(stats: Mapping[str, float]) -> float | None:
    return _preferred_metric(stats, BACKWARD_FLOW_KIND_ACTIVATION, "rms", preferred_site=BACKWARD_FLOW_SITE_OUT)


def _preferred_finite_fraction(stats: Mapping[str, float]) -> float | None:
    gradient_fraction = _preferred_metric(
        stats, BACKWARD_FLOW_KIND_GRADIENT, "finite_fraction", preferred_site=BACKWARD_FLOW_SITE_IN
    )
    if gradient_fraction is not None:
        return gradient_fraction
    return _preferred_metric(
        stats, BACKWARD_FLOW_KIND_ACTIVATION, "finite_fraction", preferred_site=BACKWARD_FLOW_SITE_OUT
    )


def _preferred_metric(
    stats: Mapping[str, float],
    kind: BackwardFlowTensorKind,
    metric: str,
    *,
    preferred_site: BackwardFlowSite,
) -> float | None:
    preferred = _metric_value(stats, preferred_site, kind, metric)
    if preferred is not None:
        return preferred
    for site in _FLOW_SITES:
        value = _metric_value(stats, site, kind, metric)
        if value is not None:
            return value
    return None


def _rms_severity(rms: float | None) -> float:
    if rms is None or not math.isfinite(rms):
        return float("-inf")
    if rms <= 0:
        return float("inf")
    return abs(math.log2(rms))


def _mix_hex(start: str, end: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    start_rgb = tuple(int(start[i : i + 2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(a + (b - a) * amount) for a, b in zip(start_rgb, end_rgb, strict=True))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _unwrap_jaxpr(jaxpr_like: Any) -> Any:
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr
    return jaxpr_like


def _is_literal(value: Any) -> bool:
    return isinstance(value, JaxprLiteral)
