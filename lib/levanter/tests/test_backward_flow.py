# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.nn import ArrayStacked

import levanter.tracker
from levanter.analysis.backward_flow import (
    BackwardFlowConfig,
    BackwardFlowEdge,
    BackwardFlowGraph,
    BackwardFlowPlate,
    BWD_IN,
    BWD_OUT,
    SummaryStats,
    backward_flow_graph_from_jaxpr,
    capture_backward_flow,
    collapse_backward_flow_graph,
    infer_backward_flow_render_hints,
    is_backward_flow_active,
    log_backward_activation,
    normalize_name_stack,
    render_backward_flow_html,
    trace_backward_activation,
)


def test_summary_stats_computes_tensor_metrics():
    stats = SummaryStats.from_tensor(jnp.array([-1.0, 2.0, 3.0], dtype=jnp.float32))
    metrics = stats.to_metrics("tensor")

    assert jnp.allclose(metrics["tensor_norm"], jnp.sqrt(jnp.array(14.0, dtype=jnp.float32)))
    assert jnp.allclose(metrics["tensor_rms"], jnp.sqrt(jnp.array(14.0 / 3.0, dtype=jnp.float32)))
    assert jnp.allclose(metrics["tensor_mean_abs"], jnp.array(2.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["tensor_max_abs"], jnp.array(3.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["tensor_finite_fraction"], jnp.array(1.0, dtype=jnp.float32))


def test_normalize_name_stack_strips_jax_transform_wrappers():
    raw = "transpose(jvp(Transformer))/block_3/checkpoint(Block)/MLP"
    assert normalize_name_stack(raw) == "Transformer/block_3/Block/MLP"


def test_log_backward_activation_records_activation_and_gradient_metrics():
    @jax.named_call
    def inner(x):
        x = log_backward_activation(x, site=BWD_IN)
        return log_backward_activation(x * 2, site=BWD_OUT)

    @jax.jit
    def compute_grad(x):
        with capture_backward_flow(BackwardFlowConfig(interval=1)):
            with levanter.tracker.defer_tracker_for_jit() as metrics:
                grad = jax.grad(lambda z: jnp.sum(inner(z) ** 2))(x)
        return grad, metrics

    grad, metrics = compute_grad(jnp.ones((3,), dtype=jnp.float32))

    assert jnp.allclose(grad, jnp.full((3,), 8.0, dtype=jnp.float32))
    assert "backward_flow/inner/in_gradient_norm" in metrics
    assert "backward_flow/inner/out_gradient_norm" in metrics
    assert jnp.allclose(metrics["backward_flow/inner/in_gradient_norm"], jnp.sqrt(jnp.array(192.0, dtype=jnp.float32)))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_norm"], jnp.sqrt(jnp.array(48.0, dtype=jnp.float32)))
    assert jnp.allclose(metrics["backward_flow/inner/in_gradient_rms"], jnp.array(8.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_rms"], jnp.array(4.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/in_gradient_max_abs"], jnp.array(8.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_max_abs"], jnp.array(4.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/in_activation_rms"], jnp.array(1.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_activation_rms"], jnp.array(2.0, dtype=jnp.float32))


def test_log_backward_activation_records_scaled_gradient_rms_when_configured():
    @jax.named_call
    def inner(x):
        return log_backward_activation(x * 2, site=BWD_OUT)

    @jax.jit
    def compute_grad(x):
        with capture_backward_flow(BackwardFlowConfig(interval=1), gradient_scale=jnp.array(10.0, dtype=jnp.float32)):
            with levanter.tracker.defer_tracker_for_jit() as metrics:
                grad = jax.grad(lambda z: jnp.sum(inner(z) ** 2))(x)
        return grad, metrics

    grad, metrics = compute_grad(jnp.ones((3,), dtype=jnp.float32))

    assert jnp.allclose(grad, jnp.full((3,), 8.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_rms"], jnp.array(4.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_rms_scaled"], jnp.array(40.0, dtype=jnp.float32))
    assert jnp.allclose(metrics["backward_flow/inner/out_gradient_max_abs_scaled"], jnp.array(40.0, dtype=jnp.float32))


def test_trace_backward_activation_adds_named_scope_for_probe():
    @jax.named_call
    def inner(x):
        return trace_backward_activation(x * 3, "resid_post_attn")

    @jax.jit
    def compute_grad(x):
        with capture_backward_flow(BackwardFlowConfig(interval=1)):
            with levanter.tracker.defer_tracker_for_jit() as metrics:
                grad = jax.grad(lambda z: jnp.sum(inner(z) ** 2))(x)
        return grad, metrics

    grad, metrics = compute_grad(jnp.ones((3,), dtype=jnp.float32))

    assert jnp.allclose(grad, jnp.full((3,), 18.0, dtype=jnp.float32))
    assert "backward_flow/inner/resid_post_attn/out_activation_rms" in metrics
    assert "backward_flow/inner/resid_post_attn/out_gradient_rms" in metrics
    assert jnp.allclose(
        metrics["backward_flow/inner/resid_post_attn/out_activation_rms"], jnp.array(3.0, dtype=jnp.float32)
    )
    assert jnp.allclose(
        metrics["backward_flow/inner/resid_post_attn/out_gradient_rms"], jnp.array(6.0, dtype=jnp.float32)
    )


def test_log_backward_activation_allows_callers_to_skip_checkpoint_when_active():
    @jax.named_call
    def inner(x):
        x = log_backward_activation(x, site=BWD_IN)
        return log_backward_activation(jnp.tanh(x * 2), site=BWD_OUT)

    def maybe_checkpointed_inner(x):
        if is_backward_flow_active():
            return inner(x)
        return jax.checkpoint(inner)(x)

    @jax.jit
    def compute_grad(x):
        with capture_backward_flow(BackwardFlowConfig(interval=1)):
            with levanter.tracker.defer_tracker_for_jit() as metrics:
                grad = jax.grad(lambda z: jnp.sum(maybe_checkpointed_inner(z) ** 2))(x)
        return grad, metrics

    grad, metrics = compute_grad(jnp.ones((3,), dtype=jnp.float32))

    assert jnp.all(jnp.isfinite(grad))
    assert "backward_flow/inner/in_gradient_rms" in metrics
    assert "backward_flow/inner/out_gradient_rms" in metrics


def test_log_backward_activation_works_with_array_stacked_diagnostic_unroll():
    class Layer(eqx.Module):
        weight: jax.Array

        @staticmethod
        def init(weight):
            return Layer(weight=weight)

        def step(self, carry: jax.Array) -> jax.Array:
            with jax.named_scope("ArrayBlock"):
                carry = log_backward_activation(carry, site=BWD_IN)
                return log_backward_activation(jnp.tanh(carry * self.weight), site=BWD_OUT)

    def apply_layers(stack: ArrayStacked[Layer], carry: jax.Array) -> jax.Array:
        if is_backward_flow_active():
            for index, layer in enumerate(stack.unstacked()):
                with jax.named_scope(f"block_{index}"):
                    carry = layer.step(carry)
            return carry
        return stack.fold_via(Layer.step)(carry)

    num_layers = 3
    width = 4
    weights = jnp.linspace(0.5, 1.5, num_layers * width, dtype=jnp.float32).reshape(num_layers, width)
    stacked = ArrayStacked.init(num_layers, Layer, gradient_checkpointing=True)(weight=weights)

    @jax.jit
    def compute_grad(x):
        def loss_fn(z):
            return jnp.sum(jnp.square(apply_layers(stacked, z)))

        with capture_backward_flow(BackwardFlowConfig(interval=1), gradient_scale=jnp.array(2.0, dtype=jnp.float32)):
            with levanter.tracker.defer_tracker_for_jit() as metrics:
                loss, grad = jax.value_and_grad(loss_fn)(x)
        return loss, grad, metrics

    loss, grad, metrics = compute_grad(jnp.ones((width,), dtype=jnp.float32))

    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad))
    for index in range(num_layers):
        assert f"backward_flow/block_{index}/ArrayBlock/in_gradient_rms" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/out_gradient_rms" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/in_gradient_rms_scaled" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/out_gradient_rms_scaled" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/in_gradient_max_abs_scaled" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/out_gradient_max_abs_scaled" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/in_activation_rms" in metrics
        assert f"backward_flow/block_{index}/ArrayBlock/out_activation_rms" in metrics


def test_backward_flow_graph_from_jaxpr_handles_checkpointed_subgraphs():
    @jax.named_call
    def producer(x):
        return x * 2

    @jax.named_call
    def consumer(x):
        y = jax.checkpoint(producer)(x)
        return jnp.sin(y).sum()

    graph = backward_flow_graph_from_jaxpr(jax.make_jaxpr(consumer)(jnp.ones((4,), dtype=jnp.float32)))

    assert "consumer" in graph.nodes
    producer_name = "consumer/producer" if "consumer/producer" in graph.nodes else "producer"
    assert producer_name in graph.nodes
    assert BackwardFlowEdge(source=producer_name, target="consumer", count=1) in graph.edges


def test_collapse_backward_flow_graph_shortcuts_untracked_nodes():
    graph = BackwardFlowGraph(
        nodes=("input", "hidden", "output"),
        edges=(
            BackwardFlowEdge(source="input", target="hidden"),
            BackwardFlowEdge(source="hidden", target="output"),
        ),
    )

    collapsed = collapse_backward_flow_graph(graph, {"input", "output"})

    assert collapsed.nodes == ("input", "output")
    assert collapsed.edges == (BackwardFlowEdge(source="input", target="output", count=1),)


def test_render_backward_flow_html_includes_svg_and_node_metrics():
    graph = BackwardFlowGraph(
        nodes=("Transformer/block_0/CausalSelfAttention", "Transformer/block_0/Block"),
        edges=(
            BackwardFlowEdge(source="Transformer/block_0/CausalSelfAttention", target="Transformer/block_0/Block"),
        ),
    )
    html = render_backward_flow_html(
        graph,
        {
            "Transformer/block_0/CausalSelfAttention": {
                "in_gradient_norm": 8.0,
                "out_gradient_norm": 12.0,
                "in_gradient_rms_scaled": 16.0,
                "out_gradient_rms_scaled": 6.0,
                "in_gradient_max_abs_scaled": 32.0,
                "in_gradient_rms": 4.0,
                "out_gradient_rms": 1.5,
                "out_gradient_max_abs": 3.0,
                "out_activation_rms": 1.1,
                "out_gradient_finite_fraction": 1.0,
            },
            "Transformer/block_0/Block": {
                "out_gradient_norm": 2.0,
                "out_gradient_max_abs": 1.0,
                "out_gradient_rms": 0.5,
                "out_activation_rms": 0.9,
                "out_gradient_finite_fraction": 1.0,
            },
        },
        title="Backward Flow Step 10",
    )

    assert "<svg" in html
    assert "<table" not in html
    assert "Backward Flow Step 10" in html
    assert "CausalSelfAttention" in html
    assert "scaled grad RMS" in html
    assert "scaled max abs grad" in html
    assert "1.60e+01" in html
    assert "3.20e+01" in html
    assert "block_0/CausalSelfAttention" in html
    assert html.index("</svg>") < html.index("<p>")


def test_render_backward_flow_html_honors_manual_grid_positions():
    graph = BackwardFlowGraph(
        nodes=("source", "target"),
        edges=(BackwardFlowEdge(source="source", target="target"),),
    )

    html = render_backward_flow_html(
        graph,
        node_grid_positions={
            "source": (2, 1),
            "target": (0, 0),
        },
    )

    assert "<rect x='410' y='280'" in html
    assert "<rect x='40' y='40'" in html


def test_render_backward_flow_html_can_render_left_to_right():
    graph = BackwardFlowGraph(
        nodes=("source", "target"),
        edges=(BackwardFlowEdge(source="source", target="target"),),
    )

    html = render_backward_flow_html(
        graph,
        node_grid_positions={
            "source": (2, 1),
            "target": (0, 0),
        },
        flow_direction="lr",
    )

    assert "<rect x='780' y='160'" in html
    assert "<rect x='40' y='40'" in html


def test_render_backward_flow_html_infers_depths_from_display_edges_with_lane_hints():
    nodes = (
        "Transformer/token_embed",
        "Transformer/block_0/Block/resid_in",
        "Transformer/block_0/Block/CausalSelfAttention",
        "Transformer/block_0/Block/resid_post_attn",
        "Transformer/block_0/Block/MLP",
        "Transformer/block_0/Block/resid_out",
        "Transformer/block_1/Block/resid_in",
        "Transformer/block_1/Block/CausalSelfAttention",
        "Transformer/block_1/Block/resid_post_attn",
        "Transformer/block_1/Block/MLP",
        "Transformer/block_1/Block/resid_out",
    )
    dense_edges = tuple(
        BackwardFlowEdge(source=source, target=target) for source in nodes for target in nodes if source != target
    )
    node_lanes = {
        "Transformer/token_embed": 1,
        "Transformer/block_0/Block/resid_in": 1,
        "Transformer/block_0/Block/CausalSelfAttention": 0,
        "Transformer/block_0/Block/resid_post_attn": 1,
        "Transformer/block_0/Block/MLP": 2,
        "Transformer/block_0/Block/resid_out": 1,
        "Transformer/block_1/Block/resid_in": 1,
        "Transformer/block_1/Block/CausalSelfAttention": 0,
        "Transformer/block_1/Block/resid_post_attn": 1,
        "Transformer/block_1/Block/MLP": 2,
        "Transformer/block_1/Block/resid_out": 1,
    }
    display_edges = (
        BackwardFlowEdge("Transformer/token_embed", "Transformer/block_0/Block/resid_in"),
        BackwardFlowEdge("Transformer/block_0/Block/resid_in", "Transformer/block_0/Block/CausalSelfAttention"),
        BackwardFlowEdge("Transformer/block_0/Block/resid_in", "Transformer/block_0/Block/resid_post_attn"),
        BackwardFlowEdge("Transformer/block_0/Block/CausalSelfAttention", "Transformer/block_0/Block/resid_post_attn"),
        BackwardFlowEdge("Transformer/block_0/Block/resid_post_attn", "Transformer/block_0/Block/MLP"),
        BackwardFlowEdge("Transformer/block_0/Block/resid_post_attn", "Transformer/block_0/Block/resid_out"),
        BackwardFlowEdge("Transformer/block_0/Block/MLP", "Transformer/block_0/Block/resid_out"),
        BackwardFlowEdge("Transformer/block_0/Block/resid_out", "Transformer/block_1/Block/resid_in"),
        BackwardFlowEdge("Transformer/block_1/Block/resid_in", "Transformer/block_1/Block/CausalSelfAttention"),
        BackwardFlowEdge("Transformer/block_1/Block/resid_in", "Transformer/block_1/Block/resid_post_attn"),
        BackwardFlowEdge("Transformer/block_1/Block/CausalSelfAttention", "Transformer/block_1/Block/resid_post_attn"),
        BackwardFlowEdge("Transformer/block_1/Block/resid_post_attn", "Transformer/block_1/Block/MLP"),
        BackwardFlowEdge("Transformer/block_1/Block/resid_post_attn", "Transformer/block_1/Block/resid_out"),
        BackwardFlowEdge("Transformer/block_1/Block/MLP", "Transformer/block_1/Block/resid_out"),
    )
    plates = (
        BackwardFlowPlate(
            label="Transformer/block_0",
            nodes=tuple(node for node in nodes if "/block_0/" in node),
        ),
        BackwardFlowPlate(
            label="Transformer/block_1",
            nodes=tuple(node for node in nodes if "/block_1/" in node),
        ),
    )

    html = render_backward_flow_html(
        BackwardFlowGraph(nodes=nodes, edges=dense_edges),
        node_lanes=node_lanes,
        display_edges=display_edges,
        plates=plates,
    )

    assert "viewBox='0 0 1080 1372'" in html
    assert "<rect class='flow-plate' x='22' y='130' width='1036' height='620'" in html
    assert "<text class='flow-plate-label' x='32' y='148'" in html
    assert "Transformer/block_0" in html
    assert "<rect x='410' y='40' width='260' height='92'" in html
    assert "<rect x='410' y='160' width='260' height='92'" in html
    assert "<rect x='40' y='280' width='260' height='92'" in html
    assert "<rect x='410' y='400' width='260' height='92'" in html
    assert "<rect x='780' y='520' width='260' height='92'" in html
    assert "<rect x='410' y='640' width='260' height='92'" in html
    assert "M540.0,252.0 C540.0,266.0 170.0,266.0 170.0,280.0" in html
    assert "M540.0,252.0 C540.0,326.0 540.0,326.0 540.0,400.0" in html
    assert "M170.0,372.0 C170.0,386.0 540.0,386.0 540.0,400.0" in html
    assert "M540.0,492.0 C540.0,506.0 910.0,506.0 910.0,520.0" in html
    assert "M540.0,492.0 C540.0,566.0 540.0,566.0 540.0,640.0" in html
    assert "M910.0,612.0 C910.0,626.0 540.0,626.0 540.0,640.0" in html
    assert "M540.0,732.0 C540.0,746.0 540.0,746.0 540.0,760.0" in html
    resid_skip_edge = "data-source='Transformer/block_0/Block/resid_post_attn'"
    resid_skip_edge += " data-target='Transformer/block_0/Block/resid_out'"
    assert resid_skip_edge in html


def test_render_backward_flow_html_projects_residual_back_gain_for_coloring():
    graph = BackwardFlowGraph(
        nodes=(
            "Transformer/block_0/Block/resid_in",
            "Transformer/block_0/Block/resid_out",
        ),
        edges=(BackwardFlowEdge("Transformer/block_0/Block/resid_in", "Transformer/block_0/Block/resid_out"),),
    )

    html = render_backward_flow_html(
        graph,
        {
            "Transformer/block_0/Block/resid_in": {
                "out_gradient_rms_scaled": 1.14,
                "out_activation_rms": 1.0,
            },
            "Transformer/block_0/Block/resid_out": {
                "out_gradient_rms_scaled": 1.0,
                "out_activation_rms": 1.0,
            },
        },
        residual_gain_horizon=50,
    )

    assert "1.14x" in html
    assert "7.00e+02x" not in html
    assert "block_0/Block/resid_in" in html
    assert "local gain: 1.14x" in html
    assert "fill='#f87171'" in html


def test_infer_backward_flow_render_hints_pattern_matches_residual_stream():
    nodes = (
        "Transformer/token_embed",
        "Transformer/block_0/Block/resid_in",
        "Transformer/block_0/Block/CausalSelfAttention",
        "Transformer/block_0/Block/resid_post_attn",
        "Transformer/block_0/Block/MLP",
        "Transformer/block_0/Block/resid_out",
        "Transformer",
    )
    graph = BackwardFlowGraph(
        nodes=nodes,
        edges=tuple(
            BackwardFlowEdge(source=source, target=target) for source in nodes for target in nodes if source != target
        ),
    )

    hints = infer_backward_flow_render_hints(graph)

    assert hints.display_edges is not None
    assert hints.node_lanes is not None
    edge_pairs = {(edge.source, edge.target) for edge in hints.display_edges}
    assert hints.node_lanes["Transformer/block_0/Block/resid_in"] == 1
    assert hints.node_lanes["Transformer/block_0/Block/CausalSelfAttention"] == 0
    assert hints.node_lanes["Transformer/block_0/Block/MLP"] == 2
    assert (
        "Transformer/block_0/Block/resid_in",
        "Transformer/block_0/Block/resid_post_attn",
    ) in edge_pairs
    assert (
        "Transformer/block_0/Block/resid_post_attn",
        "Transformer/block_0/Block/resid_out",
    ) in edge_pairs
    assert (
        "Transformer/block_0/Block/CausalSelfAttention",
        "Transformer/block_0/Block/resid_post_attn",
    ) in edge_pairs
    assert ("Transformer/block_0/Block/MLP", "Transformer/block_0/Block/resid_out") in edge_pairs
    assert ("Transformer/token_embed", "Transformer/block_0/Block/resid_in") in edge_pairs
    assert ("Transformer/block_0/Block/resid_out", "Transformer") in edge_pairs
    assert hints.plates
    assert "Transformer/block_0/Block/MLP" in hints.plates[0].nodes


def test_infer_backward_flow_render_hints_synthesizes_missing_final_output_edge():
    nodes = (
        "Transformer/token_embed",
        "Transformer/block_0/Block/resid_in",
        "Transformer/block_0/Block/CausalSelfAttention",
        "Transformer/block_0/Block/resid_post_attn",
        "Transformer/block_0/Block/MLP",
        "Transformer/block_0/Block/resid_out",
        "Transformer/final_norm",
    )
    graph = BackwardFlowGraph(
        nodes=nodes,
        edges=(
            BackwardFlowEdge("Transformer/token_embed", "Transformer/block_0/Block/resid_in"),
            BackwardFlowEdge("Transformer/block_0/Block/resid_in", "Transformer/block_0/Block/CausalSelfAttention"),
            BackwardFlowEdge(
                "Transformer/block_0/Block/CausalSelfAttention", "Transformer/block_0/Block/resid_post_attn"
            ),
            BackwardFlowEdge("Transformer/block_0/Block/resid_post_attn", "Transformer/block_0/Block/MLP"),
            BackwardFlowEdge("Transformer/block_0/Block/MLP", "Transformer/block_0/Block/resid_out"),
        ),
    )

    hints = infer_backward_flow_render_hints(graph)

    assert hints.display_edges is not None
    edge_pairs = {(edge.source, edge.target) for edge in hints.display_edges}
    assert ("Transformer/block_0/Block/resid_out", "Transformer/final_norm") in edge_pairs
