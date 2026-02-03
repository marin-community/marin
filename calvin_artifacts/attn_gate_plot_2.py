# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "plotly",
#     "numpy",
#     "scipy",
# ]
# ///

import json
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# 1. Styling & Helper Functions
# -----------------------------------------------------------------------------

# Base Palette
# Standard (Orange), Sink (Blue), Gated Naive (Green)
BASE_PALETTE = [
    "#d95f02",  # Standard
    "#1f78b4",  # Sink
    "#1b9e77",  # Gated (Naive)
]

# Additional Palette for LR Sweep
LR_COLORS = [
    "#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17"
]

MARKERS = ["square", "circle", "triangle-up", "diamond", "cross", "x", "pentagon", "star", "hexagram", "star-square"]

def enable_vertical_gridlines(fig: go.Figure, *, log_dtick: float = 1.0) -> None:
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    def _set_dtick(axis):
        if getattr(axis, "type", None) == "log":
            axis.update(dtick=log_dtick)
    fig.for_each_xaxis(_set_dtick)

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------

BASE_DIR_STARTER = Path.home() / "Projects/Work/Marin/marin/experiments/speedrun/hackable_transformer_starter"
BASE_DIR_GATE = Path.home() / "Projects/Work/Marin/marin-gated-attention/experiments/speedrun/hackable_transformer_attn_gate"

def load_result_file(filepath: Path) -> tuple[float, float]:
    """Load a speedrun results JSON file and return (training_hardware_flops, bpb)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    run_info = data["runs"][0]["run_info"]
    flops = run_info["training_hardware_flops"]
    bpb = run_info["eval/paloma/c4_en/bpb"]
    return flops, bpb

def load_sweep_data(base_path: Path, model_sizes: list[str], use_v5p64_for_1_2b: bool = False, result_filename: str = "speedrun_results.json") -> list[tuple[float, float, str]]:
    """Load data for a sweep across multiple model sizes.

    Args:
        base_path: Base directory path for the sweep
        model_sizes: List of model size directories (e.g., ["130m", "300m", "520m", "1_2b"])
        use_v5p64_for_1_2b: If True, use speedrun_results_v5p64.json for 1_2b models
        result_filename: Name of the result file to load (default: "speedrun_results.json")
    """
    data = []
    for size in model_sizes:
        size_dir = base_path / size
        if use_v5p64_for_1_2b and (size == "1_2b" or size == "1.2b"):
            result_file = size_dir / "speedrun_results_v5p64.json"
        else:
            result_file = size_dir / result_filename

        if not result_file.exists():
            print(f"Warning: {result_file} does not exist, skipping", file=sys.stderr)
            continue

        flops, bpb = load_result_file(result_file)
        # Format size for display
        display_size = size.replace("_", ".").upper()
        data.append((flops, bpb, display_size))

    return data

# Model sizes in order
MODEL_SIZES = ["130m", "300m", "520m", "1_2b"]

# Standard Attention
std_path = BASE_DIR_STARTER / "std_attn"
# Note: standard attention uses "1.2b" not "1_2b"
std_sizes = ["130m", "300m", "520m", "1.2b"]
data_std = load_sweep_data(std_path, std_sizes)

# Attention Sink (default_x2 variant)
sink_path = BASE_DIR_STARTER / "attnsink/splash/default_x2"
data_sink = load_sweep_data(sink_path, MODEL_SIZES)

# Gated Attention (Naive / No Fuse Q)
naive_path = BASE_DIR_GATE / "naive_no_fuse_q"
data_gated_naive = load_sweep_data(naive_path, MODEL_SIZES)

# LR Sweep Data
lr_sweeps = [
    ("lr_x1", "1"),
    ("lr_x1_5", "1.5"),
    ("lr_x2", "2"),
    ("lr_x2_5", "2.5"),
    ("lr_x3", "3"),
    ("lr_x3_5", "3.5"),
    ("lr_x4", "4"),
]

data_lr_sweeps = {}
for lr_dir, lr_mult in lr_sweeps:
    lr_path = BASE_DIR_GATE / lr_dir
    data = load_sweep_data(lr_path, MODEL_SIZES, use_v5p64_for_1_2b=False)
    data_lr_sweeps[lr_dir] = data

# Separate Gate Attention Data (LR x2 and LR x2.5 with separate gate)
lr_x2_sep_path = BASE_DIR_GATE / "lr_x2"
data_lr_x2_sep = load_sweep_data(lr_x2_sep_path, MODEL_SIZES, result_filename="speedrun_results-sep.json")

lr_x2_5_sep_path = BASE_DIR_GATE / "lr_x2_5"
data_lr_x2_5_sep = load_sweep_data(lr_x2_5_sep_path, MODEL_SIZES, result_filename="speedrun_results-sep.json")

# Additional colors for separate gate attention variants
SEP_COLORS = ["#E31A1C", "#6A3D9A"]  # Red, Purple

# Define all sweeps
sweeps = [
    {"name": "Standard Attention", "data": data_std, "color": BASE_PALETTE[0], "marker": MARKERS[0]},
    {"name": "Attention Sink", "data": data_sink, "color": BASE_PALETTE[1], "marker": MARKERS[1]},
    {"name": "Gated Attn (LR x1, no fuse Q)", "data": data_gated_naive, "color": BASE_PALETTE[2], "marker": MARKERS[2]},

    {"name": "Gated Attn (LR x1)", "data": data_lr_sweeps["lr_x1"], "color": LR_COLORS[0], "marker": MARKERS[3]},
    {"name": "Gated Attn (LR x1.5)", "data": data_lr_sweeps["lr_x1_5"], "color": LR_COLORS[1], "marker": MARKERS[4]},
    {"name": "Gated Attn (LR x2)", "data": data_lr_sweeps["lr_x2"], "color": LR_COLORS[2], "marker": MARKERS[5]},
    {"name": "Gated Attn (LR x2.5)", "data": data_lr_sweeps["lr_x2_5"], "color": LR_COLORS[3], "marker": MARKERS[6]},
    {"name": "Gated Attn (LR x3)", "data": data_lr_sweeps["lr_x3"], "color": LR_COLORS[4], "marker": MARKERS[7]},
    {"name": "Gated Attn (LR x3.5)", "data": data_lr_sweeps["lr_x3_5"], "color": LR_COLORS[5], "marker": MARKERS[8]},
    {"name": "Gated Attn (LR x4)", "data": data_lr_sweeps["lr_x4"], "color": LR_COLORS[6], "marker": MARKERS[9]},

    # Separate gate attention variants
    {"name": "Gated Attn Sep (LR x2)", "data": data_lr_x2_sep, "color": SEP_COLORS[0], "marker": "star"},
    {"name": "Gated Attn Sep (LR x2.5)", "data": data_lr_x2_5_sep, "color": SEP_COLORS[1], "marker": "hexagon"},
]

# -----------------------------------------------------------------------------
# 3. Print Data to stdout
# -----------------------------------------------------------------------------

def print_data():
    """Print all loaded data to stdout."""
    print("=" * 80)
    print("Loaded Data from Speedrun Results")
    print("=" * 80)
    print()
    
    for sweep in sweeps:
        print(f"{sweep['name']}:")
        for flops, bpb, size in sorted(sweep["data"], key=lambda x: x[0]):
            print(f"  {size:6s}: FLOPs = {flops:.6e}, BPB = {bpb:.10f}")
        print()

# -----------------------------------------------------------------------------
# 4. Plotting Logic
# -----------------------------------------------------------------------------

def main():
    # Print data first
    print_data()
    
    fig = go.Figure()
    
    # Projection target
    target_flops = 1e22
    projection_texts = []

    for sweep in sweeps:
        # Sort by FLOPs
        raw_data = sorted(sweep["data"], key=lambda x: x[0])
        if not raw_data:
            continue
        
        flops = np.array([x[0] for x in raw_data])
        bpb = np.array([x[1] for x in raw_data])
        
        name = sweep["name"]
        color = sweep["color"]
        marker = sweep["marker"]

        # 1. Calculate Log-Linear Fit (BPB = slope * log10(FLOPs) + intercept)
        log_flops = np.log10(flops)
        slope, intercept, r_value, p_value, std_err = linregress(log_flops, bpb)

        # Generate regression line points
        fit_x_flops = np.logspace(np.log10(flops[0]), np.log10(target_flops), 100)
        fit_y_bpb = slope * np.log10(fit_x_flops) + intercept

        # Projected value at exactly 1e22
        proj_bpb = slope * np.log10(target_flops) + intercept
        projection_texts.append(f"<b>{name}</b>: {proj_bpb:.4f}")

        # 2. Plot Regression Line
        fig.add_trace(
            go.Scatter(
                x=fit_x_flops,
                y=fit_y_bpb,
                mode="lines",
                name=f"{name} Fit",
                line=dict(color=color, dash="dash", width=1.5),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.6,
            )
        )

        # 3. Plot Actual Data Points
        fig.add_trace(
            go.Scatter(
                x=flops,
                y=bpb,
                mode="markers",
                name=name,
                marker=dict(symbol=marker, color=color, size=8, opacity=0.8),
                hovertemplate=f"<b>{name}</b><br>BPB: %{{y:.4f}}<br>FLOPs: %{{x:.2e}}<extra></extra>",
            )
        )

    # 4. Add Projection Summary Annotation
    col = "<br>".join(projection_texts)
    
    proj_box_text = f"<b>Proj. BPB @ 1e22:</b><br>{col}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        text=proj_box_text,
        showarrow=False,
        font=dict(size=10, color="#333333"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#cccccc",
        borderwidth=1,
    )

    # 5. Add Projection Vertical Line
    fig.add_vline(x=target_flops, line_width=1, line_dash="dot", line_color="#888888")

    # -------------------------------------------------------------------------
    # 5. Layout Configuration
    # -------------------------------------------------------------------------
    
    fig.update_layout(
        title={
            'text': "Scaling Laws: Standard vs Sink vs Gated Attention (LR Sweep)",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Training Hardware FLOPs",
        yaxis_title="C4 EN BPB (Lower is Better)",
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e5e5",
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(t=60, l=60, r=40, b=60),
        height=800, # Increased height for legibility
    )

    # Set X axis to Log Scale
    fig.update_xaxes(type="log")
    
    # Enable nice gridlines
    enable_vertical_gridlines(fig)
    
    # Adjust Y axis range
    fig.update_yaxes(range=[0.75, 1.20])

    # Save to HTML file and show the interactive plot
    output_path = Path(__file__).parent / "attn_gate_plot_2.html"
    fig.write_html(str(output_path))
    print(f"\nPlot saved to: {output_path}")
    fig.show()

if __name__ == "__main__":
    main()
