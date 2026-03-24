import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import curve_fit

# =========================
# FONT
# =========================

plt.rcParams.update(
    {
        "font.family": "Palatino",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino",
        "mathtext.it": "Palatino:italic",
        "mathtext.bf": "Palatino:bold",
    }
)

# =========================
# COLORS
# =========================

LIGHT_BLUE = "#8CD9FF"
PURPLE = "#7030A0"
GREEN = "#55A630"
RED = "#e74c3c"
CRAYOLA_BLUE = "#5178D5"
ROBIN_EGG_BLUE = "#45BFC6"
APPLE_GREEN = "#89B537"
ORANGE = "#F8961E"
GOLD = "#C7A020"
PINK = "#FD8BCE"
VIRIDIAN = "#2B8D6E"
BLACK = "#000000"
DARK_ORANGE = "#FF8C00"
REGULARIZED_COLOR = PURPLE

CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom", [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color="white", alpha=0)


# =========================
# DICTIONARIES
# =========================

REAL_DATA_TOKENS = 203_867_748

TOKEN_STR_TO_STEPS = {
    "203M": 777,
}

SYNTH_STREAM_TOKENS: dict[str, int] = {
    # Fill in: data-stream code -> total number of tokens in that synthetic stream
    "w2s": 398_936_730,
    "w2": 398_936_730, 
    "b4": 556_032_327,
    "s4": 556_032_327,
    "b8": 967_660_703,
    "s8": 967_660_703,
    "b16": 1_958_289_599,
    "s16": 1_958_289_599,
    "b32": 3_713_048_352,
    "s32": 3_713_048_352,
    "l16": 1_240_261_938,
}


# TODO: probably switch to using non embedded parameter counts
PARAM_STR_TO_COUNT = {
    "150m4k": 154147328 / 1e9,
    "300m4k": 299649792 / 1e9,
    "600m4k": 602457088 / 1e9,
    "1_5b4k": 1540732416 / 1e9,
}

PARAM_STR_COLOR_DICT = {
    "150m4k": CRAYOLA_BLUE,
    "300m4k": ROBIN_EGG_BLUE,
    "600m4k": APPLE_GREEN,
    "1_5b4k": PINK,
}
DATA_STREAM_COLOR_DICT: dict[str, str] = {
    "sdn": VIRIDIAN,
    # # Stitched streams (warm gradient: dark crimson → amber)
    # "b32": "#7B241C",
    # "b16": "#C0392B",
    # "b8": "#E25822",
    # "b4": "#E67E22",
    # "w2": "#F1C40F",
    # # Simple streams (cool gradient: deep plum → wisteria)
    # "s32": "#4A235A",
    # "s16": "#6C3483",
    # "s8": "#884EA0",
    # "s4": "#A569BD",
    # "w2s": "#BB8FCE",
    # Stitched streams
    "b32": CRAYOLA_BLUE,
    "b16": CRAYOLA_BLUE,
    "b8": CRAYOLA_BLUE,
    "b4": CRAYOLA_BLUE,
    "w2": CRAYOLA_BLUE,
    # Simple streams
    "s32": "#E67E22",
    "s16": "#E67E22",
    "s8": "#E67E22",
    "s4": "#E67E22",
    "w2s": "#E67E22",

    # latent thoughts
    "z2": "#34495E",
    "z4": "#34495E",
    "z8": "#34495E",
    "z16": "#34495E",
    "z32": "#34495E",
    
    # Special
    "w2f": CRAYOLA_BLUE,
    "f8": CRAYOLA_BLUE,
    "n8": CRAYOLA_BLUE,
    "n8s": "#E67E22",
    "l16": "#34495E",
}

# Optional per-stream styling used by bar plots.
# Colors live in DATA_STREAM_COLOR_DICT; this map encodes pattern variants for
# streams that intentionally share a base color.
DATA_STREAM_BAR_STYLE_DICT: dict[str, dict[str, str | float]] = {
    # No-real-data variants.
    "n8": {"hatch": "///", "edgecolor": "white", "linewidth": 0.0, "alpha": 0.95},
    "n8s": {"hatch": "///", "edgecolor": "white", "linewidth": 0.0, "alpha": 0.95},
    # Front-Stitched variant.
    "f8": {"hatch": "xx", "edgecolor": "white", "linewidth": 0.0, "alpha": 0.95},
}

DATA_STREAM_NAMES: dict[str, tuple[str, str | None]] = {
    "b32": ("Stitched Rephrasing", "G=32"),
    "s32": ("Simple Rephrasing", "G=32"),
    "b16": ("Stitched Rephrasing", "G=16"),
    "s16": ("Simple Rephrasing", "G=16"),
    "b8": ("Real Data Last Stitched Rephrasing", "G=8"),
    "s8": ("Simple Rephrasing", "G=8"),
    "b4": ("Stitched Rephrasing", "G=4"),
    "s4": ("Simple Rephrasing", "G=4"),
    "w2": ("Stitched Rephrasing", "G=2"),
    "w2s": ("Simple Rephrasing", "G=2"),
    "w2f": ("Real Data First Stitched Rephrasing", "G=2"),
    "f8": ("Real Data First Stitched Rephrasing", "G=8"),
    "l16": ("Latent Thoughts", "G=16"),
    "sdn": ("Self-Distill", None),
    "n8": ("Stitched Rephrasing, No Real Data", "G=8"),
    "n8s": ("Simple Rephrasing, No Real Data", "G=8"),
    "z2": ("Latent Thoughts", "G=2"),
    "z4": ("Latent Thoughts", "G=4"),
    "z8": ("Latent Thoughts", "G=8"),
    "z16": ("Latent Thoughts", "G=16"),
    "z32": ("Latent Thoughts", "G=32"),
}


def stream_display_name(ds: str) -> str:
    """Full display name with generation count in parentheses, e.g. 'Stitched Rephrasing (G=32)'."""
    entry = DATA_STREAM_NAMES.get(ds)
    if entry is None:
        return ds
    name, gen = entry
    return f"{name} ({gen})" if gen else name


def stream_legend(ds: str, loss: float, fmt: str = ".2f") -> str:
    """Legend label: 'Stitched Rephrasing: 3.45 @ G=32'."""
    entry = DATA_STREAM_NAMES.get(ds)
    if entry is None:
        return f"{ds}: {loss:{fmt}}"
    name, gen = entry
    if gen:
        return f"{name}: {loss:{fmt}} at {gen}"
    return f"{name}: {loss:{fmt}}"

DATA_STREAM_SHORT_NAMES: dict[str, str] = {
    "b32": "Stitched (32x)",
    "s32": "Simple (32x)",
    "b16": "Stitched (16x)",
    "s16": "Simple (16x)",
    "b8": "Real Data Last Stitched (8x)",
    "s8": "Simple (8x)",
    "b4": "Stitched (4x)",
    "s4": "Simple (4x)",
    "w2": "Stitched (2x)",
    "w2s": "Simple (2x)",
    "w2f": "Real Data First Stitched (2x)",
    "f8": "Real Data First Stitched (8x)",
    "l16": "Latent (16x)",
    "sdn": "Self-Distill",
    "n8": "Stitched, No Real (8x)",
    "n8s": "Simple, No Real (8x)",
    "z2": "Latent (2x)",
    "z4": "Latent (4x)",
    "z8": "Latent (8x)",
    "z16": "Latent (16x)",
    "z32": "Latent (32x)",
}

PRETTY_NAME_DICT = {
    "150m4k": "150M",
    "300m4k": "300M",
    "600m4k": "600M",
    "1_5b4k": "1.5B",
}

# =========================
# FIGSIZE
# =========================

ABLATION_FIGSIZE = (5, 4)
SEED_SCALING_FIGSIZE = (5, 5)
MEDIUM_RECTANGLE_FIGSIZE = (6, 5)
WIDE_RECTANGLE_FIGSIZE = (7, 5)
EXTRA_WIDE_RECTANGLE_FIGSIZE = (10, 5)


# =========================
# PLOT STYLE HELPERS
# =========================

def setup_plot_style():
    """Apply the shared rcParams for all plots."""
    plt.rcParams.update(
        {
            "font.family": "Palatino",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Palatino",
            "mathtext.it": "Palatino:italic",
            "mathtext.bf": "Palatino:bold",
        }
    )


def setup_axes(
    ax: plt.Axes,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Configure axes with the shared style used across plots."""
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle=":", alpha=0.3)
    ax.legend(loc="best")


# =========================
# SCALING LAWS
# =========================

class ScalingLaw:
    def __init__(self):
        self.params = None

    def func(self, x, params):
        pass  # implemented by subclass

    def __str__(self):
        pass  # implemented by subclass

    def asymptote(self):
        pass  # implemented by subclass

    def evaluate(self, x):
        return self.func(x, *self.params)

    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf)):
        popt, pcov = curve_fit(self.func, x, y, p0=p0, bounds=bounds)
        self.params = popt


class PowerScalingLaw(ScalingLaw):
    def __init__(self, var_name="x"):
        super().__init__()
        self.var_name = var_name

    def func(self, x, A, B, C):
        return A / (x**B) + C

    def __str__(self):
        return f"{self.params[0]:.2f}/{self.var_name}^{self.params[1]:.2f} + {self.params[2]:.2f}"

    def asymptote(self):
        return self.params[-1]
