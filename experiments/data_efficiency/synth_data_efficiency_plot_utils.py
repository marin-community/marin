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
    # # Sorted streams (warm gradient: dark crimson → amber)
    # "b32": "#7B241C",
    # "b16": "#C0392B",
    # "b8": "#E25822",
    # "b4": "#E67E22",
    # "w2": "#F1C40F",
    # # Shuffled streams (cool gradient: deep plum → wisteria)
    # "s32": "#4A235A",
    # "s16": "#6C3483",
    # "s8": "#884EA0",
    # "s4": "#A569BD",
    # "w2s": "#BB8FCE",
    # Sorted streams
    "b32": CRAYOLA_BLUE,
    "b16": CRAYOLA_BLUE,
    "b8": CRAYOLA_BLUE,
    "b4": CRAYOLA_BLUE,
    "w2": CRAYOLA_BLUE,
    # Shuffled streams
    "s32": "#E67E22",
    "s16": "#E67E22",
    "s8": "#E67E22",
    "s4": "#E67E22",
    "w2s": "#E67E22",
    
    # Special
    "w2f": CRAYOLA_BLUE,
    "n8": "#4A235A",
    "n8s": "#8B3800",
    "l16": "#34495E",
}

DATA_STREAM_NAMES: dict[str, str] = {
    "b32": "Sorted WRAP (32 Rephrases)",
    "s32": "Shuffled WRAP (32 Rephrases)",
    "b16": "Sorted WRAP (16 Rephrases)",
    "s16": "Shuffled WRAP (16 Rephrases)",
    "b8": "Sorted WRAP (8 Rephrases)",
    "s8": "Shuffled WRAP (8 Rephrases)",
    "b4": "Sorted WRAP (4 Rephrases)",
    "s4": "Shuffled WRAP (4 Rephrases)",
    "w2": "Sorted WRAP (2 Rephrases)",
    "w2s": "Shuffled WRAP (2 Rephrases)",
    "w2f": "Front Sorted WRAP (2 Rephrases)",
    "l16": "Latent Thoughts (16 Latents)",
    "sdn": "Self-Distill",
    "n8": "Sorted WRAP, No Real Data (8 Rephrases)",
    "n8s": "Shuffled WRAP, No Real Data (8 Rephrases)",
}

DATA_STREAM_SHORT_NAMES: dict[str, str] = {
    "b32": "Sorted (32x)",
    "s32": "Shuffled (32x)",
    "b16": "Sorted (16x)",
    "s16": "Shuffled (16x)",
    "b8": "Sorted (8x)",
    "s8": "Shuffled (8x)",
    "b4": "Sorted (4x)",
    "s4": "Shuffled (4x)",
    "w2": "Sorted (2x)",
    "w2s": "Shuffled (2x)",
    "w2f": "Front Sorted (2x)",
    "l16": "Latent (16x)",
    "sdn": "Self-Distill",
    "n8": "Sorted, No Real (8x)",
    "n8s": "Shuffled, No Real (8x)",
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
