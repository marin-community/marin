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

CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom", [LIGHT_BLUE, PURPLE])
CUSTOM_CMAP.set_bad(color="white", alpha=0)


# =========================
# DICTIONARIES 
# =========================

TOKEN_STR_TO_STEPS = {
    "196M": 750,
}

PARAM_STR_TO_COUNT = {
    "150m4k": 154147328 / 1e9, 
    "300m4k": 299649792 / 1e9, 
    "600m4k": 602457088 / 1e9, 
    "1_5b4k": 1540732416 / 1e9, 
}

BASELINE_COLOR = "red"
REGULARIZED_COLOR = PURPLE
ENSEMBLE_COLOR = PINK
SELF_DISTILL_COLOR = VIRIDIAN
DARKER_SELF_DISTILL_COLOR = "#1E5323"
NEON_GREEN = "#8FFF00"
WRAP_COLOR = BLACK
MIX_COLOR = DARK_ORANGE

PARAM_STR_COLOR_DICT = {
    "150m4k": CRAYOLA_BLUE,
    "300m4k": ROBIN_EGG_BLUE,
    "600m4k": APPLE_GREEN,
    "1_5b4k": PINK,
}
SYNTH_DATA_COLOR_DICT = {
    "hq_cpr16": WRAP_COLOR,
    "sd_cpr16": SELF_DISTILL_COLOR,
    "sd_cpr200": DARKER_SELF_DISTILL_COLOR,
    "sdn_c200": NEON_GREEN,
    "symx_c16": MIX_COLOR,
}

SYNTH_DATA_NAME_DICT = {
    "hq_cpr16": "WRAP cpr16",
    "sd_cpr16": "Self-Distill cpr16",
    "sd_cpr200": "Self-Distill cpr200",
    "symx_c16": "WRAP + Self-Distill cpr16",
    "sdn_c200": "Self-Distill (Ens Teacher) cpr200",
}

PRETTY_NAME_DICT = {
    "150m4k": "150M",
    "300m4k": "300M",
    "600m4k": "600M",
    "1_5b4k": "1.5B",
    "hq_cpr16": "WRAP",
    "sd_cpr16": "Self-Distill",
    "sdn_c200": "Self-Distill (Ens Teacher)",
    "symx_c16": "WRAP + Self-Distill",
}

# =========================
# FIGSIZE 
# =========================

WIDE_RECTANGLE_FIGSIZE = (7, 5)
EXTRA_WIDE_RECTANGLE_FIGSIZE = (10, 5)


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