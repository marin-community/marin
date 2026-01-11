# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: RUF001 RUF002 RUF003
# ↑ Allow Greek letters (α, β) and math symbols (×) in strings/comments for readability.

"""
Meta-analysis of isoflop scaling experiments across multiple compute ranges.

Aggregates results from individual isoflop sweeps and fits scaling laws to understand
how optimal model size and training tokens scale with compute budget. Supports two
fitting strategies:

- **Parabolic**: Fits a log-quadratic curve to each isoflop slice independently,
  finding the minimum via the parabola vertex. Simple but treats each budget in isolation.

- **Parametric** (Chinchilla Approach 3): Fits the full loss surface
  L(N, D) = E + A/N^α + B/D^β across all points for a dataset, then derives optimal
  N* and D* from the constraint C = k·N·D. More principled but requires more data.

Generates a 2x2 figure:
- Top row: Loss vs Parameters, Loss vs Tokens (raw data + fitted curves per budget)
- Bottom row: Optimal N* vs FLOPs, Optimal D* vs FLOPs (power-law scaling fits)

Configure DATASET, FIT_STRATEGY, and filter ranges at the top of the file.
"""

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Reuse existing validation and filtering functions
from experiments.plantcad.exp2101_plantcad_isoflop_analysis import (
    EXPLODED_BUDGETS,
    EXPLODED_RUNS,
    filter_to_finished_runs,
    fit_scaling_law,
)

logger = logging.getLogger(__name__)


class TeeStream:
    """A stream that writes to both the original stream and a file."""

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()


def setup_logging(log_path: Path) -> tuple:
    """Configure logging to write to both console and file.

    Args:
        log_path: Path to the log file.

    Returns:
        Tuple of (log_file, original_stdout, original_stderr) for cleanup.
    """
    # Create parent directory if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_file = open(log_path, "w")

    # Configure root logger with both console and file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Tee stdout and stderr to log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    logger.info(f"Logging to {log_path}")

    return log_file, original_stdout, original_stderr


def cleanup_logging(log_file, original_stdout, original_stderr) -> None:
    """Restore original stdout/stderr and close log file."""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


# Define compute ranges with their metadata
# Format: (version, display_name, steps)
DATASETS = {
    # PlantCAD compute ranges (v2.x):
    # Note: v2.3 (minimal-compute, 2K steps) excluded from analysis
    "plantcad": [
        ("v2.6", "very-low (4K steps)", 4096),
        ("v2.4", "low (8K steps)", 8192),
        ("v2.5", "mid (16K steps)", 16384),
        ("v2.2", "high (32K steps)", 32768),
    ],
    # Text (DCLM baseline) compute ranges:
    "dclm": [
        ("v2.12", "low (16K steps)", 16384),
        ("v2.13", "mid (32K steps)", 32768),
        ("v2.9", "high (64K steps)", 65536),
    ],
}

DATASET = "plantcad"
COMPUTE_RANGES = DATASETS[DATASET]

RESULTS_BASE_PATH = Path("experiments/plantcad/results")
OUTPUT_DIR = RESULTS_BASE_PATH / "meta_analysis"
DEFAULT_ARCH = "qwen"
EXPORT_DPI = 300

# Optional filters for parabola fitting (None = no filter)
# All data is plotted; these only control what's used for fitting
PARAMS_RANGE: tuple[float | None, float | None] = (None, None)  # (min, max)
TOKENS_RANGE: tuple[float | None, float | None] = (None, None)  # (min, max)
BATCH_SIZE_RANGE: tuple[int | None, int | None] = (None, None)  # (min, max), e.g. (8, 64)

# Fit strategy: "parabolic" or "parametric"
FIT_STRATEGY = "parametric"


def _format_number_short(val: float | int) -> str:
    """Format a number using compact SI-like notation.

    Examples: 1e6 -> "1M", 1.5e9 -> "1.5B", 1e12 -> "1T", 8192 -> "8K"
    """
    abs_val = abs(val)
    if abs_val >= 1e12:
        return f"{val / 1e12:.4g}T"
    elif abs_val >= 1e9:
        return f"{val / 1e9:.4g}B"
    elif abs_val >= 1e6:
        return f"{val / 1e6:.4g}M"
    elif abs_val >= 1e3:
        return f"{val / 1e3:.4g}K"
    else:
        return f"{val:.4g}"


def _format_range_filter(prefix: str, range_tuple: tuple[float | int | None, float | int | None]) -> str | None:
    """Format a range filter as shorthand string, or None if no filter active.

    Examples:
        ("N", (1e6, 1e9)) -> "N1M-1B"
        ("D", (None, 1e10)) -> "Dmax10B"
        ("bs", (8, None)) -> "bs8+"
    """
    lo, hi = range_tuple
    if lo is None and hi is None:
        return None

    lo_str = _format_number_short(lo) if lo is not None else None
    hi_str = _format_number_short(hi) if hi is not None else None

    if lo is not None and hi is not None:
        return f"{prefix}{lo_str}-{hi_str}"
    elif lo is not None:
        return f"{prefix}{lo_str}+"
    else:
        return f"{prefix}max{hi_str}"


def generate_output_basename() -> str:
    """Generate output filename base including strategy, dataset, and active filters.

    Format: isoflop_meta_{dataset}_{strategy}[_{filters}]

    Shorthand codes for filters only:
    - Filters: N (params), D (tokens), bs (batch_size)

    Examples:
        - isoflop_meta_plantcad_parabolic (plantcad, parabolic, no filters)
        - isoflop_meta_dclm_parametric_N1M-1B (dclm, parametric, params filter)
        - isoflop_meta_plantcad_parabolic_D10B+_bs8-64 (plantcad, parabolic, tokens and batch_size filters)
    """
    parts = [f"isoflop_meta_{DATASET}_{FIT_STRATEGY}"]

    # Add active filters
    filter_parts = []
    if params_filter := _format_range_filter("N", PARAMS_RANGE):
        filter_parts.append(params_filter)
    if tokens_filter := _format_range_filter("D", TOKENS_RANGE):
        filter_parts.append(tokens_filter)
    if bs_filter := _format_range_filter("bs", BATCH_SIZE_RANGE):
        filter_parts.append(bs_filter)

    if filter_parts:
        parts.append("_".join(filter_parts))

    return "_".join(parts)


# Color scheme for compute ranges (distinct colors for each range)
RANGE_COLORS = {
    # PlantCAD versions
    "v2.3": "#1f77b4",  # Blue - minimal
    "v2.6": "#ff7f0e",  # Orange - very-low
    "v2.4": "#2ca02c",  # Green - low
    "v2.5": "#d62728",  # Red - mid
    "v2.2": "#9467bd",  # Purple - high
    # Text (DCLM baseline) versions
    "v2.12": "#2ca02c",  # Green - low
    "v2.13": "#d62728",  # Red - mid
    "v2.9": "#9467bd",  # Purple - high
}


def load_all_csvs() -> dict[str, pd.DataFrame]:
    """Load all plantcad_isoflops.csv files for v2.x versions."""
    data: dict[str, pd.DataFrame] = {}
    for version, name, steps in COMPUTE_RANGES:
        csv_path = RESULTS_BASE_PATH / version / "plantcad_isoflops.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["version"] = version
            df["compute_range"] = name
            df["range_steps"] = steps
            data[version] = df
            logger.info(f"Loaded {len(df)} runs from {csv_path}")
        else:
            logger.warning(f"CSV not found: {csv_path}")
    return data


def filter_exploded_runs_for_version(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Filter out runs where training exploded for a specific version."""
    # EXPLODED_BUDGETS/RUNS use version keys without "v" prefix (e.g., "2.2" not "v2.2")
    version_key = version.lstrip("v")
    exploded_runs = EXPLODED_RUNS.get(version_key, [])
    exploded_budgets = EXPLODED_BUDGETS.get(version_key, [])

    run_mask = df["run_name"].isin(exploded_runs)
    budget_mask = df["flops_budget"].isin(exploded_budgets)

    n_filtered = run_mask.sum() + budget_mask.sum()
    if n_filtered > 0:
        logger.warning(f"Filtered {n_filtered} exploded runs/budgets for {version}")

    return df[~run_mask & ~budget_mask]


def fit_log_quadratic(x_vals: np.ndarray, loss_vals: np.ndarray) -> tuple[float, np.ndarray]:
    """Fits L = a*(ln x)² + b*(ln x) + c and returns the optimal x."""
    log_x = np.log(x_vals)
    coeffs = np.polyfit(log_x, loss_vals, 2)  # [a, b, c]
    a, b, _ = coeffs
    ln_x_opt = -b / (2 * a)
    return np.exp(ln_x_opt), coeffs


def fit_parabolic(df: pd.DataFrame) -> pd.DataFrame:
    """Fit each curve_id separately with np.polyfit."""
    rows = []
    for curve_id, group in df.groupby("curve_id"):
        if len(group) < 3:
            continue
        opt_N, coeffs_N = fit_log_quadratic(group["params"].values, group["eval_loss"].values)
        opt_D, coeffs_D = fit_log_quadratic(group["tokens"].values, group["eval_loss"].values)
        rows.append(
            {
                "curve_id": curve_id,
                "version": group["version"].iloc[0],
                "budget": group["flops_budget"].iloc[0],
                "opt_N": opt_N,
                "opt_D": opt_D,
                "coeffs_N": coeffs_N,
                "coeffs_D": coeffs_D,
            }
        )
    return pd.DataFrame(rows)


def _compute_huber_scale(L: np.ndarray, scale_factor: float = 2.0) -> float:
    """Compute a data-driven f_scale for Huber loss.

    Uses the median absolute deviation (MAD) of the loss values, which is
    robust to outliers. The scale_factor (default 2.0) determines how many
    MADs a residual can be before being treated as an outlier.

    Args:
        L: Array of loss values.
        scale_factor: Multiplier for MAD (default 2.0 means residuals > 2*MAD are outliers).

    Returns:
        f_scale value for scipy.optimize.least_squares with loss='huber'.
    """
    # MAD = median(|L - median(L)|)
    # For normal data, std ≈ 1.4826 * MAD
    mad = np.median(np.abs(L - np.median(L)))
    return scale_factor * mad


def _grid_search_alpha_beta(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, alpha_range: np.ndarray, beta_range: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Grid search over (α, β) with linear least squares for (E, A, B).

    For fixed α, β, the model L = E + A/N^α + B/D^β is linear in (E, A, B),
    so we can solve it efficiently with linear least squares.

    Returns best (E, A, B, α, β).
    """
    from scipy.optimize import nnls

    best_cost = np.inf
    best_params = (0.0, 0.0, 0.0, 0.0, 0.0)

    for alpha in alpha_range:
        for beta in beta_range:
            # Design matrix: L = E*1 + A*(1/N^α) + B*(1/D^β)
            X = np.column_stack([np.ones_like(N), 1.0 / N**alpha, 1.0 / D**beta])

            # Non-negative least squares to enforce E, A, B >= 0
            coeffs, cost = nnls(X, L)
            E, A, B = coeffs

            if cost < best_cost:
                best_cost = cost
                best_params = (E, A, B, alpha, beta)

    return best_params


def _huber_cost(residuals: np.ndarray, f_scale: float) -> float:
    """Compute Huber cost matching scipy.optimize.least_squares convention."""
    z = (residuals / f_scale) ** 2
    rho = np.where(z <= 1, z, 2 * np.sqrt(z) - 1)
    return 0.5 * np.sum(rho) * f_scale**2


def _check_optimization_progress(
    x0: list[float],
    result,
    residuals_fn,
    param_names: list[str],
    context: str,
    f_scale: float,
    min_rel_improvement: float = 0.0,
) -> None:
    """Log parameter changes and verify the optimization improved the objective.

    Args:
        x0: Initial parameter values.
        result: Result from scipy.optimize.least_squares.
        residuals_fn: The residuals function used in optimization.
        param_names: Names for each parameter (for logging).
        context: Description string for error messages (e.g., version name).
        f_scale: Scale parameter for Huber loss.
        min_rel_improvement: Minimum relative reduction in cost (e.g., 1e-4 = 0.01%).

    Raises:
        RuntimeError: If cost did not decrease by at least min_rel_improvement.
    """
    initial_cost = _huber_cost(residuals_fn(x0), f_scale)
    final_cost = _huber_cost(result.fun, f_scale)
    rel_improvement = (initial_cost - final_cost) / (initial_cost + 1e-12)

    # Log parameter changes
    changes = [f"{name}: {x0[i]:.4g} → {result.x[i]:.4g}" for i, name in enumerate(param_names)]
    logger.info(f"  Parameters: {', '.join(changes)}")
    logger.info(f"  Huber cost: {initial_cost:.6g} → {final_cost:.6g} ({rel_improvement:.2%} reduction)")

    if rel_improvement < min_rel_improvement:
        raise RuntimeError(
            f"Optimization did not improve objective for {context}: "
            f"cost {initial_cost:.6g} → {final_cost:.6g} ({rel_improvement:.2%} < {min_rel_improvement:.2%} required)"
        )


def _check_bound_violations(
    params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    context: str,
    rel_tol: float = 0.01,
) -> None:
    """Check if any parameters are hitting their bounds.

    Args:
        params: Dictionary of parameter name to value.
        bounds: Dictionary of parameter name to (lower, upper) bounds.
        context: Description string for error messages.
        rel_tol: Relative tolerance for detecting bound proximity.

    Raises:
        RuntimeError: If any parameter is near its bounds.
    """
    violations = []
    for name, value in params.items():
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        if np.isfinite(lo) and value < lo + rel_tol * max(abs(lo), 1):
            violations.append(f"{name}={value:.4g} near lower bound {lo}")
        if np.isfinite(hi) and value > hi - rel_tol * max(abs(hi), 1):
            violations.append(f"{name}={value:.4g} near upper bound {hi}")

    if violations:
        raise RuntimeError(
            f"Parameters hitting bounds for {context}: {', '.join(violations)}. "
            "Consider relaxing bounds or investigating data quality."
        )


def fit_parametric(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Chinchilla Approach 3: parametric loss model per experiment.

    Model: L(N, D) = E + A/N^α + B/D^β

    Uses a two-stage approach:
    1. Grid search over (α, β) with linear least squares for (E, A, B)
    2. Refine with nonlinear optimization starting from grid search result

    Then computes per-curve optimal N* and D* using the 6ND compute assumption.

    Returns DataFrame with per-curve results compatible with parabolic output format.
    """
    rows = []

    # Grid for α, β search
    alpha_range = np.linspace(0.05, 1.0, 512)
    beta_range = np.linspace(0.05, 1.0, 512)

    for version, version_group in df.groupby("version"):
        N = version_group["params"].values
        D = version_group["tokens"].values
        L = version_group["eval_loss"].values

        # Log data diagnostics
        mad = np.median(np.abs(L - np.median(L)))
        logger.info(
            f"Fitting {version}: n={len(L)}, loss=[{L.min():.4f}, {L.max():.4f}], " f"std={L.std():.4f}, MAD={mad:.4f}"
        )

        # Stage 1: Grid search for good initial values
        E_init, A_init, B_init, alpha_init, beta_init = _grid_search_alpha_beta(N, D, L, alpha_range, beta_range)
        logger.info(
            f"  Grid search: E={E_init:.4f}, A={A_init:.2f}, B={B_init:.2f}, " f"α={alpha_init:.4f}, β={beta_init:.4f}"
        )

        # Check grid search didn't hit boundaries
        _check_bound_violations(
            {"α_grid": alpha_init, "β_grid": beta_init},
            {"α_grid": (alpha_range[0], alpha_range[-1]), "β_grid": (beta_range[0], beta_range[-1])},
            f"{version} grid search",
        )

        # Stage 2: Refine with nonlinear optimization
        def residuals(params, N=N, D=D, L=L):
            E, log_A, log_B, alpha, beta = params
            A = np.exp(log_A)
            B = np.exp(log_B)
            L_pred = E + A / N**alpha + B / D**beta
            return L_pred - L

        # Use grid search result as initial guess (convert A, B to log space)
        x0 = [E_init, np.log(max(A_init, 1e-10)), np.log(max(B_init, 1e-10)), alpha_init, beta_init]

        # Compute data-driven f_scale for Huber loss
        f_scale = _compute_huber_scale(L)
        logger.info(f"  Huber f_scale={f_scale:.6f} (2×MAD)")

        # Parameter bounds: [E, log_A, log_B, alpha, beta]
        E_bounds = (0.0, np.inf)
        alpha_bounds = (0.01, 2.0)
        beta_bounds = (0.01, 2.0)
        lower_bounds = [E_bounds[0], -np.inf, -np.inf, alpha_bounds[0], beta_bounds[0]]
        upper_bounds = [E_bounds[1], np.inf, np.inf, alpha_bounds[1], beta_bounds[1]]

        # Fit with Huber loss (robust to outliers)
        result = least_squares(
            residuals,
            x0,
            loss="huber",
            f_scale=f_scale,
            bounds=(lower_bounds, upper_bounds),
        )

        # Check convergence diagnostics
        if not result.success:
            raise RuntimeError(
                f"Optimization did not converge for {version}: " f"status={result.status}, message='{result.message}'"
            )
        logger.info(
            f"  Optimization converged: status={result.status}, "
            f"nfev={result.nfev}, optimality={result.optimality:.2e}"
        )

        # Verify optimization actually improved and log changes
        _check_optimization_progress(x0, result, residuals, ["E", "log_A", "log_B", "α", "β"], version, f_scale)

        E, log_A, log_B, alpha, beta = result.x
        A = np.exp(log_A)
        B = np.exp(log_B)

        # Check if any bounded parameters are hitting their bounds
        _check_bound_violations(
            {"E": E, "α": alpha, "β": beta},
            {"E": E_bounds, "α": alpha_bounds, "β": beta_bounds},
            version,
        )

        # Compute fit residuals for diagnostics
        L_pred = E + A / N**alpha + B / D**beta
        residuals_final = L - L_pred
        rmse = np.sqrt(np.mean(residuals_final**2))
        mae = np.mean(np.abs(residuals_final))

        logger.info(f"  Refined fit: E={E:.6f}, A={A:.6f}, B={B:.6f}, " f"α={alpha:.6f}, β={beta:.6f}")
        logger.info(f"  L(N, D) = {E:.6f} + {A:.6f}/N^{alpha:.6f} + {B:.6f}/D^{beta:.6f}")
        logger.info(f"  Fit quality: RMSE={rmse:.6f}, MAE={mae:.6f}, cost={result.cost:.6f}")

        # Compute scaling exponents from C = k*N*D assumption
        # a = β/(α+β), b = α/(α+β)
        a = beta / (alpha + beta)
        b = alpha / (alpha + beta)
        logger.info(f"  Scaling exponents: a={a:.4f} (N* ∝ C^a), b={b:.4f} (D* ∝ C^b)")

        # Compute per-curve optimal N* and D*
        # N* = (αA/βB)^(1/(α+β)) * (C/k)^(β/(α+β))
        # D* = (βB/αA)^(1/(α+β)) * (C/k)^(α/(α+β))
        coeff_N = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
        coeff_D = (beta * B / (alpha * A)) ** (1 / (alpha + beta))

        # Log per-curve k values
        k_values = []
        for curve_id, curve_group in version_group.groupby("curve_id"):
            if len(curve_group) < 3:
                continue

            C = curve_group["flops_budget"].iloc[0]
            N_curve = curve_group["params"].values
            D_curve = curve_group["tokens"].values

            # Compute k via log-space regression: log(C) = log(k) + log(N) + log(D)
            # => k = C / geometric_mean(N * D)
            log_ND = np.log(N_curve * D_curve)
            k = C / np.exp(log_ND.mean())
            k_values.append(k)

            opt_N = coeff_N * (C / k) ** a
            opt_D = coeff_D * (C / k) ** b

            # Store parametric params as coeffs for plotting
            # Format: (E, A, B, alpha, beta, C, k)
            parametric_coeffs = (E, A, B, alpha, beta, C, k)

            rows.append(
                {
                    "curve_id": curve_id,
                    "version": version,
                    "budget": C,
                    "opt_N": opt_N,
                    "opt_D": opt_D,
                    "coeffs_N": parametric_coeffs,
                    "coeffs_D": parametric_coeffs,
                }
            )

        # Log k statistics for this version
        if k_values:
            k_arr = np.array(k_values)
            logger.info(
                f"  FLOPs coefficient k: mean={k_arr.mean():.2f}, "
                f"range=[{k_arr.min():.2f}, {k_arr.max():.2f}] (C = k·N·D)"
            )

    return pd.DataFrame(rows)


def fit_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Route to appropriate fitting function based on FIT_STRATEGY."""
    if FIT_STRATEGY == "parabolic":
        return fit_parabolic(df)
    elif FIT_STRATEGY == "parametric":
        return fit_parametric(df)
    else:
        raise ValueError(f"Unknown FIT_STRATEGY: {FIT_STRATEGY}")


def iter_filtered_data(
    data: dict[str, pd.DataFrame],
) -> Iterator[tuple[str, str, int, pd.DataFrame]]:
    """Iterate over compute ranges, yielding filtered dataframes.

    Applies standard filtering (exploded runs, finished runs) to each version's data.

    Yields:
        Tuples of (version, display_name, steps, filtered_df).
    """
    for version, name, steps in COMPUTE_RANGES:
        if version not in data:
            continue
        df = data[version].copy()
        df = filter_exploded_runs_for_version(df, version)
        df = filter_to_finished_runs(df)
        yield version, name, steps, df


def print_budget_summary(data: dict[str, pd.DataFrame]) -> None:
    """Print a sorted list of compute range, version, and flop budgets."""
    rows: list[tuple[str, str, float]] = []

    for version, name, _steps, df in iter_filtered_data(data):
        for budget in sorted(df["flops_budget"].unique()):
            rows.append((name, version, budget))

    # Sort by compute range name (which includes step count), then by budget
    rows.sort(key=lambda x: (x[0], x[2]))

    logger.info("\n" + "=" * 60)
    logger.info("Compute Range / Version / FLOPs Budget Summary")
    logger.info("=" * 60)
    logger.info(f"{'Compute Range':<20} {'Version':<10} {'FLOPs Budget':<15}")
    logger.info("-" * 60)
    for name, version, budget in rows:
        logger.info(f"{name:<20} {version:<10} {budget:.2e}")
    logger.info("=" * 60 + "\n")


def apply_fit_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PARAMS_RANGE, TOKENS_RANGE, and BATCH_SIZE_RANGE filters for fitting."""
    filtered = df.copy()
    if PARAMS_RANGE[0] is not None:
        filtered = filtered[filtered["params"] >= PARAMS_RANGE[0]]
    if PARAMS_RANGE[1] is not None:
        filtered = filtered[filtered["params"] <= PARAMS_RANGE[1]]
    if TOKENS_RANGE[0] is not None:
        filtered = filtered[filtered["tokens"] >= TOKENS_RANGE[0]]
    if TOKENS_RANGE[1] is not None:
        filtered = filtered[filtered["tokens"] <= TOKENS_RANGE[1]]
    if BATCH_SIZE_RANGE[0] is not None:
        filtered = filtered[filtered["batch_size"] >= BATCH_SIZE_RANGE[0]]
    if BATCH_SIZE_RANGE[1] is not None:
        filtered = filtered[filtered["batch_size"] <= BATCH_SIZE_RANGE[1]]
    return filtered


def prepare_combined_data(data: dict[str, pd.DataFrame], architecture: str = DEFAULT_ARCH) -> pd.DataFrame:
    """Combine all versions into one DataFrame, filter, add curve_id and budget_index."""
    dfs = []
    for _version, name, _steps, df in iter_filtered_data(data):
        df = df[(df["architecture"] == architecture) & (df["epochs"] == 1)].dropna(
            subset=["eval_loss", "tokens", "params", "flops_budget"]
        )
        if df.empty:
            logger.warning(f"No valid data for {name}")
            continue
        # Compute budget_index within each version (0 = smallest budget, 1 = next, etc.)
        sorted_budgets = sorted(df["flops_budget"].unique())
        budget_to_index = {b: i for i, b in enumerate(sorted_budgets)}
        df = df.copy()
        df["budget_index"] = df["flops_budget"].map(budget_to_index)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined["curve_id"] = combined["version"] + "_" + combined["flops_budget"].astype(str)
    return combined


def prepare_data(
    data: dict[str, pd.DataFrame], architecture: str = DEFAULT_ARCH
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare data for plotting. Returns dict[version -> (data_df, analysis_df)]."""
    combined = prepare_combined_data(data, architecture)
    if combined.empty:
        return {}

    filtered = apply_fit_filters(combined)
    analysis = fit_curves(filtered)

    # Split back per-version, adding group_data for plotting compatibility
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for version in combined["version"].unique():
        version_data = combined[combined["version"] == version]
        version_analysis = analysis[analysis["version"] == version].copy()
        version_analysis["group_data"] = version_analysis["curve_id"].apply(
            lambda cid: combined[combined["curve_id"] == cid]
        )
        results[version] = (version_data, version_analysis)
    return results


def plot_loss_vs_variable(
    ax,
    range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    variable: str,
):
    """Plot Loss vs a variable (params or tokens) with all compute ranges overlaid.

    Args:
        ax: Matplotlib axes to plot on.
        range_data: Dictionary mapping version to (dataframe, analysis) tuples.
        variable: Either "params" or "tokens".
    """
    if variable == "params":
        col_name = "params"
        coeffs_key = "coeffs_N"
        opt_key = "opt_N"
        xlabel = "Parameters (N)"
        title = "Loss vs Parameters"
        fit_range = PARAMS_RANGE
    else:
        col_name = "tokens"
        coeffs_key = "coeffs_D"
        opt_key = "opt_D"
        xlabel = "Tokens (D)"
        title = "Loss vs Tokens"
        fit_range = TOKENS_RANGE

    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        df, analysis = range_data[version]
        color = RANGE_COLORS[version]

        # Plot all data points for this range (more transparent)
        ax.scatter(df[col_name], df["eval_loss"], color=color, alpha=0.25, s=15, label=name)

        # Plot fits and optimum points for each budget
        for _, row in analysis.iterrows():
            group = row["group_data"]
            val_min, val_max = group[col_name].min(), group[col_name].max()

            # Draw fit curve within actual data range
            x_range = np.logspace(np.log10(val_min), np.log10(val_max), 100)

            if FIT_STRATEGY == "parabolic":
                # Parabolic: L = a*(ln x)² + b*(ln x) + c
                L_pred = np.polyval(row[coeffs_key], np.log(x_range))
            elif FIT_STRATEGY == "parametric":
                # Parametric: L(N, D) = E + A/N^α + B/D^β with constraint D = C/(kN)
                # k is computed per-curve from actual data
                E, A, B, alpha, beta, C, k = row[coeffs_key]
                if variable == "params":
                    # L(N) with D = C/(kN)
                    L_pred = E + A / x_range**alpha + B * (k * x_range / C) ** beta
                else:
                    # L(D) with N = C/(kD)
                    L_pred = E + A * (k * x_range / C) ** alpha + B / x_range**beta
            else:
                raise ValueError(f"Unknown FIT_STRATEGY: {FIT_STRATEGY}")

            ax.plot(x_range, L_pred, color=color, linestyle="--", alpha=0.5, linewidth=1)

            # Plot optimum point
            if pd.notna(row[opt_key]):
                if FIT_STRATEGY == "parabolic":
                    L_min = np.polyval(row[coeffs_key], np.log(row[opt_key]))
                elif FIT_STRATEGY == "parametric":
                    E, A, B, alpha, beta, C, k = row[coeffs_key]
                    if variable == "params":
                        L_min = E + A / row[opt_key] ** alpha + B * (k * row[opt_key] / C) ** beta
                    else:
                        L_min = E + A * (k * row[opt_key] / C) ** alpha + B / row[opt_key] ** beta
                ax.scatter([row[opt_key]], [L_min], color=color, marker="s", s=60, edgecolors="black", zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.5)

    # Draw vertical lines at filter boundaries
    if fit_range[0] is not None:
        ax.axvline(fit_range[0], color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    if fit_range[1] is not None:
        ax.axvline(fit_range[1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7)


def plot_optimal_params_vs_flops(ax, range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    """Plot Optimal Parameters vs Compute with scaling laws per range."""
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        _, analysis = range_data[version]
        color = RANGE_COLORS[version]

        valid_data = analysis.dropna(subset=["opt_N", "opt_D"])
        if len(valid_data) < 2:
            continue

        budgets = valid_data["budget"].values
        opt_N = valid_data["opt_N"].values

        # Fit and plot scaling law
        m_N, _c_N, B_smooth, N_smooth = fit_scaling_law(budgets, opt_N)

        ax.scatter(budgets, opt_N, color=color, marker="s", s=60, edgecolors="black", zorder=5)
        ax.plot(
            B_smooth, N_smooth, color=color, linestyle="--", alpha=0.7, label=f"{name}: $N^* \\propto C^{{{m_N:.2f}}}$"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget (FLOPs)")
    ax.set_ylabel("Optimal Parameters (N*)")
    ax.set_title("Optimal Parameters vs Compute")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.5)


def plot_optimal_tokens_vs_flops(ax, range_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    """Plot Optimal Tokens vs Compute with scaling laws per range."""
    for version, name, _ in COMPUTE_RANGES:
        if version not in range_data:
            continue

        _, analysis = range_data[version]
        color = RANGE_COLORS[version]

        valid_data = analysis.dropna(subset=["opt_N", "opt_D"])
        if len(valid_data) < 2:
            continue

        budgets = valid_data["budget"].values
        opt_D = valid_data["opt_D"].values

        # Fit and plot scaling law
        m_D, _c_D, B_smooth, D_smooth = fit_scaling_law(budgets, opt_D)

        ax.scatter(budgets, opt_D, color=color, marker="s", s=60, edgecolors="black", zorder=5)
        ax.plot(
            B_smooth, D_smooth, color=color, linestyle="--", alpha=0.7, label=f"{name}: $D^* \\propto C^{{{m_D:.2f}}}$"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget (FLOPs)")
    ax.set_ylabel("Optimal Tokens (D*)")
    ax.set_title("Optimal Tokens vs Compute")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.5)


def create_combined_figure(data: dict[str, pd.DataFrame], architecture: str = DEFAULT_ARCH):
    """Create a 4-facet figure showing isoflop analysis grouped by compute range.

    Layout (2x2):
    - Top-left: Loss vs Parameters
    - Top-right: Loss vs Tokens
    - Bottom-left: Optimal Params vs FLOPs
    - Bottom-right: Optimal Tokens vs FLOPs
    """
    range_data = prepare_data(data, architecture)

    if not range_data:
        raise ValueError("No valid data found for any compute range")

    logger.info(f"Prepared data for {len(range_data)} compute ranges")

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: Loss curves
    plot_loss_vs_variable(axes[0, 0], range_data, "params")
    plot_loss_vs_variable(axes[0, 1], range_data, "tokens")

    # Bottom row: Scaling laws
    plot_optimal_params_vs_flops(axes[1, 0], range_data)
    plot_optimal_tokens_vs_flops(axes[1, 1], range_data)

    plt.suptitle("PlantCAD Isoflop Analysis: Bias by Compute Range (Step Count)", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def save_combined_data(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save all combined data to CSV."""
    all_dfs = [df for _version, _name, _steps, df in iter_filtered_data(data)]

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} rows to {output_path}")


def main():
    """Main entry point."""
    # Generate output filename with config info (needed for log file path)
    output_base = generate_output_basename()

    # Create output directory and set up logging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"{output_base}.log"
    log_file, original_stdout, original_stderr = setup_logging(log_path)

    try:
        logger.info(f"Output basename: {output_base}")
        logger.info("Loading CSV files...")
        data = load_all_csvs()

        if not data:
            raise ValueError("No CSV files found")

        # Print sorted budget summary
        print_budget_summary(data)

        logger.info(f"Creating combined figure from {len(data)} compute ranges...")
        fig = create_combined_figure(data)

        # Save figure as PNG and PDF
        png_path = OUTPUT_DIR / f"{output_base}.png"
        pdf_path = OUTPUT_DIR / f"{output_base}.pdf"
        csv_path = OUTPUT_DIR / f"{output_base}.csv"

        fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
        logger.info(f"Saved figure to {png_path}")

        fig.savefig(pdf_path, dpi=EXPORT_DPI, bbox_inches="tight")
        logger.info(f"Saved figure to {pdf_path}")

        plt.close(fig)

        # Save combined data as CSV
        save_combined_data(data, csv_path)

    finally:
        cleanup_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
