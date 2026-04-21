# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Literal, Sequence, TypeVar, Union, cast

import jax
from jaxtyping import PyTree

import levanter.tracker
from levanter.analysis.tree_stats import summary_statistics_for_tree
from levanter.callbacks import JitCallback
from levanter.tracker.histogram import Histogram
from levanter.trainer_state import InsideJitInfo, TrainerState
from levanter.utils.tree_utils import key_path_to_str


Target = Literal["grads", "params", "opt_state", "updates"]
M = TypeVar("M", bound=PyTree)
S = TypeVar("S", bound=TrainerState)
VALID_WATCH_TARGETS = {"grads", "params", "opt_state", "updates"}


def _validate_watch_targets(watch_targets: Sequence[str]) -> None:
    invalid_targets = set(watch_targets) - VALID_WATCH_TARGETS
    if invalid_targets:
        raise ValueError(f"Invalid watch targets: {invalid_targets}. Valid targets are: {VALID_WATCH_TARGETS}")


def compute_watch_stats(
    *,
    watch_targets: Sequence[Target],
    include_norms: bool,
    include_per_parameter_norms: bool,
    include_histogram: bool,
    split_scan_layers: bool,
    include_zero_counts: bool = False,
    params: PyTree | None = None,
    grads: PyTree | None = None,
    updates: PyTree | None = None,
    opt_state: PyTree | None = None,
    model_tree_type: type | None = None,
) -> dict[str, jax.Array | Histogram]:
    """Compute watch metrics for selected training targets.

    Args:
        watch_targets: Targets to include, chosen from grads/params/updates/opt_state.
        include_norms: Whether to include norms.
        include_per_parameter_norms: Whether to include per-parameter norms.
        include_histogram: Whether to include histograms.
        split_scan_layers: Whether stacked scan layers are split for logging.
        include_zero_counts: Whether to include zero-count and zero-fraction metrics.
        params: Parameter tree for the ``params`` target.
        grads: Gradient tree for the ``grads`` target.
        updates: Update tree for the ``updates`` target.
        opt_state: Optimizer state tree for the ``opt_state`` target.
        model_tree_type: Optional type used to filter optimizer-state leaves.

    Returns:
        A dictionary of metric keys to scalar arrays or histograms.
    """
    _validate_watch_targets(watch_targets)

    to_log: dict[str, jax.Array | Histogram] = {}
    tree_targets: dict[Target, tuple[str, PyTree | None]] = {
        "grads": ("grad", grads),
        "params": ("params", params),
        "updates": ("updates", updates),
    }

    for target in watch_targets:
        if target in tree_targets:
            prefix, tree = tree_targets[target]
            if tree is None:
                raise ValueError(f"{target} must be provided when watch_targets includes '{target}'")
            stats = summary_statistics_for_tree(
                prefix,
                tree,
                split_scan_layers,
                include_histogram=include_histogram,
                include_norms=include_norms,
                include_per_parameter_norms=include_per_parameter_norms,
                include_zero_counts=include_zero_counts,
            )
            to_log.update(stats)
            continue

        if target == "opt_state":
            if opt_state is None:
                raise ValueError("opt_state must be provided when watch_targets includes 'opt_state'")

            if model_tree_type is None:
                leaves = jax.tree.leaves_with_path(opt_state)
            else:
                leaves = jax.tree.leaves_with_path(opt_state, is_leaf=lambda m: isinstance(m, model_tree_type))

            for path, value in leaves:
                if model_tree_type is not None and not isinstance(value, model_tree_type):
                    continue

                name = key_path_to_str(path)
                name_to_log = f"opt_state/{name}" if name else "opt_state"
                this_stats = summary_statistics_for_tree(
                    name_to_log,
                    value,
                    split_scan_layers,
                    include_histogram=include_histogram,
                    include_norms=include_norms,
                    include_per_parameter_norms=include_per_parameter_norms,
                    include_zero_counts=include_zero_counts,
                )
                to_log.update(this_stats)

    return to_log


@dataclass(frozen=True)
class WatchConfig:
    watch_targets: Union[list[Target], Target] = dataclasses.field(default_factory=lambda: ["grads", "params"])
    """
    What to watch during training. Can be a single target or a list of targets.
    Valid targets are: 'grads', 'params', 'opt_state', 'updates'.
    """
    include_norms: bool = True
    include_per_parameter_norms: bool = True
    include_histograms: bool = False
    include_zero_counts: bool = False
    split_scan_layers: bool = True

    interval: int = 10

    @property
    def is_enabled(self) -> bool:
        return (
            len(self.watch_targets) > 0
            and self.interval > 0
            and (self.include_norms or self.include_histograms or self.include_zero_counts)
        )

    def build(self) -> "WatchCallback":
        return WatchCallback(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histograms,
            include_zero_counts=self.include_zero_counts,
            split_scan_layers=self.split_scan_layers,
        )


class WatchCallback(JitCallback[S, M, dict[str, jax.Array | Histogram]]):
    """
    A unified callback for watching various aspects of training (gradients, parameters, optimizer state, updates).
    This callback combines the functionality of GradWatchCallback, ParamWatchCallback, OptStateWatchCallback,
    and UpdatesWatchCallback into a single callback.

    Args:
        watch_targets (Union[Sequence[str], str]): What to watch. Can be a comma-separated string or list of strings.
            Valid targets are: 'grads', 'params', 'opt_state', 'updates'.

        include_norms (bool): Whether to include norms in the logging.
        include_histogram (bool): Whether to include histograms in the logging.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms.
    """

    def __init__(
        self,
        watch_targets: Union[Sequence[str], str] = ("grads", "params"),
        include_norms: bool = True,
        include_per_parameter_norms: bool = True,
        include_histogram: bool = False,
        include_zero_counts: bool = False,
        split_scan_layers: bool = True,
    ):
        if isinstance(watch_targets, str):
            watch_targets = [t.strip() for t in watch_targets.split(",")]
        else:
            watch_targets = list(watch_targets)

        self.watch_targets = cast(Sequence[Target], watch_targets)
        self.include_norms = include_norms
        self.include_per_parameter_norms = include_per_parameter_norms
        self.include_histogram = include_histogram
        self.include_zero_counts = include_zero_counts
        self.split_scan_layers = split_scan_layers

        # Validate watch targets
        _validate_watch_targets(watch_targets)

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]) -> dict[str, jax.Array | Histogram]:
        return compute_watch_stats(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histogram,
            include_zero_counts=self.include_zero_counts,
            split_scan_layers=self.split_scan_layers,
            params=state.trainable_model,
            grads=inside_info.grads,
            updates=inside_info.updates,
            opt_state=state.opt_state,
            model_tree_type=type(state.model),
        )

    def on_step(self, step_info: S, cb_info: dict[str, jax.Array | Histogram]):
        levanter.tracker.log(cb_info, step=int(step_info.step))
