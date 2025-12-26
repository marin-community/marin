# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeVar, Union

import jax
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from jaxtyping import PyTree

import levanter.tracker
from levanter.analysis.tree_stats import summary_statistics_for_tree
from levanter.callbacks import JitCallback
from levanter.tracker.histogram import Histogram
from levanter.trainer_state import InsideJitInfo, TrainerState


Target = Literal["grads", "params", "opt_state", "updates", "adam_mu", "adam_nu"]
M = TypeVar("M", bound=PyTree)
S = TypeVar("S", bound=TrainerState)


@dataclass(frozen=True)
class WatchConfig:
    watch_targets: Union[list[Target], Target] = dataclasses.field(default_factory=lambda: ["grads", "params"])
    """
    What to watch during training. Can be a single target or a list of targets.
    Valid targets are: 'grads', 'params', 'opt_state', 'updates', 'adam_mu', 'adam_nu'.
    """
    include_norms: bool = True
    include_per_parameter_norms: bool = True
    include_histograms: bool = False
    split_scan_layers: bool = True

    interval: int = 10

    @property
    def is_enabled(self) -> bool:
        return len(self.watch_targets) > 0 and self.interval > 0 and (self.include_norms or self.include_histograms)

    def build(self) -> "WatchCallback":
        return WatchCallback(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histograms,
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
        split_scan_layers: bool = True,
    ):
        if isinstance(watch_targets, str):
            watch_targets = [t.strip() for t in watch_targets.split(",")]

        self.watch_targets = watch_targets
        self.include_norms = include_norms
        self.include_per_parameter_norms = include_per_parameter_norms
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

        # Validate watch targets
        valid_targets = {"grads", "params", "opt_state", "updates", "adam_mu", "adam_nu"}
        invalid_targets = set(watch_targets) - valid_targets
        if invalid_targets:
            raise ValueError(f"Invalid watch targets: {invalid_targets}. Valid targets are: {valid_targets}")

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]) -> dict[str, jax.Array | Histogram]:
        to_log = {}

        #print(f"*** DEBUG: WatchCallback processing targets: {self.watch_targets}")

        for target in self.watch_targets:
            #print(f"*** DEBUG: Processing target: {target}")
            if target == "grads":
                stats = summary_statistics_for_tree(
                    "grad",
                    inside_info.grads,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "params":
                stats = summary_statistics_for_tree(
                    "params",
                    state.trainable_model,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "updates":
                stats = summary_statistics_for_tree(
                    "updates",
                    inside_info.updates,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "opt_state":
                # Special handling for optimizer state
                leaves = jax.tree.leaves_with_path(state.opt_state, is_leaf=lambda m: isinstance(m, type(state.model)))
                for path, v in leaves:
                    if not isinstance(v, type(state.model)):
                        continue

                    name = self._munge_key_name(path)
                    name_to_log = f"opt_state/{name}" if name else "opt_state"
                    this_stats = summary_statistics_for_tree(
                        name_to_log,
                        v,
                        self.split_scan_layers,
                        include_histogram=self.include_histogram,
                        include_norms=self.include_norms,
                        include_per_parameter_norms=self.include_per_parameter_norms,
                    )
                    to_log.update(this_stats)

            elif target == "adam_mu":
                #print(f"*** DEBUG: Attempting to extract adam_mu")
                # Extract Adam momentum (mu) from optimizer state
                adam_mu = self._extract_adam_component(state.opt_state, "mu")
                #print(f"*** DEBUG: adam_mu extraction result: {adam_mu is not None}")
                if adam_mu is not None:
                    #print(f"*** DEBUG: Computing stats for adam_mu")
                    stats = summary_statistics_for_tree(
                        "adam_mu",
                        adam_mu,
                        self.split_scan_layers,
                        include_histogram=self.include_histogram,
                        include_norms=self.include_norms,
                        include_per_parameter_norms=self.include_per_parameter_norms,
                    )
                    #print(f"*** DEBUG: adam_mu stats keys: {list(stats.keys())}")
                    to_log.update(stats)
                else:
                    print(f"*** DEBUG: adam_mu extraction failed, no stats logged")

            elif target == "adam_nu":
                #print(f"*** DEBUG: Attempting to extract adam_nu")
                # Extract Adam velocity (nu) from optimizer state
                adam_nu = self._extract_adam_component(state.opt_state, "nu")
                #print(f"*** DEBUG: adam_nu extraction result: {adam_nu is not None}")
                if adam_nu is not None:
                    #print(f"*** DEBUG: Computing stats for adam_nu")
                    stats = summary_statistics_for_tree(
                        "adam_nu",
                        adam_nu,
                        self.split_scan_layers,
                        include_histogram=self.include_histogram,
                        include_norms=self.include_norms,
                        include_per_parameter_norms=self.include_per_parameter_norms,
                    )
                    #print(f"*** DEBUG: adam_nu stats keys: {list(stats.keys())}")
                    to_log.update(stats)
                else:
                    print(f"*** DEBUG: adam_nu extraction failed, no stats logged")

        return to_log

    def on_step(self, step_info: S, cb_info: dict[str, jax.Array | Histogram]):
        levanter.tracker.log(cb_info, step=int(step_info.step))

    def _extract_adam_component(self, opt_state: Any, component: str) -> Any:
        """Extract Adam mu or nu component from optimizer state."""
        #   print(f"*** DEBUG: Attempting to extract '{component}' from optimizer state")
        #print(f"*** DEBUG: opt_state type: {type(opt_state)}")
        #print(f"*** DEBUG: opt_state has inner_state: {hasattr(opt_state, 'inner_state')}")

        #if hasattr(opt_state, 'inner_state'):
            #print(f"*** DEBUG: inner_state length: {len(opt_state.inner_state)}")
            #if len(opt_state.inner_state) > 0:
                #print(f"*** DEBUG: inner_state[0] type: {type(opt_state.inner_state[0])}")
                #print(f"*** DEBUG: inner_state[0] attributes: {dir(opt_state.inner_state[0])}")

        try:
            # Try to access Adam state through common patterns
            # Pattern 1: opt_state.inner_state[0].{mu|nu} (for scale_by_adam)
            if hasattr(opt_state, 'inner_state') and len(opt_state.inner_state) > 0:
                adam_state = opt_state.inner_state[0]
                #print(f"*** DEBUG: Pattern 1 - adam_state has {component}: {hasattr(adam_state, component)}")
                if hasattr(adam_state, component):
                    result = getattr(adam_state, component)
                    #print(f"*** DEBUG: Pattern 1 SUCCESS - extracted {component}, type: {type(result)}")
                    return result

            # Pattern 2: Direct access opt_state[0].{mu|nu}
            if isinstance(opt_state, (list, tuple)) and len(opt_state) > 0:
                adam_state = opt_state[0]
                #print(f"*** DEBUG: Pattern 2 - adam_state has {component}: {hasattr(adam_state, component)}")
                if hasattr(adam_state, component):
                    result = getattr(adam_state, component)
                    #print(f"*** DEBUG: Pattern 2 SUCCESS - extracted {component}, type: {type(result)}")
                    return result

            # Pattern 3: Try to find in the full optimizer state tree
            #print(f"*** DEBUG: Pattern 3 - Searching tree for {component}")
            leaves_with_paths = jax.tree.leaves_with_path(opt_state)
            for path, leaf in leaves_with_paths:
                if any(component in str(key) for key in path):
                    #       print(f"*** DEBUG: Pattern 3 - Found {component} in path: {path}")
                    # Check if this leaf has the same structure as the model
                    if hasattr(leaf, 'shape') or (hasattr(leaf, '__len__') and not isinstance(leaf, str)):
                        #print(f"*** DEBUG: Pattern 3 SUCCESS - extracted {component}, type: {type(leaf)}")
                        return leaf

            #print(f"*** DEBUG: All patterns failed to extract {component}")
            return None
        except (AttributeError, IndexError, TypeError) as e:
            #print(f"*** Failed to extract {component} from optimizer state: {e}")
            return None

    # Optimizer states can have weird/arbitrary structures, but the states we care about
    # are PyTrees with the same class as our model parameters (e.g., NamedArray, jax arrays, etc.)
    # opt_state/inner_state/1/nu/ --> we want nu
    def _munge_key_name(self, path: Sequence[Any]) -> str:
        """Helper method to format optimizer state keys."""
        if not path:
            return ""
        path_elem = path[-1]
        if isinstance(path_elem, SequenceKey):  # type: ignore
            out = f"{path_elem.idx}"
        elif isinstance(path_elem, DictKey):  # type: ignore
            out = f"{path_elem.key}"
        elif isinstance(path_elem, GetAttrKey):  # type: ignore
            out = str(path_elem)
        elif isinstance(path_elem, FlattenedIndexKey):  # type: ignore
            out = f"{path_elem.idx}"
        else:
            path_elem = str(path_elem)
            out = f"{path_elem}"

        if out.startswith("."):
            out = out[1:]

        return out
