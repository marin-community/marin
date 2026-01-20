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

"""Configuration classes for n-domain, n-phase mixture experiments.

This module provides flexible configuration primitives for defining:
- Data domains (groups of related datasets with shared weighting)
- Training phases (time segments with different mixture weights)
- Experiment configurations (combining domains, phases, and model settings)
"""

from dataclasses import dataclass
from collections.abc import Callable

from marin.execution.executor import ExecutorStep


@dataclass
class DatasetComponent:
    """A single dataset component within a domain.

    Attributes:
        name: Unique identifier for the component.
        step_fn: Callable that returns the ExecutorStep for this dataset.
            Using a callable allows lazy initialization.
        weight: Relative weight within the domain (will be normalized).
    """

    name: str
    step_fn: Callable[[], ExecutorStep]
    weight: float = 1.0

    def get_step(self) -> ExecutorStep:
        """Get the ExecutorStep for this component."""
        return self.step_fn()


@dataclass
class Domain:
    """A domain represents a group of related datasets.

    Domains are the units over which mixture weights are sampled.
    Each domain contains one or more dataset components that share
    a common purpose (e.g., pretraining, instruction tuning).

    Attributes:
        name: Unique identifier for the domain.
        components: List of dataset components in this domain.
        natural_proportion: Prior proportion for Dirichlet sampling.
            This reflects the typical/expected weight for this domain.
        description: Optional human-readable description.
    """

    name: str
    components: list[DatasetComponent]
    natural_proportion: float = 1.0
    description: str = ""

    def get_component_weights(self) -> dict[str, float]:
        """Get normalized weights for components within this domain.

        Returns:
            Dictionary mapping component names to their normalized weights.
        """
        total = sum(c.weight for c in self.components)
        if total == 0:
            # Uniform weights if all are zero
            n = len(self.components)
            return {c.name: 1.0 / n for c in self.components}
        return {c.name: c.weight / total for c in self.components}

    def get_all_steps(self) -> dict[str, ExecutorStep]:
        """Get all ExecutorSteps for this domain.

        Returns:
            Dictionary mapping component names to their ExecutorSteps.
        """
        return {c.name: c.get_step() for c in self.components}


@dataclass
class PhaseConfig:
    """Configuration for a single training phase.

    Attributes:
        name: Identifier for the phase (e.g., "phase1", "early", "late").
        start_fraction: Start point as fraction of total training [0, 1].
        end_fraction: End point as fraction of total training [0, 1].
    """

    name: str
    start_fraction: float
    end_fraction: float

    def __post_init__(self):
        if not 0 <= self.start_fraction < self.end_fraction <= 1:
            raise ValueError(
                f"Invalid phase boundaries: start={self.start_fraction}, end={self.end_fraction}. "
                "Must satisfy 0 <= start < end <= 1."
            )

    def get_start_step(self, total_steps: int) -> int:
        """Get the starting step index for this phase."""
        return int(total_steps * self.start_fraction)

    def get_end_step(self, total_steps: int) -> int:
        """Get the ending step index for this phase."""
        return int(total_steps * self.end_fraction)

    def get_start_sequence(self, total_steps: int, batch_size: int) -> int:
        """Get the starting sequence index for this phase.

        This is used by lm_varying_mixture_data_config which uses sequence indices.
        """
        return self.get_start_step(total_steps) * batch_size


@dataclass
class PhaseSchedule:
    """A schedule defining multiple training phases.

    Attributes:
        phases: List of phase configurations, must cover [0, 1] without gaps.
    """

    phases: list[PhaseConfig]

    def __post_init__(self):
        if not self.phases:
            raise ValueError("PhaseSchedule must have at least one phase.")

        # Sort phases by start fraction
        self.phases = sorted(self.phases, key=lambda p: p.start_fraction)

        # Validate coverage
        if self.phases[0].start_fraction != 0:
            raise ValueError("First phase must start at 0.")
        if self.phases[-1].end_fraction != 1:
            raise ValueError("Last phase must end at 1.")

        # Check for gaps
        for i in range(len(self.phases) - 1):
            if self.phases[i].end_fraction != self.phases[i + 1].start_fraction:
                raise ValueError(
                    f"Gap between phases: {self.phases[i].name} ends at "
                    f"{self.phases[i].end_fraction} but {self.phases[i + 1].name} "
                    f"starts at {self.phases[i + 1].start_fraction}."
                )

    @property
    def n_phases(self) -> int:
        """Number of phases in the schedule."""
        return len(self.phases)

    @property
    def phase_names(self) -> list[str]:
        """Names of all phases."""
        return [p.name for p in self.phases]

    @classmethod
    def uniform(cls, n_phases: int) -> "PhaseSchedule":
        """Create a schedule with n uniform phases.

        Args:
            n_phases: Number of equal-length phases.

        Returns:
            PhaseSchedule with n equal phases named "phase_0", "phase_1", etc.
        """
        phases = []
        for i in range(n_phases):
            start = i / n_phases
            end = (i + 1) / n_phases
            phases.append(PhaseConfig(name=f"phase_{i}", start_fraction=start, end_fraction=end))
        return cls(phases=phases)

    @classmethod
    def from_boundaries(cls, boundaries: list[float], names: list[str] | None = None) -> "PhaseSchedule":
        """Create a schedule from phase boundary fractions.

        Args:
            boundaries: List of boundary fractions, e.g., [0.33, 0.67] creates
                3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0].
            names: Optional names for phases. If None, uses "phase_0", etc.

        Returns:
            PhaseSchedule with the specified boundaries.
        """
        all_points = [0.0, *sorted(boundaries), 1.0]
        n_phases = len(all_points) - 1

        if names is None:
            names = [f"phase_{i}" for i in range(n_phases)]
        elif len(names) != n_phases:
            raise ValueError(f"Expected {n_phases} names but got {len(names)}.")

        phases = []
        for i in range(n_phases):
            phases.append(
                PhaseConfig(
                    name=names[i],
                    start_fraction=all_points[i],
                    end_fraction=all_points[i + 1],
                )
            )
        return cls(phases=phases)


@dataclass
class WeightConfig:
    """Configuration for domain weights across all phases.

    Attributes:
        run_id: Unique identifier for this configuration.
        phase_weights: Dictionary mapping phase names to domain weight dicts.
            e.g., {"phase_0": {"pretrain": 0.8, "sft": 0.2}, ...}
    """

    run_id: int
    phase_weights: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "phase_weights": self.phase_weights,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightConfig":
        """Create from dictionary."""
        return cls(
            run_id=d["run_id"],
            phase_weights=d["phase_weights"],
        )

    def get_weights_for_phase(self, phase_name: str) -> dict[str, float]:
        """Get domain weights for a specific phase."""
        return self.phase_weights[phase_name]


@dataclass
class ExperimentConfig:
    """Configuration for a mixture experiment.

    Attributes:
        name: Name of the experiment.
        domains: List of data domains to use.
        phase_schedule: Schedule defining training phases.
        total_steps: Total number of training steps.
        batch_size: Training batch size.
        seq_len: Sequence length.
        target_budget: Target token budget for simulated epoching.
        description: Optional experiment description.
    """

    name: str
    domains: list[Domain]
    phase_schedule: PhaseSchedule
    total_steps: int
    batch_size: int
    seq_len: int = 2048
    target_budget: int | None = None
    description: str = ""

    @property
    def domain_names(self) -> list[str]:
        """Names of all domains."""
        return [d.name for d in self.domains]

    @property
    def n_domains(self) -> int:
        """Number of domains."""
        return len(self.domains)

    @property
    def n_phases(self) -> int:
        """Number of phases."""
        return self.phase_schedule.n_phases

    @property
    def tokens_per_step(self) -> int:
        """Tokens processed per training step."""
        return self.batch_size * self.seq_len

    @property
    def experiment_budget(self) -> int:
        """Total tokens in this experiment."""
        return self.total_steps * self.tokens_per_step

    def get_natural_proportions(self) -> dict[str, float]:
        """Get natural proportions for all domains (normalized)."""
        total = sum(d.natural_proportion for d in self.domains)
        return {d.name: d.natural_proportion / total for d in self.domains}

    def get_all_components(self) -> dict[str, ExecutorStep]:
        """Get all dataset components across all domains."""
        components = {}
        for domain in self.domains:
            components.update(domain.get_all_steps())
        return components

    def expand_domain_weights(self, domain_weights: dict[str, float]) -> dict[str, float]:
        """Expand domain-level weights to component-level weights.

        Args:
            domain_weights: Dictionary mapping domain names to weights.

        Returns:
            Dictionary mapping component names to weights.
        """
        component_weights = {}
        for domain in self.domains:
            domain_weight = domain_weights.get(domain.name, 0.0)
            domain_component_weights = domain.get_component_weights()
            for comp_name, comp_weight in domain_component_weights.items():
                component_weights[comp_name] = domain_weight * comp_weight
        return component_weights
