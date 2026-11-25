"""Configuration for selecting the trajectories that reach learner workers."""

from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.runtime_options import TRAJECTORY_SELECTOR_TYPE_PATH

BEST_OF_N_LEARNER_ROWS = 1


class TrajectorySelectorKind(StrEnum):
    BEST_OF_N = "best_of_n"


def require_best_of_n_samples(n_samples_per_prompt: int) -> None:
    """Reject group sizes that leave no choice of trajectory."""
    if n_samples_per_prompt <= 1:
        raise ValueError("best-of-N selection requires n_samples_per_prompt greater than one")


def best_of_n_sample_count(cfg: DictConfig) -> int | None:
    """Read and validate the physical group size for an enabled selector."""
    selector_type = OmegaConf.select(cfg, TRAJECTORY_SELECTOR_TYPE_PATH, default=None)
    if selector_type is None:
        return None
    try:
        selector_kind = TrajectorySelectorKind(selector_type)
    except ValueError as error:
        raise ValueError(f"unsupported trajectory selector {selector_type!r}") from error
    if selector_kind is TrajectorySelectorKind.BEST_OF_N:
        samples = int(cfg.generator.n_samples_per_prompt)
        require_best_of_n_samples(samples)
        return samples
    raise AssertionError(f"unhandled trajectory selector {selector_kind!r}")


def optimization_samples_per_prompt(cfg: DictConfig) -> int:
    """Return the number of rows per prompt that reach learner workers."""
    return int(cfg.generator.n_samples_per_prompt) if best_of_n_sample_count(cfg) is None else BEST_OF_N_LEARNER_ROWS
