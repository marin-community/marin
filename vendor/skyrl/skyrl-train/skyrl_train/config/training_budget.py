"""The per-step training budget every method resolves from the same config fields.

Rollout training and fixed-data training must spend the same budget per optimizer step: the same
number of sequences, the same response cap for each task, and the same sequence cap. Both
resolve it here, so the validator logs one comparable record per run and the dataset filter, the
fixed-completion runner, and rollout requests read one response cap per ``data_source``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from omegaconf import DictConfig, OmegaConf

RESPONSE_CAPS_KEY = "max_generate_length_by_data_source"


@dataclass(frozen=True)
class ResponseCaps:
    """The response-token cap of each task, keyed by row ``data_source``.

    ``default`` is ``generator.sampling_params.max_generate_length``, which stays the engine and
    tensor bound. With per-source caps configured, every row must name a configured source.
    """

    default: int
    by_data_source: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def for_source(self, data_source: str | None) -> int:
        if not self.by_data_source:
            return self.default
        if data_source not in self.by_data_source:
            raise ValueError(
                f"data_source {data_source!r} has no response cap in generator.{RESPONSE_CAPS_KEY}; "
                f"configured sources: {sorted(self.by_data_source)}"
            )
        return self.by_data_source[data_source]


@dataclass(frozen=True)
class TrainingBudget:
    """What one optimizer step consumes, counting each sampled rollout as one sequence."""

    sequences_per_step: int
    sequences_per_mini_batch: int
    max_input_tokens: int
    response_caps: ResponseCaps

    def sequence_token_cap(self, data_source: str | None) -> int:
        """The configured prompt-plus-response budget; fixed-data runs launch no engine, so
        ``engine_init_kwargs.max_model_len`` does not enter it."""
        return self.max_input_tokens + self.response_caps.for_source(data_source)


def response_caps(config: DictConfig) -> ResponseCaps:
    """Resolve and check ``generator.max_generate_length_by_data_source`` against the global cap."""
    default = int(config.generator.sampling_params.max_generate_length)
    configured = OmegaConf.select(config, f"generator.{RESPONSE_CAPS_KEY}", default=None) or {}
    caps = {}
    for source, cap in configured.items():
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= default:
            raise ValueError(
                f"generator.{RESPONSE_CAPS_KEY}.{source} must be an integer in [1, {default}] "
                f"(generator.sampling_params.max_generate_length), got {cap!r}"
            )
        caps[str(source)] = cap
    return ResponseCaps(default=default, by_data_source=MappingProxyType(caps))


def training_budget(config: DictConfig) -> TrainingBudget:
    samples = int(config.generator.n_samples_per_prompt)
    return TrainingBudget(
        sequences_per_step=int(config.trainer.train_batch_size) * samples,
        sequences_per_mini_batch=int(config.trainer.policy_mini_batch_size) * samples,
        max_input_tokens=int(config.generator.max_input_length),
        response_caps=response_caps(config),
    )
