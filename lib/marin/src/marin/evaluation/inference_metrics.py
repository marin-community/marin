# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped metrics from cumulative counters in a shared vLLM lifecycle."""

import time
from dataclasses import dataclass

from prometheus_client.core import Metric as PrometheusMetric
from rigging.telemetry.prometheus import PrometheusScraper

from marin.evaluation.records import InferenceMetrics, SpeculativeDecodingMetrics
from marin.inference.iris import RemoteInferenceSession

_PROMPT_TOKENS = "vllm:prompt_tokens_total"
_GENERATION_TOKENS = "vllm:generation_tokens_total"
_SPECULATIVE_DRAFTS = "vllm:spec_decode_num_drafts_total"
_SPECULATIVE_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
_SPECULATIVE_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens_total"


@dataclass(frozen=True)
class _CumulativeCounters:
    prompt_tokens: int
    generation_tokens: int
    speculative_drafts: int = 0
    speculative_draft_tokens: int = 0
    speculative_accepted_tokens: int = 0


def _counter(families: tuple[PrometheusMetric, ...], name: str) -> int:
    values = [float(sample.value) for family in families for sample in family.samples if sample.name == name]
    if not values:
        raise ValueError(f"vLLM metrics response omitted {name}")
    value = sum(values)
    if value < 0 or not value.is_integer():
        raise ValueError(f"vLLM counter {name} must be a non-negative integer, got {value}")
    return int(value)


def _metrics_url(session: RemoteInferenceSession) -> str:
    if session.metrics_url is None:
        raise ValueError("inference session does not expose cumulative metrics")
    return session.metrics_url


def _scrape(session: RemoteInferenceSession, *, speculative: bool) -> _CumulativeCounters:
    families = PrometheusScraper(_metrics_url(session)).scrape()
    return _CumulativeCounters(
        prompt_tokens=_counter(families, _PROMPT_TOKENS),
        generation_tokens=_counter(families, _GENERATION_TOKENS),
        speculative_drafts=_counter(families, _SPECULATIVE_DRAFTS) if speculative else 0,
        speculative_draft_tokens=_counter(families, _SPECULATIVE_DRAFT_TOKENS) if speculative else 0,
        speculative_accepted_tokens=_counter(families, _SPECULATIVE_ACCEPTED_TOKENS) if speculative else 0,
    )


def _delta(after: int, before: int, name: str) -> int:
    value = after - before
    if value < 0:
        raise ValueError(f"vLLM counter {name} reset during evaluation ({before} -> {after})")
    return value


@dataclass(frozen=True)
class InferenceMetricWindow:
    """A before/after counter window over one evaluator sharing a vLLM server."""

    session: RemoteInferenceSession
    baseline: _CumulativeCounters
    started_at: float
    speculative: bool

    @classmethod
    def start(cls, session: RemoteInferenceSession, *, speculative: bool) -> "InferenceMetricWindow":
        baseline = _scrape(session, speculative=speculative)
        return cls(
            session=session,
            baseline=baseline,
            started_at=time.monotonic(),
            speculative=speculative,
        )

    def finish(self) -> InferenceMetrics:
        finished_at = time.monotonic()
        current = _scrape(self.session, speculative=self.speculative)
        wall_time = finished_at - self.started_at
        if wall_time <= 0:
            raise ValueError(f"evaluation wall time must be positive, got {wall_time}")

        prompt_tokens = _delta(current.prompt_tokens, self.baseline.prompt_tokens, _PROMPT_TOKENS)
        generation_tokens = _delta(
            current.generation_tokens,
            self.baseline.generation_tokens,
            _GENERATION_TOKENS,
        )
        speculative_decoding = None
        if self.speculative:
            drafts = _delta(current.speculative_drafts, self.baseline.speculative_drafts, _SPECULATIVE_DRAFTS)
            draft_tokens = _delta(
                current.speculative_draft_tokens,
                self.baseline.speculative_draft_tokens,
                _SPECULATIVE_DRAFT_TOKENS,
            )
            accepted_tokens = _delta(
                current.speculative_accepted_tokens,
                self.baseline.speculative_accepted_tokens,
                _SPECULATIVE_ACCEPTED_TOKENS,
            )
            speculative_decoding = SpeculativeDecodingMetrics(
                drafts=drafts,
                draft_tokens=draft_tokens,
                accepted_tokens=accepted_tokens,
                mean_acceptance_length=1 + accepted_tokens / drafts if drafts else None,
                draft_acceptance_rate=accepted_tokens / draft_tokens if draft_tokens else None,
            )

        return InferenceMetrics(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            wall_time_seconds=wall_time,
            generation_tokens_per_second=generation_tokens / wall_time,
            speculative_decoding=speculative_decoding,
        )
