# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent next-token parity against frozen scores."""

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from marin.inference.vllm_server import VllmRuntimeFingerprint

OBSERVATION_SCHEMA_VERSION = 1


def source_digest(*module_paths: str | Path) -> str:
    """Hash capture modules without retaining controller-local paths."""
    digest = hashlib.sha256()
    for path in sorted(Path(path) for path in module_paths):
        relative_name = path.name.encode()
        digest.update(len(relative_name).to_bytes(8, "little"))
        digest.update(relative_name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenScore:
    logprob: float
    token_id: int


@dataclass(frozen=True)
class GoldenTokenObservation:
    """One canonical token as observed by a backend.

    Serving backends may omit a canonical token from their returned top-N, in which
    case ``logprob`` and ``rank`` are ``None`` and the report remains diagnostic.
    """

    token_id: int
    logprob: float | None
    rank: int | None


@dataclass(frozen=True)
class NextTokenObservation:
    """Auditable next-token output for one case in one backend run."""

    case_id: str
    bucket_max_tokens: int
    repeat_index: int
    backend_index: int
    greedy_token_id: int
    top_logprobs: tuple[TokenScore, ...]
    golden_tokens: tuple[GoldenTokenObservation, ...]
    capacity_overflow: tuple[float, ...]
    has_nonfinite: bool


@dataclass(frozen=True)
class RunProvenance:
    """Inputs and effective execution choices shared by one process report."""

    backend: str
    platform: str
    process_id: str
    code_digest: str
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    requested_attention: str
    effective_attention: str
    requested_moe: str
    effective_moe: str
    mesh_shape: tuple[tuple[str, int], ...]
    device_kind: str
    golden_digest: str = ""
    # Full Git commit digests passed to the isolated serving environment. A
    # numerical contract that depends on external fork code pins both entries.
    fork_source_revisions: tuple[tuple[str, str], ...] = ()
    runtime_fingerprint: VllmRuntimeFingerprint | None = None


@dataclass(frozen=True)
class ParityDiscovery:
    """Measurements from which one cell's holdout bound was frozen."""

    candidate_headroom: float
    clean_process_count: int
    max_probability_error: float
    repeatability_probability_error: float
    same_process_repeat_count: int
    summary_sha256: str


@dataclass(frozen=True)
class ParityContract:
    """Frozen numerical and provenance contract for one backend/platform cell."""

    schema_version: int
    name: str
    backend: str
    platform: str
    max_probability_error: float
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    canonical_golden_digest: str
    requested_attention: str
    effective_attention: str
    requested_moe: str
    effective_moe: str
    # A contract may pin only semantically relevant axes; extra size-one axes
    # in a backend's reported mesh do not make it a different numerical cell.
    mesh_shape: tuple[tuple[str, int], ...]
    discovery: ParityDiscovery
    # Some serving kernels accumulate materially different error at long
    # context. A reviewed cell may therefore tighten specific prompt buckets
    # while retaining ``max_probability_error`` as the fallback/global cap.
    max_probability_error_by_bucket: tuple[tuple[int, float], ...] = ()
    fork_source_revisions: tuple[tuple[str, str], ...] = ()
    runtime_fingerprint_digest: str = ""


@dataclass(frozen=True)
class ObservationReport:
    provenance: RunProvenance
    observations: tuple[NextTokenObservation, ...]
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "ObservationReport":
        raw = json.loads(payload)
        schema_version = int(raw["schema_version"])
        if schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema {schema_version}; expected {OBSERVATION_SCHEMA_VERSION}")
        provenance = RunProvenance(
            **{
                **raw["provenance"],
                "mesh_shape": tuple(tuple(axis) for axis in raw["provenance"]["mesh_shape"]),
                "golden_digest": raw["provenance"]["golden_digest"],
                "fork_source_revisions": tuple(
                    tuple(revision) for revision in raw["provenance"].get("fork_source_revisions", ())
                ),
                "runtime_fingerprint": (
                    VllmRuntimeFingerprint.from_dict(raw["provenance"]["runtime_fingerprint"])
                    if raw["provenance"].get("runtime_fingerprint") is not None
                    else None
                ),
            }
        )
        observations = tuple(
            NextTokenObservation(
                case_id=observation["case_id"],
                bucket_max_tokens=int(observation["bucket_max_tokens"]),
                repeat_index=int(observation["repeat_index"]),
                backend_index=int(observation["backend_index"]),
                greedy_token_id=int(observation["greedy_token_id"]),
                top_logprobs=tuple(TokenScore(**score) for score in observation["top_logprobs"]),
                golden_tokens=tuple(GoldenTokenObservation(**golden) for golden in observation["golden_tokens"]),
                capacity_overflow=tuple(float(value) for value in observation["capacity_overflow"]),
                has_nonfinite=bool(observation["has_nonfinite"]),
            )
            for observation in raw["observations"]
        )
        return cls(provenance=provenance, observations=observations, schema_version=schema_version)


def _float32_bits(value: float | None) -> int | None:
    if value is None:
        return None
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _observation_bitwise_signature(observation: NextTokenObservation) -> tuple:
    return (
        observation.case_id,
        observation.bucket_max_tokens,
        observation.greedy_token_id,
        tuple((score.token_id, _float32_bits(score.logprob)) for score in observation.top_logprobs),
        tuple((token.token_id, _float32_bits(token.logprob), token.rank) for token in observation.golden_tokens),
        tuple(_float32_bits(value) for value in observation.capacity_overflow),
        observation.has_nonfinite,
    )


def observations_bitwise_equal(left: NextTokenObservation, right: NextTokenObservation) -> bool:
    """Compare token IDs and exact float32 score/diagnostic bit patterns."""
    return _observation_bitwise_signature(left) == _observation_bitwise_signature(right)


def observation_from_completion_response(
    payload: dict,
    *,
    case_id: str,
    prompt_token_ids: tuple[int, ...],
    expected_top_logprobs: tuple[TokenScore, ...],
    bucket_max_tokens: int,
    repeat_index: int,
    backend_index: int,
) -> NextTokenObservation:
    """Normalize one vLLM completions response into the shared report schema."""
    (choice,) = payload["choices"]
    assert choice["prompt_token_ids"] == list(prompt_token_ids), case_id
    (greedy_token_id,) = choice["token_ids"]
    (raw_top_logprobs,) = choice["logprobs"]["top_logprobs"]
    scores_by_id = {int(token.removeprefix("token_id:")): float(logprob) for token, logprob in raw_top_logprobs.items()}
    assert int(greedy_token_id) in scores_by_id, f"{case_id}: greedy token is absent from returned logprobs"
    top_logprobs = tuple(
        TokenScore(token_id=token_id, logprob=logprob)
        for token_id, logprob in sorted(scores_by_id.items(), key=lambda item: (-item[1], item[0]))
    )
    assert top_logprobs[0].token_id == int(greedy_token_id), f"{case_id}: greedy token is not top-ranked"
    rank_by_id = {score.token_id: rank for rank, score in enumerate(top_logprobs)}
    golden_tokens = tuple(
        GoldenTokenObservation(
            token_id=expected.token_id,
            logprob=scores_by_id.get(expected.token_id),
            rank=rank_by_id.get(expected.token_id),
        )
        for expected in expected_top_logprobs
    )
    return NextTokenObservation(
        case_id=case_id,
        bucket_max_tokens=bucket_max_tokens,
        repeat_index=repeat_index,
        backend_index=backend_index,
        greedy_token_id=int(greedy_token_id),
        top_logprobs=top_logprobs,
        golden_tokens=golden_tokens,
        capacity_overflow=(),
        has_nonfinite=any(not math.isfinite(score.logprob) for score in top_logprobs),
    )


def request_next_token_observation(
    completions_url: str,
    model_id: str,
    *,
    case_id: str,
    prompt_token_ids: tuple[int, ...],
    expected_top_logprobs: tuple[TokenScore, ...],
    bucket_max_tokens: int,
    repeat_index: int,
    backend_index: int,
    returned_logprobs: int,
    headers: dict[str, str] | None = None,
    timeout: tuple[float, float] = (30.0, 300.0),
) -> NextTokenObservation:
    """Request one deterministic token from an OpenAI completions endpoint."""
    import requests  # noqa: PLC0415 -- endpoint clients need not import requests for offline analysis

    context = f"case={case_id} backend_index={backend_index}"
    try:
        response = requests.post(
            completions_url,
            headers=headers,
            json={
                "model": model_id,
                "prompt": list(prompt_token_ids),
                "add_special_tokens": False,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": returned_logprobs,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return observation_from_completion_response(
            response.json(),
            case_id=case_id,
            prompt_token_ids=prompt_token_ids,
            expected_top_logprobs=expected_top_logprobs,
            bucket_max_tokens=bucket_max_tokens,
            repeat_index=repeat_index,
            backend_index=backend_index,
        )
    except Exception as error:
        error.add_note(context)
        raise


@dataclass(frozen=True)
class NextTokenParity:
    """One backend observation against one frozen next-token distribution."""

    case_id: str
    backend_rank: int
    greedy_token_id: int
    golden_top_token_ids: tuple[int, ...]
    golden_probability_gap_to_greedy: float
    max_probability_error: float
    top_probability_l1_error: float

    def assert_matches(self, *, max_probability_error: float) -> None:
        """Require golden-token coverage and a probability-supported winner."""
        assert self.greedy_token_id in self.golden_top_token_ids, self
        assert self.golden_probability_gap_to_greedy <= 2 * self.max_probability_error, self
        assert self.max_probability_error <= max_probability_error, self


def parity_from_logprob_map(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
    """Score a vLLM-style ``{token_id: logprob}`` response."""
    assert greedy_token_id in actual_logprobs, f"{case_id} rank {backend_rank}: greedy token missing from logprobs"
    maximum_actual_logprob = max(actual_logprobs.values())
    assert (
        actual_logprobs[greedy_token_id] == maximum_actual_logprob
    ), f"{case_id} rank {backend_rank}: greedy token does not have maximum returned logprob"
    return _parity_from_token_scores(
        case_id,
        expected_top_logprobs,
        greedy_token_id,
        actual_logprobs,
        backend_rank=backend_rank,
    )


def parity_from_observation(
    observation: NextTokenObservation,
    expected_top_logprobs: tuple[TokenScore, ...],
) -> NextTokenParity:
    """Score one backend-neutral observation against a frozen distribution."""
    actual_logprobs: dict[int, float] = {}
    for token in observation.golden_tokens:
        assert token.logprob is not None, f"{observation.case_id}: missing canonical-token logprob"
        assert token.token_id not in actual_logprobs, f"{observation.case_id}: duplicate canonical token"
        actual_logprobs[token.token_id] = token.logprob
    return _parity_from_token_scores(
        observation.case_id,
        expected_top_logprobs,
        observation.greedy_token_id,
        actual_logprobs,
        backend_rank=observation.backend_index,
    )


def assert_report_matches_goldens(
    report: ObservationReport,
    expected_by_id: dict[str, tuple[TokenScore, ...]],
    *,
    max_probability_error: float,
    max_probability_error_by_bucket: dict[int, float] | None = None,
) -> tuple[NextTokenParity, ...]:
    """Apply the shared numerical and health contract to every report repeat."""
    observed_by_repeat: dict[int, list[str]] = {}
    parities = []
    failures = []
    bucket_maxima: dict[int, tuple[float, float, str]] = {}
    for observation in report.observations:
        observed_by_repeat.setdefault(observation.repeat_index, []).append(observation.case_id)
        try:
            assert not observation.has_nonfinite, "nonfinite scores"
            assert not any(value != 0.0 for value in observation.capacity_overflow), "router capacity overflow"
            assert observation.top_logprobs, "no top logprobs"
            assert (
                observation.greedy_token_id == observation.top_logprobs[0].token_id
            ), "greedy token is not the returned top token"
            assert len({score.token_id for score in observation.top_logprobs}) == len(
                observation.top_logprobs
            ), "duplicate returned token"
            expected = expected_by_id.get(observation.case_id)
            assert expected is not None, "case is absent from the golden"
            parity = parity_from_observation(observation, expected)
        except AssertionError as error:
            failures.append(f"case={observation.case_id} repeat={observation.repeat_index}: {error}")
            continue

        bucket_bound = (max_probability_error_by_bucket or {}).get(
            observation.bucket_max_tokens,
            max_probability_error,
        )
        parities.append(parity)
        previous_maximum = bucket_maxima.get(observation.bucket_max_tokens)
        if previous_maximum is None or parity.max_probability_error > previous_maximum[0]:
            bucket_maxima[observation.bucket_max_tokens] = (
                parity.max_probability_error,
                bucket_bound,
                observation.case_id,
            )

        reasons = []
        if parity.greedy_token_id not in parity.golden_top_token_ids:
            reasons.append("greedy token is outside the canonical top tokens")
        if parity.golden_probability_gap_to_greedy > 2 * parity.max_probability_error:
            reasons.append("greedy-token change is not explained by the measured probability error")
        if parity.max_probability_error > bucket_bound:
            reasons.append(
                f"maximum probability error {parity.max_probability_error:.17g} exceeds bound {bucket_bound:.17g}"
            )
        if reasons:
            failures.append(
                f"case={observation.case_id} repeat={observation.repeat_index} "
                f"bucket={observation.bucket_max_tokens}: {'; '.join(reasons)}; parity={parity}"
            )

    try:
        _assert_complete_repeats(observed_by_repeat, expected_by_id.keys())
    except AssertionError as error:
        failures.append(f"report coverage: {error}")

    if failures:
        bucket_summary = "\n".join(
            f"bucket={bucket}: max_probability_error={maximum:.17g} bound={bound:.17g} case={case_id}"
            for bucket, (maximum, bound, case_id) in sorted(bucket_maxima.items())
        )
        raise AssertionError(
            "Report parity failures after evaluating all observations.\n"
            f"Bucket maxima:\n{bucket_summary}\n"
            f"Violations:\n" + "\n".join(failures)
        )
    return tuple(parities)


def assert_report_matches_contract(
    report: ObservationReport,
    expected_by_id: dict[str, tuple[TokenScore, ...]],
    contract: ParityContract,
) -> tuple[NextTokenParity, ...]:
    """Apply one cell's frozen provenance, topology, health, and numerical gate."""
    provenance = report.provenance
    expected_provenance = {
        "backend": contract.backend,
        "platform": contract.platform,
        "parameter_digest": contract.parameter_digest,
        "model_config_digest": contract.model_config_digest,
        "prompt_fixture_digest": contract.prompt_fixture_digest,
        "golden_digest": contract.canonical_golden_digest,
        "requested_attention": contract.requested_attention,
        "effective_attention": contract.effective_attention,
        "requested_moe": contract.requested_moe,
        "effective_moe": contract.effective_moe,
        "fork_source_revisions": contract.fork_source_revisions,
    }
    for field, expected in expected_provenance.items():
        actual = getattr(provenance, field)
        assert actual == expected, f"Unexpected {field}: expected {expected!r}, got {actual!r}"

    if contract.runtime_fingerprint_digest:
        assert provenance.runtime_fingerprint is not None, "Report is missing its vLLM runtime fingerprint"
        actual_runtime_digest = provenance.runtime_fingerprint.digest()
        assert actual_runtime_digest == contract.runtime_fingerprint_digest, (
            "Unexpected runtime_fingerprint_digest: "
            f"expected {contract.runtime_fingerprint_digest!r}, got {actual_runtime_digest!r}"
        )

    actual_mesh = dict(provenance.mesh_shape)
    for axis, expected_size in contract.mesh_shape:
        actual_size = actual_mesh.get(axis)
        assert actual_size == expected_size, (
            f"Unexpected mesh axis {axis}: expected {expected_size}, got {actual_size}; "
            f"full mesh={provenance.mesh_shape}"
        )
    return assert_report_matches_goldens(
        report,
        expected_by_id,
        max_probability_error=contract.max_probability_error,
        max_probability_error_by_bucket=dict(contract.max_probability_error_by_bucket),
    )


def assert_report_matches_exact_goldens(
    report: ObservationReport,
    expected_by_id: dict[str, tuple[TokenScore, ...]],
    *,
    score_source: str,
) -> None:
    """Compare an exact cell snapshot using the shared observation schema."""
    observed_by_repeat: dict[int, list[str]] = {}
    failures = []
    for observation in report.observations:
        observed_by_repeat.setdefault(observation.repeat_index, []).append(observation.case_id)
        expected = expected_by_id.get(observation.case_id)
        if expected is None:
            failures.append(
                f"case={observation.case_id} repeat={observation.repeat_index}: case is absent from the exact golden"
            )
            continue
        if score_source == "top_logprobs":
            actual = observation.top_logprobs[: len(expected)]
        elif score_source == "canonical_tokens":
            actual = tuple(
                TokenScore(token_id=token.token_id, logprob=token.logprob)
                for token in observation.golden_tokens
                if token.logprob is not None
            )
        else:
            raise ValueError(f"Unsupported exact score source {score_source!r}")
        expected_signature = tuple((score.token_id, _float32_bits(score.logprob)) for score in expected)
        actual_signature = tuple((score.token_id, _float32_bits(score.logprob)) for score in actual)
        if actual_signature != expected_signature:
            failures.append(f"case={observation.case_id} repeat={observation.repeat_index}: exact scores differ")

    try:
        _assert_complete_repeats(observed_by_repeat, expected_by_id.keys())
    except AssertionError as error:
        failures.append(f"report coverage: {error}")
    if failures:
        raise AssertionError("Exact-golden failures:\n" + "\n".join(failures))


def _assert_complete_repeats(
    observed_by_repeat: dict[int, list[str]],
    expected_case_ids: Iterable[str],
) -> None:
    """Require every repeat to cover each expected case exactly once."""
    assert observed_by_repeat, "report contains no observations"
    expected_case_ids = set(expected_case_ids)
    for repeat_index, case_ids in observed_by_repeat.items():
        assert len(case_ids) == len(set(case_ids)), f"repeat {repeat_index} contains duplicate cases"
        assert set(case_ids) == expected_case_ids, f"repeat {repeat_index} does not cover the golden cases"


def _parity_from_token_scores(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
    golden_logprobs = {entry.token_id: entry.logprob for entry in expected_top_logprobs}
    missing = golden_logprobs.keys() - actual_logprobs.keys()
    assert not missing, f"{case_id} rank {backend_rank}: golden tokens missing from backend logprobs: {sorted(missing)}"
    probability_errors = tuple(
        abs(math.exp(actual_logprobs[token_id]) - math.exp(golden_logprob))
        for token_id, golden_logprob in golden_logprobs.items()
    )
    maximum_golden_logprob = max(golden_logprobs.values())
    selected_golden_logprob = golden_logprobs.get(greedy_token_id, -math.inf)
    return NextTokenParity(
        case_id=case_id,
        backend_rank=backend_rank,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=tuple(golden_logprobs),
        golden_probability_gap_to_greedy=math.exp(maximum_golden_logprob) - math.exp(selected_golden_logprob),
        max_probability_error=max(probability_errors),
        top_probability_l1_error=sum(probability_errors),
    )
