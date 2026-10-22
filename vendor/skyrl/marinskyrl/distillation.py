"""Typed, fail-closed configuration for teacher-scored distillation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar
from urllib.parse import urlsplit

from omegaconf import DictConfig, OmegaConf
from rigging.secrets import is_secret_reference

from marinskyrl.runtime_options import ftpo_requested


class DistillationObjectiveKind(StrEnum):
    SAMPLED_REVERSE_KL = "sampled_reverse_kl"
    SPARSE_FORWARD_KL = "sparse_forward_kl"
    SPARSE_REVERSE_KL = "sparse_reverse_kl"
    SPARSE_JSD = "sparse_jsd"
    STUDENT_TOPK_POLICY_SURROGATE = "student_topk_policy_surrogate"


class DistillationRewardMode(StrEnum):
    ADD = "add"
    REPLACE = "replace"


class TeacherEvidenceKind(StrEnum):
    CHOSEN_TOKEN = "chosen_token"
    TOPK_DISTRIBUTION = "topk_distribution"
    STUDENT_SELECTED_TOPK = "student_selected_topk"


class TeacherSource(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL_INFERENCE = "local_inference"
    FROZEN_WORKER = "frozen_worker"
    RESIDENT = "resident"


class TeacherPlacement(StrEnum):
    EXTERNAL = "external"
    PINNED = "pinned"
    ROTATING = "rotating"
    CO_RESIDENT = "co_resident"


@dataclass(frozen=True)
class TeacherModelSpec:
    path: str
    revision: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path.strip()
            or not isinstance(self.revision, str)
            or not self.revision.strip()
        ):
            raise ValueError("teacher model path and revision must be non-empty strings")


@dataclass(frozen=True)
class TeacherEndpointSpec:
    url: str
    auth: str | None
    max_concurrency: int


@dataclass(frozen=True)
class TeacherResourceSpec:
    num_nodes: int
    gpus_per_node: int
    tensor_parallel_size: int
    colocation_group: str
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    # One engine spans tensor_parallel_size x data_parallel_size GPUs. Expert parallelism shards MoE
    # experts across that engine's GPUs; the vLLM fork serves GrugMoE only with tensor_parallel_size=1.
    data_parallel_size: int = 1
    expert_parallel_size: int = 1

    @property
    def gpus_per_engine(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size


class TeacherSpec(Protocol):
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    resources: TeacherResourceSpec | None
    top_k: int | None


@dataclass(frozen=True)
class OpenAICompatibleTeacherSpec:
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    endpoints: tuple[TeacherEndpointSpec, ...]
    tokenizer_fingerprint: str
    max_sequence_length: int
    request_timeout_seconds: float
    resources: TeacherResourceSpec | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class LocalInferenceTeacherSpec:
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    backend: str
    resources: TeacherResourceSpec | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class FrozenWorkerTeacherSpec:
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    resources: TeacherResourceSpec | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class ResidentTeacherSpec:
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    resources: TeacherResourceSpec | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class TeacherRouteSpec:
    key: str
    teacher_id: str
    weight: float


@dataclass(frozen=True)
class TeacherRoutingPlan:
    name: str
    revision: str
    routes: tuple[TeacherRouteSpec, ...]


@dataclass(frozen=True)
class TeacherResidencySpec:
    max_resident: int = 1
    minimum_residency_seconds: float = 60.0


@dataclass(frozen=True)
class DomainGradientBalanceSpec:
    """Per-batch target domain shares with first-observed-gap anchoring."""

    target_shares: tuple[tuple[str, float], ...]
    gap_scale_alpha: float


@dataclass(frozen=True)
class DistillationPlan:
    objective: DistillationObjectiveKind
    coefficient: float
    reward_mode: DistillationRewardMode
    teachers: tuple[TeacherSpec, ...]
    routing: TeacherRoutingPlan
    residency: TeacherResidencySpec = TeacherResidencySpec()
    domain_gradient_balance: DomainGradientBalanceSpec | None = None
    advantage_clip: float | None = None
    jsd_beta: float | None = None
    entry_clip: float | None = None


def validate_distillation_runtime_support(plan: DistillationPlan | None) -> None:
    """Fail before allocation unless every teacher has a production scoring adapter."""
    if plan is None:
        return
    rotating_teachers = tuple(teacher for teacher in plan.teachers if teacher.placement is TeacherPlacement.ROTATING)
    if rotating_teachers and plan.residency.max_resident != 1:
        raise ValueError("the local teacher resource plan currently supports exactly one rotating residency slot")
    pinned_groups: set[str] = set()
    rotating_groups: set[str] = set()
    rotating_footprint: TeacherResourceSpec | None = None
    for teacher in plan.teachers:
        if teacher.source is TeacherSource.OPENAI_COMPATIBLE:
            assert isinstance(teacher, OpenAICompatibleTeacherSpec)
            continue
        if teacher.source is not TeacherSource.LOCAL_INFERENCE:
            raise ValueError(
                "the distillation runtime currently supports openai_compatible and local_inference teachers"
            )
        if teacher.placement not in {TeacherPlacement.PINNED, TeacherPlacement.ROTATING}:
            raise ValueError("local_inference teachers must use pinned or rotating placement")
        if teacher.resources is None:
            raise ValueError(f"teachers.{teacher.id}.resources is required for a local teacher runtime")
        total_gpus = teacher.resources.num_nodes * teacher.resources.gpus_per_node
        if total_gpus % teacher.resources.gpus_per_engine != 0:
            raise ValueError(
                f"teachers.{teacher.id}.resources reserves {total_gpus} GPUs, which is not divisible by "
                f"tensor_parallel_size={teacher.resources.tensor_parallel_size} x "
                f"data_parallel_size={teacher.resources.data_parallel_size}"
            )
        expert_parallel = teacher.resources.expert_parallel_size
        if expert_parallel not in (1, teacher.resources.gpus_per_engine):
            raise ValueError(
                f"teachers.{teacher.id}.resources.expert_parallel_size must be 1 or equal to "
                f"tensor_parallel_size x data_parallel_size ({teacher.resources.gpus_per_engine}); got {expert_parallel}"
            )

        group = teacher.resources.colocation_group
        if teacher.placement is TeacherPlacement.PINNED:
            if group in pinned_groups:
                raise ValueError(f"pinned local teachers must use distinct colocation groups; duplicate {group!r}")
            pinned_groups.add(group)
            continue

        rotating_groups.add(group)
        if rotating_footprint is None:
            rotating_footprint = teacher.resources
        elif teacher.resources != rotating_footprint:
            raise ValueError("rotating local teachers must share one identical resource footprint and colocation group")
    overlapping_groups = pinned_groups & rotating_groups
    if overlapping_groups:
        group = min(overlapping_groups)
        raise ValueError(f"pinned and rotating local teachers cannot share colocation group {group!r}")


_OBJECTIVE_EVIDENCE = {
    DistillationObjectiveKind.SAMPLED_REVERSE_KL: TeacherEvidenceKind.CHOSEN_TOKEN,
    DistillationObjectiveKind.SPARSE_FORWARD_KL: TeacherEvidenceKind.TOPK_DISTRIBUTION,
    DistillationObjectiveKind.SPARSE_REVERSE_KL: TeacherEvidenceKind.TOPK_DISTRIBUTION,
    DistillationObjectiveKind.SPARSE_JSD: TeacherEvidenceKind.TOPK_DISTRIBUTION,
    DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE: TeacherEvidenceKind.STUDENT_SELECTED_TOPK,
}
_TOKENIZER_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GCP_SECRET_REFERENCE_PATTERN = re.compile(
    r"^gcp-secret://projects/[^/]+/secrets/[^/]+/versions/(?:[1-9][0-9]*|latest)$"
)
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_unknown(config: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(str(key) for key in config.keys() - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")


def _required_string(config: Mapping[str, object], key: str, path: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(config: Mapping[str, object], key: str, path: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string when set")
    return value.strip()


def _enum_value(enum_type: type[_EnumT], config: Mapping[str, object], key: str, path: str) -> _EnumT:
    value = _required_string(config, key, path)
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{path}.{key} must be one of {choices}; got {value!r}") from error


def _positive_float(config: Mapping[str, object], key: str, path: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path}.{key} must be a positive number; got {value!r}")
    return result


def _positive_integer(config: Mapping[str, object], key: str, path: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")
    return value


def _nonnegative_float(config: Mapping[str, object], key: str, path: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{path}.{key} must be a non-negative number; got {value!r}")
    return result


def _gpu_memory_utilization(config: Mapping[str, object], path: str) -> float:
    result = _positive_float(config, "gpu_memory_utilization", path)
    if result > 1:
        raise ValueError(f"{path}.gpu_memory_utilization must be at most 1; got {result!r}")
    return result


def _teacher_residency(config: Mapping[str, object]) -> TeacherResidencySpec:
    raw = config.get("residency")
    if raw is None:
        return TeacherResidencySpec()
    path = "trainer.algorithm.distillation.residency"
    residency = _mapping(raw, path)
    _reject_unknown(residency, frozenset({"max_resident", "minimum_residency_seconds"}), path)
    return TeacherResidencySpec(
        max_resident=_positive_integer(residency, "max_resident", path),
        minimum_residency_seconds=_nonnegative_float(residency, "minimum_residency_seconds", path),
    )


def _teacher_resources(config: Mapping[str, object], path: str) -> TeacherResourceSpec | None:
    raw = config.get("resources")
    if raw is None:
        return None
    resources = _mapping(raw, f"{path}.resources")
    _reject_unknown(
        resources,
        frozenset(
            {
                "num_nodes",
                "gpus_per_node",
                "tensor_parallel_size",
                "colocation_group",
                "max_num_batched_tokens",
                "gpu_memory_utilization",
                "data_parallel_size",
                "expert_parallel_size",
            }
        ),
        f"{path}.resources",
    )
    return TeacherResourceSpec(
        num_nodes=_positive_integer(resources, "num_nodes", f"{path}.resources"),
        gpus_per_node=_positive_integer(resources, "gpus_per_node", f"{path}.resources"),
        tensor_parallel_size=_positive_integer(resources, "tensor_parallel_size", f"{path}.resources"),
        colocation_group=_required_string(resources, "colocation_group", f"{path}.resources"),
        max_num_batched_tokens=(
            None
            if resources.get("max_num_batched_tokens") is None
            else _positive_integer(resources, "max_num_batched_tokens", f"{path}.resources")
        ),
        gpu_memory_utilization=(
            None
            if resources.get("gpu_memory_utilization") is None
            else _gpu_memory_utilization(resources, f"{path}.resources")
        ),
        data_parallel_size=(
            1
            if resources.get("data_parallel_size") is None
            else _positive_integer(resources, "data_parallel_size", f"{path}.resources")
        ),
        expert_parallel_size=(
            1
            if resources.get("expert_parallel_size") is None
            else _positive_integer(resources, "expert_parallel_size", f"{path}.resources")
        ),
    )


def _teacher_model(config: Mapping[str, object], path: str) -> TeacherModelSpec:
    model = _mapping(config.get("model"), f"{path}.model")
    _reject_unknown(model, frozenset({"path", "revision"}), f"{path}.model")
    return TeacherModelSpec(
        path=_required_string(model, "path", f"{path}.model"),
        revision=_required_string(model, "revision", f"{path}.model"),
    )


def _teacher_endpoints(config: Mapping[str, object], path: str) -> tuple[TeacherEndpointSpec, ...]:
    raw_endpoints = config.get("endpoints", ())
    if not isinstance(raw_endpoints, (list, tuple)):
        raise ValueError(f"{path}.endpoints must be a list")
    endpoints = []
    for index, raw_endpoint in enumerate(raw_endpoints):
        endpoint_path = f"{path}.endpoints[{index}]"
        endpoint = _mapping(raw_endpoint, endpoint_path)
        _reject_unknown(endpoint, frozenset({"url", "auth", "max_concurrency"}), endpoint_path)
        url = _required_string(endpoint, "url", endpoint_path)
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path.rstrip("/").split("/")[-1] != "v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{endpoint_path}.url must be an HTTP(S) /v1 base endpoint; got {url!r}")
        auth = _optional_string(endpoint, "auth", endpoint_path)
        if auth is not None:
            valid_auth = (
                (auth.startswith("env:") and len(auth) > len("env:"))
                or (auth.startswith("file:") and len(auth) > len("file:"))
                or _GCP_SECRET_REFERENCE_PATTERN.fullmatch(auth) is not None
            )
            if not is_secret_reference(auth) or not valid_auth:
                raise ValueError(f"{endpoint_path}.auth must be an env:, file:, or versioned gcp-secret:// reference")
        endpoints.append(
            TeacherEndpointSpec(
                url=url.rstrip("/"),
                auth=auth,
                max_concurrency=_positive_integer(endpoint, "max_concurrency", endpoint_path),
            )
        )
    return tuple(endpoints)


def _tokenizer_fingerprint(config: Mapping[str, object], path: str) -> str:
    fingerprint = _required_string(config, "tokenizer_fingerprint", path)
    if not _TOKENIZER_FINGERPRINT_PATTERN.fullmatch(fingerprint):
        raise ValueError(f"{path}.tokenizer_fingerprint must be a sha256: fingerprint")
    return fingerprint


def _teacher_spec(teacher_id: str, raw: object) -> TeacherSpec:
    path = f"teachers.{teacher_id}"
    config = _mapping(raw, path)
    _reject_unknown(
        config,
        frozenset(
            {
                "source",
                "placement",
                "model",
                "evidence",
                "top_k",
                "endpoints",
                "backend",
                "resources",
                "tokenizer_fingerprint",
                "max_sequence_length",
                "request_timeout_seconds",
            }
        ),
        path,
    )
    source = _enum_value(TeacherSource, config, "source", path)
    placement_defaults = {
        TeacherSource.OPENAI_COMPATIBLE: TeacherPlacement.EXTERNAL,
        TeacherSource.LOCAL_INFERENCE: TeacherPlacement.PINNED,
        TeacherSource.FROZEN_WORKER: TeacherPlacement.PINNED,
        TeacherSource.RESIDENT: TeacherPlacement.CO_RESIDENT,
    }
    raw_placement = config.get("placement")
    placement = (
        placement_defaults[source]
        if raw_placement is None
        else _enum_value(TeacherPlacement, config, "placement", path)
    )
    evidence = _enum_value(TeacherEvidenceKind, config, "evidence", path)
    top_k = None if config.get("top_k") is None else _positive_integer(config, "top_k", path)
    endpoints = _teacher_endpoints(config, path)
    backend = _optional_string(config, "backend", path)
    resources = _teacher_resources(config, path)

    if source is TeacherSource.OPENAI_COMPATIBLE:
        if not endpoints:
            raise ValueError(f"{path}.endpoints must contain at least one endpoint for openai_compatible teachers")
        if placement is not TeacherPlacement.EXTERNAL:
            raise ValueError(f"{path}.placement must be external for openai_compatible teachers")
    elif endpoints:
        raise ValueError(f"{path}.endpoints is only valid for openai_compatible teachers")

    external_fields = ("tokenizer_fingerprint", "max_sequence_length", "request_timeout_seconds")
    if source is not TeacherSource.OPENAI_COMPATIBLE:
        unexpected_external_fields = [field for field in external_fields if config.get(field) is not None]
        if unexpected_external_fields:
            raise ValueError(f"{path}.{unexpected_external_fields[0]} is only valid for openai_compatible teachers")

    if source is TeacherSource.LOCAL_INFERENCE:
        if backend == "sglang":
            raise ValueError(f"{path}.backend='sglang' cannot provide the prompt logprobs required for teacher scoring")
        if backend != "vllm":
            raise ValueError(f"{path}.backend must be vllm for local_inference teachers; got {backend!r}")
        if placement not in {TeacherPlacement.PINNED, TeacherPlacement.ROTATING}:
            raise ValueError(f"{path}.placement must be pinned or rotating for local_inference teachers")
    elif backend is not None:
        raise ValueError(f"{path}.backend is only valid for local_inference teachers")

    if source is TeacherSource.RESIDENT and placement is not TeacherPlacement.CO_RESIDENT:
        raise ValueError(f"{path}.placement must be co_resident for resident teachers")
    if source is TeacherSource.FROZEN_WORKER and placement is not TeacherPlacement.PINNED:
        raise ValueError(f"{path}.placement must be pinned for frozen_worker teachers")
    if placement is TeacherPlacement.EXTERNAL and resources is not None:
        raise ValueError(f"{path}.resources cannot reserve Iris capacity for an external teacher")
    if evidence in {TeacherEvidenceKind.TOPK_DISTRIBUTION, TeacherEvidenceKind.STUDENT_SELECTED_TOPK} and top_k is None:
        raise ValueError(f"{path}.top_k is required for {evidence.value} evidence")
    if evidence is TeacherEvidenceKind.CHOSEN_TOKEN and top_k is not None:
        raise ValueError(f"{path}.top_k is only valid for top-K evidence")

    common = {
        "id": teacher_id,
        "source": source,
        "placement": placement,
        "model": _teacher_model(config, path),
        "evidence": evidence,
        "resources": resources,
        "top_k": top_k,
    }
    if source is TeacherSource.OPENAI_COMPATIBLE:
        return OpenAICompatibleTeacherSpec(
            **common,
            endpoints=endpoints,
            tokenizer_fingerprint=_tokenizer_fingerprint(config, path),
            max_sequence_length=_positive_integer(config, "max_sequence_length", path),
            request_timeout_seconds=_positive_float(config, "request_timeout_seconds", path),
        )
    if source is TeacherSource.LOCAL_INFERENCE:
        assert backend is not None
        return LocalInferenceTeacherSpec(**common, backend=backend)
    if source is TeacherSource.FROZEN_WORKER:
        return FrozenWorkerTeacherSpec(**common)
    return ResidentTeacherSpec(**common)


def _routing_plan(name: str, raw: object, teacher_ids: frozenset[str]) -> TeacherRoutingPlan:
    path = f"teacher_routing.{name}"
    config = _mapping(raw, path)
    _reject_unknown(config, frozenset({"revision", "routes"}), path)
    revision = _required_string(config, "revision", path)
    raw_routes = _mapping(config.get("routes"), f"{path}.routes")
    if not raw_routes:
        raise ValueError(f"{path}.routes must contain at least one route")
    if any(not isinstance(route_key, str) or not route_key.strip() for route_key in raw_routes):
        raise ValueError(f"{path}.routes keys must be non-empty strings")

    routes = []
    for route_key, raw_route in sorted(raw_routes.items()):
        route_path = f"{path}.routes.{route_key}"
        route = _mapping(raw_route, route_path)
        _reject_unknown(route, frozenset({"teacher", "weight"}), route_path)
        teacher_id = _required_string(route, "teacher", route_path)
        if teacher_id not in teacher_ids:
            raise ValueError(f"{route_path}.teacher references unknown teacher {teacher_id!r}")
        routes.append(
            TeacherRouteSpec(
                key=route_key,
                teacher_id=teacher_id,
                weight=_positive_float(route, "weight", route_path),
            )
        )
    return TeacherRoutingPlan(name=name, revision=revision, routes=tuple(routes))


def _domain_gradient_balance(raw: object, routing: TeacherRoutingPlan) -> DomainGradientBalanceSpec:
    path = "trainer.algorithm.distillation.domain_gradient_balance"
    config = _mapping(raw, path)
    _reject_unknown(config, frozenset({"target_shares", "gap_scale_alpha"}), path)
    shares = _mapping(config.get("target_shares"), f"{path}.target_shares")
    expected = {route.key for route in routing.routes}
    if set(shares) != expected:
        raise ValueError(f"{path}.target_shares must name exactly the routed domains {sorted(expected)}")
    target_shares = tuple(
        (route, _positive_float(shares, route, f"{path}.target_shares")) for route in sorted(expected)
    )
    alpha = _nonnegative_float(config, "gap_scale_alpha", path)
    if alpha > 2:
        raise ValueError(f"{path}.gap_scale_alpha must be at most 2")
    return DomainGradientBalanceSpec(target_shares=target_shares, gap_scale_alpha=alpha)


def compile_distillation_plan(config: Mapping[str, object]) -> DistillationPlan | None:
    """Compile an immutable plan, or return ``None`` when distillation is absent."""
    if "teacher" in config:
        raise ValueError(
            "legacy teacher configuration is not supported; use top-level teachers and teacher_routing with "
            "trainer.algorithm.distillation"
        )

    trainer = _mapping(config.get("trainer", {}), "trainer")
    algorithm = _mapping(trainer.get("algorithm", {}), "trainer.algorithm")
    raw_distillation = algorithm.get("distillation")
    raw_teachers = _mapping(config.get("teachers", {}), "teachers")
    raw_routing = _mapping(config.get("teacher_routing", {}), "teacher_routing")

    if raw_distillation is None:
        if raw_teachers:
            raise ValueError("teachers requires trainer.algorithm.distillation")
        if raw_routing:
            raise ValueError("teacher_routing requires trainer.algorithm.distillation")
        return None

    distillation = _mapping(raw_distillation, "trainer.algorithm.distillation")
    _reject_unknown(
        distillation,
        frozenset(
            {
                "objective",
                "routing_plan",
                "coefficient",
                "reward_mode",
                "residency",
                "domain_gradient_balance",
                "advantage_clip",
                "jsd_beta",
                "entry_clip",
            }
        ),
        "trainer.algorithm.distillation",
    )
    objective = _enum_value(
        DistillationObjectiveKind,
        distillation,
        "objective",
        "trainer.algorithm.distillation",
    )
    reward_mode = _enum_value(
        DistillationRewardMode,
        distillation,
        "reward_mode",
        "trainer.algorithm.distillation",
    )
    coefficient = _positive_float(distillation, "coefficient", "trainer.algorithm.distillation")
    advantage_clip = (
        _positive_float(distillation, "advantage_clip", "trainer.algorithm.distillation")
        if distillation.get("advantage_clip") is not None
        else None
    )
    jsd_beta = (
        _positive_float(distillation, "jsd_beta", "trainer.algorithm.distillation")
        if distillation.get("jsd_beta") is not None
        else None
    )
    if objective is DistillationObjectiveKind.SPARSE_JSD:
        if jsd_beta is None or jsd_beta >= 1:
            raise ValueError("sparse_jsd requires jsd_beta in (0, 1)")
    elif jsd_beta is not None:
        raise ValueError("jsd_beta requires sparse_jsd")
    entry_clip = (
        _nonnegative_float(distillation, "entry_clip", "trainer.algorithm.distillation")
        if distillation.get("entry_clip") is not None
        else None
    )
    if entry_clip is not None and objective is not DistillationObjectiveKind.SPARSE_FORWARD_KL:
        raise ValueError("entry_clip requires sparse_forward_kl")
    routing_name = _required_string(distillation, "routing_plan", "trainer.algorithm.distillation")
    residency = _teacher_residency(distillation)

    if not raw_teachers:
        raise ValueError("teachers must contain at least one teacher when distillation is configured")
    if any(not isinstance(teacher_id, str) or not teacher_id.strip() for teacher_id in raw_teachers):
        raise ValueError("teachers keys must be non-empty strings")
    teachers = tuple(_teacher_spec(teacher_id, raw) for teacher_id, raw in sorted(raw_teachers.items()))
    teachers_by_id = {teacher.id: teacher for teacher in teachers}
    if routing_name not in raw_routing:
        raise ValueError(
            f"trainer.algorithm.distillation.routing_plan references unknown teacher_routing plan {routing_name!r}"
        )
    routing = _routing_plan(routing_name, raw_routing[routing_name], frozenset(teachers_by_id))
    balance = None
    if "domain_gradient_balance" in distillation:
        if objective is not DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE:
            raise ValueError("domain_gradient_balance requires student_topk_policy_surrogate")
        balance = _domain_gradient_balance(distillation["domain_gradient_balance"], routing)

    expected_evidence = _OBJECTIVE_EVIDENCE[objective]
    for teacher in teachers:
        if teacher.evidence is not expected_evidence:
            raise ValueError(
                f"teachers.{teacher.id}.evidence must be {expected_evidence.value} for objective {objective.value}; "
                f"got {teacher.evidence.value}"
            )

    routed_teachers = {route.teacher_id for route in routing.routes}
    unused_teachers = sorted(set(teachers_by_id) - routed_teachers)
    if unused_teachers:
        raise ValueError(f"teachers contains roles with no route consumer: {', '.join(unused_teachers)}")

    return DistillationPlan(
        objective=objective,
        coefficient=coefficient,
        reward_mode=reward_mode,
        teachers=teachers,
        routing=routing,
        residency=residency,
        domain_gradient_balance=balance,
        advantage_clip=advantage_clip,
        jsd_beta=jsd_beta,
        entry_clip=entry_clip,
    )


def compile_distillation_plan_from_config(cfg: DictConfig) -> DistillationPlan | None:
    """Compile a plan from one resolved OmegaConf boundary."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return compile_distillation_plan(resolved)


def validate_generation_logprobs(cfg: DictConfig) -> None:
    """Validate generation log probabilities against the configured distillation plan."""
    generator = cfg.get("generator", {})
    logprobs = generator.get("sampling_params", {}).get("logprobs")
    if logprobs is None:
        return
    assert isinstance(logprobs, int)
    if logprobs > 0 and not ftpo_requested(cfg.get("trainer", {}).get("algorithm", {})):
        plan = compile_distillation_plan_from_config(cfg)
        widths = {teacher.top_k for teacher in plan.teachers} if plan is not None else set()
        if (
            plan is None
            or plan.objective is not DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE
            or widths != {logprobs}
            or generator.get("backend") != "vllm"
        ):
            raise ValueError(
                "positive generator.sampling_params.logprobs requires a local vLLM "
                "student_topk_policy_surrogate plan with matching teacher top_k, or FTPO"
            )
    if not generator.get("run_engines_locally", False):
        raise NotImplementedError("Remote inference mode doesn't support `sampling_params.logprobs`")
