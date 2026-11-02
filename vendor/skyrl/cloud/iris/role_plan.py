"""Derive the canonical SkyRL role plan and Iris topology from Hydra config.

All execution geometry is derived from the SkyRL configuration. This module owns the
role claims, physical bundles, and placement arithmetic used before Iris submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from marinskyrl.runtime_options import inference_engines_required, reference_model_required
from marinskyrl.distillation import (
    LocalInferenceTeacherSpec,
    OpenAICompatibleTeacherSpec,
    TeacherPlacement,
    compile_distillation_plan,
)
from cloud.iris.runtime_environment import RuntimeProfile


class ModelRoleKind(StrEnum):
    POLICY = "policy"
    REFERENCE = "reference"
    CRITIC = "critic"
    ROLLOUT = "rollout"
    DRAFT_TRAINER = "draft_trainer"
    TEACHER = "teacher"


ALL_ROLES_COLOCATION_GROUP = "all"


class RoleExecution(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class ModelRoleClaim:
    """One active logical model role and its resolved execution claim."""

    role_id: str
    kind: ModelRoleKind
    execution: RoleExecution
    backend: str
    colocation_group: str | None
    num_nodes: int
    gpus_per_node: int
    replicas: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int

    def __post_init__(self) -> None:
        if not self.role_id or not self.backend:
            raise ValueError("model role claims require non-empty role_id and backend")
        if self.execution is RoleExecution.REMOTE:
            if self.colocation_group is not None or self.num_nodes != 0 or self.gpus_per_node != 0:
                raise ValueError(f"remote role {self.role_id!r} cannot reserve an Iris bundle")
            if self.replicas <= 0:
                raise ValueError(f"remote role {self.role_id!r} requires at least one endpoint replica")
            return
        if not self.colocation_group:
            raise ValueError(f"local role {self.role_id!r} requires a named colocation group")
        geometry = (
            self.num_nodes,
            self.gpus_per_node,
            self.replicas,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.data_parallel_size,
            self.expert_parallel_size,
        )
        if min(geometry) <= 0:
            raise ValueError(f"local role {self.role_id!r} requires positive execution geometry")
        if self.tensor_parallel_size > self.num_nodes * self.gpus_per_node:
            raise ValueError(
                f"local role {self.role_id!r} tensor parallelism {self.tensor_parallel_size} exceeds its "
                f"{self.num_nodes}x{self.gpus_per_node} physical footprint"
            )


@dataclass(frozen=True)
class RoleBundle:
    """One physical whole-node footprint shared by compatible local roles."""

    name: str
    role_ids: tuple[str, ...]
    num_nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        if not self.name or not self.role_ids or min(self.num_nodes, self.gpus_per_node) <= 0:
            raise ValueError("role bundles require a name, roles, and positive whole-node geometry")
        if len(set(self.role_ids)) != len(self.role_ids):
            raise ValueError(f"role bundle {self.name!r} contains duplicate roles")


@dataclass(frozen=True)
class SkyRLRolePlan:
    claims: tuple[ModelRoleClaim, ...]
    bundles: tuple[RoleBundle, ...]
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int

    def __post_init__(self) -> None:
        role_ids = [claim.role_id for claim in self.claims]
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("role plan contains duplicate role claims")
        bundle_names = [bundle.name for bundle in self.bundles]
        if len(set(bundle_names)) != len(bundle_names):
            raise ValueError("role plan contains duplicate physical bundles")
        claims_by_id = {claim.role_id: claim for claim in self.claims}
        bundled_roles = [role_id for bundle in self.bundles for role_id in bundle.role_ids]
        local_roles = [claim.role_id for claim in self.claims if claim.execution is RoleExecution.LOCAL]
        if sorted(bundled_roles) != sorted(local_roles):
            raise ValueError("role bundles must cover every local role exactly once and no remote roles")
        for bundle in self.bundles:
            for role_id in bundle.role_ids:
                claim = claims_by_id[role_id]
                if claim.colocation_group != bundle.name:
                    raise ValueError(f"role {role_id!r} does not name its containing bundle {bundle.name!r}")
                if (claim.num_nodes, claim.gpus_per_node) != (bundle.num_nodes, bundle.gpus_per_node):
                    raise ValueError(f"role {role_id!r} physical geometry does not match bundle {bundle.name!r}")
        gpu_widths = {bundle.gpus_per_node for bundle in self.bundles}
        if len(gpu_widths) > 1:
            raise ValueError(f"Iris role bundles require one GPUs-per-node value; got {sorted(gpu_widths)}")

    def claim(self, role_id: str) -> ModelRoleClaim:
        matches = [claim for claim in self.claims if claim.role_id == role_id]
        if len(matches) != 1:
            raise ValueError(f"role plan must contain exactly one {role_id!r} claim; found {len(matches)}")
        return matches[0]

    @property
    def colocate_all(self) -> bool:
        """Whether the policy-side roles and rollout share one bundle."""
        rollout_claims = [claim for claim in self.claims if claim.kind is ModelRoleKind.ROLLOUT]
        if not rollout_claims:
            # Static preference-pair launches provision no rollout role at all.
            return False
        if len(rollout_claims) > 1:
            raise ValueError("role plan contains duplicate rollout claims")
        if rollout_claims[0].execution is RoleExecution.REMOTE:
            return False
        groups = {
            claim.colocation_group
            for claim in self.claims
            if claim.execution is RoleExecution.LOCAL
            and claim.kind not in {ModelRoleKind.TEACHER, ModelRoleKind.DRAFT_TRAINER}
        }
        return len(groups) == 1


@dataclass(frozen=True)
class _RolePlanValues:
    colocate_all: bool
    colocate_policy_ref: bool
    policy_num_nodes: int
    policy_num_gpus_per_node: int
    run_engines_locally: bool
    num_inference_engines: int
    inference_engine_tensor_parallel_size: int
    inference_engine_pipeline_parallel_size: int
    inference_engine_data_parallel_size: int
    inference_engine_expert_parallel_size: int
    inference_engine_mp_backend: bool
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int


def _at(config: dict[str, Any], path: str) -> Any:
    """Fetch a required nested key, raising with the full dotted path when absent."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{path} missing from the RL config (looked up {part!r})")
        node = node[part]
    return node


def _optional_at(config: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _role_plan_values(config: dict[str, Any]) -> _RolePlanValues:
    return _RolePlanValues(
        colocate_all=bool(_at(config, "trainer.placement.colocate_all")),
        colocate_policy_ref=bool(_optional_at(config, "trainer.placement.colocate_policy_ref", True)),
        policy_num_nodes=int(_at(config, "trainer.placement.policy_num_nodes")),
        policy_num_gpus_per_node=int(_at(config, "trainer.placement.policy_num_gpus_per_node")),
        run_engines_locally=bool(_at(config, "generator.run_engines_locally")),
        num_inference_engines=int(_at(config, "generator.num_inference_engines")),
        inference_engine_tensor_parallel_size=int(_at(config, "generator.inference_engine_tensor_parallel_size")),
        inference_engine_pipeline_parallel_size=int(
            _optional_at(config, "generator.inference_engine_pipeline_parallel_size", 1)
        ),
        inference_engine_data_parallel_size=int(
            _optional_at(config, "generator.inference_engine_data_parallel_size", 1)
        ),
        inference_engine_expert_parallel_size=int(
            _optional_at(config, "generator.inference_engine_expert_parallel_size", 1)
        ),
        inference_engine_mp_backend=bool(_optional_at(config, "generator.inference_engine_mp_backend", False)),
        train_batch_size=int(_at(config, "trainer.train_batch_size")),
        policy_mini_batch_size=int(_at(config, "trainer.policy_mini_batch_size")),
        micro_train_batch_size_per_gpu=int(_at(config, "trainer.micro_train_batch_size_per_gpu")),
        n_samples_per_prompt=int(_at(config, "generator.n_samples_per_prompt")),
    )


def _policy_claim(config: dict[str, Any], values: _RolePlanValues) -> ModelRoleClaim:
    return ModelRoleClaim(
        role_id=ModelRoleKind.POLICY.value,
        kind=ModelRoleKind.POLICY,
        execution=RoleExecution.LOCAL,
        backend=derive_strategy(config) or RuntimeProfile.MEGATRON.value,
        colocation_group=ALL_ROLES_COLOCATION_GROUP if values.colocate_all else ModelRoleKind.POLICY.value,
        num_nodes=values.policy_num_nodes,
        gpus_per_node=values.policy_num_gpus_per_node,
        replicas=values.policy_num_nodes * values.policy_num_gpus_per_node,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=values.policy_num_nodes * values.policy_num_gpus_per_node,
        expert_parallel_size=1,
    )


def _core_model_claims(config: dict[str, Any], values: _RolePlanValues) -> list[ModelRoleClaim]:
    placement = _at(config, "trainer.placement")
    use_reference = reference_model_required(_at(config, "trainer.algorithm"))
    use_critic = bool(_optional_at(config, "trainer.critic.model.path"))
    strategy = derive_strategy(config) or RuntimeProfile.MEGATRON.value
    ref_num_nodes = int(placement.get("ref_num_nodes") or values.policy_num_nodes)
    ref_num_gpus_per_node = int(placement.get("ref_num_gpus_per_node") or values.policy_num_gpus_per_node)

    shared_group = ALL_ROLES_COLOCATION_GROUP
    policy = _policy_claim(config, values)
    policy_group = policy.colocation_group
    claims = [policy]
    if use_reference:
        reference_group = (
            policy_group if values.colocate_all or values.colocate_policy_ref else ModelRoleKind.REFERENCE.value
        )
        claims.append(
            ModelRoleClaim(
                role_id=ModelRoleKind.REFERENCE.value,
                kind=ModelRoleKind.REFERENCE,
                execution=RoleExecution.LOCAL,
                backend=strategy,
                colocation_group=reference_group,
                num_nodes=ref_num_nodes,
                gpus_per_node=ref_num_gpus_per_node,
                replicas=ref_num_nodes * ref_num_gpus_per_node,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=ref_num_nodes * ref_num_gpus_per_node,
                expert_parallel_size=1,
            )
        )
    if use_critic:
        critic_num_nodes = int(_at(config, "trainer.placement.critic_num_nodes"))
        critic_num_gpus_per_node = int(_at(config, "trainer.placement.critic_num_gpus_per_node"))
        claims.append(
            ModelRoleClaim(
                role_id=ModelRoleKind.CRITIC.value,
                kind=ModelRoleKind.CRITIC,
                execution=RoleExecution.LOCAL,
                backend=strategy,
                colocation_group=shared_group if values.colocate_all else ModelRoleKind.CRITIC.value,
                num_nodes=critic_num_nodes,
                gpus_per_node=critic_num_gpus_per_node,
                replicas=critic_num_nodes * critic_num_gpus_per_node,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=critic_num_nodes * critic_num_gpus_per_node,
                expert_parallel_size=1,
            )
        )
    return claims


def _minimum_disaggregated_rollout_nodes(
    *,
    replicas: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    gpus_per_node: int,
    multiprocessing_backend: bool,
) -> int:
    """Return the smallest whole-node pool that SkyRL's current Ray placement can realize."""
    tensor_pipeline_size = tensor_parallel_size * pipeline_parallel_size
    per_engine_gpus = tensor_pipeline_size * data_parallel_size

    if tensor_pipeline_size > gpus_per_node:
        raise ValueError(
            f"each rollout TP*PP rank group requires {tensor_pipeline_size} GPUs but one node has {gpus_per_node}"
        )
    # Mirror the runtime's placement atoms, including otherwise wasted GPUs when an atom does not divide a node.
    # The ray/uni path STRICT_PACKs each multi-GPU engine; the mp path packs node-atomic TP*PP actors per DP rank.
    if not multiprocessing_backend and per_engine_gpus > gpus_per_node:
        raise ValueError(
            f"each ray-backed rollout engine requires {per_engine_gpus} GPUs in one {gpus_per_node}-GPU node; "
            "reduce TP*PP*DP or use generator.inference_engine_mp_backend=true"
        )
    if multiprocessing_backend:
        placement_atom_gpus = tensor_pipeline_size
        placement_atoms = replicas * data_parallel_size
    elif per_engine_gpus > 1:
        placement_atom_gpus = per_engine_gpus
        placement_atoms = replicas
    else:
        placement_atom_gpus = 1
        placement_atoms = replicas
    atoms_per_node = gpus_per_node // placement_atom_gpus
    return (placement_atoms + atoms_per_node - 1) // atoms_per_node


def _rollout_claim(config: dict[str, Any], values: _RolePlanValues) -> ModelRoleClaim:
    rollout_backend = _at(config, "generator.backend")
    if not isinstance(rollout_backend, str) or not rollout_backend.strip():
        raise ValueError("generator.backend must be a non-empty string")
    rollout_is_local = values.run_engines_locally
    rollout_nodes = (
        values.policy_num_nodes
        if rollout_is_local and values.colocate_all
        else _minimum_disaggregated_rollout_nodes(
            replicas=values.num_inference_engines,
            tensor_parallel_size=values.inference_engine_tensor_parallel_size,
            pipeline_parallel_size=values.inference_engine_pipeline_parallel_size,
            data_parallel_size=values.inference_engine_data_parallel_size,
            gpus_per_node=values.policy_num_gpus_per_node,
            multiprocessing_backend=values.inference_engine_mp_backend,
        )
        if rollout_is_local
        else 0
    )
    claim = ModelRoleClaim(
        role_id=ModelRoleKind.ROLLOUT.value,
        kind=ModelRoleKind.ROLLOUT,
        execution=RoleExecution.LOCAL if rollout_is_local else RoleExecution.REMOTE,
        backend=rollout_backend,
        colocation_group=(ALL_ROLES_COLOCATION_GROUP if values.colocate_all else ModelRoleKind.ROLLOUT.value)
        if rollout_is_local
        else None,
        num_nodes=rollout_nodes,
        gpus_per_node=values.policy_num_gpus_per_node if rollout_is_local else 0,
        replicas=values.num_inference_engines,
        tensor_parallel_size=values.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=values.inference_engine_pipeline_parallel_size,
        data_parallel_size=values.inference_engine_data_parallel_size,
        expert_parallel_size=values.inference_engine_expert_parallel_size,
    )
    if claim.execution is RoleExecution.LOCAL:
        rollout_gpus = (
            claim.replicas * claim.tensor_parallel_size * claim.pipeline_parallel_size * claim.data_parallel_size
        )
        rollout_capacity = claim.num_nodes * claim.gpus_per_node
        if values.colocate_all and rollout_gpus != rollout_capacity:
            raise ValueError(
                "colocated rollout geometry must consume the policy bundle exactly: "
                f"{claim.replicas} replicas x TP {claim.tensor_parallel_size} x PP {claim.pipeline_parallel_size} "
                f"x DP {claim.data_parallel_size} != "
                f"{claim.num_nodes} nodes x {claim.gpus_per_node} GPUs"
            )
        if rollout_gpus > rollout_capacity:
            raise ValueError(
                f"rollout geometry requires {rollout_gpus} GPUs but its bundle contains {rollout_capacity}"
            )
    return claim


def derive_role_plan(config: dict[str, Any]) -> SkyRLRolePlan:
    """Derive every role-plan field from the RL config.

    Required geometry raises before submission, while omitted reference dimensions
    deliberately inherit the resolved policy dimensions. A missing trainer strategy
    uses the launcher's Megatron default. Only active roles receive claims,
    remote roles receive no Iris bundle, and each local colocation group becomes one
    physical whole-node footprint.
    """
    values = _role_plan_values(config)
    claims = _core_model_claims(config, values)
    if inference_engines_required(config):
        claims.append(_rollout_claim(config, values))
    claims.extend(_teacher_claims(config))
    claims.extend(_draft_trainer_claims(config, values))
    return _role_plan(tuple(claims), values)


def derive_policy_export_role_plan(config: dict[str, Any]) -> SkyRLRolePlan:
    """Derive only the saved policy's footprint for checkpoint conversion."""
    values = _role_plan_values(config)
    return _role_plan((_policy_claim(config, values),), values)


def _role_plan(claims: tuple[ModelRoleClaim, ...], values: _RolePlanValues) -> SkyRLRolePlan:
    return SkyRLRolePlan(
        claims=claims,
        bundles=_physical_bundles(claims),
        train_batch_size=values.train_batch_size,
        policy_mini_batch_size=values.policy_mini_batch_size,
        micro_train_batch_size_per_gpu=values.micro_train_batch_size_per_gpu,
        n_samples_per_prompt=values.n_samples_per_prompt,
    )


def _teacher_claims(config: dict[str, Any]) -> tuple[ModelRoleClaim, ...]:
    plan = compile_distillation_plan(config)
    if plan is None:
        return ()
    claims = []
    for teacher in plan.teachers:
        role_id = f"teacher:{teacher.id}"
        if teacher.placement is TeacherPlacement.EXTERNAL:
            if not isinstance(teacher, OpenAICompatibleTeacherSpec):
                raise TypeError(f"external teacher {teacher.id!r} must declare compatible endpoints")
            claims.append(
                ModelRoleClaim(
                    role_id=role_id,
                    kind=ModelRoleKind.TEACHER,
                    execution=RoleExecution.REMOTE,
                    backend=teacher.source.value,
                    colocation_group=None,
                    num_nodes=0,
                    gpus_per_node=0,
                    replicas=len(teacher.endpoints),
                    tensor_parallel_size=0,
                    pipeline_parallel_size=0,
                    data_parallel_size=0,
                    expert_parallel_size=0,
                )
            )
            continue

        resources = teacher.resources
        if resources is None:
            raise ValueError(f"teachers.{teacher.id}.resources is required for a local teacher role claim")
        backend = teacher.backend if isinstance(teacher, LocalInferenceTeacherSpec) else teacher.source.value
        claims.append(
            ModelRoleClaim(
                role_id=role_id,
                kind=ModelRoleKind.TEACHER,
                execution=RoleExecution.LOCAL,
                backend=backend,
                colocation_group=resources.colocation_group,
                num_nodes=resources.num_nodes,
                gpus_per_node=resources.gpus_per_node,
                replicas=1,
                tensor_parallel_size=resources.tensor_parallel_size,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                expert_parallel_size=1,
            )
        )
    return tuple(claims)


def _draft_trainer_claims(config: dict[str, Any], values: _RolePlanValues) -> tuple[ModelRoleClaim, ...]:
    speculative_decoding = _optional_at(config, "generator.speculative_decoding")
    if not isinstance(speculative_decoding, dict) or speculative_decoding.get("training") is None:
        return ()
    return (
        ModelRoleClaim(
            role_id=ModelRoleKind.DRAFT_TRAINER.value,
            kind=ModelRoleKind.DRAFT_TRAINER,
            execution=RoleExecution.LOCAL,
            backend="torch",
            colocation_group=ModelRoleKind.DRAFT_TRAINER.value,
            num_nodes=1,
            gpus_per_node=values.policy_num_gpus_per_node,
            replicas=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=1,
        ),
    )


def _physical_bundles(claims: tuple[ModelRoleClaim, ...]) -> tuple[RoleBundle, ...]:
    grouped: dict[str, list[ModelRoleClaim]] = {}
    for claim in claims:
        if claim.execution is RoleExecution.REMOTE:
            continue
        assert claim.colocation_group is not None
        grouped.setdefault(claim.colocation_group, []).append(claim)

    bundles = []
    for name, group_claims in grouped.items():
        footprints = {(claim.num_nodes, claim.gpus_per_node) for claim in group_claims}
        if len(footprints) != 1:
            details = ", ".join(f"{claim.role_id}={claim.num_nodes}x{claim.gpus_per_node}" for claim in group_claims)
            raise ValueError(f"colocation group {name!r} has incompatible physical footprints: {details}")
        num_nodes, gpus_per_node = footprints.pop()
        bundles.append(
            RoleBundle(
                name=name,
                role_ids=tuple(claim.role_id for claim in group_claims),
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
            )
        )
    return tuple(bundles)


def derive_num_nodes(plan: SkyRLRolePlan) -> int:
    """Return the whole-node gang size from unique local physical bundles."""
    return sum(bundle.num_nodes for bundle in plan.bundles)


def derive_strategy(config: dict[str, Any]) -> str | None:
    """Return ``trainer.strategy`` from the config, or ``None`` when absent."""
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    strategy = trainer.get("strategy")
    return strategy if isinstance(strategy, str) else None
