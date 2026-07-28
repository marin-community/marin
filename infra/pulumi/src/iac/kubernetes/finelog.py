# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A Finelog server built and deployed to Kubernetes.

The component owns the image, PersistentVolumeClaim, Deployment, and Service. Secret
values remain outside Pulumi: callers name a pre-existing Kubernetes Secret, and the
Deployment references it through ``envFrom``.
"""

from dataclasses import dataclass

import pulumi
import pulumi_docker_build as docker_build
import pulumi_kubernetes as k8s
from finelog.deploy.config import FinelogConfig, auth_policy_json

CACHE_MOUNT_PATH = "/var/cache/finelog"
CACHE_VOLUME_NAME = "cache"
FINELOG_USER_ID = 1000
HEALTH_PATH = "/health"
NODE_ARCH = "amd64"


@dataclass(frozen=True)
class FinelogServerArgs:
    """Configuration for a Kubernetes Finelog deployment."""

    config: FinelogConfig
    build_context: str
    dockerfile: str
    cargo_profile: str
    cache_image: str
    env_secret_name: str | None
    deploy_generation: int
    adopt: bool = False

    def __post_init__(self) -> None:
        if self.config.deployment.k8s is None:
            raise ValueError("FinelogServer requires a Kubernetes deployment config")
        if (self.config.remote_log_dir.startswith("s3://") or self.config.forwarding) and not self.env_secret_name:
            raise ValueError("S3 archives and forwarding require env_secret_name")


@dataclass(frozen=True)
class FinelogResourceArgs:
    """Typed inputs for the Kubernetes objects owned by ``FinelogServer``."""

    pvc: k8s.core.v1.PersistentVolumeClaimInitArgs
    deployment: k8s.apps.v1.DeploymentInitArgs
    service: k8s.core.v1.ServiceInitArgs


def _container_env(config: FinelogConfig) -> list[k8s.core.v1.EnvVarArgs]:
    env = [
        k8s.core.v1.EnvVarArgs(name="FINELOG_PORT", value=str(config.port)),
        k8s.core.v1.EnvVarArgs(name="FINELOG_REMOTE_DIR", value=config.remote_log_dir),
    ]
    if config.query_metadata_cache_mb is not None:
        env.append(
            k8s.core.v1.EnvVarArgs(
                name="FINELOG_QUERY_METADATA_CACHE_MB",
                value=str(config.query_metadata_cache_mb),
            )
        )
    if config.auth:
        env.append(k8s.core.v1.EnvVarArgs(name="FINELOG_AUTH_POLICY", value=auth_policy_json(config.auth)))
    if config.forwarding:
        env.append(k8s.core.v1.EnvVarArgs(name="FINELOG_FORWARDING", value=config.forwarding.to_env_json()))
    return env


def finelog_resource_args(args: FinelogServerArgs, image_ref: pulumi.Input[str]) -> FinelogResourceArgs:
    """Build typed PVC, Deployment, and Service inputs for a Finelog server."""
    config = args.config
    assert config.deployment.k8s is not None
    deployment = config.deployment.k8s
    labels = {"app": config.name}
    cache_pvc_name = f"{config.name}-cache"
    metadata = k8s.meta.v1.ObjectMetaArgs(
        name=config.name,
        namespace=deployment.namespace,
        labels=labels,
    )
    health_action = k8s.core.v1.HTTPGetActionArgs(path=HEALTH_PATH, port=config.port)
    container = k8s.core.v1.ContainerArgs(
        name="finelog",
        image=image_ref,
        image_pull_policy="IfNotPresent",
        security_context=k8s.core.v1.SecurityContextArgs(capabilities=k8s.core.v1.CapabilitiesArgs(add=["SYS_PTRACE"])),
        ports=[k8s.core.v1.ContainerPortArgs(name="rpc", container_port=config.port, protocol="TCP")],
        env=_container_env(config),
        env_from=(
            [
                k8s.core.v1.EnvFromSourceArgs(
                    secret_ref=k8s.core.v1.SecretEnvSourceArgs(name=args.env_secret_name),
                )
            ]
            if args.env_secret_name
            else None
        ),
        volume_mounts=[
            k8s.core.v1.VolumeMountArgs(
                name=CACHE_VOLUME_NAME,
                mount_path=CACHE_MOUNT_PATH,
            )
        ],
        resources=k8s.core.v1.ResourceRequirementsArgs(
            requests={"cpu": deployment.cpu_request, "memory": deployment.memory_request},
            limits={"cpu": deployment.cpu_limit, "memory": deployment.memory_limit},
        ),
        # Opening an existing store on a network-backed PVC can exceed one minute.
        startup_probe=k8s.core.v1.ProbeArgs(
            http_get=health_action,
            period_seconds=10,
            timeout_seconds=15,
            failure_threshold=30,
        ),
        liveness_probe=k8s.core.v1.ProbeArgs(
            http_get=health_action,
            initial_delay_seconds=15,
            period_seconds=30,
            timeout_seconds=15,
            failure_threshold=3,
        ),
        readiness_probe=k8s.core.v1.ProbeArgs(
            http_get=health_action,
            initial_delay_seconds=5,
            period_seconds=10,
            timeout_seconds=15,
            failure_threshold=3,
        ),
    )
    pod_spec = k8s.core.v1.PodSpecArgs(
        # Native dependencies and rollback images are not uniformly multi-architecture.
        node_selector={"kubernetes.io/arch": NODE_ARCH},
        # The image runs as UID/GID 1000; fsGroup makes the mounted PVC writable.
        security_context=k8s.core.v1.PodSecurityContextArgs(fs_group=FINELOG_USER_ID),
        priority_class_name=deployment.priority_class_name,
        containers=[container],
        volumes=[
            k8s.core.v1.VolumeArgs(
                name=CACHE_VOLUME_NAME,
                persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(claim_name=cache_pvc_name),
            )
        ],
    )

    return FinelogResourceArgs(
        pvc=k8s.core.v1.PersistentVolumeClaimInitArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                name=cache_pvc_name,
                namespace=deployment.namespace,
                labels=labels,
            ),
            spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
                access_modes=["ReadWriteOnce"],
                storage_class_name=deployment.storage_class,
                resources=k8s.core.v1.VolumeResourceRequirementsArgs(requests={"storage": f"{deployment.storage_gb}Gi"}),
            ),
        ),
        deployment=k8s.apps.v1.DeploymentInitArgs(
            metadata=metadata,
            spec=k8s.apps.v1.DeploymentSpecArgs(
                replicas=1,
                # The store permits one writer; rollouts must stop the old pod first.
                strategy=k8s.apps.v1.DeploymentStrategyArgs(type="Recreate"),
                selector=k8s.meta.v1.LabelSelectorArgs(match_labels=labels),
                template=k8s.core.v1.PodTemplateSpecArgs(
                    metadata=k8s.meta.v1.ObjectMetaArgs(
                        labels=labels,
                        annotations={
                            "finelog.marin/deploy-generation": str(args.deploy_generation),
                        },
                    ),
                    spec=pod_spec,
                ),
            ),
        ),
        service=k8s.core.v1.ServiceInitArgs(
            metadata=metadata,
            spec=k8s.core.v1.ServiceSpecArgs(
                type="ClusterIP",
                selector=labels,
                ports=[
                    k8s.core.v1.ServicePortArgs(
                        name="rpc",
                        port=config.port,
                        target_port=config.port,
                        protocol="TCP",
                    )
                ],
            ),
        ),
    )


class FinelogServer(pulumi.ComponentResource):
    """Build and run a single-writer Finelog server on Kubernetes."""

    image_ref: pulumi.Output[str]
    namespace: pulumi.Output[str]
    service_name: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: FinelogServerArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:kubernetes:FinelogServer", name, None, opts)
        config = args.config
        assert config.deployment.k8s is not None
        namespace = config.deployment.k8s.namespace

        image = docker_build.Image(
            "image",
            context=docker_build.BuildContextArgs(location=args.build_context),
            dockerfile=docker_build.DockerfileArgs(location=f"{args.build_context}/{args.dockerfile}"),
            build_args={"CARGO_PROFILE": args.cargo_profile},
            cache_from=[
                docker_build.CacheFromArgs(
                    registry=docker_build.CacheFromRegistryArgs(ref=args.cache_image),
                )
            ],
            cache_to=[
                docker_build.CacheToArgs(
                    registry=docker_build.CacheToRegistryArgs(
                        ref=args.cache_image,
                        mode=docker_build.CacheMode.MAX,
                        compression=docker_build.CompressionType.ZSTD,
                        compression_level=3,
                        oci_media_types=True,
                        image_manifest=True,
                    )
                )
            ],
            platforms=[docker_build.Platform.LINUX_AMD64],
            tags=[config.image],
            push=True,
            build_on_preview=False,
            opts=pulumi.ResourceOptions(parent=self),
        )
        resources = finelog_resource_args(args, image.ref)

        def child_options(
            import_id: str,
            *,
            depends_on: list[pulumi.Resource] | None = None,
            protect: bool = False,
        ) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
                import_=import_id if args.adopt else None,
                protect=protect,
            )

        pvc = k8s.core.v1.PersistentVolumeClaim(
            "pvc",
            args=resources.pvc,
            opts=child_options(f"{namespace}/{config.name}-cache", protect=True),
        )
        deployment = k8s.apps.v1.Deployment(
            "deployment",
            args=resources.deployment,
            opts=child_options(f"{namespace}/{config.name}", depends_on=[pvc]),
        )
        k8s.core.v1.Service(
            "service",
            args=resources.service,
            opts=child_options(f"{namespace}/{config.name}", depends_on=[deployment]),
        )

        self.image_ref = image.ref
        self.namespace = pulumi.Output.from_input(namespace)
        self.service_name = pulumi.Output.from_input(config.name)
        self.register_outputs(
            {
                "image_ref": self.image_ref,
                "namespace": self.namespace,
                "service_name": self.service_name,
            }
        )
