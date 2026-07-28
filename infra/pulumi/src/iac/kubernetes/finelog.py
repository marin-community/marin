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
class FinelogManifests:
    """The Kubernetes object shapes owned by ``FinelogServer``."""

    pvc: dict
    deployment: dict
    service: dict


def _container_env(config: FinelogConfig) -> list[dict[str, str]]:
    env = [
        {"name": "FINELOG_PORT", "value": str(config.port)},
        {"name": "FINELOG_REMOTE_DIR", "value": config.remote_log_dir},
    ]
    if config.query_metadata_cache_mb is not None:
        env.append({"name": "FINELOG_QUERY_METADATA_CACHE_MB", "value": str(config.query_metadata_cache_mb)})
    if config.auth:
        env.append({"name": "FINELOG_AUTH_POLICY", "value": auth_policy_json(config.auth)})
    if config.forwarding:
        env.append({"name": "FINELOG_FORWARDING", "value": config.forwarding.to_env_json()})
    return env


def finelog_manifests(args: FinelogServerArgs, image_ref: pulumi.Input[str]) -> FinelogManifests:
    """Build the PVC, Deployment, and Service manifests for a Finelog server."""
    config = args.config
    assert config.deployment.k8s is not None
    deployment = config.deployment.k8s
    labels = {"app": config.name}

    pvc_spec: dict = {
        "accessModes": ["ReadWriteOnce"],
        "resources": {"requests": {"storage": f"{deployment.storage_gb}Gi"}},
    }
    if deployment.storage_class:
        pvc_spec["storageClassName"] = deployment.storage_class

    pod_spec: dict = {
        # Native dependencies and rollback images are not uniformly multi-architecture.
        "nodeSelector": {"kubernetes.io/arch": NODE_ARCH},
        # The image runs as UID/GID 1000; fsGroup makes the mounted PVC writable.
        "securityContext": {"fsGroup": FINELOG_USER_ID},
        "containers": [
            {
                "name": "finelog",
                "image": image_ref,
                "imagePullPolicy": "IfNotPresent",
                "securityContext": {"capabilities": {"add": ["SYS_PTRACE"]}},
                "ports": [{"name": "rpc", "containerPort": config.port, "protocol": "TCP"}],
                "env": _container_env(config),
                "volumeMounts": [{"name": "cache", "mountPath": CACHE_MOUNT_PATH}],
                "resources": {
                    "requests": {
                        "cpu": deployment.cpu_request,
                        "memory": deployment.memory_request,
                    },
                    "limits": {
                        "cpu": deployment.cpu_limit,
                        "memory": deployment.memory_limit,
                    },
                },
                # Opening an existing store on a network-backed PVC can exceed one minute.
                "startupProbe": {
                    "httpGet": {"path": HEALTH_PATH, "port": config.port},
                    "periodSeconds": 10,
                    "timeoutSeconds": 15,
                    "failureThreshold": 30,
                },
                "livenessProbe": {
                    "httpGet": {"path": HEALTH_PATH, "port": config.port},
                    "initialDelaySeconds": 15,
                    "periodSeconds": 30,
                    "timeoutSeconds": 15,
                    "failureThreshold": 3,
                },
                "readinessProbe": {
                    "httpGet": {"path": HEALTH_PATH, "port": config.port},
                    "initialDelaySeconds": 5,
                    "periodSeconds": 10,
                    "timeoutSeconds": 15,
                    "failureThreshold": 3,
                },
            }
        ],
        "volumes": [{"name": "cache", "persistentVolumeClaim": {"claimName": f"{config.name}-cache"}}],
    }
    if deployment.priority_class_name:
        pod_spec["priorityClassName"] = deployment.priority_class_name
    if args.env_secret_name:
        pod_spec["containers"][0]["envFrom"] = [{"secretRef": {"name": args.env_secret_name}}]

    return FinelogManifests(
        pvc={
            "metadata": {
                "name": f"{config.name}-cache",
                "namespace": deployment.namespace,
                "labels": labels,
            },
            "spec": pvc_spec,
        },
        deployment={
            "metadata": {
                "name": config.name,
                "namespace": deployment.namespace,
                "labels": labels,
            },
            "spec": {
                "replicas": 1,
                # The store permits one writer; rollouts must stop the old pod first.
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {
                        "labels": labels,
                        "annotations": {
                            "finelog.marin/deploy-generation": str(args.deploy_generation),
                        },
                    },
                    "spec": pod_spec,
                },
            },
        },
        service={
            "metadata": {
                "name": config.name,
                "namespace": deployment.namespace,
                "labels": labels,
            },
            "spec": {
                "type": "ClusterIP",
                "selector": labels,
                "ports": [
                    {
                        "name": "rpc",
                        "port": config.port,
                        "targetPort": config.port,
                        "protocol": "TCP",
                    }
                ],
            },
        },
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
        manifests = finelog_manifests(args, image.ref)

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
            metadata=manifests.pvc["metadata"],
            spec=manifests.pvc["spec"],
            opts=child_options(f"{namespace}/{config.name}-cache", protect=True),
        )
        deployment = k8s.apps.v1.Deployment(
            "deployment",
            metadata=manifests.deployment["metadata"],
            spec=manifests.deployment["spec"],
            opts=child_options(f"{namespace}/{config.name}", depends_on=[pvc]),
        )
        k8s.core.v1.Service(
            "service",
            metadata=manifests.service["metadata"],
            spec=manifests.service["spec"],
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
