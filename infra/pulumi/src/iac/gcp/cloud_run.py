# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An IAP-gated internal Cloud Run service, built from a local Dockerfile.

The generic shape behind Marin's single-instance internal web services: build the image
from a Dockerfile, push it digest-pinned to a per-service Artifact Registry repo, and run
it on Cloud Run v2 with Direct VPC egress so it reaches cluster-internal IPs, gated by
Identity-Aware Proxy.

The component owns the runtime service account, Artifact Registry repo and image, service,
and IAP settings. The ``marin`` infrastructure stack owns every IAM grant for the service
and runtime account. The project-level OAuth consent screen remains external.
"""

import re
from dataclasses import dataclass, field

import pulumi
import pulumi_docker_build as docker_build
import pulumi_gcp as gcp
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT

BUILD_CACHE_TAG = "buildcache"
BUILD_CACHE_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class SecretEnv:
    """A Secret Manager secret exposed to the container as an environment variable.

    ``name`` is the variable the container reads; ``secret`` is the Secret Manager secret id
    in the service's project; ``version`` is the version to mount ("latest" or a number). The
    ``iam_data.yaml`` grants the runtime service account access. This component references the
    secret and never creates it or holds its value.

    When the same stack creates the secret (or its version), pass those resources in
    ``wait_for``: the string id carries no dependency edge, so the service/job that mounts
    the secret can race its creation on a fresh deploy.
    """

    name: str
    secret: str
    version: str = "latest"
    wait_for: tuple[pulumi.Resource, ...] = ()


@dataclass(frozen=True)
class CloudRunServiceArgs:
    project: str
    region: str
    service_name: str

    # Image build. `build_context` is the directory sent to buildx; `dockerfile` is
    # resolved within it. The pushed image is referenced by digest so a redeploy that
    # rebuilds identical bytes is a no-op.
    build_context: str
    dockerfile: str = "Dockerfile"

    # Container runtime. Cloud Run injects PORT and expects the app to listen on it;
    # `port` is the advertised container port (Cloud Run's PORT matches it).
    port: int = 8080
    env: dict[str, str] = field(default_factory=dict)
    cpu: str = "2"
    memory: str = "2Gi"
    # Keep CPU allocated between requests. Cloud Run's default throttles CPU to near-zero
    # off the request path, which stalls a service whose background work runs while idle
    # (an apiserver, indexers, reconcilers). True also enables startup CPU boost.
    cpu_always_allocated: bool = False
    request_timeout: int = 60
    max_instance_request_concurrency: int | None = None
    # Service-level min == max == 1 for a service whose local SQLite is per-instance:
    # >1 diverges alert and dashboard state, while 0 stops alert evaluation and makes
    # first paint a cold start. Cloud Run can temporarily overlap instances while a new
    # revision becomes ready, but only one instance serves traffic in steady state.
    min_instances: int = 1
    max_instances: int = 1

    # Direct VPC egress: the service dials cluster-internal IPs, so it needs an interface
    # on the VPC with private-ranges-only egress (public traffic still goes direct).
    network: str = "default"
    subnet: str = "default"

    # Runtime service-account id. Defaults to service_name. Override to keep an existing
    # account: GCP cannot rename a service account in place, so a service whose account was
    # created under a different name pins that name here rather than orphaning it.
    service_account_id: str | None = None

    # Secret Manager secrets mounted as container env vars. IAM access is declared centrally;
    # the component references each secret and never creates it or holds its value.
    secrets: tuple[SecretEnv, ...] = ()
    # Additional OAuth client IDs IAP accepts as programmatic-token audiences beyond the
    # shared Marin desktop client. This lets a CLI or agent reach the service with a
    # Google-signed ID token instead of an interactive browser session. Each id must be an
    # OAuth client that already exists.
    iap_programmatic_clients: tuple[str, ...] = ()
    # Cloud SQL connection names (project:region:instance) to attach. The service mounts the
    # connector socket at /cloudsql; its centrally declared IAM includes cloudsql.client.
    cloudsql_instances: tuple[str, ...] = ()


def resource_slug(identifier: str) -> str:
    """Return a stable Pulumi resource-name-safe identifier."""
    return re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")


def runtime_service_account(
    *,
    account_id: str,
    display_name: str,
    project: str,
    opts: pulumi.ResourceOptions,
) -> gcp.serviceaccount.Account:
    """Create the runtime service account shared by Cloud Run services and jobs."""
    return gcp.serviceaccount.Account(
        "sa",
        account_id=account_id,
        project=project,
        display_name=display_name,
        opts=opts,
    )


def dockerfile_image(
    *,
    image_name: str,
    description: str,
    project: str,
    region: str,
    build_context: str,
    dockerfile: str,
    parent: pulumi.ComponentResource,
    gcp_provider: pulumi.ProviderResource,
) -> docker_build.Image:
    """Per-deployable Artifact Registry repo + digest-pinned linux/amd64 image from a Dockerfile."""
    repo = gcp.artifactregistry.Repository(
        "repo",
        project=project,
        location=region,
        repository_id=image_name,
        format="DOCKER",
        description=description,
        opts=pulumi.ResourceOptions(parent=parent, provider=gcp_provider),
    )
    image_tag = repo.repository_id.apply(
        lambda repo_id: f"{region}-docker.pkg.dev/{project}/{repo_id}/{image_name}:latest"
    )
    cache_ref = repo.repository_id.apply(
        lambda repo_id: f"{region}-docker.pkg.dev/{project}/{repo_id}/{image_name}:{BUILD_CACHE_TAG}"
    )
    return docker_build.Image(
        "image",
        context=docker_build.BuildContextArgs(location=build_context),
        dockerfile=docker_build.DockerfileArgs(location=f"{build_context}/{dockerfile}"),
        # GitHub-hosted runners have no durable local BuildKit state. Export the full
        # cache beside the image so later CI and local builds can reuse every stage.
        cache_from=[
            docker_build.CacheFromArgs(
                registry=docker_build.CacheFromRegistryArgs(ref=cache_ref),
            )
        ],
        cache_to=[
            docker_build.CacheToArgs(
                registry=docker_build.CacheToRegistryArgs(
                    ref=cache_ref,
                    mode=docker_build.CacheMode.MAX,
                    compression=docker_build.CompressionType.ZSTD,
                    compression_level=BUILD_CACHE_COMPRESSION_LEVEL,
                    oci_media_types=True,
                    image_manifest=True,
                ),
            )
        ],
        # Cloud Run is linux/amd64; pin it so a build from an arm64 workstation still
        # produces a runnable image.
        platforms=[docker_build.Platform.LINUX_AMD64],
        tags=[image_tag],
        push=True,
        # Preview plans the graph without invoking buildx; the build + push happen on up.
        build_on_preview=False,
        opts=pulumi.ResourceOptions(parent=parent, provider=gcp_provider, depends_on=[repo]),
    )


class CloudRunService(pulumi.ComponentResource):
    """Build, push, and run a Dockerfile as an IAP-gated Cloud Run v2 service.

    Exposes ``uri`` (the service URL, reachable only through IAP) and ``image_ref`` (the
    digest-pinned image the service runs).
    """

    uri: pulumi.Output[str]
    image_ref: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: CloudRunServiceArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:CloudRunService", name, None, opts)
        child = pulumi.ResourceOptions(parent=self, provider=gcp_provider)

        service_account = runtime_service_account(
            account_id=args.service_account_id or args.service_name,
            display_name=f"{args.service_name} (Cloud Run)",
            project=args.project,
            opts=child,
        )
        image = dockerfile_image(
            image_name=args.service_name,
            description=f"Images for the {args.service_name} Cloud Run service.",
            project=args.project,
            region=args.region,
            build_context=args.build_context,
            dockerfile=args.dockerfile,
            parent=self,
            gcp_provider=gcp_provider,
        )

        # Cloud SQL connector: a "cloudsql" volume exposes the auth-proxy sockets under
        # /cloudsql, one per attached connection name. Empty when no instances are attached.
        cloudsql_volumes = (
            [
                gcp.cloudrunv2.ServiceTemplateVolumeArgs(
                    name="cloudsql",
                    cloud_sql_instance=gcp.cloudrunv2.ServiceTemplateVolumeCloudSqlInstanceArgs(
                        instances=list(args.cloudsql_instances),
                    ),
                )
            ]
            if args.cloudsql_instances
            else []
        )
        cloudsql_volume_mounts = (
            [gcp.cloudrunv2.ServiceTemplateContainerVolumeMountArgs(name="cloudsql", mount_path="/cloudsql")]
            if args.cloudsql_instances
            else []
        )

        service = gcp.cloudrunv2.Service(
            "service",
            name=args.service_name,
            project=args.project,
            location=args.region,
            # IAP is the gate; ingress stays open so IAP (not the network) authorizes.
            ingress="INGRESS_TRAFFIC_ALL",
            iap_enabled=True,
            scaling=gcp.cloudrunv2.ServiceScalingArgs(
                min_instance_count=args.min_instances,
                max_instance_count=args.max_instances,
            ),
            template=gcp.cloudrunv2.ServiceTemplateArgs(
                service_account=service_account.email,
                timeout=f"{args.request_timeout}s",
                max_instance_request_concurrency=args.max_instance_request_concurrency,
                vpc_access=gcp.cloudrunv2.ServiceTemplateVpcAccessArgs(
                    egress="PRIVATE_RANGES_ONLY",
                    network_interfaces=[
                        gcp.cloudrunv2.ServiceTemplateVpcAccessNetworkInterfaceArgs(
                            network=args.network,
                            subnetwork=args.subnet,
                        )
                    ],
                ),
                volumes=cloudsql_volumes,
                containers=[
                    gcp.cloudrunv2.ServiceTemplateContainerArgs(
                        image=image.ref,
                        ports=gcp.cloudrunv2.ServiceTemplateContainerPortsArgs(container_port=args.port),
                        envs=[
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=key, value=value)
                            for key, value in args.env.items()
                        ]
                        + [
                            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                                name=secret_env.name,
                                value_source=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                                    secret_key_ref=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                        secret=secret_env.secret,
                                        version=secret_env.version,
                                    )
                                ),
                            )
                            for secret_env in args.secrets
                        ],
                        resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                            limits={"cpu": args.cpu, "memory": args.memory},
                            cpu_idle=not args.cpu_always_allocated,
                            startup_cpu_boost=args.cpu_always_allocated,
                        ),
                        volume_mounts=cloudsql_volume_mounts,
                    )
                ],
            ),
            # Cloud Run validates secret access and version existence at deploy. IAM access
            # comes from the marin stack; wait here for secrets created by this stack.
            opts=pulumi.ResourceOptions.merge(
                child,
                pulumi.ResourceOptions(depends_on=[r for secret in args.secrets for r in secret.wait_for]),
            ),
        )

        project_number = gcp.organizations.get_project(
            project_id=args.project, opts=pulumi.InvokeOptions(provider=gcp_provider)
        ).number
        # Register programmatic-token audiences on the service's IAP resource. IAP then admits
        # an ID token whose `aud` is one of these client ids and attributes the caller by its
        # email claim — the path a CLI or agent uses instead of the interactive browser sign-in.
        programmatic_clients = (
            MARIN_DESKTOP_OAUTH_CLIENT.client_id,
            *args.iap_programmatic_clients,
        )
        gcp.iap.Settings(
            "iap-settings",
            name=service.name.apply(
                lambda name: f"projects/{project_number}/iap_web/cloud_run-{args.region}/services/{name}"
            ),
            access_settings=gcp.iap.SettingsAccessSettingsArgs(
                oauth_settings=gcp.iap.SettingsAccessSettingsOauthSettingsArgs(
                    programmatic_clients=list(dict.fromkeys(programmatic_clients)),
                ),
            ),
            opts=child,
        )

        self.uri = service.uri
        self.image_ref = image.ref
        self.register_outputs({"uri": self.uri, "image_ref": self.image_ref})
