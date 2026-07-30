# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An IAP-gated internal Cloud Run service, built from a local Dockerfile.

The generic shape behind Marin's single-instance internal web services: build the image
from a Dockerfile, push it digest-pinned to a per-service Artifact Registry repo, and run
it on Cloud Run v2 with Direct VPC egress so it reaches cluster-internal IPs, gated by
Identity-Aware Proxy.

The component owns everything a deploy needs: the runtime service account and its
project roles, the Artifact Registry repo and image, the service, and the IAP wiring
(the service is invokable only by the IAP service agent; people reach it through IAP's
``httpsResourceAccessor``). The one project-level prerequisite it does not own is the
OAuth consent screen, which is shared across a project's IAP services.
"""

import re
from dataclasses import dataclass, field

import pulumi
import pulumi_docker_build as docker_build
import pulumi_gcp as gcp

# Cloud Run terminates the browser session as the IAP service agent, so that agent —
# not the end user — is what invokes the service. People are admitted separately, through
# IAP's httpsResourceAccessor role.
IAP_SERVICE_AGENT = "serviceAccount:service-{project_number}@gcp-sa-iap.iam.gserviceaccount.com"
OPENATHENA_IAP_MEMBER = "domain:openathena.ai"


@dataclass(frozen=True)
class SecretEnv:
    """A Secret Manager secret exposed to the container as an environment variable.

    ``name`` is the variable the container reads; ``secret`` is the Secret Manager secret id
    in the service's project; ``version`` is the version to mount ("latest" or a number). The
    component grants the runtime service account roles/secretmanager.secretAccessor on the
    secret — it references the secret, and never creates it or holds its value.

    When the same stack creates the secret (or its version), pass those resources in
    ``wait_for``: the string id carries no dependency edge, so without it the accessor
    grant and the service/job that mounts the secret can race its creation on a fresh
    deploy (Cloud Run validates secret access and version existence at deploy time).
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

    # Project roles granted to the runtime service account (e.g. roles/compute.viewer for
    # a service that lists VM internal IPs).
    service_account_roles: tuple[str, ...] = ()
    # Secret Manager secrets mounted as container env vars. Each grants the runtime service
    # account roles/secretmanager.secretAccessor on its secret; the component references the
    # secret and never creates it or holds its value.
    secrets: tuple[SecretEnv, ...] = ()
    # Additional people admitted through IAP beyond the organization-wide OpenAthena domain
    # grant. Each entry is a bare email ("alice@x.com"), a domain wildcard ("*@example.com"),
    # or an already-qualified IAM member ("group:eng@example.com"). Each grant is its own
    # resource, so re-running with a changed list updates only the added/removed grants —
    # never the service.
    iap_members: tuple[str, ...] = ()
    # OAuth client IDs IAP accepts as a programmatic-token audience, so a CLI or agent can
    # reach the service with a Google-signed ID token (browser desktop-login or a
    # service-account-minted token) instead of the interactive browser session. Empty leaves
    # the service browser-only. Each id must be an OAuth client that already exists.
    iap_programmatic_clients: tuple[str, ...] = ()
    # Cloud SQL connection names (project:region:instance) to attach. When non-empty the
    # service mounts the connector socket at /cloudsql and the runtime service account gets
    # roles/cloudsql.client on the project.
    cloudsql_instances: tuple[str, ...] = ()


IAM_MEMBER_PREFIXES = ("user:", "group:", "domain:", "serviceAccount:")
IAM_SPECIAL_MEMBERS = ("allUsers", "allAuthenticatedUsers")


def normalize_iap_member(entry: str) -> str:
    """Map a friendly IAP access entry to an IAM member.

    Passes an already-qualified member ("group:eng@x.com") or special token
    ("allAuthenticatedUsers") through unchanged; maps "*@domain" to "domain:domain" and a
    bare email to "user:email".
    """
    entry = entry.strip()
    if entry in IAM_SPECIAL_MEMBERS or entry.startswith(IAM_MEMBER_PREFIXES):
        return entry
    if entry.startswith("*@"):
        return f"domain:{entry[2:]}"
    if "@" in entry:
        return f"user:{entry}"
    raise ValueError(f"cannot read IAP access entry {entry!r}: use an email, *@domain, or a prefixed IAM member")


def iap_access_members(additional_members: tuple[str, ...]) -> tuple[str, ...]:
    """Add the OpenAthena Workspace domain to a service's normalized IAP exceptions."""
    members = (OPENATHENA_IAP_MEMBER, *(normalize_iap_member(member) for member in additional_members))
    return tuple(dict.fromkeys(members))


def _role_slug(role: str) -> str:
    """Pulumi resource-name-safe slug for an IAM role id (roles/compute.viewer -> compute-viewer)."""
    return role.removeprefix("roles/").replace(".", "-").replace("/", "-")


def resource_slug(identifier: str) -> str:
    """Stable resource-name-safe slug for an identifier (IAM member, secret id), so each
    grant is its own resource."""
    return re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")


def runtime_service_account(
    *,
    account_id: str,
    display_name: str,
    project: str,
    roles: tuple[str, ...],
    secrets: tuple[SecretEnv, ...],
    cloudsql_instances: tuple[str, ...],
    opts: pulumi.ResourceOptions,
) -> tuple[gcp.serviceaccount.Account, list[pulumi.Resource]]:
    """Runtime service account with its project roles, cloudsql.client, and secret accessor grants.

    Returns the account and its IAM grant resources: Cloud Run validates secret access at
    deploy time, so the service/job resource must depends_on the grants or a fresh deploy
    can race them. Shared between the Cloud Run service and job components; child resource
    names ("sa", "sa-<role>", "sa-cloudsql-client", "secret-<slug>") are part of existing
    stacks' state, so they must stay stable.
    """
    service_account = gcp.serviceaccount.Account(
        "sa",
        account_id=account_id,
        project=project,
        display_name=display_name,
        opts=opts,
    )
    member = service_account.email.apply(lambda email: f"serviceAccount:{email}")
    grants: list[pulumi.Resource] = []
    for role in roles:
        grants.append(
            gcp.projects.IAMMember(
                f"sa-{_role_slug(role)}",
                project=project,
                role=role,
                member=member,
                opts=opts,
            )
        )
    if cloudsql_instances:
        grants.append(
            gcp.projects.IAMMember(
                "sa-cloudsql-client",
                project=project,
                role="roles/cloudsql.client",
                member=member,
                opts=opts,
            )
        )
    for secret_env in secrets:
        grants.append(
            gcp.secretmanager.SecretIamMember(
                f"secret-{resource_slug(secret_env.secret)}",
                project=project,
                secret_id=secret_env.secret,
                role="roles/secretmanager.secretAccessor",
                member=member,
                opts=pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(depends_on=list(secret_env.wait_for))),
            )
        )
    return service_account, grants


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
    return docker_build.Image(
        "image",
        context=docker_build.BuildContextArgs(location=build_context),
        dockerfile=docker_build.DockerfileArgs(location=f"{build_context}/{dockerfile}"),
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

        service_account, sa_grants = runtime_service_account(
            account_id=args.service_account_id or args.service_name,
            display_name=f"{args.service_name} (Cloud Run)",
            project=args.project,
            roles=args.service_account_roles,
            secrets=args.secrets,
            cloudsql_instances=args.cloudsql_instances,
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
            # Cloud Run validates secret access and version existence at deploy, so the
            # grants and any stack-created secrets must exist before the service.
            opts=pulumi.ResourceOptions.merge(
                child,
                pulumi.ResourceOptions(depends_on=sa_grants + [r for s in args.secrets for r in s.wait_for]),
            ),
        )

        # IAP invokes the service as its own service agent; only that agent gets run.invoker.
        # People are admitted separately through IAP (httpsResourceAccessor). The shared
        # organization grant keeps every internal site consistent; `iap_members` adds only
        # service-specific exceptions.
        project_number = gcp.organizations.get_project(
            project_id=args.project, opts=pulumi.InvokeOptions(provider=gcp_provider)
        ).number
        gcp.cloudrunv2.ServiceIamMember(
            "iap-invoker",
            project=args.project,
            location=args.region,
            name=service.name,
            role="roles/run.invoker",
            member=IAP_SERVICE_AGENT.format(project_number=project_number),
            opts=child,
        )
        for member in iap_access_members(args.iap_members):
            gcp.iap.WebCloudRunServiceIamMember(
                f"iap-access-{resource_slug(member)}",
                project=args.project,
                location=args.region,
                cloud_run_service_name=service.name,
                role="roles/iap.httpsResourceAccessor",
                member=member,
                opts=child,
            )

        # Register programmatic-token audiences on the service's IAP resource. IAP then admits
        # an ID token whose `aud` is one of these client ids and attributes the caller by its
        # email claim — the path a CLI or agent uses instead of the interactive browser sign-in.
        if args.iap_programmatic_clients:
            gcp.iap.Settings(
                "iap-settings",
                name=service.name.apply(
                    lambda name: f"projects/{project_number}/iap_web/cloud_run-{args.region}/services/{name}"
                ),
                access_settings=gcp.iap.SettingsAccessSettingsArgs(
                    oauth_settings=gcp.iap.SettingsAccessSettingsOauthSettingsArgs(
                        programmatic_clients=list(args.iap_programmatic_clients),
                    ),
                ),
                opts=child,
            )

        self.uri = service.uri
        self.image_ref = image.ref
        self.register_outputs({"uri": self.uri, "image_ref": self.image_ref})
