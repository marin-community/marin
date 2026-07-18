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


@dataclass(frozen=True)
class SecretEnv:
    """A Secret Manager secret exposed to the container as an environment variable.

    ``name`` is the variable the container reads; ``secret`` is the Secret Manager secret id
    in the service's project; ``version`` is the version to mount ("latest" or a number). The
    component grants the runtime service account roles/secretmanager.secretAccessor on the
    secret — it references the secret, and never creates it or holds its value.
    """

    name: str
    secret: str
    version: str = "latest"


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
    # min == max == 1 for a service whose local SQLite is per-instance: >1 diverges alert
    # and dashboard state, 0 stops alert evaluation and makes first paint a cold start.
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
    # People admitted through IAP. Each entry is a bare email ("alice@x.com"), a domain
    # wildcard ("*@openathena.ai"), or an already-qualified IAM member ("group:eng@x.com").
    # Each grant is its own resource, so re-running with a changed list updates only the
    # added/removed grants — never the service.
    iap_members: tuple[str, ...] = ()


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


def _role_slug(role: str) -> str:
    """Pulumi resource-name-safe slug for an IAM role id (roles/compute.viewer -> compute-viewer)."""
    return role.removeprefix("roles/").replace(".", "-").replace("/", "-")


def _member_slug(member: str) -> str:
    """Stable resource-name-safe slug for an IAM member, so each grant is its own resource."""
    return re.sub(r"[^a-z0-9]+", "-", member.lower()).strip("-")


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

        service_account = gcp.serviceaccount.Account(
            "sa",
            account_id=args.service_account_id or args.service_name,
            project=args.project,
            display_name=f"{args.service_name} (Cloud Run)",
            opts=child,
        )
        member = service_account.email.apply(lambda email: f"serviceAccount:{email}")
        for role in args.service_account_roles:
            gcp.projects.IAMMember(
                f"sa-{_role_slug(role)}",
                project=args.project,
                role=role,
                member=member,
                opts=child,
            )
        for secret_env in args.secrets:
            gcp.secretmanager.SecretIamMember(
                f"secret-{_member_slug(secret_env.secret)}",
                project=args.project,
                secret_id=secret_env.secret,
                role="roles/secretmanager.secretAccessor",
                member=member,
                opts=child,
            )

        repo = gcp.artifactregistry.Repository(
            "repo",
            project=args.project,
            location=args.region,
            repository_id=args.service_name,
            format="DOCKER",
            description=f"Images for the {args.service_name} Cloud Run service.",
            opts=child,
        )
        image_tag = repo.repository_id.apply(
            lambda repo_id: f"{args.region}-docker.pkg.dev/{args.project}/{repo_id}/{args.service_name}:latest"
        )
        image = docker_build.Image(
            "image",
            context=docker_build.BuildContextArgs(location=args.build_context),
            dockerfile=docker_build.DockerfileArgs(location=f"{args.build_context}/{args.dockerfile}"),
            # Cloud Run is linux/amd64; pin it so a build from an arm64 workstation still
            # produces a runnable image.
            platforms=[docker_build.Platform.LINUX_AMD64],
            tags=[image_tag],
            push=True,
            # Preview plans the graph without invoking buildx; the build + push happen on up.
            build_on_preview=False,
            opts=pulumi.ResourceOptions(parent=self, provider=gcp_provider, depends_on=[repo]),
        )

        service = gcp.cloudrunv2.Service(
            "service",
            name=args.service_name,
            project=args.project,
            location=args.region,
            # IAP is the gate; ingress stays open so IAP (not the network) authorizes.
            ingress="INGRESS_TRAFFIC_ALL",
            iap_enabled=True,
            template=gcp.cloudrunv2.ServiceTemplateArgs(
                service_account=service_account.email,
                timeout=f"{args.request_timeout}s",
                scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                    min_instance_count=args.min_instances,
                    max_instance_count=args.max_instances,
                ),
                vpc_access=gcp.cloudrunv2.ServiceTemplateVpcAccessArgs(
                    egress="PRIVATE_RANGES_ONLY",
                    network_interfaces=[
                        gcp.cloudrunv2.ServiceTemplateVpcAccessNetworkInterfaceArgs(
                            network=args.network,
                            subnetwork=args.subnet,
                        )
                    ],
                ),
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
                    )
                ],
            ),
            opts=child,
        )

        # IAP invokes the service as its own service agent; only that agent gets run.invoker.
        # People are admitted separately through IAP (httpsResourceAccessor); an empty
        # `iap_members` leaves the service reachable by nobody until access is granted.
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
        for raw_member in args.iap_members:
            member = normalize_iap_member(raw_member)
            gcp.iap.WebCloudRunServiceIamMember(
                f"iap-access-{_member_slug(member)}",
                project=args.project,
                location=args.region,
                cloud_run_service_name=service.name,
                role="roles/iap.httpsResourceAccessor",
                member=member,
                opts=child,
            )

        self.uri = service.uri
        self.image_ref = image.ref
        self.register_outputs({"uri": self.uri, "image_ref": self.image_ref})
