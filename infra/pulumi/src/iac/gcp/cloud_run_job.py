# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scheduled Cloud Run job, built from a local Dockerfile.

The generic shape behind Marin's periodic batch work: build the image from a Dockerfile,
push it digest-pinned to a per-job Artifact Registry repo, run it as a Cloud Run v2 job,
and trigger it on a cron schedule through Cloud Scheduler.

The component owns everything a deploy needs: the runtime service account and its grants,
the Artifact Registry repo and image, the job, and the Scheduler trigger (which invokes the
Cloud Run Admin API as the job's own service account via an OAuth token). Secret Manager
secrets are referenced, never created — the component grants the service account accessor
on each and mounts them as env vars.
"""

from dataclasses import dataclass, field

import pulumi
import pulumi_gcp as gcp

from iac.gcp.cloud_run import SecretEnv, dockerfile_image, runtime_service_account

# The Cloud Run Admin API endpoint Scheduler POSTs to; *.googleapis.com targets take an
# OAuth token (not OIDC, which is for end-user-facing URLs like Cloud Run services).
RUN_JOB_URI = "https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job}:run"


@dataclass(frozen=True)
class ScheduledCloudRunJobArgs:
    project: str
    region: str
    job_name: str

    # Image build. `build_context` is the directory sent to buildx; `dockerfile` is
    # resolved within it. The pushed image is referenced by digest so a redeploy that
    # rebuilds identical bytes is a no-op.
    build_context: str
    dockerfile: str = "Dockerfile"

    # Cron schedule (Cloud Scheduler syntax) and its interpretation timezone.
    schedule: str = "0 */6 * * *"
    time_zone: str = "Etc/UTC"

    env: dict[str, str] = field(default_factory=dict)
    cpu: str = "1"
    memory: str = "2Gi"
    # Per-attempt timeout; a hung run is killed rather than blocking the next trigger.
    timeout: int = 1800
    max_retries: int = 1

    # Secret Manager secrets mounted as container env vars. Each grants the runtime service
    # account roles/secretmanager.secretAccessor on its secret.
    secrets: tuple[SecretEnv, ...] = ()
    # Cloud SQL connection names (project:region:instance) to attach. When non-empty the
    # job mounts the connector socket at /cloudsql and the runtime service account gets
    # roles/cloudsql.client on the project.
    cloudsql_instances: tuple[str, ...] = ()


class ScheduledCloudRunJob(pulumi.ComponentResource):
    """Build, push, and schedule a Dockerfile as a Cloud Run v2 job.

    Exposes ``job_name`` and ``image_ref`` (the digest-pinned image the job runs).
    """

    job_name: pulumi.Output[str]
    image_ref: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: ScheduledCloudRunJobArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:ScheduledCloudRunJob", name, None, opts)
        child = pulumi.ResourceOptions(parent=self, provider=gcp_provider)

        service_account = runtime_service_account(
            account_id=args.job_name,
            display_name=f"{args.job_name} (Cloud Run job)",
            project=args.project,
            roles=(),
            secrets=args.secrets,
            cloudsql_instances=args.cloudsql_instances,
            opts=child,
        )
        member = service_account.email.apply(lambda email: f"serviceAccount:{email}")
        image = dockerfile_image(
            image_name=args.job_name,
            description=f"Images for the {args.job_name} Cloud Run job.",
            project=args.project,
            region=args.region,
            build_context=args.build_context,
            dockerfile=args.dockerfile,
            parent=self,
            gcp_provider=gcp_provider,
        )

        cloudsql_volumes = (
            [
                gcp.cloudrunv2.JobTemplateTemplateVolumeArgs(
                    name="cloudsql",
                    cloud_sql_instance=gcp.cloudrunv2.JobTemplateTemplateVolumeCloudSqlInstanceArgs(
                        instances=list(args.cloudsql_instances),
                    ),
                )
            ]
            if args.cloudsql_instances
            else []
        )
        cloudsql_volume_mounts = (
            [gcp.cloudrunv2.JobTemplateTemplateContainerVolumeMountArgs(name="cloudsql", mount_path="/cloudsql")]
            if args.cloudsql_instances
            else []
        )

        job = gcp.cloudrunv2.Job(
            "job",
            name=args.job_name,
            project=args.project,
            location=args.region,
            template=gcp.cloudrunv2.JobTemplateArgs(
                template=gcp.cloudrunv2.JobTemplateTemplateArgs(
                    service_account=service_account.email,
                    timeout=f"{args.timeout}s",
                    max_retries=args.max_retries,
                    volumes=cloudsql_volumes,
                    containers=[
                        gcp.cloudrunv2.JobTemplateTemplateContainerArgs(
                            image=image.ref,
                            envs=[
                                gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(name=key, value=value)
                                for key, value in args.env.items()
                            ]
                            + [
                                gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
                                    name=secret_env.name,
                                    value_source=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceArgs(
                                        secret_key_ref=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                            secret=secret_env.secret,
                                            version=secret_env.version,
                                        )
                                    ),
                                )
                                for secret_env in args.secrets
                            ],
                            resources=gcp.cloudrunv2.JobTemplateTemplateContainerResourcesArgs(
                                limits={"cpu": args.cpu, "memory": args.memory},
                            ),
                            volume_mounts=cloudsql_volume_mounts,
                        )
                    ],
                ),
            ),
            opts=child,
        )

        # Scheduler runs the job as the job's own service account, which therefore needs
        # run.invoker on the job (jobs.run permission).
        gcp.cloudrunv2.JobIamMember(
            "sa-invoker",
            project=args.project,
            location=args.region,
            name=job.name,
            role="roles/run.invoker",
            member=member,
            opts=child,
        )
        gcp.cloudscheduler.Job(
            "trigger",
            name=f"{args.job_name}-trigger",
            project=args.project,
            region=args.region,
            schedule=args.schedule,
            time_zone=args.time_zone,
            http_target=gcp.cloudscheduler.JobHttpTargetArgs(
                http_method="POST",
                uri=RUN_JOB_URI.format(project=args.project, region=args.region, job=args.job_name),
                oauth_token=gcp.cloudscheduler.JobHttpTargetOauthTokenArgs(
                    service_account_email=service_account.email,
                ),
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=gcp_provider, depends_on=[job]),
        )

        self.job_name = job.name
        self.image_ref = image.ref
        self.register_outputs({"job_name": self.job_name, "image_ref": self.image_ref})
