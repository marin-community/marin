# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marina: one Cloud Run service serving every app under infra/marina/apps.

The stack also carries the ``context`` database and the Loom VM's Cloud SQL login, which the
codehealth review workbench (infra/codehealth) uses directly, and the group login people
read the apps' schemas with.

The stack owns what the apps share: the ``marina`` database on the ``marin-metadata`` Cloud
SQL instance (one schema per Python app, all owned by the service account), the
``marin-marina`` bucket the data root points at, the service, and two Cloud Run jobs that
run the same image as the service under the same account: ``marina-migrate`` (executed at
the end of every ``pulumi up`` that changes the image) and ``marina-echo-sync`` (scheduled,
keeps Echo's corpus mirror current). IAP is the front door; the container verifies IAP's
signed assertion against the service's audience so each request carries the caller's
email. IAM grants live in the ``marin`` stack (iac.gcp.marina).

Vanity hosts: the apps are served from ``marina.oa.dev``. ``echo.oa.dev`` and
``evaldash.oa.dev`` map to the same service, where every request redirects to
``marina.oa.dev/echo/`` or ``marina.oa.dev/evaldash/`` with its path, so the old links keep
resolving and one origin holds the apps.
"""

from pathlib import Path

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_command as command
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marina"
INSTANCE = "marin-metadata"
CONNECTION_NAME = f"{PROJECT}:{REGION}:{INSTANCE}"
DATABASE = "marina"
DATA_BUCKET = "marin-marina"
# The runtime service account CloudRunService creates as <service>@<project>; its Cloud SQL
# IAM database user is the principal minus the ".gserviceaccount.com" suffix.
SERVICE_ACCOUNT = f"{SERVICE}@{PROJECT}.iam.gserviceaccount.com"
DATABASE_USER = SERVICE_ACCOUNT.removesuffix(".gserviceaccount.com")
# The codehealth workbench's database and its writer (infra/codehealth/review_store.py).
CODEHEALTH_DATABASE = "context"
LOOM_DATABASE_USER = "loom-vm@hai-gcp-models.iam"
# Cloud SQL group login for people: members read any app's schema under their own Google
# identity, without a database user each. ``marina migrate`` grants it.
READER_GROUP = "eng-all@openathena.ai"
MIGRATE_JOB = "marina-migrate"
ECHO_SYNC_JOB = "marina-echo-sync"
# marinmirror bearer token: a GitHub PAT (read:org) of an Open-Athena member.
MARINMIRROR_TOKEN_SECRET = "marinmirror-token"
# Google's shared frontend for Cloud Run domain mappings; vanity CNAMEs point here.
CLOUD_RUN_FRONTEND = "ghs.googlehosted.com"
HOST_APPS = {"echo.oa.dev": "echo", "evaldash.oa.dev": "evaldash"}
MARINA_HOST = "marina.oa.dev"
GRANTS_SCRIPT = Path(__file__).parent / "database_grants.py"

DATABASE_ENV = {"CLOUDSQL_CONNECTION": CONNECTION_NAME, "PGDATABASE": DATABASE, "PGUSER": DATABASE_USER}


def iap_audience(project_number: str) -> str:
    """The ``aud`` claim IAP signs into X-Goog-IAP-JWT-Assertion for a Cloud Run service."""
    return f"/projects/{project_number}/locations/{REGION}/services/{SERVICE}"


def job_template(
    image: pulumi.Output[str],
    args: list[str],
    env: dict[str, str],
    secrets: tuple[SecretEnv, ...],
    cpu: str,
    memory: str,
) -> gcp.cloudrunv2.JobTemplateTemplateArgs:
    """The service image run as a job under the service account, with the Cloud SQL socket attached."""
    return gcp.cloudrunv2.JobTemplateTemplateArgs(
        service_account=SERVICE_ACCOUNT,
        max_retries=0,
        timeout="7200s",
        volumes=[
            gcp.cloudrunv2.JobTemplateTemplateVolumeArgs(
                name="cloudsql",
                cloud_sql_instance=gcp.cloudrunv2.JobTemplateTemplateVolumeCloudSqlInstanceArgs(
                    instances=[CONNECTION_NAME]
                ),
            )
        ],
        containers=[
            gcp.cloudrunv2.JobTemplateTemplateContainerArgs(
                image=image,
                args=args,
                envs=[gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(name=k, value=v) for k, v in env.items()]
                + [
                    gcp.cloudrunv2.JobTemplateTemplateContainerEnvArgs(
                        name=secret.name,
                        value_source=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceArgs(
                            secret_key_ref=gcp.cloudrunv2.JobTemplateTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                secret=secret.secret, version=secret.version
                            )
                        ),
                    )
                    for secret in secrets
                ],
                resources=gcp.cloudrunv2.JobTemplateTemplateContainerResourcesArgs(
                    limits={"cpu": cpu, "memory": memory}
                ),
                volume_mounts=[
                    gcp.cloudrunv2.JobTemplateTemplateContainerVolumeMountArgs(name="cloudsql", mount_path="/cloudsql")
                ],
            )
        ],
    )


def main() -> None:
    config = pulumi.Config()
    gcp_provider = gcp.Provider("gcp", project=PROJECT)
    child = pulumi.ResourceOptions(provider=gcp_provider)

    project_number = gcp.organizations.get_project(
        project_id=PROJECT, opts=pulumi.InvokeOptions(provider=gcp_provider)
    ).number

    database = gcp.sql.Database("database", name=DATABASE, instance=INSTANCE, project=PROJECT, opts=child)
    codehealth_database = gcp.sql.Database(
        "codehealth-database",
        name=CODEHEALTH_DATABASE,
        instance=INSTANCE,
        project=PROJECT,
        opts=child,
    )
    loom_user = gcp.sql.User(
        "loom-db-user",
        name=LOOM_DATABASE_USER,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_SERVICE_ACCOUNT",
        opts=child,
    )
    reader_group = gcp.sql.User(
        "reader-group",
        name=READER_GROUP,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_GROUP",
        opts=child,
    )
    bucket = gcp.storage.Bucket(
        "data",
        name=DATA_BUCKET,
        project=PROJECT,
        location=REGION.upper(),
        uniform_bucket_level_access=True,
        opts=child,
    )
    mirror_token = gcp.secretmanager.Secret(
        "marinmirror-token",
        secret_id=MARINMIRROR_TOKEN_SECRET,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=child,
    )
    # CoreWeave object-storage keys for evaldash's s3:// record prefixes. Values stay in Secret Manager.
    coreweave_keys = (
        SecretEnv(name="CW_KEY_ID", secret="cw-object-storage-key-id"),
        SecretEnv(name="CW_KEY_SECRET", secret="cw-object-storage-key-secret"),
    )

    # The one IAM database user; Cloud SQL registers the name without needing the account yet.
    database_user = gcp.sql.User(
        "service-db-user",
        name=DATABASE_USER,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_SERVICE_ACCOUNT",
        opts=pulumi.ResourceOptions.merge(child, pulumi.ResourceOptions(depends_on=[database])),
    )

    # IAM users can connect but not create schemas; grant the service account the database
    # and the Loom VM the codehealth schema, as the native admin user, once per user change.
    grants = command.local.Command(
        "database-grants",
        create=f"uv run {GRANTS_SCRIPT}",
        triggers=[GRANTS_SCRIPT.read_text()],
        environment={"GOOGLE_CLOUD_QUOTA_PROJECT": PROJECT},
        opts=pulumi.ResourceOptions(depends_on=[database_user, loom_user, reader_group, database, codehealth_database]),
    )

    # Every app's migrate() is idempotent, so the job runs after each image change and the
    # new revision only starts once its schema is in place.
    def migrate_before_deploy(image_ref: pulumi.Output[str]) -> tuple[pulumi.Resource, ...]:
        migrate = gcp.cloudrunv2.Job(
            "migrate",
            name=MIGRATE_JOB,
            project=PROJECT,
            location=REGION,
            deletion_protection=False,
            template=gcp.cloudrunv2.JobTemplateArgs(
                template=job_template(
                    image_ref,
                    ["marina", "migrate", "--reader", READER_GROUP],
                    DATABASE_ENV,
                    (),
                    cpu="1",
                    memory="1Gi",
                )
            ),
            opts=pulumi.ResourceOptions.merge(child, pulumi.ResourceOptions(depends_on=[database_user, grants])),
        )
        run = command.local.Command(
            "run-migrate",
            create=f"gcloud run jobs execute {MIGRATE_JOB} --project {PROJECT} --region {REGION} --wait",
            triggers=[image_ref],
            opts=pulumi.ResourceOptions(depends_on=[migrate]),
        )
        return (run,)

    service = CloudRunService(
        "service",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context="../..",
            dockerfile="infra/marina/Dockerfile",
            env={
                "MARINA_IAP_AUDIENCE": iap_audience(project_number),
                "MARINA_DATA_ROOT": f"gs://{DATA_BUCKET}",
                "MARINA_HOST_APPS": ",".join(f"{host}={app}" for host, app in HOST_APPS.items()),
                "MARINA_CANONICAL_ORIGIN": f"https://{MARINA_HOST}",
                **DATABASE_ENV,
            },
            secrets=coreweave_keys,
            # Echo's search is CPU-bound inference and evaldash keeps an ingest loop running
            # between requests, so CPU stays allocated and a warm instance is always up.
            cpu_always_allocated=True,
            startup_cpu_boost=True,
            min_instances=1,
            max_instances=4,
            max_instance_request_concurrency=8,
            cpu="4",
            memory="4Gi",
            cloudsql_instances=(CONNECTION_NAME,),
            before_deploy=migrate_before_deploy,
        ),
        gcp_provider=gcp_provider,
    )

    sync = gcp.cloudrunv2.Job(
        "echo-sync",
        name=ECHO_SYNC_JOB,
        project=PROJECT,
        location=REGION,
        deletion_protection=False,
        template=gcp.cloudrunv2.JobTemplateArgs(
            template=job_template(
                service.image_ref,
                ["python", "-m", "echo.sync.main"],
                DATABASE_ENV,
                (SecretEnv(name="MARINMIRROR_TOKEN", secret=MARINMIRROR_TOKEN_SECRET, wait_for=(mirror_token,)),),
                # Four CPUs keep a first repository embedding build within one attempt.
                cpu="4",
                memory="4Gi",
            )
        ),
        opts=pulumi.ResourceOptions.merge(child, pulumi.ResourceOptions(depends_on=[database_user, mirror_token])),
    )
    gcp.cloudscheduler.Job(
        "echo-sync-trigger",
        name=f"{ECHO_SYNC_JOB}-trigger",
        project=PROJECT,
        region=REGION,
        # Activity checks keep a ten-minute cadence; the repository phase advances one turn per run.
        schedule="*/10 * * * *",
        time_zone="Etc/UTC",
        http_target=gcp.cloudscheduler.JobHttpTargetArgs(
            http_method="POST",
            uri=f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/jobs/{ECHO_SYNC_JOB}:run",
            oauth_token=gcp.cloudscheduler.JobHttpTargetOauthTokenArgs(service_account_email=SERVICE_ACCOUNT),
        ),
        opts=pulumi.ResourceOptions.merge(child, pulumi.ResourceOptions(depends_on=[sync])),
    )

    # Vanity hosts: a Cloud Run domain mapping per host routes it to the service and provisions
    # the managed cert; a DNS-only Cloudflare CNAME points the host at Cloud Run's frontend
    # (a Cloudflare proxy would block cert issuance). Mappings are immutable and carry
    # server-set metadata, so those fields are ignored. Set marin-marina:dns_zone_id to enable.
    dns_zone_id = config.get("dns_zone_id")
    if dns_zone_id:
        for host in (MARINA_HOST, *HOST_APPS):
            slug = host.split(".")[0]
            gcp.cloudrun.DomainMapping(
                f"{slug}-domain",
                name=host,
                location=REGION,
                metadata=gcp.cloudrun.DomainMappingMetadataArgs(namespace=PROJECT),
                spec=gcp.cloudrun.DomainMappingSpecArgs(route_name=SERVICE),
                opts=pulumi.ResourceOptions.merge(
                    child,
                    pulumi.ResourceOptions(depends_on=[service], ignore_changes=["metadata", "spec", "statuses"]),
                ),
            )
            cloudflare.DnsRecord(
                f"{slug}-dns",
                zone_id=dns_zone_id,
                name=host,
                type="CNAME",
                content=CLOUD_RUN_FRONTEND,
                ttl=1,  # 1 = automatic
                proxied=False,
            )

    pulumi.export("uri", service.uri)
    pulumi.export("image", service.image_ref)
    pulumi.export("database", database.name)
    pulumi.export("data_root", bucket.name.apply(lambda name: f"gs://{name}"))


main()
