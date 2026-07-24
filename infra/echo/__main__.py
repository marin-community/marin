# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for echo: Marin's shared agent-context database and its sync job.

Declares the `context` database on the shared `marin-metadata` Cloud SQL instance
(infra/cloudsql owns the instance), its Cloud SQL IAM database users, and the `echo-sync`
scheduled Cloud Run job that keeps the corpus mirror current (sync/main.py).

Access is IAM, not passwords: the `eng-all@openathena.ai` group reads the corpus and
appends to the logbook, and the sync job's service account writes `chunks`/`sync_state`.
Every principal authenticates through the Cloud SQL connector with a short-lived OAuth
token, so no database password exists. Table grants are applied by migrate.py (see
README.md); this program owns the users and their login IAM roles.
"""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv
from iac.gcp.cloud_run_job import ScheduledCloudRunJob, ScheduledCloudRunJobArgs

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-metadata"
CONNECTION_NAME = f"{PROJECT}:{REGION}:{INSTANCE}"
DATABASE = "context"
# Google group whose members read the corpus and append to work_log, via IAM group auth.
AGENTS_GROUP = "eng-all@openathena.ai"
# The Cloud Run runtime service accounts (created by their components as <name>@<project>).
SYNC_SA = f"echo-sync@{PROJECT}.iam.gserviceaccount.com"
API_SA = f"echo-api@{PROJECT}.iam.gserviceaccount.com"
# A Cloud SQL IAM database user's Postgres name is its principal minus the SA suffix.
SYNC_DB_USER = SYNC_SA.removesuffix(".gserviceaccount.com")
API_DB_USER = API_SA.removesuffix(".gserviceaccount.com")
# marinmirror bearer token: a GitHub PAT (read:org) of an Open-Athena member.
MARINMIRROR_TOKEN_SECRET = "marinmirror-token"

# Login roles for Cloud SQL IAM auth: instanceUser carries the cloudsql.instances.login
# permission, client lets the connector reach the instance.
LOGIN_ROLES = ("roles/cloudsql.instanceUser", "roles/cloudsql.client")


def role_slug(role: str) -> str:
    return role.removeprefix("roles/").replace(".", "-")


def main() -> None:
    config = pulumi.Config("marin-iac")
    # Adoption recon: `pulumi config set marin-iac:import true` stamps import_ on
    # pre-existing resources (one-shot; see infra/pulumi/README.md).
    adopt = config.get_bool("import") or False
    gcp_provider = gcp.Provider("gcp", project=PROJECT)
    child = pulumi.ResourceOptions(provider=gcp_provider)

    database = gcp.sql.Database("context", name=DATABASE, instance=INSTANCE, project=PROJECT, opts=child)

    # IAM database users. The group is added once; its members inherit its grants without
    # per-user registration. The sync SA is created by the job component below.
    gcp.sql.User(
        "agents-group",
        name=AGENTS_GROUP,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_GROUP",
        opts=pulumi.ResourceOptions.merge(
            child,
            pulumi.ResourceOptions(import_=f"{PROJECT}/{INSTANCE}//{AGENTS_GROUP}" if adopt else None),
        ),
    )
    for member, roles in (
        (f"group:{AGENTS_GROUP}", LOGIN_ROLES),
        (f"serviceAccount:{SYNC_SA}", ("roles/cloudsql.instanceUser",)),
        (f"serviceAccount:{API_SA}", ("roles/cloudsql.instanceUser",)),
    ):
        for role in roles:
            gcp.projects.IAMMember(
                f"login-{role_slug(member.split(':', 1)[1])}-{role_slug(role)}",
                project=PROJECT,
                role=role,
                member=member,
                opts=child,
            )

    mirror_token = gcp.secretmanager.Secret(
        "marinmirror-token",
        secret_id=MARINMIRROR_TOKEN_SECRET,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=child,
    )

    sync = ScheduledCloudRunJob(
        "sync",
        ScheduledCloudRunJobArgs(
            project=PROJECT,
            region=REGION,
            job_name="echo-sync",
            build_context=".",
            dockerfile="sync/Dockerfile",
            # Cheap when nothing changed: the job exits on the manifest watermark check, and
            # only a new upstream corpus build (~every 90 min) triggers the full download+upsert.
            schedule="*/10 * * * *",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": SYNC_DB_USER,
            },
            secrets=(SecretEnv(name="MARINMIRROR_TOKEN", secret=MARINMIRROR_TOKEN_SECRET, wait_for=(mirror_token,)),),
            cloudsql_instances=(CONNECTION_NAME,),
            # The sync holds a ~650 MB corpus download plus batch buffers in memory.
            memory="2Gi",
        ),
        gcp_provider=gcp_provider,
    )

    # The echo HTTP API: one IAP-gated service holding the DB identity and the query model,
    # so agents reach the corpus over HTTP without their own database grants.
    api = CloudRunService(
        "api",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name="echo-api",
            build_context=".",
            dockerfile="api/Dockerfile",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": API_DB_USER,
            },
            # Keep one instance warm: it holds the ~130 MB embedding model and the DB pool.
            min_instances=1,
            max_instances=1,
            cpu_always_allocated=True,
            memory="2Gi",
            iap_members=(f"group:{AGENTS_GROUP}",),
            cloudsql_instances=(CONNECTION_NAME,),
        ),
        gcp_provider=gcp_provider,
    )

    # The service SAs exist only after their components; register them as IAM database users.
    for resource_name, db_user, component in (("sync-sa", SYNC_DB_USER, sync), ("api-sa", API_DB_USER, api)):
        gcp.sql.User(
            resource_name,
            name=db_user,
            instance=INSTANCE,
            project=PROJECT,
            type="CLOUD_IAM_SERVICE_ACCOUNT",
            opts=pulumi.ResourceOptions.merge(
                child,
                pulumi.ResourceOptions(
                    depends_on=[component],
                    import_=f"{PROJECT}/{INSTANCE}//{db_user}" if adopt else None,
                ),
            ),
        )

    pulumi.export("connection_name", CONNECTION_NAME)
    pulumi.export("database", database.name)
    pulumi.export("sync_job", sync.job_name)
    pulumi.export("api_uri", api.uri)


main()
