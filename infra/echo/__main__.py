# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Echo's context database, sync job, API, and dashboard.

Declares the `context` database on the shared `marin-metadata` Cloud SQL instance
(infra/cloudsql owns the instance), its Cloud SQL IAM database users, and the `echo-sync`
scheduled Cloud Run job that keeps the corpus mirror current (sync/main.py).

Access is IAM, not passwords: the `eng-all@openathena.ai` group reads the corpus
and appends to the logbook, and the sync job's service account writes
`chunks`/`sync_state`. The API owns wiki writes and serves the compiled Vue dashboard.
Every principal authenticates through the Cloud SQL connector with a short-lived OAuth
token, so no database password exists. Table grants are applied by migrate.py (see
README.md); this program owns the users and their login IAM roles.
"""

import hashlib
from pathlib import Path

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_command as command
import pulumi_gcp as gcp
import search_config
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv
from iac.gcp.cloud_run_job import ScheduledCloudRunJob, ScheduledCloudRunJobArgs
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT

ECHO_DIR = Path(__file__).parent
MIGRATIONS_DIR = ECHO_DIR / "migrations"

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-metadata"
CONNECTION_NAME = f"{PROJECT}:{REGION}:{INSTANCE}"
DATABASE = "context"
# The IAP-gated Cloud Run service serving the API + dashboard; also the domain-mapping route.
API_SERVICE = "echo-api"
# Google's shared frontend for Cloud Run domain mappings; the vanity CNAME points here.
CLOUD_RUN_FRONTEND = "ghs.googlehosted.com"
# Cloud Identity group whose members read the corpus and append to work_log through
# Cloud SQL IAM group authentication. Cloud SQL does not support domain principals as
# database users, so the organization-wide group represents *@openathena.ai at the
# database boundary.
OPENATHENA_GROUP = "eng-all@openathena.ai"
# The bootstrap migration chain requires this role to exist from its initial grants
# through their revocation. It receives no IAM login roles or IAP access.
LEGACY_ECHO_GROUP = "echo@openathena.ai"
# Echo's Cloud Run runtime service accounts (created by their components as <name>@<project>).
SYNC_SA = f"echo-sync@{PROJECT}.iam.gserviceaccount.com"
API_SA = f"echo-api@{PROJECT}.iam.gserviceaccount.com"
LOOM_VM_SA = f"loom-vm@{PROJECT}.iam.gserviceaccount.com"
# A Cloud SQL IAM database user's Postgres name is its principal minus the SA suffix.
SYNC_DB_USER = SYNC_SA.removesuffix(".gserviceaccount.com")
API_DB_USER = API_SA.removesuffix(".gserviceaccount.com")
LOOM_VM_DB_USER = LOOM_VM_SA.removesuffix(".gserviceaccount.com")
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

    # IAM database groups are added once; their members inherit database grants without
    # per-user registration. Keep the legacy group present for migration ordering, but
    # grant login only to the organization-wide group.
    legacy_echo_group = gcp.sql.User(
        "agents-group",
        name=LEGACY_ECHO_GROUP,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_GROUP",
        opts=pulumi.ResourceOptions.merge(
            child,
            pulumi.ResourceOptions(import_=f"{PROJECT}/{INSTANCE}//{LEGACY_ECHO_GROUP}" if adopt else None),
        ),
    )
    openathena_group = gcp.sql.User(
        "openathena-group",
        name=OPENATHENA_GROUP,
        instance=INSTANCE,
        project=PROJECT,
        type="CLOUD_IAM_GROUP",
        opts=pulumi.ResourceOptions.merge(
            child,
            pulumi.ResourceOptions(import_=f"{PROJECT}/{INSTANCE}//{OPENATHENA_GROUP}" if adopt else None),
        ),
    )
    login_grants: list[pulumi.Resource] = []
    for member, roles in (
        (f"group:{OPENATHENA_GROUP}", LOGIN_ROLES),
        (f"serviceAccount:{SYNC_SA}", ("roles/cloudsql.instanceUser",)),
        (f"serviceAccount:{API_SA}", ("roles/cloudsql.instanceUser",)),
        (f"serviceAccount:{LOOM_VM_SA}", LOGIN_ROLES),
    ):
        for role in roles:
            login_grants.append(
                gcp.projects.IAMMember(
                    f"login-{role_slug(member.split(':', 1)[1])}-{role_slug(role)}",
                    project=PROJECT,
                    role=role,
                    member=member,
                    opts=child,
                )
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
            # Activity checks retain their ten-minute cadence. The repository phase uses its
            # database watermark to query GitHub no more than once per hour.
            schedule="*/10 * * * *",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": SYNC_DB_USER,
                "GITHUB_REPOSITORY": search_config.INDEXED_REPOSITORY,
                "GITHUB_BRANCH": search_config.INDEXED_BRANCH,
            },
            secrets=(SecretEnv(name="MARINMIRROR_TOKEN", secret=MARINMIRROR_TOKEN_SECRET, wait_for=(mirror_token,)),),
            cloudsql_instances=(CONNECTION_NAME,),
            # Four CPUs keep the first repository embedding build within its two-hour
            # attempt; incremental hourly refreshes normally embed only changed files.
            cpu="4",
            memory="4Gi",
            timeout=7200,
        ),
        gcp_provider=gcp_provider,
    )

    api = CloudRunService(
        "api",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=API_SERVICE,
            build_context=".",
            dockerfile="api/Dockerfile",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": API_DB_USER,
                "GITHUB_REPOSITORY": search_config.INDEXED_REPOSITORY,
                "GITHUB_BRANCH": search_config.INDEXED_BRANCH,
            },
            # Keep one instance warm. Four concurrent embedding/reranking requests peak
            # around 2.7 GiB; bound concurrency and leave headroom for the DB pool.
            min_instances=1,
            max_instances=1,
            cpu_always_allocated=True,
            memory="4Gi",
            max_instance_request_concurrency=4,
            # Admit CLI/agent tokens (cli.py) whose audience is the shared Marin desktop OAuth
            # client, so the same rigging login that reaches iris also reaches echo-api.
            iap_programmatic_clients=(MARIN_DESKTOP_OAUTH_CLIENT.client_id,),
            cloudsql_instances=(CONNECTION_NAME,),
        ),
        gcp_provider=gcp_provider,
    )

    # The service SAs exist only after their components; register them as IAM database users.
    db_users = [legacy_echo_group, openathena_group]
    for resource_name, db_user, component in (("sync-sa", SYNC_DB_USER, sync), ("api-sa", API_DB_USER, api)):
        db_users.append(
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
        )
    db_users.append(
        gcp.sql.User(
            "loom-vm-sa",
            name=LOOM_VM_DB_USER,
            instance=INSTANCE,
            project=PROJECT,
            type="CLOUD_IAM_SERVICE_ACCOUNT",
            opts=pulumi.ResourceOptions.merge(
                child,
                pulumi.ResourceOptions(depends_on=login_grants),
            ),
        )
    )

    # Apply pending migrations as the last step of `pulumi up`: migrate.py creates the tables
    # and grants the IAM users above (so it must follow them). It is idempotent — it skips
    # migrations recorded in schema_migrations — and re-runs only when a migration file
    # changes, keyed by their combined hash. Runs on the operator's machine as part of the
    # deploy, so that host needs gcloud ADC with connector + secret access (same as pulumi).
    migration_files = sorted(MIGRATIONS_DIR.glob("m[0-9]*.py"))
    migration_hash = hashlib.sha256(b"".join(f.read_bytes() for f in migration_files)).hexdigest()
    command.local.Command(
        "migrate",
        create=str(ECHO_DIR / "migrate.py"),
        triggers=[migration_hash],
        # The ADC default quota project may lack the SQL Admin / Secret Manager APIs; pin it.
        environment={"GOOGLE_CLOUD_QUOTA_PROJECT": PROJECT},
        opts=pulumi.ResourceOptions(depends_on=db_users),
    )

    # Optional vanity domain (echo.oa.dev): a Cloud Run domain mapping routes the host to
    # echo-api and provisions the managed cert, and a DNS-only Cloudflare CNAME points the
    # host at Cloud Run's shared frontend. Cloud Run terminates TLS and IAP, so a Cloudflare
    # proxy would block cert issuance — the record stays unproxied. Set marin-echo:custom_domain
    # and marin-echo:dns_zone_id to enable. The domain mapping is immutable and adopted with
    # server-set metadata, so ignore those fields to keep `up` from planning unsupported updates.
    site_config = pulumi.Config()
    custom_domain = site_config.get("custom_domain")
    if custom_domain:
        dns_zone_id = site_config.require("dns_zone_id")
        gcp.cloudrun.DomainMapping(
            "api-domain",
            name=custom_domain,
            location=REGION,
            metadata=gcp.cloudrun.DomainMappingMetadataArgs(namespace=PROJECT),
            spec=gcp.cloudrun.DomainMappingSpecArgs(route_name=API_SERVICE),
            opts=pulumi.ResourceOptions.merge(
                child,
                pulumi.ResourceOptions(depends_on=[api], ignore_changes=["metadata", "spec", "statuses"]),
            ),
        )
        cloudflare.DnsRecord(
            "api-dns",
            zone_id=dns_zone_id,
            name=custom_domain,
            type="CNAME",
            content=CLOUD_RUN_FRONTEND,
            ttl=1,  # 1 = automatic
            proxied=False,
        )
        pulumi.export("custom_domain", f"https://{custom_domain}")

    pulumi.export("connection_name", CONNECTION_NAME)
    pulumi.export("database", database.name)
    pulumi.export("sync_job", sync.job_name)
    pulumi.export("api_uri", api.uri)


main()
