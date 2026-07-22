# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi-managed PostgreSQL roles, credentials, and grants for the ops service."""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp
import pulumi_postgresql as postgresql
import pulumi_random as random

from iac.gcp.cloud_sql import CloudSqlPostgres

POSTGRES_VERSION = "16"
POSTGRES_CONNECT_TIMEOUT = 15
POSTGRES_MAX_CONNECTIONS = 4
PASSWORD_LENGTH = 32
PULUMI_ADMIN = "pulumi_db_admin"
OPS_SCHEMA = "public"


@dataclass(frozen=True)
class LoginRole:
    name: str
    group: str
    secret: str
    search_paths: tuple[str, ...]


@dataclass(frozen=True)
class ManagedPassword:
    value: pulumi.Output[str]
    version: str
    secret_version: gcp.secretmanager.SecretVersion


LOGIN_ROLES = (
    LoginRole(
        name="ops_migrator",
        group="ops_migrator_role",
        secret="cloudsql-ops-migrator-password",
        search_paths=("public", "pg_catalog"),
    ),
    LoginRole(
        name="ops_app",
        group="ops_app_role",
        secret="cloudsql-ops-app-password",
        search_paths=("public", "pg_catalog"),
    ),
)


def managed_password(
    name: str,
    *,
    secret: gcp.secretmanager.Secret,
    generation: int,
    gcp_provider: pulumi.ProviderResource,
) -> ManagedPassword:
    password = random.RandomPassword(
        name,
        length=PASSWORD_LENGTH,
        special=False,
        keepers={"generation": str(generation)},
    )
    secret_version = gcp.secretmanager.SecretVersion(
        name,
        secret=secret.id,
        secret_data_wo=password.result,
        secret_data_wo_version=generation,
        deletion_policy="ABANDON",
        opts=pulumi.ResourceOptions(provider=gcp_provider),
    )
    return ManagedPassword(value=password.result, version=str(generation), secret_version=secret_version)


def postgres_provider(
    name: str,
    *,
    database: str,
    connection_name: pulumi.Input[str],
    username: str,
    password: pulumi.Input[str],
    depends_on: list[pulumi.Resource],
) -> postgresql.Provider:
    return postgresql.Provider(
        name,
        scheme="gcppostgres",
        host=connection_name,
        database=database,
        username=username,
        password=password,
        expected_version=POSTGRES_VERSION,
        connect_timeout=POSTGRES_CONNECT_TIMEOUT,
        max_connections=POSTGRES_MAX_CONNECTIONS,
        superuser=False,
        opts=pulumi.ResourceOptions(depends_on=depends_on),
    )


def configure_ops_database(
    *,
    project: str,
    postgres: CloudSqlPostgres,
    gcp_provider: pulumi.ProviderResource,
    admin_password_generation: int,
    admin_secret: str,
    ops_password_generation: int,
) -> None:
    admin_password = managed_password(
        "pulumi-db-admin-password",
        secret=postgres.password_secrets[admin_secret],
        generation=admin_password_generation,
        gcp_provider=gcp_provider,
    )
    admin_user = gcp.sql.User(
        "pulumi-db-admin",
        name=PULUMI_ADMIN,
        instance=postgres.instance.name,
        project=project,
        type="BUILT_IN",
        password_wo=admin_password.value,
        password_wo_version=admin_password_generation,
        opts=pulumi.ResourceOptions(
            provider=gcp_provider,
            depends_on=[admin_password.secret_version],
            protect=True,
        ),
    )

    postgres_admin = postgres_provider(
        "postgres-admin",
        database="postgres",
        connection_name=postgres.connection_name,
        username=PULUMI_ADMIN,
        password=admin_password.value,
        depends_on=[admin_user],
    )
    ops_admin = postgres_provider(
        "ops-admin",
        database="ops",
        connection_name=postgres.connection_name,
        username=PULUMI_ADMIN,
        password=admin_password.value,
        depends_on=[admin_user, postgres.databases["ops"]],
    )

    group_roles: dict[str, postgresql.Role] = {}
    for login in LOGIN_ROLES:
        group_roles[login.group] = postgresql.Role(
            login.group.replace("_", "-"),
            name=login.group,
            login=False,
            opts=pulumi.ResourceOptions(provider=postgres_admin, protect=True),
        )

    for login in LOGIN_ROLES:
        password = managed_password(
            f"{login.name.replace('_', '-')}-password",
            secret=postgres.password_secrets[login.secret],
            generation=ops_password_generation,
            gcp_provider=gcp_provider,
        )
        group = group_roles[login.group]
        postgresql.Role(
            login.name.replace("_", "-"),
            name=login.name,
            login=True,
            roles=[group.name],
            assume_role=group.name,
            search_paths=list(login.search_paths),
            password_wo=password.value,
            password_wo_version=password.version,
            opts=pulumi.ResourceOptions(
                provider=postgres_admin,
                depends_on=[password.secret_version],
                protect=True,
            ),
        )

    migrator_role = group_roles["ops_migrator_role"]
    app_role = group_roles["ops_app_role"]

    postgresql.Grant(
        "ops-migrator-database",
        database="ops",
        object_type="database",
        role=migrator_role.name,
        privileges=["CONNECT", "CREATE"],
        opts=pulumi.ResourceOptions(provider=ops_admin),
    )
    postgresql.Grant(
        "ops-app-database",
        database="ops",
        object_type="database",
        role=app_role.name,
        privileges=["CONNECT"],
        opts=pulumi.ResourceOptions(provider=ops_admin),
    )
    ops_schema = postgresql.Schema(
        "ops-public",
        database="ops",
        name=OPS_SCHEMA,
        owner=migrator_role.name,
        opts=pulumi.ResourceOptions(
            provider=ops_admin,
            protect=True,
            # Cloud SQL creates public with each database; every new stack adopts it.
            import_="ops.public",
        ),
    )
    postgresql.Grant(
        "ops-public-default",
        database="ops",
        schema=OPS_SCHEMA,
        object_type="schema",
        role="public",
        privileges=["USAGE"],
        opts=pulumi.ResourceOptions(provider=ops_admin, depends_on=[ops_schema]),
    )
    postgresql.Grant(
        "ops-app-schema",
        database="ops",
        schema=OPS_SCHEMA,
        object_type="schema",
        role=app_role.name,
        privileges=["USAGE"],
        opts=pulumi.ResourceOptions(provider=ops_admin, depends_on=[ops_schema]),
    )
    postgresql.DefaultPrivileges(
        "ops-app-tables",
        database="ops",
        schema=OPS_SCHEMA,
        owner=migrator_role.name,
        object_type="table",
        role=app_role.name,
        privileges=["SELECT", "INSERT", "UPDATE", "DELETE"],
        opts=pulumi.ResourceOptions(provider=ops_admin, depends_on=[ops_schema]),
    )
    postgresql.DefaultPrivileges(
        "ops-app-sequences",
        database="ops",
        schema=OPS_SCHEMA,
        owner=migrator_role.name,
        object_type="sequence",
        role=app_role.name,
        privileges=["USAGE", "SELECT", "UPDATE"],
        opts=pulumi.ResourceOptions(provider=ops_admin, depends_on=[ops_schema]),
    )
