# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Privileged Postgres setup for Marina's dynamic applets."""

import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

APPLET_READER_ROLE = "marina_reader"
PROVISION_APPLET_FUNCTION = "marina.provision_applet"
POSTGRES_ROLE_PATTERN = re.compile(r"^[A-Za-z0-9_@.\-]+$")


def _quoted_role(role: str) -> str:
    if not POSTGRES_ROLE_PATTERN.fullmatch(role):
        raise ValueError(f"invalid Postgres role {role!r}")
    return '"' + role.replace('"', '""') + '"'


def applet_provisioning_statements(service_role: str) -> tuple[str, ...]:
    """SQL an administrator runs once to let Marina provision constrained applet roles."""
    service = _quoted_role(service_role)
    return (
        f"CREATE SCHEMA IF NOT EXISTS marina AUTHORIZATION {service}",
        f"GRANT ALL ON SCHEMA marina TO {service}",
        f"""DO $$
        BEGIN
            CREATE ROLE {APPLET_READER_ROLE} NOLOGIN;
        EXCEPTION WHEN duplicate_object THEN
            ALTER ROLE {APPLET_READER_ROLE} NOLOGIN;
        END
        $$""",
        "REVOKE CREATE ON SCHEMA public FROM PUBLIC",
        f"GRANT USAGE ON SCHEMA public TO {APPLET_READER_ROLE}",
        f"""CREATE OR REPLACE FUNCTION {PROVISION_APPLET_FUNCTION}(applet_id UUID)
        RETURNS TEXT
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $$
        DECLARE
            applet_role TEXT := 'applet_' || replace(applet_id::TEXT, '-', '');
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = applet_role) THEN
                EXECUTE format('CREATE ROLE %I NOLOGIN', applet_role);
            END IF;
            EXECUTE format('GRANT {APPLET_READER_ROLE} TO %I', applet_role);
            EXECUTE format('GRANT %I TO %I', applet_role, {service_role!r});
            EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I AUTHORIZATION %I', applet_role, applet_role);
            EXECUTE format('ALTER SCHEMA %I OWNER TO %I', applet_role, applet_role);
            EXECUTE format('GRANT USAGE ON SCHEMA %I TO {APPLET_READER_ROLE}', applet_role);
            RETURN applet_role;
        END
        $$""",
        f"REVOKE ALL ON FUNCTION {PROVISION_APPLET_FUNCTION}(UUID) FROM PUBLIC",
        f"GRANT EXECUTE ON FUNCTION {PROVISION_APPLET_FUNCTION}(UUID) TO {service}",
    )


def ensure_applet_provisioning(engine: Engine) -> None:
    """Install provisioning as a local admin, or verify an operator already installed it."""
    with engine.begin() as connection:
        service_role, can_create_role = connection.execute(
            text("SELECT current_user, rolcreaterole OR rolsuper FROM pg_roles " "WHERE rolname = current_user")
        ).one()
        installed = connection.execute(
            text("SELECT to_regprocedure('marina.provision_applet(uuid)') IS NOT NULL")
        ).scalar_one()
        if installed:
            return
        if not can_create_role:
            raise RuntimeError("an administrator must install marina.provision_applet before marina migrate")
        for statement in applet_provisioning_statements(service_role):
            connection.execute(text(statement))
