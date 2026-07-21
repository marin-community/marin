-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

-- Run with psql as the existing Grafana database owner before creating the
-- ops_* login users. Cloud SQL assigns its broad cloudsqlsuperuser role when a
-- built-in user is created without --database-roles, so the login users are
-- created separately and attached to these NOLOGIN roles.
\set ON_ERROR_STOP on

SELECT 'CREATE ROLE ops_migrator_role NOLOGIN'
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ops_migrator_role')
\gexec

SELECT 'CREATE ROLE ops_app_role NOLOGIN'
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ops_app_role')
\gexec

SELECT 'CREATE ROLE ops_grafana_reader_role NOLOGIN'
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ops_grafana_reader_role')
\gexec

GRANT CONNECT, CREATE ON DATABASE ops TO ops_migrator_role;
GRANT CONNECT ON DATABASE ops TO ops_app_role;
GRANT CONNECT ON DATABASE grafana TO ops_grafana_reader_role;

-- PostgreSQL requires the current admin to be able to SET ROLE to a new schema
-- owner. Cloud SQL does not grant that option to the creator automatically.
-- Keep the membership only for this ownership and default-privilege bootstrap.
GRANT ops_migrator_role TO CURRENT_USER;

\connect ops

ALTER SCHEMA public OWNER TO ops_migrator_role;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO ops_app_role;
ALTER DEFAULT PRIVILEGES FOR ROLE ops_migrator_role IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ops_app_role;
ALTER DEFAULT PRIVILEGES FOR ROLE ops_migrator_role IN SCHEMA public
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO ops_app_role;

\connect grafana

GRANT USAGE ON SCHEMA public TO ops_grafana_reader_role;
GRANT SELECT ON TABLE public.alert_instance, public.alert_rule TO ops_grafana_reader_role;

REVOKE ops_migrator_role FROM CURRENT_USER;
