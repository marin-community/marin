-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

-- Run with psql as the Grafana database owner after gcloud creates the three
-- login users with their matching --database-roles assignments. Setting role at
-- login makes objects created by the migrator belong to the durable NOLOGIN
-- owner and ensures runtime sessions exercise only their narrow group role.
\set ON_ERROR_STOP on

ALTER ROLE ops_migrator SET role = 'ops_migrator_role';
ALTER ROLE ops_migrator SET search_path = public, pg_catalog;

ALTER ROLE ops_app SET role = 'ops_app_role';
ALTER ROLE ops_app SET search_path = public, pg_catalog;

ALTER ROLE ops_grafana_reader SET role = 'ops_grafana_reader_role';
ALTER ROLE ops_grafana_reader SET search_path = pg_catalog;
