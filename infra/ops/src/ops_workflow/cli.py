# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local and production command line for the ops workflow spike."""

import argparse
import logging
import os
from pathlib import Path
from typing import cast

import psycopg
import uvicorn
from rigging.log_setup import LogBuffer, configure_logging

from ops_workflow.grafana_source import PostgresGrafanaAlertSource
from ops_workflow.loom import LoomGateway, StubAgentGateway
from ops_workflow.migrations import Connection as MigrationConnection
from ops_workflow.migrations import apply_migrations, migration_plan
from ops_workflow.repository import OpsRepository
from ops_workflow.service import OpsService
from ops_workflow.slack import SlackDispatcher, SlackWebhook
from ops_workflow.web import WebConfig, create_app

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATIONS = PACKAGE_ROOT / "migrations"
DEFAULT_STATIC = PACKAGE_ROOT / "dashboard" / "dist"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    log_buffer = configure_logging(level=logging.INFO)
    if args.command == "migrate":
        _migrate(args.database_url, args.migrations)
        return
    if args.command == "serve":
        _serve(args, log_buffer)
        return
    parser.error(f"unknown command {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--database-url", required=True)
    migrate.add_argument("--migrations", type=Path, default=DEFAULT_MIGRATIONS)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--database-url", required=True)
    serve.add_argument("--migrations", type=Path, default=DEFAULT_MIGRATIONS)
    serve.add_argument("--migrate-on-start", action="store_true")
    grafana_url = serve.add_mutually_exclusive_group(required=True)
    grafana_url.add_argument("--grafana-database-url")
    grafana_url.add_argument("--grafana-database-url-env")
    serve.add_argument("--grafana-database-password")
    serve.add_argument("--grafana-database-password-env")
    serve.add_argument("--grafana-poll-interval", type=float, default=60.0)
    serve.add_argument("--public-url", default="http://127.0.0.1:8088")
    serve.add_argument("--slack-webhook-url-env")
    serve.add_argument("--auth-mode", choices=("local", "iap"), default="local")
    serve.add_argument("--agent-mode", choices=("stub", "loom"), default="stub")
    serve.add_argument("--loom-api-url")
    serve.add_argument("--loom-token")
    serve.add_argument("--loom-token-env")
    serve.add_argument("--loom-agent", default="codex")
    serve.add_argument("--loom-model")
    serve.add_argument("--loom-effort", default="low")
    serve.add_argument("--repo-root", default=str(PACKAGE_ROOT.parents[1]))
    serve.add_argument("--loom-base", default="origin/main")
    serve.add_argument("--repo-revision", default="local-spike")
    serve.add_argument("--skill-revision", default="working-tree")
    serve.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    return parser


def _migrate(database_url: str, migrations: Path) -> None:
    with psycopg.connect(database_url) as connection:
        apply_migrations(cast(MigrationConnection, connection), migration_plan(migrations))


def _serve(args: argparse.Namespace, log_buffer: LogBuffer) -> None:
    if args.auth_mode == "local" and args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("local auth mode may only bind to loopback")
    grafana_database_url = _secret_argument(
        value=args.grafana_database_url,
        environment_name=args.grafana_database_url_env,
        option="--grafana-database-url",
    )
    assert grafana_database_url is not None
    grafana_database_password = _secret_argument(
        value=args.grafana_database_password,
        environment_name=args.grafana_database_password_env,
        option="--grafana-database-password",
    )
    loom_token = _secret_argument(
        value=args.loom_token,
        environment_name=args.loom_token_env,
        option="--loom-token",
    )
    slack_webhook_url = _secret_argument(
        value=None,
        environment_name=args.slack_webhook_url_env,
        option="--slack-webhook-url",
    )
    if args.grafana_poll_interval <= 0:
        raise SystemExit("--grafana-poll-interval must be positive")
    if args.migrate_on_start:
        _migrate(args.database_url, args.migrations)
    repository = OpsRepository(
        args.database_url,
        repo_revision=args.repo_revision,
        skill_revision=args.skill_revision,
    )
    if args.agent_mode == "loom":
        if not args.loom_api_url or not loom_token:
            raise SystemExit("--loom-api-url and a Loom token are required for --agent-mode loom")
        gateway = LoomGateway(
            api_url=args.loom_api_url,
            token=loom_token,
            repo_root=args.repo_root,
            base=args.loom_base,
            agent=args.loom_agent,
            model=args.loom_model,
            effort=args.loom_effort,
        )
    else:
        gateway = StubAgentGateway()
    slack_dispatcher = SlackDispatcher(repository, SlackWebhook(slack_webhook_url)) if slack_webhook_url else None
    app = create_app(
        OpsService(repository, gateway, public_url=args.public_url),
        repository,
        PostgresGrafanaAlertSource(grafana_database_url, password=grafana_database_password),
        log_buffer,
        slack_dispatcher,
        WebConfig(
            auth_mode=args.auth_mode,
            static_dir=args.static_dir,
            poll_interval=args.grafana_poll_interval,
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _secret_argument(*, value: str | None, environment_name: str | None, option: str) -> str | None:
    if value and environment_name:
        raise SystemExit(f"use either {option} or {option}-env, not both")
    if environment_name:
        secret = os.environ.get(environment_name)
        if not secret:
            raise SystemExit(f"environment variable {environment_name} is empty or unset")
        return secret
    return value


if __name__ == "__main__":
    main()
