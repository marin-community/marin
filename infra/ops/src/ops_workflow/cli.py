# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local and production command line for the ops workflow spike."""

import argparse
import hashlib
import hmac
import logging
import os
import time
from pathlib import Path
from typing import cast

import httpx
import psycopg
import uvicorn

from ops_workflow.loom import LoomGateway, StubAgentGateway
from ops_workflow.migrations import Connection as MigrationConnection
from ops_workflow.migrations import apply_migrations, migration_plan
from ops_workflow.repository import OpsRepository
from ops_workflow.service import OpsService
from ops_workflow.web import WebConfig, create_app

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATIONS = PACKAGE_ROOT / "migrations"
DEFAULT_STATIC = PACKAGE_ROOT / "dashboard" / "dist"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "migrate":
        _migrate(args.database_url, args.migrations)
        return
    if args.command == "send-fixture":
        _send_fixture(args.url, args.secret, args.fixture)
        return
    if args.command == "serve":
        _serve(args)
        return
    parser.error(f"unknown command {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--database-url", required=True)
    migrate.add_argument("--migrations", type=Path, default=DEFAULT_MIGRATIONS)

    fixture = subparsers.add_parser("send-fixture")
    fixture.add_argument("--url", default="http://127.0.0.1:8088/api/ingest/grafana")
    fixture.add_argument("--secret", required=True)
    fixture.add_argument("fixture", type=Path)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--database-url", required=True)
    serve.add_argument("--migrations", type=Path, default=DEFAULT_MIGRATIONS)
    serve.add_argument("--migrate-on-start", action="store_true")
    serve.add_argument("--grafana-webhook-secret")
    serve.add_argument("--grafana-webhook-secret-env")
    serve.add_argument("--grafana-key-id", default="grafana-v1")
    serve.add_argument("--auth-mode", choices=("local", "iap"), default="local")
    serve.add_argument("--surface", choices=("all", "ingest", "ui"), default="all")
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


def _send_fixture(url: str, secret: str, fixture: Path) -> None:
    body = fixture.read_bytes()
    timestamp = str(int(time.time()))
    signature = hmac.new(secret.encode(), timestamp.encode() + b":" + body, hashlib.sha256).hexdigest()
    response = httpx.post(
        url,
        content=body,
        headers={
            "content-type": "application/json",
            "x-grafana-alerting-signature": signature,
            "x-grafana-alerting-signature-timestamp": timestamp,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(response.json())


def _serve(args: argparse.Namespace) -> None:
    if args.auth_mode == "local" and args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("local auth mode may only bind to loopback")
    grafana_webhook_secret = _secret_argument(
        value=args.grafana_webhook_secret,
        environment_name=args.grafana_webhook_secret_env,
        option="--grafana-webhook-secret",
    )
    loom_token = _secret_argument(
        value=args.loom_token,
        environment_name=args.loom_token_env,
        option="--loom-token",
    )
    if args.surface in ("all", "ingest") and not grafana_webhook_secret:
        raise SystemExit("a Grafana webhook secret is required for the ingest surface")
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
    app = create_app(
        OpsService(repository, gateway),
        repository,
        WebConfig(
            grafana_webhook_secret=grafana_webhook_secret.encode() if grafana_webhook_secret else None,
            grafana_key_id=args.grafana_key_id,
            auth_mode=args.auth_mode,
            static_dir=args.static_dir,
            surface=args.surface,
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
