#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[pg8000]>=1.9",
# ]
# ///
"""Read and append the shared agent work_log on the echo context database.

The work_log is a team-wide, agent-written logbook: one row per distilled milestone
(a result, a decision, a blocker, a handoff), never session transcripts. See
`.agents/skills/work-log/SKILL.md` for when to read and write, and
`infra/echo/README.md` for the database itself.

    scripts/work_log.py recent [--days 7] [--project P] [--limit 30]
    scripts/work_log.py show <id>
    scripts/work_log.py add --project P --title T [--body -|TEXT] [--author A]

Auth: gcloud ADC with roles/cloudsql.client plus accessor on the
cloudsql-agents-password secret in hai-gcp-models.
"""

import argparse
import getpass
import subprocess
import sys

from google.cloud.sql.connector import Connector

PROJECT = "hai-gcp-models"
INSTANCE = "hai-gcp-models:us-central1:marin-metadata"
DATABASE = "context"
DB_USER = "agents"
PASSWORD_SECRET = "cloudsql-agents-password"


def db_password() -> str:
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={PASSWORD_SECRET}", f"--project={PROJECT}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def connect(connector: Connector):
    return connector.connect(INSTANCE, "pg8000", user=DB_USER, password=db_password(), db=DATABASE)


def cmd_recent(cur, args: argparse.Namespace) -> None:
    filters, params = ["at > now() - make_interval(days => %s)"], [args.days]
    if args.project:
        filters.append("project = %s")
        params.append(args.project)
    cur.execute(
        # work_log.at, not bare at: the selected at::date is also named "at", and ORDER BY
        # prefers output-column names — sorting by the truncated date ties same-day entries.
        f"SELECT id, at::date, author, project, title FROM work_log WHERE {' AND '.join(filters)} "
        "ORDER BY work_log.at DESC LIMIT %s",
        [*params, args.limit],
    )
    rows = cur.fetchall()
    if not rows:
        scope = f" for project {args.project}" if args.project else ""
        print(f"no entries in the last {args.days} days{scope}")
        return
    for row_id, date, author, project, title in rows:
        print(f"#{row_id:<5} {date}  {author:<24} {project:<24} {title}")


def cmd_show(cur, args: argparse.Namespace) -> None:
    cur.execute("SELECT at, author, project, title, body FROM work_log WHERE id = %s", [args.id])
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no work_log entry #{args.id}")
    at, author, project, title, body = row
    print(f"#{args.id} {at:%Y-%m-%d %H:%M}Z {author} [{project}]\n{title}\n")
    if body:
        print(body)


def cmd_add(cur, args: argparse.Namespace) -> None:
    body = sys.stdin.read() if args.body == "-" else args.body
    cur.execute(
        "INSERT INTO work_log (author, project, title, body) VALUES (%s, %s, %s, %s) RETURNING id",
        [args.author, args.project, args.title, body],
    )
    print(f"logged #{cur.fetchone()[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and append the shared agent work_log.")
    sub = parser.add_subparsers(dest="command", required=True)

    recent = sub.add_parser("recent", help="list recent entries")
    recent.add_argument("--days", type=int, default=7)
    recent.add_argument("--project", help="filter to one project slug")
    recent.add_argument("--limit", type=int, default=30)
    recent.set_defaults(func=cmd_recent)

    show = sub.add_parser("show", help="print one entry with its body")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)

    add = sub.add_parser("add", help="append one entry")
    add.add_argument("--project", required=True, help="stable slug for the thread of work")
    add.add_argument("--title", required=True, help="one-line summary")
    add.add_argument("--body", help="short markdown; '-' reads stdin")
    add.add_argument("--author", default=f"{getpass.getuser()}/claude-code", help="whose agent: <user>/<agent>")
    add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    connector = Connector(quota_project=PROJECT)
    conn = connect(connector)
    conn.autocommit = True
    try:
        args.func(conn.cursor(), args)
    finally:
        conn.close()
        connector.close()


if __name__ == "__main__":
    main()
