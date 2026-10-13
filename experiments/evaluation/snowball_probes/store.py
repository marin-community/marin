# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Crash-safe token reservations, traces, and the Marina upload outbox."""

import json
import sqlite3
import time
import zlib
from pathlib import Path


class TokenCapReached(Exception):
    pass


class Store:
    def __init__(self, path: Path, token_cap: int):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER, generated INTEGER);
            CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY, amount INTEGER);
            CREATE TABLE IF NOT EXISTS objects(
                path TEXT, key TEXT, payload BLOB, case_id TEXT, sequence INTEGER, model TEXT, state TEXT,
                PRIMARY KEY(path,key)
            );
            CREATE INDEX IF NOT EXISTS objects_case ON objects(path,case_id,sequence);
            CREATE INDEX IF NOT EXISTS objects_state ON objects(path,state);
            CREATE TABLE IF NOT EXISTS outbox(
                path TEXT, key TEXT, payload TEXT, generation INTEGER, priority INTEGER,
                PRIMARY KEY(path,key)
            );
        """
        )
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO budget VALUES(1,?,0)", (token_cap,))
        if self.connection.execute("SELECT cap FROM budget").fetchone()[0] != token_cap:
            raise ValueError("Cannot change an existing run's token cap")

    def recover(self) -> None:
        """Charge requests interrupted by a crash at their reserved maximum."""
        with self.connection:
            self.connection.execute(
                "UPDATE budget SET generated=generated+(SELECT COALESCE(SUM(amount),0) FROM reservations)"
            )
            self.connection.execute("DELETE FROM reservations")
        for case in self.objects("/cases", ("generating", "reviewing")):
            self.put("/cases", case["case_id"], {**case, "state": "interrupted"})
        for trace in self.objects("/traces", ("streaming",)):
            payload = {**trace["payload"], "error": "Worker interrupted; reservation charged conservatively"}
            self.put("/traces", trace["message_id"], {**trace, "state": "interrupted", "payload": payload})

    def totals(self) -> tuple[int, int]:
        generated = self.connection.execute("SELECT generated FROM budget").fetchone()[0]
        reserved = self.connection.execute("SELECT COALESCE(SUM(amount),0) FROM reservations").fetchone()[0]
        return generated, reserved

    def restore_budget(self, generated: int) -> None:
        """Restore durable accounting before an empty worker starts on another host."""
        if self.totals() != (0, 0) or self.count("/runs"):
            raise ValueError("Cannot restore a budget over existing worker state")
        with self.connection:
            self.connection.execute("UPDATE budget SET generated=? WHERE id=1", (generated,))

    def reserve(self, request_id: str, maximum: int) -> None:
        with self.connection:
            generated, reserved = self.totals()
            cap = self.connection.execute("SELECT cap FROM budget").fetchone()[0]
            if generated + reserved + maximum > cap:
                raise TokenCapReached
            self.connection.execute("INSERT INTO reservations VALUES(?,?)", (request_id, maximum))

    def settle(self, request_id: str, completion_tokens: int | None) -> int:
        """Count reported completions, including thinking; conservatively charge missing usage."""
        with self.connection:
            row = self.connection.execute("SELECT amount FROM reservations WHERE id=?", (request_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown token reservation {request_id}")
            amount = row[0] if completion_tokens is None else completion_tokens
            if amount < 0:
                raise ValueError("Negative completion usage")
            self.connection.execute("UPDATE budget SET generated=generated+?", (amount,))
            self.connection.execute("DELETE FROM reservations WHERE id=?", (request_id,))
        return amount

    def put(self, path: str, key: str, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        priority = {"/runs": 0, "/cases": 1, "/traces": 2}[path]
        with self.connection:
            self.connection.execute(
                "INSERT INTO objects VALUES(?,?,?,?,?,?,?) ON CONFLICT(path,key) DO UPDATE SET "
                "payload=excluded.payload,state=excluded.state",
                (
                    path,
                    key,
                    zlib.compress(payload.encode()),
                    value.get("case_id"),
                    value.get("sequence"),
                    value.get("model"),
                    value.get("state"),
                ),
            )
            self.connection.execute(
                "INSERT INTO outbox VALUES(?,?,?,?,?) ON CONFLICT(path,key) DO UPDATE SET "
                "payload=excluded.payload,generation=excluded.generation",
                (path, key, payload, time.time_ns(), priority),
            )

    def pending(self, limit: int = 64) -> list[tuple]:
        return self.connection.execute(
            "SELECT path,key,payload,generation FROM outbox ORDER BY priority,generation LIMIT ?", (limit,)
        ).fetchall()

    def acknowledge(self, path: str, key: str, generation: int) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM outbox WHERE path=? AND key=? AND generation=?", (path, key, generation)
            )

    def count(self, path: str) -> int:
        return self.connection.execute("SELECT count(*) FROM objects WHERE path=?", (path,)).fetchone()[0]

    def objects(self, path: str, states: tuple[str, ...] = ()) -> list[dict]:
        condition = " AND state IN (" + ",".join("?" for _ in states) + ")" if states else ""
        return [
            json.loads(zlib.decompress(row[0]))
            for row in self.connection.execute("SELECT payload FROM objects WHERE path=?" + condition, (path, *states))
        ]

    def get(self, path: str, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM objects WHERE path=? AND key=?", (path, key)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def case_traces(self, case_id: str) -> list[dict]:
        return [
            json.loads(zlib.decompress(row[0]))
            for row in self.connection.execute(
                "SELECT payload FROM objects WHERE path='/traces' AND case_id=? ORDER BY sequence,model,key",
                (case_id,),
            )
        ]

    def close(self) -> None:
        self.connection.close()
