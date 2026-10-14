# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Postgres transport through the lab's restricted worker functions."""

import base64
import json
import zlib
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from uuid import UUID

from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.marina.applets.snowball_probes.server.records import case_record, searchable_metadata


@dataclass(frozen=True)
class DatabaseConfig:
    connection_name: str
    database: str
    user: str


class WorkerDatabase:
    def __init__(self, engine: Engine, applet: str):
        self.engine = engine
        self.schema = "applet_" + UUID(applet).hex

    def call(self, function: str, parameters: dict) -> dict:
        arguments = ",".join(
            f"CAST(:{name} AS JSONB)" if name in {"record", "records"} else f":{name}" for name in parameters
        )
        with self.engine.begin() as connection:
            result = connection.execute(text(f"SELECT {self.schema}.{function}({arguments})"), parameters).scalar_one()
        return result

    def run(self, run_id: str) -> dict:
        return self.call("worker_run", {"run_name": run_id})

    def cases(self, run_id: str, state: str = "", limit: int = 200) -> dict:
        result = self.call("worker_cases", {"run_name": run_id, "case_state": state, "page_limit": limit})
        result["cases"] = [self.decode_case(case) for case in result["cases"]]
        return result

    def case(self, case_id: str, trace_state: str) -> dict:
        result = self.decode_case(self.call("worker_case", {"case_name": case_id, "trace_state": trace_state}))
        for trace in result["traces"]:
            trace["payload"] = json.loads(zlib.decompress(base64.b64decode(trace["payload"])))
        return result

    @staticmethod
    def decode_case(case: dict) -> dict:
        encoded = case["metadata_payload"]
        case["metadata_payload"] = base64.b64decode(encoded) if encoded is not None else None
        return case_record(case)

    def save(self, path: str, records: list[dict]) -> dict:
        if path == "/runs":
            if len(records) != 1:
                raise ValueError("Save one run snapshot at a time")
            return self.call("worker_save_run", {"record": json.dumps(records[0], ensure_ascii=False)})
        if path == "/cases":
            values = [
                {
                    **record,
                    "seed": searchable_metadata(record["seed"]),
                    "review": searchable_metadata(record["review"]),
                    "metadata_payload": (
                        base64.b64encode(
                            zlib.compress(json.dumps({"seed": record["seed"], "review": record["review"]}).encode())
                        ).decode()
                    ),
                }
                for record in records
            ]
            return self.call("worker_save_cases", {"records": json.dumps(values, ensure_ascii=False)})
        if path != "/traces":
            raise ValueError(f"Unknown worker record path: {path}")
        values = []
        for record in records:
            payload = json.dumps(record["payload"], ensure_ascii=False).encode()
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("One trace message exceeds 4 MiB")
            values.append({**record, "payload": base64.b64encode(zlib.compress(payload)).decode()})
        return self.call("worker_save_traces", {"records": json.dumps(values, ensure_ascii=False)})

    def claim(self, run_id: str, case_id: str) -> dict:
        return self.call("worker_claim", {"run_name": run_id, "case_name": case_id})


@contextmanager
def cloud_database(config: DatabaseConfig, applet: str) -> Iterator[WorkerDatabase]:
    """Use renewable machine IAM credentials with a restricted database login."""
    with Connector(refresh_strategy="lazy", timeout=30) as connector:

        def connect():
            connection = connector.connect(
                config.connection_name,
                "pg8000",
                user=config.user,
                db=config.database,
                enable_iam_auth=True,
                timeout=60,
            )
            with closing(connection.cursor()) as cursor:
                cursor.execute("SET statement_timeout TO '60s'")
            connection.commit()
            return connection

        engine = create_engine(
            "postgresql+pg8000://",
            creator=connect,
            pool_size=4,
            max_overflow=0,
            pool_pre_ping=True,
            pool_timeout=30,
            hide_parameters=True,
        )
        try:
            yield WorkerDatabase(engine, applet)
        finally:
            engine.dispose()
