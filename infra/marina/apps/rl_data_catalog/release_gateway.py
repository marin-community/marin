# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run candidate read-only API projections through Marina's bounded SQL transport."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from fastapi.testclient import TestClient
from marina.applets import AppletServices
from marina.client import AppletQuery, MarinaRequestError
from sqlalchemy.engine import Engine

from .server.app import create_api


@dataclass
class GatewayRows:
    rows: list[Any]

    def __iter__(self):
        return iter(self.rows)

    def all(self):
        return self.rows

    def one(self):
        if len(self.rows) != 1:
            raise ValueError(f"Expected one SQL row, got {len(self.rows)}")
        return self.rows[0]

    def one_or_none(self):
        return self.one() if self.rows else None

    def first(self):
        return self.rows[0] if self.rows else None


@dataclass
class GatewayResult:
    rows: list[dict[str, Any]]

    def __iter__(self):
        return iter(self.tuples())

    def mappings(self):
        return GatewayRows(self.rows)

    def scalars(self):
        return GatewayRows([next(iter(row.values())) for row in self.rows])

    def tuples(self):
        return GatewayRows([tuple(row.values()) for row in self.rows])

    def one(self):
        return self.tuples().one()

    def scalar_one(self):
        return self.scalars().one()

    def scalar_one_or_none(self):
        return self.scalars().one_or_none()


@dataclass
class GatewayConnection:
    query: AppletQuery

    def execute(self, statement, parameters=None):
        sql = str(statement)
        if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError("Candidate browser projection cannot mutate the catalog")
        return GatewayResult(self.query(sql, parameters or {})["rows"])


@dataclass
class GatewayEngine:
    query: AppletQuery

    @contextmanager
    def connect(self):
        yield GatewayConnection(self.query)


@dataclass
class GatewayServices:
    query: AppletQuery

    def engine(self) -> Engine:
        return cast(Engine, GatewayEngine(self.query))


class CandidateApi:
    """Exercise candidate route code against the frozen shared SQL, without publishing it."""

    def __init__(self, query: AppletQuery):
        self.client = TestClient(create_api(cast(AppletServices, GatewayServices(query))))

    def get(self, path: str) -> Any:
        response = self.client.get("/" + path.removeprefix("api/"))
        if not response.is_success:
            raise MarinaRequestError(response.status_code, response.text)
        return response.json()

    def close(self) -> None:
        self.client.close()
