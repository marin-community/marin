# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A one-table app: who is calling, and a list of notes in the app's own schema."""

from fastapi import FastAPI
from marina.apps import RegisteredApi, Services, registered_api
from marina.mcp import OperationRisk, operation_extension
from pydantic import BaseModel, RootModel
from rigging.server_auth import get_verified_identity
from sqlalchemy import Engine, text


class IdentityResponse(BaseModel):
    user: str
    data: str


class NoteCreate(BaseModel):
    body: str


class NoteCreated(BaseModel):
    ok: bool


class NoteList(RootModel[list[str]]):
    pass


def create_api(services: Services) -> RegisteredApi:
    api = FastAPI()
    engine = services.engine()

    @api.get(
        "/whoami",
        operation_id="identity",
        description="Return the verified caller and this application's data root.",
        response_model=IdentityResponse,
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def whoami() -> IdentityResponse:
        return IdentityResponse(user=get_verified_identity().user_id, data=services.data_url)

    @api.post(
        "/notes",
        operation_id="create_note",
        description="Create one note in this application's database schema.",
        response_model=NoteCreated,
        openapi_extra=operation_extension(OperationRisk.WRITE),
    )
    def add(body: NoteCreate) -> NoteCreated:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO notes (body) VALUES (:body)"), {"body": body.body})
        return NoteCreated(ok=True)

    @api.get(
        "/notes",
        operation_id="list_notes",
        description="List notes from this application's database schema in creation order.",
        response_model=NoteList,
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def list_notes() -> NoteList:
        with engine.connect() as conn:
            return NoteList([row[0] for row in conn.execute(text("SELECT body FROM notes ORDER BY id"))])

    return registered_api(api)


def migrate(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS notes"))
        conn.execute(text("CREATE TABLE notes (id serial PRIMARY KEY, body text NOT NULL)"))
