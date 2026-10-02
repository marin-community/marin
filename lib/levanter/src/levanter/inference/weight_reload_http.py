# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL weight-transfer control endpoints for native inference."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from levanter.inference.draft_reload import DraftReloadSession
from levanter.inference.weight_reload import (
    WeightPublication,
    WeightPublicationContext,
    WeightReloadSession,
    WeightTransferConfig,
)
from levanter.trainer import TrainerConfig


class WeightGroupRequest(BaseModel):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str
    override_existing: bool = False


class WeightPublicationRequest(BaseModel):
    publication_id: str
    model_version: int


class WeightTensorRequest(WeightPublicationRequest):
    name: str
    dtype: str
    shape: tuple[int, ...]


class DraftCheckpointInfo(BaseModel):
    weights_path: str


class DraftCheckpointRequest(WeightPublicationRequest):
    update_info: DraftCheckpointInfo


def add_weight_reload_routes(
    app: FastAPI,
    context: WeightPublicationContext,
    trainer: TrainerConfig,
    config: WeightTransferConfig | None,
    *,
    draft_session: DraftReloadSession | None = None,
) -> None:
    session = None
    receiver = None
    if config is not None:
        # Torch is optional and is loaded only when tensor broadcast is enabled.
        from levanter.inference.torch_weight_transfer import TorchWeightReceiver  # noqa: PLC0415 -- optional Torch

        session = WeightReloadSession(context, trainer, config)
        receiver = TorchWeightReceiver(config)
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application):
            try:
                async with original_lifespan(application) as state:
                    yield {} if state is None else state
            finally:
                await asyncio.to_thread(session.reset_transport, receiver.close)

        app.router.lifespan_context = lifespan

        @app.post("/init_weight_update_communicator")
        async def initialize(request: WeightGroupRequest):
            await asyncio.to_thread(session.reset_transport, lambda: receiver.initialize(**request.model_dump()))
            return {"status": "ok"}

        @app.post("/begin_weight_reload")
        async def begin():
            publication = await asyncio.to_thread(session.begin)
            return asdict(publication)

        @app.post("/finish_weight_reload")
        async def finish(request: WeightPublicationRequest):
            try:
                version = await asyncio.to_thread(session.finish, WeightPublication(**request.model_dump()))
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return {"model_version": version}

        @app.post("/destroy_weights_update_group")
        async def destroy():
            await asyncio.to_thread(session.reset_transport, receiver.close)
            return {"status": "ok"}

    @app.post("/update_weights")
    async def receive(request: WeightTensorRequest | DraftCheckpointRequest):
        publication = WeightPublication(request.publication_id, request.model_version)
        try:
            if isinstance(request, DraftCheckpointRequest):
                if draft_session is None:
                    raise ValueError("Remote draft checkpoint updates are disabled")
                await asyncio.to_thread(draft_session.receive, publication, request.update_info.weights_path)
            else:
                if session is None or receiver is None:
                    raise ValueError("Remote tensor broadcasts are disabled")
                await asyncio.to_thread(
                    session.receive, publication, request.name, request.dtype, request.shape, receiver
                )
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "ok"}

    if draft_session is not None:

        @app.post("/start_draft_weight_update")
        async def begin_draft():
            try:
                return asdict(await asyncio.to_thread(draft_session.begin))
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error

        @app.post("/finish_weight_update")
        async def finish_draft(request: WeightPublicationRequest):
            try:
                await asyncio.to_thread(draft_session.finish, WeightPublication(**request.model_dump()))
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return {"active": True, "model_version": request.model_version}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        await asyncio.to_thread(context.reset_prefix_cache)
        return {"status": "ok"}
