# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL weight-transfer control endpoints for native inference."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


def add_weight_reload_routes(
    app: FastAPI, context: WeightPublicationContext, trainer: TrainerConfig, config: WeightTransferConfig
) -> None:
    # Torch is optional and is loaded only when the server explicitly enables transfer.
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

    @app.post("/update_weights")
    async def receive(request: WeightTensorRequest):
        publication = WeightPublication(request.publication_id, request.model_version)
        try:
            await asyncio.to_thread(session.receive, publication, request.name, request.dtype, request.shape, receiver)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "ok"}

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

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        await asyncio.to_thread(context.reset_prefix_cache)
        return {"status": "ok"}
