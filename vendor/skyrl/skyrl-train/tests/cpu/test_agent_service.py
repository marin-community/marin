import asyncio

import httpx
import pytest
from fastapi import FastAPI

from skyrl_train.rollouts.agent_service import serve_agent_app


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier_reward", [0.0, 1.0])
async def test_finished_trial_keeps_verifier_reward_while_pending_model_callback_stops(verifier_reward):
    app = FastAPI()
    entered = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()

    @app.get("/completion")
    async def completion():
        entered.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(timeout=5) as client:
        pending = None

        async def finished_trial():
            nonlocal pending
            async with serve_agent_app(app, host="127.0.0.1", startup_timeout=5, shutdown_timeout=1) as port:
                pending = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/completion"))
                await asyncio.wait_for(entered.wait(), timeout=5)
                return {"verifier_reward": verifier_reward}

        try:
            result = await finished_trial()
            assert result == {"verifier_reward": verifier_reward}
            assert stopped.is_set()
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
