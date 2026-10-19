"""The episode's end of a stand-in agent: the driver calls of ``urn:agentenv:driver/v1``, and a stand-in served in this
process for an outside agent that works in no sandbox."""

from __future__ import annotations

import asyncio
import secrets
import threading
from typing import Any

import httpx
import uvicorn
from agentenv_protocol.a2a_agent import DRIVER_V1
from agentenv_protocol.a2a_agent.stand_in import StandInAgent

from agent_env.a2a_agent.a2a_agent import A2AAgent

_CALL_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_SERVE_TIMEOUT_SECONDS = 30.0
# Short, so a long-poll in flight when the episode ends finishes within the server's shutdown.
_IN_PROCESS_LONG_POLL_SECONDS = 1.0


class StandInClient:
    """One stand-in, claimed with a token of this episode's, at the endpoints its card names."""

    def __init__(self, a2a_url: str, card: dict[str, Any]):
        driver = A2AAgent.find_extension(card, DRIVER_V1.uri)
        if driver is None:
            raise RuntimeError(f"the agent at {a2a_url} isn't a stand-in: its card has no {DRIVER_V1.uri}")
        self._endpoints = {operation: a2a_url + A2AAgent.operation(driver, operation)[1]
                           for operation in DRIVER_V1.operations}
        self._token = secrets.token_urlsafe(24)
        self._client = httpx.AsyncClient(timeout=_CALL_TIMEOUT)

    async def claim(self) -> None:
        await self._call("claim")

    async def next(self, after: int) -> dict[str, Any]:
        return await self._call("next", after=after)

    async def reply(self, task_id: str, **reply: Any) -> bool:
        """Whether the reply reached an open turn: a turn its sender stopped waiting for is gone."""
        response = await self._client.post(self._endpoints["reply"], json={"token": self._token, "task_id": task_id,
                                                                           **reply})
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    async def run(self, command: str) -> str:
        exec_id = (await self._call("exec", command=command))["exec_id"]
        while not (output := await self._call("output", exec_id=exec_id))["done"]:
            pass
        return output["output"]

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(self, operation: str, **body: Any) -> dict[str, Any]:
        response = await self._client.post(self._endpoints[operation], json={"token": self._token, **body})
        response.raise_for_status()
        return response.json()


class InProcessStandIn:
    """A stand-in served on loopback in a thread of its own, with no shell: one for an outside agent that works in no
    sandbox, which only this process's steps message."""

    def __init__(self) -> None:
        agent = StandInAgent(shell=False, long_poll_seconds=_IN_PROCESS_LONG_POLL_SECONDS)
        self._server = uvicorn.Server(uvicorn.Config(agent.create_app(), host="127.0.0.1", port=0, log_level="warning",
                                                     timeout_graceful_shutdown=2 * _IN_PROCESS_LONG_POLL_SECONDS))
        self._thread = threading.Thread(target=self._server.run, name="agentenv-stand-in", daemon=True)

    async def start(self) -> str:
        """Serve it, and return its URL once it answers."""
        self._thread.start()
        async with asyncio.timeout(_SERVE_TIMEOUT_SECONDS):
            while not self._server.started:
                if not self._thread.is_alive():
                    raise RuntimeError("the in-process stand-in agent stopped as it started")
                await asyncio.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        """Have it stop; its thread ends once the requests in flight have."""
        self._server.should_exit = True
