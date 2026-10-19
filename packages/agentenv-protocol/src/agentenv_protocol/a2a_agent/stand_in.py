"""The stand-in: an agent with no model, played by a driver outside its sandbox.

Each message it gets is an open turn until the driver replies to it. Through ``urn:agentenv:driver/v1`` the driver
reads the open turns, replies to them and runs commands where the stand-in works, as it; its first ``claim`` sets the
token every later call carries. A turn its sender cancels closes without a reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
import signal
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from starlette.exceptions import HTTPException

from .extensions import (
    DRIVER_V1,
    MCP_CONFIG_V1,
    SKILL_CONFIG_V1,
    TRAJECTORY_V1,
    BundleSkillRequest,
    DriverClaimRequest,
    DriverExecRequest,
    DriverExecResponse,
    DriverNextRequest,
    DriverNextResponse,
    DriverOutputRequest,
    DriverOutputResponse,
    DriverReplyRequest,
    DriverSkill,
    DriverTurn,
    InlineSkillRequest,
    extension,
)
from .framework import AgentEnvAgent, AgentIdentity, a2a_agent
from .tasks.v1 import AgentConfig, TaskRequest, TaskResult, TextPart, Usage
from ..transfers import download

LONG_POLL_SECONDS = 25.0
COMMAND_TIMEOUT_SECONDS = 3600
SKILLS_DIR = Path.home() / "skills"
TRAJECTORY_FORMAT = "agentenv-stand-in/v1"


class StandInConfig(AgentConfig):
    system_prompt: str | None = None


@dataclass
class _Turn:
    task_id: str
    context_id: str
    text: str
    system_prompt: str | None
    skills: list[DriverSkill]
    reply: asyncio.Future[DriverReplyRequest] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


IDENTITY = AgentIdentity(
    name="agentenv-stand-in",
    description="Stands in for an agent played from outside AgentEnv: a driver answers its turns.",
    version="1.0.0",
)
EXTENSIONS = (MCP_CONFIG_V1, TRAJECTORY_V1, DRIVER_V1)


@a2a_agent(identity=IDENTITY, config=StandInConfig, extensions=EXTENSIONS)
class StandInAgent(AgentEnvAgent):
    """``shell=False`` is a stand-in for an agent that works in no sandbox: it keeps no skills and runs no commands.
    ``long_poll_seconds`` bounds how long ``next`` and ``output`` wait for something to report."""

    def __init__(self, *, shell: bool = True, long_poll_seconds: float = LONG_POLL_SECONDS):
        self._shell = shell
        self._long_poll_seconds = long_poll_seconds
        self._token: str | None = None
        self._turns: dict[str, _Turn] = {}
        self._version = 0
        self._changed: asyncio.Future[None] | None = None
        self._commands: dict[str, asyncio.Task[str]] = {}

    async def run(self, request: TaskRequest[StandInConfig]) -> TaskResult:
        turn = _Turn(
            task_id=request.task_id,
            context_id=request.context_id,
            text="\n".join(part.text for part in request.parts if isinstance(part, TextPart)),
            system_prompt=request.config.system_prompt,
            skills=[DriverSkill(name=skill["name"], description=skill.get("description") or "",
                                path=str(SKILLS_DIR / skill["name"]) if self._shell else None)
                    for skill in request.skills],
        )
        self._turns[turn.task_id] = turn
        self._bump()
        try:
            reply = await turn.reply
        finally:
            del self._turns[turn.task_id]
            self._bump()
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        for span in reply.trajectory:
            request.trajectory.append(span)
        if reply.error is not None:
            return TaskResult.failure("stand_in.driver_error", reply.error)
        return TaskResult.text(reply.text, usage=Usage(tool_call_count=reply.tool_call_count))

    @extension(DRIVER_V1.claim)
    async def claim(self, request: DriverClaimRequest) -> None:
        if self._token is None:
            self._token = request.token
        elif not hmac.compare_digest(self._token, request.token):
            raise HTTPException(status_code=409, detail="this stand-in is already claimed")

    @extension(DRIVER_V1.next)
    async def next_turns(self, request: DriverNextRequest) -> DriverNextResponse:
        self._check(request.token)
        if self._version <= request.after:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._next_change()), self._long_poll_seconds)
        return DriverNextResponse(version=self._version, turns=[
            DriverTurn(task_id=turn.task_id, context_id=turn.context_id, text=turn.text,
                       system_prompt=turn.system_prompt, skills=turn.skills)
            for turn in self._turns.values() if not turn.reply.done()
        ])

    @extension(DRIVER_V1.reply)
    async def reply(self, request: DriverReplyRequest) -> None:
        self._check(request.token)
        turn = self._turns.get(request.task_id)
        if turn is None or turn.reply.done():
            raise HTTPException(status_code=404, detail=f"no open turn {request.task_id!r}")
        turn.reply.set_result(request)

    @extension(DRIVER_V1.exec)
    async def exec_command(self, request: DriverExecRequest) -> DriverExecResponse:
        self._check(request.token)
        if not self._shell:
            raise HTTPException(status_code=409, detail="this stand-in works in no sandbox, so it runs no commands")
        command_id = uuid.uuid4().hex
        self._commands[command_id] = asyncio.create_task(_run(request.command))
        return DriverExecResponse(exec_id=command_id)

    @extension(DRIVER_V1.output)
    async def output(self, request: DriverOutputRequest) -> DriverOutputResponse:
        self._check(request.token)
        if (command := self._commands.get(request.exec_id)) is None:
            raise HTTPException(status_code=404, detail=f"no command {request.exec_id!r}")
        await asyncio.wait((command,), timeout=self._long_poll_seconds)
        if not command.done():
            return DriverOutputResponse(done=False)
        self._commands.pop(request.exec_id, None)
        return DriverOutputResponse(done=True, output=command.result())

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, str]:
        folder = self._skill_folder(request.name)
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, str]:
        folder = self._skill_folder(request.name)
        for item in request.skill_bundle.files:
            await download(item.object, folder / item.path)
        return {"name": request.name}

    def _check(self, token: str) -> None:
        if self._token is None or not hmac.compare_digest(self._token, token):
            raise HTTPException(status_code=401, detail="the driver's token is missing or wrong")

    def _skill_folder(self, name: str) -> Path:
        if not self._shell:
            raise HTTPException(status_code=409, detail="this stand-in works in no sandbox, so it keeps no skills")
        return SKILLS_DIR / name

    def _next_change(self) -> asyncio.Future[None]:
        if self._changed is None:
            self._changed = asyncio.get_running_loop().create_future()
        return self._changed

    def _bump(self) -> None:
        self._version += 1
        if self._changed is not None:
            self._changed.set_result(None)
            self._changed = None


async def _run(command: str) -> str:
    """``command``'s stdout then its stderr, as verifiers' bash tool returns them. It runs in a process group of its
    own, stopped with whatever it started if it outlives ``COMMAND_TIMEOUT_SECONDS``."""
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), COMMAND_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return f"error: the command timed out after {COMMAND_TIMEOUT_SECONDS}s"
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    return (stdout + stderr).decode(errors="replace")


if __name__ == "__main__":
    StandInAgent().serve()
