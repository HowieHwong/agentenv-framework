"""The proxy: an agent with no model, whose turns a driver outside its sandbox answers.

Each message it gets is an open turn until the driver replies to it. Through ``urn:agentenv:proxy/v1`` the driver
reads the open turns, replies to them and runs commands where the proxy works, as it; its first ``claim`` sets the
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
    PROXY_V1,
    SKILL_CONFIG_V1,
    TRAJECTORY_V1,
    BundleSkillRequest,
    InlineSkillRequest,
    ProxyClaimRequest,
    ProxyExecRequest,
    ProxyExecResponse,
    ProxyNextRequest,
    ProxyNextResponse,
    ProxyOutputRequest,
    ProxyOutputResponse,
    ProxyReplyRequest,
    ProxySkill,
    ProxyTurn,
    extension,
)
from .framework import AgentEnvAgent, AgentIdentity, a2a_agent
from .tasks.v1 import AgentConfig, TaskRequest, TaskResult, TextPart, Usage
from ..transfers import download

LONG_POLL_SECONDS = 25.0
COMMAND_TIMEOUT_SECONDS = 3600
SKILLS_DIR = Path.home() / "skills"
TRAJECTORY_FORMAT = "agentenv-proxy/v1"


class ProxyConfig(AgentConfig):
    system_prompt: str | None = None
    # The role a deploy names, which the driver acts as in the envs it reaches itself.
    role: str | None = None


@dataclass
class _Turn:
    task_id: str
    context_id: str
    text: str
    system_prompt: str | None
    skills: list[ProxySkill]
    reply: asyncio.Future[ProxyReplyRequest] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


IDENTITY = AgentIdentity(
    name="agentenv-proxy",
    description="An agent played from outside AgentEnv: a driver answers its turns.",
    version="1.0.0",
)
EXTENSIONS = (TRAJECTORY_V1, PROXY_V1)


@a2a_agent(identity=IDENTITY, config=ProxyConfig, extensions=EXTENSIONS)
class ProxyAgent(AgentEnvAgent):
    """``long_poll_seconds`` bounds how long ``next`` and ``output`` wait for something to report."""

    def __init__(self, *, long_poll_seconds: float = LONG_POLL_SECONDS):
        self._long_poll_seconds = long_poll_seconds
        self._token: str | None = None
        self._turns: dict[str, _Turn] = {}
        self._version = 0
        self._changed: asyncio.Future[None] | None = None
        self._commands: dict[str, asyncio.Task[str]] = {}

    async def run(self, request: TaskRequest[ProxyConfig]) -> TaskResult:
        turn = _Turn(
            task_id=request.task_id,
            context_id=request.context_id,
            text="\n".join(part.text for part in request.parts if isinstance(part, TextPart)),
            system_prompt=request.config.system_prompt,
            skills=[ProxySkill(name=skill["name"], description=skill.get("description") or "",
                               path=str(SKILLS_DIR / skill["name"]))
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
            return TaskResult.failure("proxy.driver_error", reply.error)
        return TaskResult.text(reply.text, usage=Usage(tool_call_count=reply.tool_call_count))

    @extension(PROXY_V1.claim)
    async def claim(self, request: ProxyClaimRequest) -> None:
        if self._token is None:
            self._token = request.token
        elif not hmac.compare_digest(self._token, request.token):
            raise HTTPException(status_code=409, detail="this proxy is already claimed")

    @extension(PROXY_V1.next)
    async def next_turns(self, request: ProxyNextRequest) -> ProxyNextResponse:
        self._check(request.token)
        if self._version <= request.after:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._next_change()), self._long_poll_seconds)
        return ProxyNextResponse(version=self._version, turns=[
            ProxyTurn(task_id=turn.task_id, context_id=turn.context_id, text=turn.text,
                      system_prompt=turn.system_prompt, skills=turn.skills)
            for turn in self._turns.values() if not turn.reply.done()
        ])

    @extension(PROXY_V1.reply)
    async def reply(self, request: ProxyReplyRequest) -> None:
        self._check(request.token)
        turn = self._turns.get(request.task_id)
        if turn is None or turn.reply.done():
            raise HTTPException(status_code=404, detail=f"no open turn {request.task_id!r}")
        turn.reply.set_result(request)

    @extension(PROXY_V1.exec)
    async def exec_command(self, request: ProxyExecRequest) -> ProxyExecResponse:
        self._check(request.token)
        command_id = uuid.uuid4().hex
        self._commands[command_id] = asyncio.create_task(_run(request.command))
        return ProxyExecResponse(exec_id=command_id)

    @extension(PROXY_V1.output)
    async def output(self, request: ProxyOutputRequest) -> ProxyOutputResponse:
        self._check(request.token)
        if (command := self._commands.get(request.exec_id)) is None:
            raise HTTPException(status_code=404, detail=f"no command {request.exec_id!r}")
        await asyncio.wait((command,), timeout=self._long_poll_seconds)
        if not command.done():
            return ProxyOutputResponse(done=False)
        self._commands.pop(request.exec_id, None)
        return ProxyOutputResponse(done=True, output=command.result())

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, str]:
        folder = SKILLS_DIR / request.name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, str]:
        for item in request.skill_bundle.files:
            await download(item.object, SKILLS_DIR / request.name / item.path)
        return {"name": request.name}

    def _check(self, token: str) -> None:
        if self._token is None or not hmac.compare_digest(self._token, token):
            raise HTTPException(status_code=401, detail="the driver's token is missing or wrong")

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
    own, stopped with whatever it started if its output isn't done within ``COMMAND_TIMEOUT_SECONDS``, or if the call
    is cancelled, even when ``bash`` has exited and a child it left holds the output open."""
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), COMMAND_TIMEOUT_SECONDS)
    except BaseException as e:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        if isinstance(e, asyncio.TimeoutError):
            return f"error: the command timed out after {COMMAND_TIMEOUT_SECONDS}s"
        raise
    return (stdout + stderr).decode(errors="replace")
