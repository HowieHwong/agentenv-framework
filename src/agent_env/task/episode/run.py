from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from agentenv_protocol import EnvironmentTool, ToolResult, ToolSession

from agent_env.providers.sandbox_providers.sandbox_provider import sandbox_request_headers_for_url
from agent_env.store.routing import namespace_routing
from agent_env.task.episode.stand_in import InProcessStandIn, StandInClient
from agent_env.task.store import record_task_cancelled
from agent_env.task.task import Task, recorded_scores, substitute_seed_in_steps
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import DeployedAgent, TaskStepContext

logger = logging.getLogger(__name__)

_live: dict[str, Episode] = {}
_DRAIN_RETRY_SECONDS = 1.0
_CARD_TIMEOUT_SECONDS = 10.0

BASH = EnvironmentTool(
    name="bash",
    description="Run a bash command where the agent works, as its user. Returns its stdout, then its stderr.",
    inputSchema={"type": "object", "properties": {"command": {"type": "string", "description": "The command to run."}},
                 "required": ["command"]},
)


class EpisodeError(RuntimeError):
    """A task that can't run as an episode, or an episode asked for what it can't give."""


@dataclass(frozen=True)
class Skill:
    """A skill the task gave the agent: its instructions are ``path``'s ``SKILL.md``, in the agent's sandbox."""

    name: str
    description: str
    path: str | None


class Turn:
    """An outside agent's turn. Its model reads ``system_prompt``, ``prompt`` and ``skills``, acts through ``tools()``
    (or an MCP client of its own on ``mcp_urls``, sending ``headers`` and ``role``), and hands its reply back with
    ``end()``."""

    def __init__(
        self, *, agent: str, step_id: str, prompt: str, system_prompt: str | None, role: str | None,
        mcp_urls: dict[str, str], skills: list[Skill], shell: bool, stand_in: StandInClient, task_id: str,
    ):
        self.agent = agent
        self.step_id = step_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.role = role
        self.mcp_urls = mcp_urls
        self.headers = {env_id: sandbox_request_headers_for_url(url) for env_id, url in mcp_urls.items()}
        self.skills = skills
        self._shell = shell
        self._stand_in = stand_in
        self._task_id = task_id
        self._spans: list[dict[str, Any]] = []
        self._started = _now()
        self._open = True

    def tools(self, *, timeout: float = 600.0) -> TurnTools:
        """A client for the tools of the envs this agent acts in, calling them as its role with the headers each URL
        needs, and ``bash`` when it works in a sandbox; use it with ``async with``. ``timeout`` bounds each env tool's
        call, in seconds."""
        return TurnTools(self, timeout)

    async def end(self, response: str | None = None) -> None:
        """Hand the turn back, and the run carries on. No reply records an empty one, as an agent that answers with
        nothing leaves. Once the turn is over, ended or not, this does nothing."""
        if not self._open:
            return
        self._open = False
        reply = response or ""
        spans = [*self._spans, _span("turn", "chain", self._started, completion={"content": [
            {"type": "text", "text": reply}]})]
        await self._stand_in.reply(self._task_id, text=reply, trajectory=spans, tool_call_count=len(self._spans))

    async def _fail(self, error: Exception) -> None:
        if self._open:
            self._open = False
            await self._stand_in.reply(self._task_id, error=str(error))


class TurnTools:
    """The tools of a turn, listed once when it opens: its envs', and ``bash`` when its agent works in a sandbox.
    ``list()`` names every one, ``call()`` routes a call to what offers it, and each call goes in the turn's
    trajectory."""

    def __init__(self, turn: Turn, timeout: float):
        self._turn = turn
        self._sessions = {
            env_id: ToolSession(mcp_url=url, role=turn.role, headers=turn.headers[env_id], timeout=timeout)
            for env_id, url in turn.mcp_urls.items()
        }
        self._tools: list[EnvironmentTool] = []
        self._offered_by: dict[str, str | None] = {}

    async def __aenter__(self) -> TurnTools:
        try:
            for env_id, session in self._sessions.items():
                await session.open()
                for tool in await session.list_tools():
                    if (other := self._offered_by.get(tool.name)) is not None:
                        raise EpisodeError(f"tool {tool.name!r} is offered by both env {other!r} and env {env_id!r}")
                    self._offered_by[tool.name] = env_id
                    self._tools.append(tool)
            if self._turn._shell:
                if (env_id := self._offered_by.get(BASH.name)) is not None:
                    raise EpisodeError(f"tool {BASH.name!r} is offered by env {env_id!r}, and the agent's sandbox "
                                       "offers its own")
                self._offered_by[BASH.name] = None
                self._tools.append(BASH)
        except BaseException:
            await self._close()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._close()

    def list(self) -> list[EnvironmentTool]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name not in self._offered_by:
            raise EpisodeError(
                f"no env of turn {self._turn.step_id!r} offers tool {name!r}; they offer {sorted(self._offered_by)}")
        started = _now()
        env_id = self._offered_by[name]
        if env_id is None:
            result = await self._bash(arguments or {})
        else:
            result = await self._sessions[env_id].call_tool(name, arguments)
        self._turn._spans.append(_span(name, "execute_tool", started, prompt={"input": arguments or {}},
                                       completion={"output": result.content}))
        return result

    async def _bash(self, arguments: dict[str, Any]) -> ToolResult:
        if not isinstance(command := arguments.get("command"), str):
            raise EpisodeError(f"tool {BASH.name!r} takes a command string, not {command!r}")
        return ToolResult(content=[{"type": "text", "text": await self._turn._stand_in.run(command)}])

    async def _close(self) -> None:
        await asyncio.gather(*(session.close() for session in self._sessions.values()))


class Episode:
    """One run of an ``EpisodeSpec``'s task, which waits at each outside turn for the caller to play it.

    Cancelling a caller of ``next_turn()``, ``result()`` or ``abandon()`` cancels that call, never the run. Use an
    episode on the event loop that started it."""

    def __init__(self, id: str, task: Task, context: TaskStepContext):
        self.id = id
        self._task = task
        self._context = context
        self._run: asyncio.Task[None]
        self._waiting: deque[Turn] = deque()
        self._turns: set[Turn] = set()
        self._changed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tearing_down = False
        self._abandoned: str | None = None
        self._stand_ins: list[StandInClient] = []
        self._drains: list[asyncio.Task[None]] = []
        self._in_process: list[InProcessStandIn] = []

    async def next_turn(self) -> Turn | None:
        """The next outside turn the run reaches, or None once the run has finished. Raises the run's error if it
        failed first. End the turns you hold before waiting on one that depends on them."""
        while True:
            changed = self._changed
            while self._waiting and self._abandoned is None:
                turn = self._waiting.popleft()
                if turn._open:
                    return turn
            if self._run.done():
                self._raise_unless_completed()
                return None
            await self._wait(changed)

    async def result(self) -> dict[str, float]:
        """Every score the run recorded, by the id of the step that recorded it, once the run and its teardown are
        over: what ``agent-env run`` reports. Making a reward of them is the caller's."""
        while not self._run.done():
            changed = self._changed
            if open_turn := next((turn for turn in self._turns if turn._open), None):
                raise EpisodeError(
                    f"turn {open_turn.step_id!r} of agent {open_turn.agent!r} isn't ended, so the run can't finish"
                )
            await self._wait(changed)
        self._raise_unless_completed()
        return recorded_scores(self._seeded_steps(), self._context)

    async def abandon(self, reason: str) -> None:
        """Stop the run and wait for its teardown; it is recorded cancelled with ``reason``. A run a step has
        already failed is only finishing what's in flight, so this ends its open turns and the run's error stands.
        Once the run has ended, or while it tears down, this only waits."""
        if not (self._run.done() or self._tearing_down or self._abandoned is not None):
            if any(failure.get("is_fatal") for failure in self._context.metadata.get("failed_steps") or []):
                for turn in list(self._turns):
                    await turn._fail(EpisodeError(f"episode abandoned: {reason}"))
            else:
                self._abandoned = reason
                self._run.cancel(msg=reason)
        await asyncio.wait((self._run,))

    async def _launch(self) -> None:
        """Start the run and return once its record exists. Cancelled, it takes the run down and waits for teardown."""
        _live[self.id] = self
        running = asyncio.Event()
        self._run = asyncio.create_task(self._drive(running))
        try:
            await running.wait()
        except asyncio.CancelledError:
            self._abandoned = "start() was cancelled"
            self._run.cancel()
            await asyncio.wait((self._run,))
            raise

    async def _drive(self, running: asyncio.Event) -> None:
        """Run the task with namespace routing on, as ``agent-env run`` does, so the stand-in, an ``@local`` agent,
        resolves."""
        running.set()
        with namespace_routing():
            try:
                await self._task.run(context=self._context, instance_id=self.id)
            finally:
                self._tearing_down = True
                try:
                    await self._release_stand_ins()
                    report = await teardown_run(self._context)
                    if report.still_up:
                        logger.warning("episode %s: %s may still be up", self.id,
                                       [s.sandbox_id for s in report.still_up])
                    if self._abandoned is not None:
                        await asyncio.to_thread(record_task_cancelled, self.id, self._abandoned, Task._utc_now_str())
                finally:
                    del _live[self.id]

    async def _wait(self, changed: asyncio.Future[None]) -> None:
        """Until the run ends or what was ``changed`` when the caller last looked changes again."""
        await asyncio.wait((self._run, changed), return_when=asyncio.FIRST_COMPLETED)

    def _raise_unless_completed(self) -> None:
        if self._abandoned is not None:
            raise EpisodeError(f"episode abandoned: {self._abandoned}")
        if (error := self._run.exception()) is not None:
            raise error

    async def _serve_in_process(self, agent: str, role: str | None, mcp_urls: dict[str, str]) -> DeployedAgent:
        """A stand-in served in this process for ``agent``, which works in no sandbox, played as the others are."""
        server = InProcessStandIn()
        self._in_process.append(server)
        url = await server.start()
        async with httpx.AsyncClient(timeout=_CARD_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{url}/.well-known/agent.json")
            response.raise_for_status()
        deployed = DeployedAgent(agent_name=agent, api_url=url, a2a_url=url, a2a_card=response.json(), role=role)
        await self._attach(agent, deployed, role, mcp_urls, shell=False)
        return deployed

    async def _attach(
        self, agent: str, deployed: DeployedAgent, role: str | None, mcp_urls: dict[str, str], *, shell: bool,
    ) -> None:
        """Claim ``agent``'s stand-in and hand out the turns it opens."""
        stand_in = StandInClient(deployed.a2a_url, deployed.a2a_card or {})
        self._stand_ins.append(stand_in)
        await stand_in.claim()
        self._drains.append(asyncio.create_task(self._drain(agent, stand_in, role, mcp_urls, shell)))

    async def _drain(
        self, agent: str, stand_in: StandInClient, role: str | None, mcp_urls: dict[str, str], shell: bool,
    ) -> None:
        """Hand out each turn the stand-in opens, and close each one it drops, until the run ends."""
        after, open_turns = 0, {}
        while True:
            try:
                answer = await stand_in.next(after)
            except httpx.HTTPError as e:
                logger.warning("episode %s: reading agent %r's turns failed, retrying: %s", self.id, agent, e)
                await asyncio.sleep(_DRAIN_RETRY_SECONDS)
                continue
            after = answer["version"]
            listed = {item["task_id"]: item for item in answer["turns"]}
            for task_id in [task_id for task_id in open_turns if task_id not in listed]:
                turn = open_turns.pop(task_id)
                turn._open = False
                self._turns.discard(turn)
            for task_id, item in listed.items():
                if task_id not in open_turns:
                    open_turns[task_id] = turn = Turn(
                        agent=agent, step_id=self._step_of(agent, item["context_id"]), prompt=item["text"],
                        system_prompt=item.get("system_prompt"), role=role, mcp_urls=dict(mcp_urls),
                        skills=[Skill(**skill) for skill in item.get("skills", [])], shell=shell, stand_in=stand_in,
                        task_id=task_id,
                    )
                    self._waiting.append(turn)
                    self._turns.add(turn)
            self._changed.set_result(None)
            self._changed = asyncio.get_running_loop().create_future()

    def _step_of(self, agent: str, context_id: str) -> str:
        """The ``prompt_agent`` step whose conversation with ``agent`` a message on ``context_id`` is in."""
        conversations = self._context.metadata.get("a2a_conversations", {})
        for step in self._seeded_steps():
            if step.type == "prompt_agent" and step.agent_name == agent and (
                    step.context_id or conversations.get(step.id)) == context_id:
                return step.id
        return context_id

    def _seeded_steps(self) -> list:
        return substitute_seed_in_steps(self._task.steps, self._context.metadata.get("seed") or {})

    async def _release_stand_ins(self) -> None:
        for drain in self._drains:
            drain.cancel()
        await asyncio.gather(*self._drains, return_exceptions=True)
        for turn in self._turns:
            turn._open = False
        for server in self._in_process:
            server.stop()
        await asyncio.gather(*(stand_in.close() for stand_in in self._stand_ins), return_exceptions=True)


def _span(name: str, operation: str, started: str, *, prompt: Any = None, completion: Any = None) -> dict[str, Any]:
    """An OTel GenAI span, the trajectory format rubrics read."""
    attributes = {"gen_ai.operation.name": operation}
    if prompt is not None:
        attributes["gen_ai.prompt"] = json.dumps(prompt, default=str)
    if completion is not None:
        attributes["gen_ai.completion"] = json.dumps(completion, default=str)
    return {"name": name, "start_time": started, "end_time": _now(), "attributes": attributes}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_of(context: TaskStepContext) -> Episode:
    return _live[context.metadata["episode"]["id"]]
