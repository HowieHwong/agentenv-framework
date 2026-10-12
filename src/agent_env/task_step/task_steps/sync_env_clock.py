"""Arm a deployed env gateway's virtual clock, then sync its backing servers to it.

Arms the clock (set_time) and reads the gateway's server-reachable env_get_time_url (state), both as the
stored env card declares clock/v1, then invokes sync_time on each backing server whose child env card
advertises it. Non-advertisers are skipped when tolerate_missing_sync_time (default True). Place right
before prompt_agent so the virtual timeline starts at agent-start. virtual_time_from_step_id resumes the clock
from the virtual time a snapshot_env step recorded, so an env restored from that snapshot reads no earlier than
the records written into it. That time is when the snapshot completed, so on a fast clock it can be well past the
agent's last action (see snapshot_env).
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol import client as protocol_v1

from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class SyncEnvClockTaskStep(TaskStep):
    type: ClassVar[str] = "sync_env_clock"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        virtual_time: Optional[str] = None,
        virtual_seconds_per_real_second: Optional[float] = None,
        tolerate_missing_sync_time: bool = True,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
        virtual_time_from_step_id: Optional[str] = None,
    ):
        """virtual_seconds_per_real_second = clock speed (1.0 real time, 0 frozen, 86400 cap =
        1 real s -> 1 virtual day). For ~V virtual seconds over an ~R-real-second task, pass V/R.
        With virtual_time_from_step_id, virtual_time is the fallback when that step recorded no time, and an
        unset rate takes the recorded one."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not (isinstance(virtual_time, str) or (virtual_time is None and virtual_time_from_step_id is not None)):
            raise ValueError("virtual_time must be an RFC3339 string (omit it only with virtual_time_from_step_id)")
        if virtual_time_from_step_id is None and virtual_seconds_per_real_second is None:
            virtual_seconds_per_real_second = 1.0
        self.env_id = env_id
        self.virtual_time = virtual_time
        self.virtual_seconds_per_real_second = virtual_seconds_per_real_second
        self.tolerate_missing_sync_time = tolerate_missing_sync_time
        self.virtual_time_from_step_id = virtual_time_from_step_id
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["virtual_time"] = self.virtual_time
        base["virtual_seconds_per_real_second"] = self.virtual_seconds_per_real_second
        base["tolerate_missing_sync_time"] = self.tolerate_missing_sync_time
        base["timeout_seconds"] = self.timeout_seconds
        if self.virtual_time_from_step_id is not None:
            base["virtual_time_from_step_id"] = self.virtual_time_from_step_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> SyncEnvClockTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            virtual_time=data.get("virtual_time"),
            virtual_seconds_per_real_second=data.get("virtual_seconds_per_real_second"),
            tolerate_missing_sync_time=data.get("tolerate_missing_sync_time", True),
            virtual_time_from_step_id=data.get("virtual_time_from_step_id"),
            timeout_seconds=data.get("timeout_seconds", 30),
        )

    def _service_names(self, deployed) -> list[str]:
        """Backing MCP server names for the env (mirrors apply_server_config's enumeration)."""
        from agent_env.env.env import Env
        env = Env.get(deployed.env_id, deployed.env_version)
        names = [e.environment_name for e in (getattr(env, "mcp_server_envs", None) or [])]
        if not names and getattr(env, "environment_name", None):
            names = [env.environment_name]
        return names

    def context_source_step_ids(self) -> list[str]:
        return [self.virtual_time_from_step_id] if self.virtual_time_from_step_id is not None else []

    def _resolve_time(self, context: TaskStepContext) -> tuple[str, float]:
        """(virtual_time, rate) to arm: the time virtual_time_from_step_id recorded, else virtual_time; an
        explicit rate wins over the recorded one."""
        recorded = (context.metadata.get("env_snapshot_virtual_times") or {}).get(self.virtual_time_from_step_id)
        if recorded is not None:
            virtual_time, rate = recorded["virtual_time"], recorded["virtual_seconds_per_real_second"]
        elif self.virtual_time is not None:
            if self.virtual_time_from_step_id is not None:
                logger.warning(f"sync_env_clock '{self.id}': step '{self.virtual_time_from_step_id}' recorded no "
                               f"virtual time; arming at the fallback virtual_time {self.virtual_time}")
            virtual_time, rate = self.virtual_time, 1.0
        else:
            raise RuntimeError(
                f"sync_env_clock '{self.id}': step '{self.virtual_time_from_step_id}' recorded no virtual time "
                "(it did not run, or the env's clock was not armed when it snapshotted) and no virtual_time "
                "fallback is set"
            )
        if self.virtual_seconds_per_real_second is not None:
            rate = self.virtual_seconds_per_real_second
        return virtual_time, rate

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import gateway_url_of

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        virtual_time, rate = self._resolve_time(context)
        try:
            await deployed.invoke(EXT_CLOCK_URI, "set_time",
                                  {"virtual_time": virtual_time, "virtual_seconds_per_real_second": rate},
                                  timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"clock arm failed (HTTP {e.response.status_code}): {e.response.text}") from e
        state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=self.timeout_seconds)
        env_get_time_url = state.get("env_get_time_url")

        synced, skipped = [], []
        for service in self._service_names(deployed):
            outcome = await self._sync_one(deployed, gateway_url_of(deployed), service, env_get_time_url)
            (synced if outcome.get("synced") else skipped).append({"service": service, "environment": service, **outcome})

        context.metadata.setdefault("clock_configurations", []).append({
            "step_id": self.id, "env_id": self.env_id,
            "virtual_time": virtual_time, "virtual_seconds_per_real_second": rate,
            "env_get_time_url": env_get_time_url, "synced": synced, "skipped": skipped,
        })
        logger.info(f"sync_env_clock env={self.env_id}: armed at {virtual_time} rate={rate}; "
                    f"synced={[s['service'] for s in synced]} skipped={[s['service'] for s in skipped]}")
        if not synced:
            logger.warning(f"sync_env_clock env={self.env_id}: clock armed but NO server synced "
                           f"({[(s['service'], s.get('reason')) for s in skipped]}) — every tool still answers wall time")
        return context

    async def _sync_one(self, deployed, gateway_url: str, service: str, env_get_time_url: Optional[str]) -> dict:
        """Invoke sync_time on one server if its card advertises clock/v1; skip (or raise) per the flag."""
        from agent_env.env import legacy_protocol

        base_url, card = await legacy_protocol.child_env_card(deployed, gateway_url, service, timeout=self.timeout_seconds)
        if card is None:
            if self.tolerate_missing_sync_time:
                return {"synced": False, "reason": "no_env_card"}
            raise RuntimeError(f"{service} has no env card: it serves none, its card isn't named {service!r}, or the deploy couldn't read it")
        if protocol_v1.find_extension(card, EXT_CLOCK_URI) is None:
            if self.tolerate_missing_sync_time:
                logger.info(f"sync_env_clock: {service} does not advertise {EXT_CLOCK_URI}; skipping")
                return {"synced": False, "reason": "not_advertised"}
            raise RuntimeError(f"{service} does not advertise {EXT_CLOCK_URI}")
        if not env_get_time_url:
            raise RuntimeError(f"gateway did not expose env_get_time_url; cannot sync {service}")
        result = await protocol_v1.invoke_extension(base_url, card, EXT_CLOCK_URI,
                                                    params={"env_get_time_url": env_get_time_url},
                                                    timeout=self.timeout_seconds)
        return {"synced": True, "result": result}
