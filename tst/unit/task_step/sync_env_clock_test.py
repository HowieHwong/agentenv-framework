"""Unit tests for SyncEnvClockTaskStep — round-trip + the per-server sync_time fan-out (mocked)."""
import json
import logging

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import sync_env_clock as mod
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

T0 = "2026-01-01T00:00:00Z"
CLOCK_URL = "http://gateway:9000/clock/time"


def _step(**kw):
    return SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time=T0, depends_on=[], **kw)


def test_round_trip():
    s = _step(virtual_seconds_per_real_second=3600, tolerate_missing_sync_time=True)
    d = s.to_dict()
    s2 = SyncEnvClockTaskStep.from_dict(d)
    assert d["type"] == "sync_env_clock"
    assert s2.virtual_time == T0 and s2.virtual_seconds_per_real_second == 3600 and s2.tolerate_missing_sync_time is True


def test_existing_positional_arguments_keep_their_meaning():
    s = SyncEnvClockTaskStep("sync", 1, "e", T0, 2.0, False, [], False, 45)
    assert (s.virtual_seconds_per_real_second, s.tolerate_missing_sync_time, s.depends_on) == (2.0, False, [])
    assert (s.fail_task_on_error, s.timeout_seconds, s.virtual_time_from_step_id) == (False, 45, None)


def test_defaults():
    s = SyncEnvClockTaskStep(id="s", version=1, env_id="e", virtual_time=T0)
    assert s.virtual_seconds_per_real_second == 1.0 and s.tolerate_missing_sync_time is True


_LEGACY_KEYS = {"env_id", "virtual_time", "virtual_seconds_per_real_second", "tolerate_missing_sync_time", "timeout_seconds"}


def test_a_step_without_virtual_time_from_step_id_serializes_as_before():
    d = _step().to_dict()
    assert _LEGACY_KEYS <= d.keys() and "virtual_time_from_step_id" not in d
    assert d["virtual_seconds_per_real_second"] == 1.0
    legacy = {k: v for k, v in d.items() if k != "virtual_seconds_per_real_second"}
    assert SyncEnvClockTaskStep.from_dict(legacy).virtual_seconds_per_real_second == 1.0


def test_round_trip_with_virtual_time_from_step_id():
    s = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time_from_step_id="snap")
    d = s.to_dict()
    s2 = SyncEnvClockTaskStep.from_dict(d)
    assert d["virtual_time_from_step_id"] == "snap"
    assert s2.virtual_time_from_step_id == "snap" and s2.virtual_time is None and s2.virtual_seconds_per_real_second is None
    assert s2.to_dict() == d


def test_virtual_time_is_required_without_virtual_time_from_step_id():
    with pytest.raises(ValueError, match="omit it only with virtual_time_from_step_id"):
        SyncEnvClockTaskStep(id="s", version=1, env_id="e")


_RECORDED = {"env_snapshot_virtual_times": {"snap": {"virtual_time": "2026-01-01T05:30:00Z", "virtual_seconds_per_real_second": 60.0}}}


@pytest.mark.parametrize("kw, metadata, armed", [
    ({"virtual_time_from_step_id": "snap"}, _RECORDED, ("2026-01-01T05:30:00Z", 60.0)),
    ({"virtual_time_from_step_id": "snap", "virtual_seconds_per_real_second": 2.0}, _RECORDED, ("2026-01-01T05:30:00Z", 2.0)),
    ({"virtual_time_from_step_id": "snap", "virtual_time": T0}, _RECORDED, ("2026-01-01T05:30:00Z", 60.0)),
    ({"virtual_time_from_step_id": "snap", "virtual_time": T0}, {}, (T0, 1.0)),
    ({"virtual_time_from_step_id": "other", "virtual_time": T0, "virtual_seconds_per_real_second": 3.0}, _RECORDED, (T0, 3.0)),
], ids=["recorded", "explicit-rate-wins", "recorded-over-fallback", "fallback", "fallback-other-step"])
@pytest.mark.asyncio
async def test_execute_arms_at_the_recorded_time_else_the_fallback(monkeypatch, kw, metadata, armed):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr(SyncEnvClockTaskStep, "_service_names", lambda self, deployed: [])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({}))], metadata=dict(metadata))

    await SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", **kw).execute(context)

    virtual_time, rate = armed
    assert json.loads(sent[0].content) == {"virtual_time": virtual_time, "virtual_seconds_per_real_second": rate}
    entry = context.metadata["clock_configurations"][-1]
    assert (entry["virtual_time"], entry["virtual_seconds_per_real_second"]) == armed


@pytest.mark.asyncio
async def test_execute_warns_when_it_arms_at_the_fallback(monkeypatch, caplog):
    _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr(SyncEnvClockTaskStep, "_service_names", lambda self, deployed: [])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({}))])
    step = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time=T0, virtual_time_from_step_id="snap")

    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        await step.execute(context)

    assert any("step 'snap' recorded no virtual time" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)


def _clock_from(snap_id, depends_on=None):
    return SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time_from_step_id=snap_id,
                                depends_on=depends_on)


def _snap():
    return SyncEnvClockTaskStep(id="snap", version=1, env_id="e1", virtual_time=T0)


@pytest.mark.parametrize("steps", [
    lambda: [_snap(), _clock_from("snap")],
    lambda: [_snap(), _clock_from("snap", depends_on=["snap"])],
    lambda: [_snap(), SyncEnvClockTaskStep(id="mid", version=1, env_id="e1", virtual_time=T0, depends_on=["snap"]),
             _clock_from("snap", depends_on=["mid"])],
    lambda: [_clock_from("agent-run-snap", depends_on=[])],
], ids=["default-order", "direct-dependency", "transitive-dependency", "step-of-the-seeded-run"])
def test_a_task_accepts_a_recording_step_that_runs_first_or_is_outside_it(steps):
    Task(id="t", version=1, steps=steps())


@pytest.mark.parametrize("steps", [
    lambda: [_snap(), _clock_from("snap", depends_on=[])],
    lambda: [_clock_from("snap", depends_on=[]), _snap()],
], ids=["not-a-dependency", "after-it"])
def test_a_task_refuses_a_recording_step_that_could_run_after_it(steps):
    with pytest.raises(ValueError, match="reads the output of step 'snap', which is not in its dependency ancestry"):
        Task(id="t", version=1, steps=steps())


@pytest.mark.asyncio
async def test_execute_raises_before_any_request_with_no_record_and_no_fallback(monkeypatch):
    sent = _mock_http(monkeypatch)
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({}))])
    step = SyncEnvClockTaskStep(id="sync", version=1, env_id="e1", virtual_time_from_step_id="snap")
    with pytest.raises(RuntimeError, match="step 'snap' recorded no virtual time .* no virtual_time fallback"):
        await step.execute(context)
    assert sent == []


@pytest.mark.asyncio
async def test_sync_one_skips_unadvertised(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    out = await _step()._sync_one(None, "http://gw", "email", CLOCK_URL)
    assert out == {"synced": False, "reason": "not_advertised"}


@pytest.mark.asyncio
async def test_sync_one_invokes_advertised(monkeypatch):
    card = {"capabilities": {"extensions": [
        {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time",
                                          "methods": {"sync_time": {"method": "POST"}}}}]}}
    calls = {}

    async def fake_card(base_url, **kw):
        return card

    async def fake_invoke(base_url, c, uri, params=None, **kw):
        calls.update(base_url=base_url, uri=uri, params=params)
        return {"ok": True}

    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    monkeypatch.setattr(mod.protocol_v1, "invoke_extension", fake_invoke)
    out = await _step()._sync_one(None, "http://gw", "email", CLOCK_URL)
    assert out["synced"] is True
    assert calls["uri"] == EXT_CLOCK_URI and calls["params"] == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_unadvertised_strict_raises(monkeypatch):
    async def fake_card(base_url, **kw):
        return {"capabilities": {"extensions": []}}
    monkeypatch.setattr(mod.protocol_v1, "get_card", fake_card)
    with pytest.raises(RuntimeError):
        await _step(tolerate_missing_sync_time=False)._sync_one(None, "http://gw", "email", CLOCK_URL)


@pytest.mark.asyncio
async def test_sync_one_invokes_at_the_stored_child_card_with_no_card_read(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))
    out = await _step()._sync_one(record, "http://gw", "email", CLOCK_URL)

    assert out["synced"] is True
    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time")]
    assert json.loads(sent[0].content) == {"env_get_time_url": CLOCK_URL}


@pytest.mark.asyncio
async def test_sync_one_skips_or_raises_for_a_child_env_missing_from_the_stored_card(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded(await _composed_card({"mcp-slack": [_CLOCK_EXT]}))

    assert await _step()._sync_one(record, "http://gw", "email", CLOCK_URL) == {"synced": False, "reason": "no_env_card"}
    with pytest.raises(RuntimeError, match="email has no env card"):
        await _step(tolerate_missing_sync_time=False)._sync_one(record, "http://gw", "email", CLOCK_URL)
    assert sent == []


@pytest.mark.asyncio
async def test_execute_syncs_each_child_env_through_the_stored_card(monkeypatch):
    sent = _mock_http(monkeypatch, state={"env_get_time_url": CLOCK_URL})
    monkeypatch.setattr(SyncEnvClockTaskStep, "_service_names", lambda self, deployed: ["email"])
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))])

    await _step().execute(context)

    assert [(r.method, str(r.url)) for r in sent] == [
        ("PUT", f"{_CARD}/clock/set-time"), ("GET", f"{_CARD}/clock/state"),
        ("POST", f"{_CARD}/svc/mcp-email/agentenv/ext/sync_time"),
    ]
    assert json.loads(sent[0].content) == {"virtual_time": T0, "virtual_seconds_per_real_second": 1.0}
    assert [s["service"] for s in context.metadata["clock_configurations"][0]["synced"]] == ["email"]


@pytest.mark.asyncio
async def test_execute_raises_before_any_request_when_the_card_lacks_set_time(monkeypatch):
    sent = _mock_http(monkeypatch)
    record = _carded({"name": "gw", "capabilities": {"extensions": []}, "children_environments": []})
    with pytest.raises(EnvCapabilityUnsupported, match="does not offer 'set_time' on urn:agentenv:clock/v1"):
        await _step().execute(TaskStepContext(deployed_envs=[record]))
    assert sent == []


@pytest.mark.asyncio
async def test_execute_keeps_the_arm_failure_message(monkeypatch):
    _mock_http(monkeypatch, set_time_status=409)
    context = TaskStepContext(deployed_envs=[_carded(await _composed_card({"mcp-email": [_CLOCK_EXT]}))])
    with pytest.raises(RuntimeError, match=r"clock arm failed \(HTTP 409\): "):
        await _step().execute(context)


_CLOCK_EXT = {"uri": EXT_CLOCK_URI, "params": {"endpoint": "/agentenv/ext/sync_time", "methods": {"sync_time": {"method": "POST"}}}}
# The card's address differs from the gateway URL, so a call built from the stored card is told apart from one built from /svc/... itself.
_CARD = "https://sandbox.example/sb-1"


async def _composed_card(children: dict[str, list]) -> dict:
    """The real gateway's composed card over child envs at the given gateway keys, each advertising the given extensions."""
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in children})

    async def fetch(client, key, base_url):
        card = {"name": key.removeprefix("mcp-"), "url": "/agentenv", "capabilities": {"extensions": children[key]}}
        return gw._rewrite_child_card(key, card)

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _carded(card: dict) -> DeployedEnv:
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    return DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None,
                       sandbox_id="sb-1", environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card)


def _mock_http(monkeypatch, state: dict | None = None, set_time_status: int = 200):
    """Record every request: `/clock/state` answers `state`, `/clock/set-time` `set_time_status`, anything else an empty object."""
    import httpx

    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/clock/set-time"):
            return httpx.Response(set_time_status, json={})
        return httpx.Response(200, json=state if request.url.path.endswith("/clock/state") else {})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
