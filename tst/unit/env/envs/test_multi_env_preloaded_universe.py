"""A deployment handed over with its universe already loaded skips the data load.

A provider can deliver an env whose universe was loaded before anyone asked for it -- a
warm pool, say. Re-ingesting it would cost the whole load the provider exists to save.
Resume never skips a complete load, because it can't tell a caller's explicit load from a
retry; here the deployment's own record guarantees the data is the universe asked for.

The safety property is the same changelog gate resume uses: a guarantee about data that
has since been touched is void, and the universe loads as usual.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers import EnvironmentGatewayProvider
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER

SERVICES = ["a", "b", "gmail"]


def _universe(version: int = 39, metadata: dict | None = None):
    ua = MagicMock()
    ua.id, ua.version = "uni", version
    ua.get_metadata.return_value = metadata or {}
    arts = []
    for name in SERVICES:
        a = MagicMock()
        a.environment_name = name
        arts.append(a)
    ua.get_environment_artifacts.return_value = arts
    return ua


def _env(*, preloaded: dict | None, mode: str = "vm"):
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    env._instance_id = "inst-1"
    env._deployed = DeployedGatewayEnv(
        env_id="env-1", env_version=1, instance_id="inst-1", sandbox_id="sb-1",
        gateway_url="https://gw.example", preloaded_universe=preloaded,
    )
    gp = EnvironmentGatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()
    env._env_provider = gp
    sandbox = MagicMock()
    sandbox.mode = mode
    sandbox.exec_script = AsyncMock(return_value="8\n")
    sandbox.load_object_file = AsyncMock()
    env._sandbox = sandbox
    env._load_from_snapshot = AsyncMock()
    env._bake_snapshot_after_load = AsyncMock(return_value=(True, None))
    return env


async def _load(env, universe, *, changelog_empty=True, changelog_error=None):
    """Drive a load with stubbed stores and changelog; returns what it touched."""
    loaded: list[str] = []
    env.load_environment_artifact = AsyncMock(side_effect=lambda a: loaded.append(a.environment_name))
    instances = MagicMock()
    instances.get_loaded_environments.return_value = []
    snapshots = MagicMock()
    snapshots.get_clean.return_value = None
    stamp = MagicMock()
    changelog = AsyncMock(side_effect=changelog_error) if changelog_error else AsyncMock(return_value=changelog_empty)
    with patch("agent_env.env.store.get_env_instance_store", return_value=instances), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snapshots), \
         patch("agent_env.env.snapshot_store._check_changelog_empty", changelog), \
         patch("agent_env.env.store.update_env_instance_environment_universe", stamp):
        result = await env.load_environment_universe_artifact(universe, snapshot_after_load=True)
    return MagicMock(loaded=loaded, snapshots=snapshots, stamp=stamp, result=result, instances=instances)


@pytest.mark.asyncio
async def test_a_deployment_delivered_with_the_universe_loads_no_service():
    """The saving: the whole re-ingest the provider loaded ahead of time."""
    run = await _load(_env(preloaded={"id": "uni", "version": 39}), _universe())
    assert run.loaded == []


@pytest.mark.asyncio
async def test_the_universe_is_still_recorded_on_the_instance():
    """Readers of the instance record ask what it holds, however it came to hold it."""
    run = await _load(_env(preloaded={"id": "uni", "version": 39}), _universe())
    run.stamp.assert_called_once_with("inst-1", "uni", 39)


@pytest.mark.asyncio
async def test_metadata_files_are_still_staged():
    """Later steps read the paths this returns; they live under a fresh directory per load,
    so the provider's earlier load can't have produced them."""
    file_artifact = MagicMock(filename="universe.yaml", object_url="s3://b/universe.yaml")
    env = _env(preloaded={"id": "uni", "version": 39})
    run = await _load(env, _universe(metadata={"universe": file_artifact}))
    env._sandbox.load_object_file.assert_awaited_once()
    assert run.result.metadata_filepaths["universe"].endswith("/universe.yaml")


@pytest.mark.asyncio
async def test_a_preloaded_env_neither_restores_nor_bakes_a_snapshot():
    env = _env(preloaded={"id": "uni", "version": 39})
    run = await _load(env, _universe())
    run.snapshots.get_clean.assert_not_called()
    env._load_from_snapshot.assert_not_awaited()
    env._bake_snapshot_after_load.assert_not_awaited()
    assert run.result.restored_from_snapshot is False


@pytest.mark.asyncio
@pytest.mark.parametrize("preloaded", [
    None,
    {"id": "other", "version": 39},
    {"id": "uni", "version": 38},
    {},
])
async def test_a_deployment_not_delivered_with_this_universe_loads_it(preloaded):
    """Another universe, or another version of this one, is not this universe."""
    run = await _load(_env(preloaded=preloaded), _universe())
    assert sorted(run.loaded) == sorted(SERVICES)


@pytest.mark.asyncio
async def test_data_touched_since_delivery_voids_the_guarantee():
    """THE safety property: loading a universe is documented as destructive. If anything has
    changed the data since the provider loaded it, every service is re-seeded."""
    run = await _load(_env(preloaded={"id": "uni", "version": 39}), _universe(), changelog_empty=False)
    assert sorted(run.loaded) == sorted(SERVICES)


@pytest.mark.asyncio
async def test_an_unreadable_changelog_loads_rather_than_assumes():
    run = await _load(_env(preloaded={"id": "uni", "version": 39}), _universe(),
                      changelog_error=RuntimeError("exec failed"))
    assert sorted(run.loaded) == sorted(SERVICES)


@pytest.mark.asyncio
async def test_container_mode_loads_as_usual():
    """The changelog gate reads the VM's database; a container-mode env has none to read."""
    run = await _load(_env(preloaded={"id": "uni", "version": 39}, mode=SANDBOX_MODE_CONTAINER), _universe())
    assert sorted(run.loaded) == sorted(SERVICES)


@pytest.mark.asyncio
async def test_without_the_guarantee_a_snapshot_is_still_baked():
    """The bake is skipped only because nothing was ingested, not for every load."""
    env = _env(preloaded=None)
    await _load(env, _universe())
    env._bake_snapshot_after_load.assert_awaited_once()


def test_the_guarantee_survives_the_instance_record():
    """The instance store registers a record from its dict form and loads it back, which is
    how the env that runs the load comes to hold it."""
    record = DeployedGatewayEnv(
        env_id="env-1", env_version=1, sandbox_id="sb-1", gateway_url="https://gw.example",
        preloaded_universe={"id": "uni", "version": 39},
    )
    back = DeployedEnv.from_dict(dataclasses.asdict(record))
    assert type(back) is DeployedGatewayEnv
    assert back.preloaded_universe == {"id": "uni", "version": 39}


def _universe_named(name: str, version: int = 39):
    ua = _universe(version)
    ua.id = name
    return ua


@pytest.mark.asyncio
async def test_loading_another_universe_voids_the_guarantee():
    """A load reinstalls each service's changelog triggers, which empties the changelog. A
    guarantee for A surviving a load of B would pass that gate on the next request for A,
    and skip a load the data needs."""
    env = _env(preloaded={"id": "uni", "version": 39})
    await _load(env, _universe_named("other"))

    run = await _load(env, _universe_named("uni"))

    assert sorted(run.loaded) == sorted(SERVICES)


@pytest.mark.asyncio
async def test_the_void_reaches_the_stored_record():
    """So a reattach, which rebuilds the deployment from the record, can't revive it."""
    env = _env(preloaded={"id": "uni", "version": 39})

    run = await _load(env, _universe_named("other"))

    run.instances.clear_preloaded_universe.assert_called_once_with("inst-1")
    assert env._deployed.preloaded_universe is None


@pytest.mark.asyncio
async def test_a_skipped_load_keeps_the_guarantee():
    """Nothing replaced the data, so asking for the same universe again skips again."""
    env = _env(preloaded={"id": "uni", "version": 39})

    run = await _load(env, _universe())

    run.instances.clear_preloaded_universe.assert_not_called()
    assert env._deployed.preloaded_universe == {"id": "uni", "version": 39}
    assert (await _load(env, _universe())).loaded == []


@pytest.mark.asyncio
async def test_a_deployment_with_no_guarantee_writes_nothing():
    run = await _load(_env(preloaded=None), _universe())

    run.instances.clear_preloaded_universe.assert_not_called()
