"""The agents bundle's stand-in: its card, how it installs on a VM's host, and how the episode writes it."""

from __future__ import annotations

import re
import subprocess

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.bundle.installed import write_installed
from agent_env.task.episode.steps import STAND_IN_AGENT_ID
from agent_env.task_step.task_steps.install_agent import _build_param_resolvers

from tst.unit.agents.stand_in.agent import AGENT_DIR, a2a_server


def _extensions() -> dict:
    card = a2a_server.AGENT_CARD.model_dump(mode="json", exclude_none=True, by_alias=True)
    return {ext["uri"]: ext for ext in card["capabilities"]["extensions"]}


def test_the_card_offers_the_driver_and_a_host_install_only():
    extensions = _extensions()
    assert {"urn:agentenv:driver/v1", "urn:agentenv:mcp-config/v1", "urn:agentenv:trajectory/v1",
            "urn:agentenv:install/v1"} <= set(extensions)
    params = extensions["urn:agentenv:install/v1"]["params"]
    assert params["install_commands_host"] and not params.get("install_commands")


def test_the_host_install_asks_only_for_params_install_agent_supplies_and_runs_the_folders_script():
    params = _extensions()["urn:agentenv:install/v1"]["params"]
    supplied = set(_build_param_resolvers(container_name=None, work_dir="/w", agent_ctx_tar="/w/a.tgz", a2a_port=8000,
                                          config=None, agent_name="a"))
    placeholders = {name for command in params["install_commands_host"] for name in re.findall(r"\{(\w+)\}", command)}
    assert placeholders == set(params["required_params_host"]) and placeholders <= supplied
    assert params["install_commands_host"][-1] == "sh {work_dir}/agent/install.sh {agent_name} {a2a_port} {workdir} {user}"


def test_the_install_script_parses_and_the_image_keeps_it_in_its_build_context():
    subprocess.run(["sh", "-n", str(AGENT_DIR / "install.sh")], check=True)
    assert "COPY a2a_server.py install.sh ./" in (AGENT_DIR / "Dockerfile").read_text()


def test_the_image_works_in_app_as_agent():
    lines = (AGENT_DIR / "Dockerfile").read_text().splitlines()
    last = {key: next(line.split(maxsplit=1)[1] for line in reversed(lines) if line.startswith(f"{key} "))
            for key in ("USER", "WORKDIR")}
    assert last == {"USER": "agent", "WORKDIR": "/app"}


def test_the_episode_writes_it_from_the_bundle_taking_no_credentials(local_stores, cli_routing):
    version = write_installed(STAND_IN_AGENT_ID)
    agent = A2AAgent.get(STAND_IN_AGENT_ID)

    assert (agent.version, agent.metadata) == (version, {"takes_credentials": False})
    assert agent.docker_image_artifact.context_only
    assert write_installed(STAND_IN_AGENT_ID) == version
