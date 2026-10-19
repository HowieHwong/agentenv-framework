"""The stand-in agent as AgentEnv deploys it: served from ``/app`` in its image, or installed on a VM host as
``install_agent`` installs it, in the place of the agent it replaces (its ``{workdir}``, as its ``{user}``).
``AGENT_CARD`` is what ``install_agent`` reads; importing this module serves nothing. ``install_agent`` formats each
command with ``str.format``, so a literal brace is doubled."""

from agentenv_protocol.a2a_agent import INSTALL_V1, a2a_agent, enable
from agentenv_protocol.a2a_agent.stand_in import EXTENSIONS, IDENTITY, StandInAgent, StandInConfig

HOME = "/opt/agentenv-stand-in-{agent_name}"
INSTALL_COMMANDS_HOST = [
    "bash -c 'set -e; python3 -c \"import ensurepip, venv\" 2>/dev/null || {{ export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq; apt-get install -y -qq --no-install-recommends python3 python3-venv; }}'",
    f"rm -rf {HOME} && mkdir -p {HOME} && tar -xzf {{agent_ctx_tar}} -C {HOME}",
    f"python3 -m venv {HOME}/venv && {HOME}/venv/bin/pip install -q -r {HOME}/requirements.txt",
    "id -u {user} >/dev/null 2>&1 || useradd -m {user}; mkdir -p {workdir} && chown -R {user}: {workdir}",
    f"nohup runuser -u {{user}} -- env -C {{workdir}} A2A_PORT={{a2a_port}} PYTHONPATH={HOME} {HOME}/venv/bin/python "
    f"{HOME}/a2a_server.py > /tmp/agentenv-stand-in-{{agent_name}}.log 2>&1 < /dev/null &",
]


@a2a_agent(
    identity=IDENTITY,
    config=StandInConfig,
    extensions=(
        *EXTENSIONS,
        enable(INSTALL_V1, install_commands_host=INSTALL_COMMANDS_HOST,
               required_params_host=["agent_ctx_tar", "a2a_port", "agent_name", "workdir", "user"], a2a_port=8000),
    ),
)
class InstalledStandIn(StandInAgent):
    pass


AGENT_CARD = InstalledStandIn().create_app().state.agentenv_a2a.card

if __name__ == "__main__":
    InstalledStandIn().serve()
