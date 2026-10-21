"""Serves the stand-in agent. ``AGENT_CARD`` is what agent-env's install/v1 reads. ``install_agent`` formats each
command with ``str.format``."""

from agentenv_protocol.a2a_agent import INSTALL_V1, a2a_agent, enable
from agentenv_protocol.a2a_agent.stand_in import EXTENSIONS, IDENTITY, StandInAgent, StandInConfig

_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
# In the place of the agent it replaces: its {workdir}, as its {user}.
_INSTALL_COMMANDS_HOST = [_UNPACK, "sh {work_dir}/agent/install.sh {agent_name} {a2a_port} {workdir} {user}"]


@a2a_agent(
    identity=IDENTITY,
    config=StandInConfig,
    extensions=(
        *EXTENSIONS,
        enable(INSTALL_V1, install_commands_host=_INSTALL_COMMANDS_HOST,
               required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", "workdir", "user"],
               a2a_port=8000),
    ),
)
class InstalledStandIn(StandInAgent):
    pass


agent = InstalledStandIn()
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
