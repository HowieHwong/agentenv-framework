"""The stand-in agent the Episode API deploys in place of an agent played from outside: written to the store as an
image's build context, and written again only when that context changes."""

from importlib.metadata import version

from agent_env.a2a_agent import stand_in
from agent_env.a2a_agent.a2a_agent import TAKES_CREDENTIALS, A2AAgent
from agent_env.a2a_agent.stand_in import STAND_IN_AGENT_ID, stand_in_agent


def test_the_stand_in_is_a_build_context_that_takes_no_credentials(local_stores, cli_routing):
    agent = stand_in_agent()

    assert agent.id == STAND_IN_AGENT_ID
    assert agent.docker_image_artifact.context_only
    assert (agent.default_env_vars, agent.metadata) == ({}, {TAKES_CREDENTIALS: False})


class _NoModelConfig:
    def __getattr__(self, name):
        raise AssertionError(f"an agent that takes no credentials asked config for {name}")


def test_an_agent_that_takes_no_credentials_gets_no_model_endpoint_key_or_headers_whoever_passes_them():
    agent = A2AAgent(id="a", version=1, docker_image_artifact=None, metadata={TAKES_CREDENTIALS: False})
    passed = {"LITELLM_API_KEY": "sk-from-the-step", "LITELLM_BASE_URL": "http://model", "LITELLM_EXTRA_HEADERS": "{}",
              "GITHUB_TOKEN": "the task's own"}

    assert agent._with_model_credentials(passed, _NoModelConfig()) == {"GITHUB_TOKEN": "the task's own"}


def test_an_unchanged_context_reuses_the_stand_in_and_a_changed_one_writes_it_again(
        local_stores, cli_routing, monkeypatch):
    first = stand_in_agent()
    assert stand_in_agent().version == first.version

    monkeypatch.setattr(stand_in, "_requirements", lambda: ["httpx==0.0.1"])

    assert stand_in_agent().version == first.version + 1


def test_its_context_holds_the_server_the_protocol_package_and_the_pinned_requirements_to_serve(tmp_path):
    context = stand_in._stage(tmp_path)

    assert sorted(path.name for path in context.iterdir()) == [
        "Dockerfile", "a2a_server.py", "agentenv_protocol", "requirements.txt"]
    assert (context / "agentenv_protocol" / "a2a_agent" / "stand_in.py").is_file()
    assert not list(context.rglob("__pycache__"))
    pinned = dict(line.split("==") for line in (context / "requirements.txt").read_text().splitlines())
    assert {"a2a-sdk[http-server]", "httpx", "pydantic", "starlette", "uvicorn"} <= pinned.keys()
    assert not pinned.keys() & {"fastmcp", "pytest", "pytest-asyncio"}
    assert all(pinned[name] == version(name.split("[")[0]) for name in pinned)
