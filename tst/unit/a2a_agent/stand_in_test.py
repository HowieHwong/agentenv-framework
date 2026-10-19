"""The stand-in agent the Episode API deploys in place of an agent played from outside: written to the store as an
image's build context, and written again only when that context changes."""

from agent_env.a2a_agent import stand_in
from agent_env.a2a_agent.a2a_agent import SHARE_STORE_CREDENTIALS
from agent_env.a2a_agent.stand_in import STAND_IN_AGENT_ID, stand_in_agent


def test_the_stand_in_is_a_build_context_given_no_model_key_or_store_credentials(local_stores, cli_routing):
    agent = stand_in_agent()

    assert agent.id == STAND_IN_AGENT_ID
    assert agent.docker_image_artifact.context_only
    assert agent.default_env_vars == {"LITELLM_API_KEY": "unused", "LITELLM_BASE_URL": "http://unused.invalid"}
    assert agent.metadata == {SHARE_STORE_CREDENTIALS: False}


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
    requirements = (context / "requirements.txt").read_text().splitlines()
    assert {line.split("==")[0] for line in requirements} == {
        "a2a-sdk[http-server]", "httpx", "pydantic", "python-multipart", "regex", "starlette", "uvicorn"}
    assert all(line.split("==")[1] for line in requirements)
