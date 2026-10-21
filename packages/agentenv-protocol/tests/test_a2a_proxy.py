from __future__ import annotations

import subprocess
import time
from typing import Any

import pytest
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import stand_in
from agentenv_protocol.a2a_agent.stand_in import StandInAgent

TOKEN = "t" * 32


def _send(client: TestClient, text: str, *, context_id: str = "context-1") -> str:
    response = client.post("/a2a", json={
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {
            "message": {"kind": "message", "messageId": f"message-{text}", "role": "user", "contextId": context_id,
                        "parts": [{"kind": "text", "text": text}]},
            "configuration": {"blocking": False},
        },
    })
    return response.json()["result"]["id"]


def _task(client: TestClient, task_id: str) -> dict[str, Any]:
    return client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get",
                                     "params": {"id": task_id}}).json()["result"]


def _settled(client: TestClient, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while (task := _task(client, task_id))["status"]["state"] not in ("completed", "failed", "canceled"):
        assert time.monotonic() < deadline, task
        time.sleep(0.02)
    return task


def _driver(client: TestClient, operation: str, **body: Any):
    return client.post(f"/ext/driver/{operation}", json={"token": TOKEN, **body})


def _turns(client: TestClient, after: int = 0) -> dict[str, Any]:
    response = _driver(client, "next", after=after)
    assert response.status_code == 200, response.text
    return response.json()


def _ran(client: TestClient, command: str) -> str:
    exec_id = _driver(client, "exec", command=command).json()["exec_id"]
    while not (result := _driver(client, "output", exec_id=exec_id).json())["done"]:
        pass
    return result["output"]


def _stopped(pid: int) -> None:
    """Until ``pid`` is gone, or a zombie: a killed child whose parent has gone stays one until what adopted it reaps it."""
    deadline = time.monotonic() + 10
    while (state := _state(pid)) and not state.startswith("Z"):
        assert time.monotonic() < deadline, state
        time.sleep(0.02)


def _state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(stand_in, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.chdir(tmp_path)
    with TestClient(StandInAgent(long_poll_seconds=0.2).create_app()) as client:
        assert client.post("/ext/driver/claim", json={"token": TOKEN}).status_code == 200
        yield client


def test_the_card_advertises_the_driver_and_takes_a_system_prompt():
    card = StandInAgent().create_app().state.agentenv_a2a.card.model_dump(mode="json")
    extensions = {extension["uri"]: extension for extension in card["capabilities"]["extensions"]}

    assert sorted(extensions["urn:agentenv:driver/v1"]["params"]["methods"]) == [
        "claim", "exec", "next", "output", "reply"]
    assert "system_prompt" in extensions["urn:agentenv:agent-config/v1"]["params"]["methods"]["set"]["request"][
        "supported"]


def test_the_first_claim_sets_the_token_that_every_driver_call_needs():
    with TestClient(StandInAgent(long_poll_seconds=0.2).create_app()) as client:
        assert _driver(client, "next").status_code == 401
        assert client.post("/ext/driver/claim", json={"token": "short"}).status_code == 400
        assert client.post("/ext/driver/claim", json={"token": TOKEN}).status_code == 200
        assert client.post("/ext/driver/claim", json={"token": TOKEN}).status_code == 200
        assert client.post("/ext/driver/claim", json={"token": "u" * 32}).status_code == 409
        assert client.post("/ext/driver/next", json={"token": "u" * 32}).status_code == 401
        assert _driver(client, "next").status_code == 200


def test_a_message_is_an_open_turn_until_the_driver_replies(client):
    client.post("/ext/agent-config", json={"system_prompt": "be brief"})
    task_id = _send(client, "fix the build")

    seen = _turns(client)
    assert seen["turns"] == [{"task_id": task_id, "context_id": "context-1", "text": "fix the build",
                              "system_prompt": "be brief", "skills": []}]
    span = {"name": "bash", "attributes": {"gen_ai.operation.name": "execute_tool"}}
    assert _driver(client, "reply", task_id=task_id, text="fixed", trajectory=[span], tool_call_count=1).status_code \
        == 200

    task = _settled(client, task_id)
    assert task["status"]["state"] == "completed"
    parts = task["status"]["message"]["parts"]
    assert parts[0] == {"kind": "text", "text": "fixed"}
    assert parts[1]["data"]["usage"]["tool_call_count"] == 1
    assert client.post("/ext/trajectory", json={"task_id": task_id}).json()["trajectory"] == [span]
    assert _turns(client, after=seen["version"])["turns"] == []


def test_an_error_reply_fails_the_turn(client):
    task_id = _send(client, "do it")

    _driver(client, "reply", task_id=task_id, error="the driver gave up")

    task = _settled(client, task_id)
    assert task["status"]["state"] == "failed"
    assert "the driver gave up" in task["status"]["message"]["parts"][0]["text"]


def test_a_reply_names_exactly_one_open_turn(client):
    task_id = _send(client, "do it")

    assert _driver(client, "reply", task_id=task_id).status_code == 400
    assert _driver(client, "reply", task_id="nope", text="done").status_code == 404
    _driver(client, "reply", task_id=task_id, text="done")
    _settled(client, task_id)
    assert _driver(client, "reply", task_id=task_id, text="again").status_code == 404


def test_a_turn_its_sender_cancels_is_withdrawn(client):
    task_id = _send(client, "do it")
    assert [turn["task_id"] for turn in _turns(client)["turns"]] == [task_id]

    client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/cancel", "params": {"id": task_id}})

    assert _settled(client, task_id)["status"]["state"] == "canceled"
    assert _turns(client)["turns"] == []


def test_next_waits_for_a_change_after_the_version_it_was_given(client):
    version = _turns(client)["version"]

    started = time.monotonic()
    unchanged = _turns(client, after=version)

    assert unchanged["version"] == version
    assert time.monotonic() - started >= 0.2


def test_exec_returns_stdout_then_stderr_from_the_folder_it_works_in(client, tmp_path):
    (tmp_path / "inputs").mkdir()

    assert _ran(client, "ls; echo out; echo err >&2; exit 3") == "inputs\nout\nerr\n"


def test_a_command_that_outlives_its_timeout_is_stopped_with_what_it_started(client, monkeypatch, tmp_path):
    monkeypatch.setattr(stand_in, "COMMAND_TIMEOUT_SECONDS", 0.5)

    output = _ran(client, "sleep 60 & echo $! > child; wait")

    assert output == "error: the command timed out after 0.5s"
    _stopped(int((tmp_path / "child").read_text()))


def test_a_child_left_holding_the_output_is_stopped_when_the_command_times_out(client, monkeypatch, tmp_path):
    monkeypatch.setattr(stand_in, "COMMAND_TIMEOUT_SECONDS", 0.5)

    output = _ran(client, "sleep 60 & echo $! > child")

    assert output == "error: the command timed out after 0.5s"
    _stopped(int((tmp_path / "child").read_text()))


def test_output_names_a_command_it_started(client):
    assert _driver(client, "output", exec_id="nope").status_code == 404


def test_a_skill_is_written_where_its_turns_say_it_is(client, tmp_path):
    response = client.post("/ext/skill-config", json={"name": "deploy", "description": "How to deploy.",
                                                      "skill_md": "# Deploy\n"})
    assert response.status_code == 200, response.text
    _send(client, "go")

    skill = _turns(client)["turns"][0]["skills"]
    assert skill == [{"name": "deploy", "description": "How to deploy.", "path": str(tmp_path / "skills" / "deploy")}]
    assert (tmp_path / "skills" / "deploy" / "SKILL.md").read_text() == "# Deploy\n"


def test_a_stand_in_without_a_shell_keeps_no_skills_and_runs_no_commands():
    with TestClient(StandInAgent(shell=False, long_poll_seconds=0.2).create_app()) as client:
        client.post("/ext/driver/claim", json={"token": TOKEN})

        assert _driver(client, "exec", command="true").status_code == 409
        assert client.post("/ext/skill-config", json={"name": "x", "description": "d", "skill_md": "#"}).status_code \
            == 409
