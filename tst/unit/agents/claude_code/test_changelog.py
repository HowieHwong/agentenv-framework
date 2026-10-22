"""snapshot/v1's changelog: the capture hook's increments, their replay onto a fresh agent, and the agent that serves
it only where the host's tar is GNU tar."""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agentenv_protocol.a2a_agent import NamespaceChangelogEnableRequest, ObjectChangelogApplyRequest
from agentenv_protocol.transfers import TransferError
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from tst.unit.agents.claude_code.agent import a2a_server, tar_changelog
from tst.unit.agents.claude_code.support import expiry, extension_methods, read_grant

GNU_TAR = shutil.which("tar") is not None and tar_changelog.gnu_tar()
needs_gnu_tar = pytest.mark.skipif(not GNU_TAR, reason="the changelog is a GNU incremental tar")
STAGING_PATH = "/staging/changelog"
ROOT_PATH = "agent_changelog/run-1/solver"


def namespace_grant(**limits) -> dict:
    """A namespace grant on the agent's own staging, so its uploads reach a local server over loopback."""
    return {
        "root_path": ROOT_PATH,
        "expires_at": expiry(),
        "max_objects": limits.get("max_objects", 10),
        "max_object_bytes": limits.get("max_object_bytes", 1 << 20),
        "max_total_bytes": limits.get("max_total_bytes", 1 << 22),
        "write": {
            "kind": "http-post-policy",
            "url": f"https://bucket.example.com{STAGING_PATH}",
            "fields": {"policy": "p", "signature": "s"},
            "path_field": "key",
            "file_field": "file",
            "headers": {"AgentEnv-Staging-Path": STAGING_PATH},
        },
    }


class Staging:
    """A local server taking the namespace's multipart POSTs: each object's bytes, by the path it was posted to."""

    def __init__(self, status: int = 204) -> None:
        self.objects: dict[str, bytes] = {}
        self.status = status
        staging = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["content-length"]))
                message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    f"content-type: {self.headers['content-type']}\r\n\r\n".encode() + body)
                fields = {part.get_param("name", header="content-disposition"): part.get_content()
                          for part in message.iter_parts()}
                if staging.status < 300:
                    staging.objects[fields["key"]] = fields["file"]
                self.send_response(staging.status)
                self.send_header("content-length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server.server_port

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def staging(socket_enabled):
    """The hook uploads to a server on loopback, as it reaches the agent's own staging."""
    server = Staging()
    yield server
    server.close()


def _changelog(tmp_path, roots=None):
    installed: list[str] = []
    after: list[bool] = []
    resumed: list[str | None] = []

    async def resume(target_context_id):
        resumed.append(target_context_id)
        return target_context_id or "minted"

    log = tar_changelog.Changelog(
        default_roots=roots or [str(tmp_path / "default-root")],
        install_hook=installed.append,
        resume=resume,
        directory=tmp_path / "changelog",
        after_apply=lambda: after.append(True),
    )
    return log, installed, after, resumed


def _enable(log, roots=None, **limits):
    body = {"write_namespace": namespace_grant(**limits), **({"roots": roots} if roots is not None else {})}
    return asyncio.run(log.enable(NamespaceChangelogEnableRequest.model_validate(body)))


def _capture(log, port: int, tool: str = "Bash") -> subprocess.CompletedProcess:
    """One run of the post-tool hook, as the CLI starts it: the agent's environment, the hook's JSON on stdin."""
    return subprocess.run(
        [sys.executable, log.capture_path], input=json.dumps({"tool_name": tool}), capture_output=True, text=True,
        env={**os.environ, "A2A_PORT": str(port)}, timeout=120,
    )


def _members(archive: bytes) -> list[str]:
    listing = subprocess.run(["tar", "--incremental", "-tzf", "-"], input=archive, capture_output=True, check=True)
    return [line for line in listing.stdout.decode().splitlines() if not line.endswith("/")]


def _tgz(*paths) -> bytes:
    return subprocess.run(["tar", "-czf", "-", *map(str, paths)], capture_output=True, check=True).stdout


def test_the_hook_waits_long_enough_for_its_turn_and_the_script_reads_its_own_config(tmp_path):
    log, _, _, _ = _changelog(tmp_path)
    assert tar_changelog.HOOK_TIMEOUT_S == 600
    assert f"CFG = {json.dumps(str(tmp_path / 'changelog' / 'config.json'))}" in log.capture_script()
    assert log.hook_command.endswith(f"{tmp_path / 'changelog' / 'capture.py'}")
    assert log.hook_command.split()[0] == sys.executable


def test_the_changelog_directory_is_one_per_port():
    assert tar_changelog.changelog_dir({"A2A_PORT": "8123"}).name == "claude-code-changelog-8123"
    assert tar_changelog.changelog_dir({"A2A_PORT": "8123"}) != tar_changelog.changelog_dir({"A2A_PORT": "8124"})
    assert tar_changelog.changelog_dir({}).name == "claude-code-changelog-default"


def test_outermost_keeps_each_path_once_and_none_inside_another():
    assert tar_changelog.outermost(["/home/agent/.claude", "/app", "/home/agent", "/app/", "/etc"]) == [
        "/app", "/home/agent", "/etc"]


def test_merge_adds_the_hook_once_and_keeps_the_rest(tmp_path):
    path = tmp_path / "cli" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"model": "m", "hooks": {"Other": [{"matcher": "x", "hooks": []}]}}))

    for _ in range(2):
        tar_changelog.merge_settings_hook(path, event="PostToolUse", matcher="", command="py /c.py", timeout=600)

    assert json.loads(path.read_text()) == {
        "model": "m",
        "hooks": {
            "Other": [{"matcher": "x", "hooks": []}],
            "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "py /c.py", "timeout": 600}]}],
        },
    }


def test_merge_replaces_an_unreadable_settings_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json")

    tar_changelog.merge_settings_hook(path, event="PostToolUse", matcher="", command="c", timeout=1)

    assert json.loads(path.read_text()) == {
        "hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "c", "timeout": 1}]}]}
    }


def test_remove_drops_only_the_hooks_whose_command_matches(tmp_path):
    path = tmp_path / "settings.json"
    keep = {"matcher": "", "hooks": [{"type": "command", "command": "other"}]}
    drop = {"matcher": "", "hooks": [{"type": "command", "command": "py /tmp/claude-code-changelog-8000/capture.py"}]}
    path.write_text(json.dumps({"hooks": {"PostToolUse": [keep, drop]}}))

    assert tar_changelog.remove_settings_hooks(path, event="PostToolUse", matches=tar_changelog.is_capture_command) is True
    assert json.loads(path.read_text()) == {"hooks": {"PostToolUse": [keep]}}
    assert tar_changelog.remove_settings_hooks(path, event="PostToolUse", matches=tar_changelog.is_capture_command) is False
    assert tar_changelog.remove_settings_hooks(tmp_path / "absent.json", event="PostToolUse",
                                           matches=tar_changelog.is_capture_command) is False


@needs_gnu_tar
def test_enable_writes_the_script_and_config_baselines_and_installs_the_hook(tmp_path):
    log, installed, _, _ = _changelog(tmp_path)
    (tmp_path / "default-root").mkdir()
    directory = tmp_path / "changelog"
    directory.mkdir()
    for stale in ("seq", "turn", "namespace.json", "held.3", "state.snar.2"):
        (directory / stale).write_text("0")

    response = _enable(log)

    assert response.roots == [str(tmp_path / "default-root")]
    assert installed == [log.hook_command]
    assert (directory / "capture.py").read_text() == log.capture_script()
    config = json.loads((directory / "config.json").read_text())
    assert set(config) == {"dir", "snar", "roots", "epoch", "write_namespace"}
    assert (config["dir"], config["snar"]) == (str(directory), str(directory / "state.snar"))
    assert config["write_namespace"]["root_path"] == ROOT_PATH
    assert (directory / "state.snar").is_file()
    assert not any((directory / stale).exists() for stale in ("seq", "turn", "namespace.json", "held.3", "state.snar.2"))
    # The config holds the grant, so no other user reads the directory.
    assert directory.stat().st_mode & 0o777 == 0o700


def test_enable_refuses_an_empty_root_list(tmp_path):
    log, installed, _, _ = _changelog(tmp_path)

    with pytest.raises(HTTPException) as raised:
        _enable(log, roots=[])

    assert (raised.value.status_code, raised.value.detail) == (
        400, "roots must be a non-empty list of strings (omit to use defaults)")
    assert installed == []


@needs_gnu_tar
def test_each_enable_starts_a_new_epoch(tmp_path):
    log, _, _, _ = _changelog(tmp_path)
    config = tmp_path / "changelog" / "config.json"

    _enable(log, roots=[str(tmp_path)])
    first = json.loads(config.read_text())["epoch"]
    _enable(log, roots=[str(tmp_path)])

    assert json.loads(config.read_text())["epoch"] != first


@needs_gnu_tar
def test_each_tool_call_uploads_its_own_writes_and_the_increments_replay_them(tmp_path, staging, store):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "before.txt").write_text("there before the capture")
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log)

    (root / "first.txt").write_text("one")
    first = _capture(log, staging.port)
    (root / "second.txt").write_text("two")
    (root / "first.txt").unlink()
    second = _capture(log, staging.port, tool="Write")

    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert "captured seq=0 tool=Bash" in first.stderr and "captured seq=1 tool=Write" in second.stderr
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000000.tgz", f"{ROOT_PATH}/000001.tgz"]
    assert _members(staging.objects[f"{ROOT_PATH}/000000.tgz"]) == [str(root / "first.txt").lstrip("/")]
    assert _members(staging.objects[f"{ROOT_PATH}/000001.tgz"]) == [str(root / "second.txt").lstrip("/")]
    assert not list((tmp_path / "changelog").glob("*.tgz"))

    # Replayed onto the state before the capture, the increments reproduce the state after the last tool call.
    (root / "second.txt").unlink()
    (root / "first.txt").write_text("stale")
    increments = []
    for sequence in (0, 1):
        key = f"ns/{sequence:06d}.tgz"
        store.objects[key] = staging.objects[f"{ROOT_PATH}/{sequence:06d}.tgz"]
        increments.append({"sequence": sequence, "object": read_grant(key, len(store.objects[key]))})
    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.count == 2
    assert sorted(p.name for p in root.iterdir()) == ["before.txt", "second.txt"]
    assert (root / "second.txt").read_text() == "two"


@needs_gnu_tar
def test_a_capture_past_the_grants_object_limit_is_skipped_across_hook_processes(tmp_path, staging):
    root = tmp_path / "workspace"
    root.mkdir()
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log, max_objects=1)

    (root / "first.txt").write_text("one")
    first = _capture(log, staging.port)
    (root / "second.txt").write_text("two")
    second = _capture(log, staging.port)

    assert first.returncode == second.returncode == 0
    assert sorted(staging.objects) == [f"{ROOT_PATH}/000000.tgz"]
    assert "skipped seq=1: past the grant's limits" in second.stderr
    assert not list((tmp_path / "changelog").glob("*.tgz"))


@needs_gnu_tar
def test_a_refused_upload_gives_back_its_reservation_and_exits_zero(tmp_path, socket_enabled):
    root = tmp_path / "workspace"
    root.mkdir()
    log, _, _, _ = _changelog(tmp_path, roots=[str(root)])
    _enable(log, max_objects=1)
    refusing = Staging(status=403)

    (root / "first.txt").write_text("one")
    try:
        refused = _capture(log, refusing.port)
    finally:
        refusing.close()

    assert refused.returncode == 0 and "hook error (non-fatal)" in refused.stderr
    assert json.loads((tmp_path / "changelog" / "namespace.json").read_text()) == {}
    assert not list((tmp_path / "changelog").glob("*.tgz"))


@needs_gnu_tar
def test_apply_extracts_in_order_then_runs_after_apply_and_resumes_only_when_asked(tmp_path, store):
    target = tmp_path / "restored" / "file.txt"
    target.parent.mkdir()
    target.write_text("final")
    store.objects["ns/000003.tgz"] = _tgz(target)
    target.unlink()
    log, _, after, resumed = _changelog(tmp_path)
    increments = [{"sequence": 3, "object": read_grant("ns/000003.tgz", len(store.objects["ns/000003.tgz"]))}]

    plain = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))
    resumed_answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
        {"increments": [], "resume_conversation": True, "target_context_id": "ctx"})))

    assert plain.model_dump(exclude_none=True) == {"count": 1}
    assert resumed_answer.model_dump(exclude_none=True) == {"count": 0, "context_id": "ctx"}
    assert target.read_text() == "final"
    assert after == [True, True] and resumed == ["ctx"]
    assert not any((tmp_path / "changelog" / "apply").iterdir())


def test_apply_without_after_apply_or_a_resumed_transcript(tmp_path):
    async def nothing_to_resume(_target):
        return None

    log = tar_changelog.Changelog(default_roots=[], install_hook=lambda command: None, resume=nothing_to_resume,
                              directory=tmp_path / "changelog")

    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest(increments=[], resume_conversation=True)))

    assert answer.model_dump(exclude_none=True) == {"count": 0}


def test_a_failed_increment_download_is_the_frameworks_typed_error(tmp_path, store):
    log, _, after, _ = _changelog(tmp_path)

    with pytest.raises(TransferError) as raised:
        asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate(
            {"increments": [{"sequence": 0, "object": read_grant("ns/missing.tgz", 1)}]})))

    assert raised.value.code == "transfer_rejected" and after == []


@needs_gnu_tar
def test_an_increment_that_restores_only_partly_still_counts_and_is_logged(tmp_path, store, caplog):
    blocked = tmp_path / "restored" / "blocked"
    blocked.parent.mkdir()
    blocked.write_text("from the capture")
    store.objects["ns/000000.tgz"] = _tgz(blocked)
    blocked.unlink()
    blocked.mkdir()
    (blocked / "keeps-the-directory").write_text("")
    log, _, _, _ = _changelog(tmp_path)
    increments = [{"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))}]

    with caplog.at_level("WARNING", logger="tar_changelog"):
        answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.model_dump(exclude_none=True) == {"count": 1}
    warning = next(record for record in caplog.records if record.levelname == "WARNING")
    assert "increment 000000: tar exited" in warning.getMessage()
    assert not any((tmp_path / "changelog" / "apply").iterdir())


@needs_gnu_tar
def test_an_unreadable_increment_fails_the_apply(tmp_path, store):
    first = tmp_path / "restored" / "first.txt"
    first.parent.mkdir()
    first.write_text("from increment 0")
    store.objects["ns/000000.tgz"] = _tgz(first)
    first.unlink()
    store.objects["ns/000001.tgz"] = b"not a gzip archive"
    log, _, after, _ = _changelog(tmp_path)
    increments = [
        {"sequence": 0, "object": read_grant("ns/000000.tgz", len(store.objects["ns/000000.tgz"]))},
        {"sequence": 1, "object": read_grant("ns/000001.tgz", len(store.objects["ns/000001.tgz"]))},
    ]

    with pytest.raises(HTTPException) as raised:
        asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert raised.value.status_code == 422
    assert raised.value.detail.startswith("changelog increment 000001 is not a readable archive: ")
    assert after == []
    assert not any((tmp_path / "changelog" / "apply").iterdir())


@needs_gnu_tar
def test_each_increment_is_removed_before_the_next_is_fetched(tmp_path, store, monkeypatch):
    restored = [tmp_path / "restored" / f"file-{n}.txt" for n in range(3)]
    restored[0].parent.mkdir()
    for n, path in enumerate(restored):
        path.write_text(str(n))
        store.objects[f"ns/{n:06d}.tgz"] = _tgz(path)
        path.unlink()
    apply_dir = tmp_path / "changelog" / "apply"
    on_disk_at_each_fetch: list[list[str]] = []
    serve = type(store).__call__

    def observing(fake, request):
        on_disk_at_each_fetch.append(sorted(p.name for p in apply_dir.iterdir()))
        return serve(fake, request)

    monkeypatch.setattr(type(store), "__call__", observing)
    log, _, _, _ = _changelog(tmp_path)
    increments = [
        {"sequence": n, "object": read_grant(f"ns/{n:06d}.tgz", len(store.objects[f"ns/{n:06d}.tgz"]))}
        for n in range(3)
    ]

    answer = asyncio.run(log.apply(ObjectChangelogApplyRequest.model_validate({"increments": increments})))

    assert answer.model_dump(exclude_none=True) == {"count": 3}
    for n, names in enumerate(on_disk_at_each_fetch):
        assert all(name.startswith(f".{n:06d}.tgz.") and name.endswith(".part") for name in names), names
    assert [path.read_text() for path in restored] == ["0", "1", "2"]


@pytest.mark.parametrize(("gnu", "expected"), [(True, "ClaudeCodeChangelogAgent"), (False, "ClaudeCodeAgent")])
def test_the_changelog_is_served_only_where_tar_is_gnu_tar(monkeypatch, gnu, expected):
    monkeypatch.setattr(a2a_server, "gnu_tar", lambda: gnu)

    assert type(a2a_server.agent_for_this_host()).__name__ == expected


def test_the_changelog_agent_advertises_both_methods_and_the_base_agent_neither(app):
    changelog_app = a2a_server.build_app(a2a_server.ClaudeCodeChangelogAgent())

    assert extension_methods(changelog_app, "urn:agentenv:snapshot/v1")["enable-changelog"] == {
        "method": "POST", "endpoint": "/ext/snapshot/changelog",
        "request": {"required": ["write_namespace"], "optional": ["roots"]}, "response": {"required": ["roots"]},
    }
    assert extension_methods(changelog_app, "urn:agentenv:snapshot/v1")["apply-changelog"] == {
        "method": "PUT", "endpoint": "/ext/snapshot/changelog",
        "request": {"required": ["increments"], "optional": ["resume_conversation", "target_context_id"]},
        "response": {"required": ["count"], "optional": ["context_id"]},
    }
    assert sorted(extension_methods(app, "urn:agentenv:snapshot/v1")) == ["load", "save"]


def test_the_default_roots_hold_the_workspace_and_the_clis_state(agent_dirs):
    roots = a2a_server._changelog_roots()

    assert str(a2a_server.WORKSPACE_DIR) in roots
    assert any(str(agent_dirs) == root or str(agent_dirs).startswith(root + "/") for root in roots)


def _changelog_client(monkeypatch, tmp_path) -> tuple[TestClient, a2a_server.ClaudeCodeChangelogAgent]:
    monkeypatch.setenv("A2A_PORT", "8000")
    monkeypatch.setattr(a2a_server, "changelog_dir", lambda environ: tmp_path / "changelog")
    agent = a2a_server.ClaudeCodeChangelogAgent()
    return TestClient(a2a_server.build_app(agent)), agent


def test_apply_resumes_the_restored_transcript_and_drops_the_restored_capture_hook(monkeypatch, tmp_path, agent_dirs):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    sessions = a2a_server.SESSION_JSONL_DIR
    sessions.mkdir(parents=True)
    (sessions / "captured.jsonl").write_text(
        json.dumps({"sessionId": "old", "type": "user"}) + "\n" + json.dumps({"sessionId": "old", "type": "assistant"}))
    restored_hook = f"python3 {tmp_path / 'claude-code-changelog-9' / 'capture.py'}"
    (agent_dirs / "settings.json").write_text(json.dumps(
        {"hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": restored_hook}]}]}}))

    response = client.put("/ext/snapshot/changelog",
                          json={"increments": [], "resume_conversation": True, "target_context_id": "ctx-r"})

    assert response.status_code == 200, response.text
    assert response.json() == {"count": 0, "context_id": "ctx-r"}
    session = agent.session_ref_for_context("ctx-r")
    assert session and session != "old"
    assert [json.loads(line)["sessionId"] for line in (sessions / f"{session}.jsonl").read_text().splitlines()] == [
        session, session]
    assert json.loads((agent_dirs / "settings.json").read_text()) == {"hooks": {"PostToolUse": []}}


def test_apply_keeps_a_capture_hook_whose_script_this_agent_has(monkeypatch, tmp_path, agent_dirs):
    client, _ = _changelog_client(monkeypatch, tmp_path)
    own = tmp_path / "claude-code-changelog-8000" / "capture.py"
    own.parent.mkdir()
    own.write_text("")
    settings = {"hooks": {"PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": f"python3 {own}"}]}]}}
    agent_dirs.mkdir(parents=True)
    (agent_dirs / "settings.json").write_text(json.dumps(settings))

    response = client.put("/ext/snapshot/changelog", json={"increments": []})

    assert response.status_code == 200
    assert json.loads((agent_dirs / "settings.json").read_text()) == settings


def test_apply_without_a_restored_transcript_resumes_nothing(monkeypatch, tmp_path):
    client, _ = _changelog_client(monkeypatch, tmp_path)

    response = client.put("/ext/snapshot/changelog", json={"increments": [], "resume_conversation": True})

    assert response.status_code == 200 and response.json() == {"count": 0}


def test_apply_refuses_while_a_turn_runs_and_onto_a_bound_context(monkeypatch, tmp_path):
    client, agent = _changelog_client(monkeypatch, tmp_path)
    agent._active_contexts.add("busy")
    running = client.put("/ext/snapshot/changelog", json={"increments": []})
    agent._active_contexts.clear()
    agent.set_session_ref_for_context("taken", "session-1")
    bound = client.put("/ext/snapshot/changelog", json={"increments": [], "target_context_id": "taken"})
    empty = client.put("/ext/snapshot/changelog", json={"increments": [], "target_context_id": ""})

    assert (running.status_code, running.json()["detail"]) == (
        409, "cannot apply a changelog while an agent turn is running")
    assert bound.status_code == 409 and "already bound" in bound.json()["detail"]
    assert empty.status_code == 400
