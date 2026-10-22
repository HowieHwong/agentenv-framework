"""snapshot/v1's changelog: each tool call's writes captured, and replayed onto a fresh agent.

``enable`` writes a capture script and baselines a GNU incremental tar of the capture roots; the CLI's post-tool hook
then archives each tool call's writes and uploads them through agent-env's namespace grant as
``<root_path>/<position:06d>.tgz``. ``apply`` extracts granted increments in order onto a fresh agent of the same
image, and can make the restored conversation resumable.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path

from agentenv_protocol.a2a_agent import (
    ChangelogIncrement,
    NamespaceChangelogEnableRequest,
    NamespaceChangelogEnableResponse,
    ObjectChangelogApplyRequest,
    ObjectChangelogApplyResponse,
)
from agentenv_protocol.transfers import download
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)

# A capture hook can wait its turn behind other hooks (parallel tool calls), then tar and upload with retries; a CLI
# that kills it past its timeout leaves that tool call's increment missing.
HOOK_TIMEOUT_S = 600
_DIR_PREFIX = "claude-code-changelog-"
_TAR_STDERR_TAIL = 2000

# The post-tool hook, run by the agent's own Python, as a child of the CLI after every tool call, so an increment
# includes that tool's writes. It exits 0 on a capture error, so the error never reaches the model.
CAPTURE_SCRIPT = r'''import asyncio, fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

from agentenv_protocol.transfers import NamespaceUploader, WriteNamespaceGrant

CFG = "/tmp/changelog/config.json"


def _read_int(path):
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


def _write_atomic(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _epoch():
    try:
        return json.load(open(CFG)).get("epoch")
    except Exception:
        return None


def _alive(changelog_dir, k):
    """Whether the hook holding position k is still running (the CLI kills one past its timeout)."""
    try:
        pid = int(open(os.path.join(changelog_dir, "held.%d" % k)).read())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _committed(changelog_dir):
    try:
        return json.load(open(os.path.join(changelog_dir, "namespace.json")))
    except Exception:
        return {}


def _reserve(changelog_dir, grant, name, size):
    """Reserve an object's bytes against the grant's limits, which every hook process shares."""
    with open(os.path.join(changelog_dir, "namespace.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prospective = {**_committed(changelog_dir), name: size}
        if (size > grant.max_object_bytes or len(prospective) > grant.max_objects
                or sum(prospective.values()) > grant.max_total_bytes):
            return False
        _write_atomic(os.path.join(changelog_dir, "namespace.json"), json.dumps(prospective))
        return True


def _release(changelog_dir, name):
    with open(os.path.join(changelog_dir, "namespace.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        committed = _committed(changelog_dir)
        committed.pop(name, None)
        _write_atomic(os.path.join(changelog_dir, "namespace.json"), json.dumps(committed))


def main():
    cfg = json.load(open(CFG))
    changelog_dir, snar, roots, epoch = cfg["dir"], cfg["snar"], cfg["roots"], cfg.get("epoch")
    try:
        hook = json.load(sys.stdin)
    except Exception:
        hook = {}
    tool = str(hook.get("tool_name") or "tool").replace("/", "_")[:40]
    seqf = os.path.join(changelog_dir, "seq")
    turnf = os.path.join(changelog_dir, "turn")
    held = os.path.join(changelog_dir, "held.%d")
    # Parallel tool calls run their hooks at once. Each draws its position in arrival order, under a lock held only
    # for the draw, and records itself as that position's holder.
    with open(os.path.join(changelog_dir, "seq.lock"), "a") as seq_lock:
        fcntl.flock(seq_lock, fcntl.LOCK_EX)
        if _epoch() != epoch:  # enable started another capture since this hook loaded its config
            return
        k = _read_int(seqf)
        _write_atomic(seqf, str(k + 1))
        _write_atomic(held % k, str(os.getpid()))
    local = os.path.join(changelog_dir, "%04d.tgz" % k)
    present = [r for r in roots if os.path.exists(r)]
    # The archives are incremental, so they're made in position order: wait for this position's turn, passing over
    # positions whose hook died first. A killed hook leaves a gap at its own position instead of shifting later ones.
    archived = False
    while True:
        with open(os.path.join(changelog_dir, "lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if _epoch() != epoch:  # its capture was replaced; this position is no longer its to write
                return
            turn = _read_int(turnf)
            while turn < k and not _alive(changelog_dir, turn):
                turn += 1
            if turn == k:
                # tar works on a copy of the incremental state, promoted only once tar has finished: a tar that
                # outlives a killed hook advances nothing the next archive relies on. tar inherits the lock too.
                work = "%s.%d" % (snar, k)
                if os.path.exists(snar):
                    shutil.copyfile(snar, work)
                subprocess.run(
                    ["tar", "--listed-incremental=" + work, "--exclude=" + changelog_dir, "-czf", local, *present],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(lock.fileno(),),
                )
                os.replace(work, snar)
                archived = True
            if turn >= k:
                _write_atomic(turnf, str(max(turn, k + 1)))
                break
        time.sleep(0.05)
    try:
        os.remove(held % k)
    except OSError:
        pass
    if not archived:  # a later position went first, so this one would replay out of order
        sys.stderr.write("[changelog] skipped seq=%d: a later position was archived first\n" % k)
        return
    # agent-env's namespace grant, named by the absolute tool-call position.
    grant = WriteNamespaceGrant.model_validate(cfg["write_namespace"])
    name = "%06d.tgz" % k
    try:
        if not _reserve(changelog_dir, grant, name, os.path.getsize(local)):
            sys.stderr.write("[changelog] skipped seq=%d: past the grant's limits\n" % k)
            return
        try:
            asyncio.run(NamespaceUploader(grant).upload(name, Path(local)))
        except BaseException:
            _release(changelog_dir, name)
            raise
    finally:
        os.remove(local)
    sys.stderr.write("[changelog] captured seq=%d tool=%s -> %s\n" % (k, tool, name))


if __name__ == "__main__":
    # The CLI stops a hook past its timeout with SIGTERM; exiting through SystemExit takes back its reservation.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    try:
        main()
    except Exception as e:
        sys.stderr.write("[changelog] hook error (non-fatal): %s\n" % e)
    sys.exit(0)
'''
_DEFAULT_CONFIG_LINE = 'CFG = "/tmp/changelog/config.json"'


def gnu_tar() -> bool:
    """Whether this host's tar is GNU tar, which the changelog's incremental archives need."""
    try:
        version = subprocess.run(["tar", "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GNU tar" in version


def changelog_dir(environ: Mapping[str, str]) -> Path:
    """This agent's changelog directory: one per port, since agents installed on one host listen on their own, and
    outside every capture root, so a capture never archives the changelog itself."""
    return Path(tempfile.gettempdir()) / f"{_DIR_PREFIX}{environ.get('A2A_PORT') or 'default'}"


def outermost(paths: Iterable[str]) -> list[str]:
    """``paths`` without duplicates or any inside another, so no file is archived twice."""
    kept: list[str] = []
    for path in dict.fromkeys(os.path.normpath(p) for p in paths):
        if not any(path == k or path.startswith(k.rstrip("/") + "/") for k in kept):
            kept = [k for k in kept if not k.startswith(path.rstrip("/") + "/")] + [path]
    return kept


def merge_settings_hook(path: Path, *, event: str, matcher: str, command: str, timeout: int) -> None:
    """Add a hook running ``command`` on ``event`` to a CLI's JSON settings file, once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(settings, dict):
            settings = {}
    except Exception:
        settings = {}
    entries = settings.setdefault("hooks", {}).setdefault(event, [])
    already = any(
        isinstance(e, dict) and any(h.get("command") == command for h in e.get("hooks", []))
        for e in entries
    )
    if not already:
        entries.append({"matcher": matcher, "hooks": [{"type": "command", "command": command, "timeout": timeout}]})
    path.write_text(json.dumps(settings, indent=2))


def remove_settings_hooks(path: Path, *, event: str, matches: Callable[[str], bool]) -> bool:
    """Drop every ``event`` hook entry whose command ``matches``; whether the file changed."""
    if not path.exists():
        return False
    try:
        settings = json.loads(path.read_text())
    except Exception:
        return False
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    entries = hooks.get(event) if isinstance(hooks, dict) else None
    if not isinstance(entries, list):
        return False
    kept = [
        e for e in entries
        if not (isinstance(e, dict) and any(matches(str(h.get("command", ""))) for h in e.get("hooks", [])))
    ]
    if len(kept) == len(entries):
        return False
    hooks[event] = kept
    path.write_text(json.dumps(settings, indent=2))
    return True


def is_capture_command(command: str) -> bool:
    """Whether ``command`` runs some agent's capture script: a restored capture agent's hook names its own."""
    return f"/{_DIR_PREFIX}" in command and command.endswith("capture.py")


def is_missing_capture_command(command: str) -> bool:
    """Whether ``command`` runs a capture script this agent doesn't have: a hook restored from the agent that
    captured, rather than this agent's own."""
    return is_capture_command(command) and not Path(shlex.split(command)[-1]).exists()


def _stderr_tail(stderr: bytes) -> str:
    return stderr.decode(errors="replace").strip()[-_TAR_STDERR_TAIL:]


class Changelog:
    """The ``enable-changelog`` / ``apply-changelog`` pair.

    ``install_hook(command)`` registers ``command`` as the CLI's post-tool hook. ``resume(target_context_id)`` makes
    the restored conversation resumable and returns the context it's bound to, or None when there's nothing to
    resume. ``after_apply()`` runs once increments are extracted, before any resume.
    """

    def __init__(
        self,
        default_roots: Iterable[str],
        install_hook: Callable[[str], None],
        resume: Callable[[str | None], Awaitable[str | None]],
        directory: Path | str,
        after_apply: Callable[[], None] | None = None,
    ) -> None:
        self.default_roots = list(default_roots)
        self.install_hook = install_hook
        self.resume = resume
        self.after_apply = after_apply
        self.directory = str(directory)
        self.snar = f"{self.directory}/state.snar"
        self.config_path = f"{self.directory}/config.json"
        self.capture_path = f"{self.directory}/capture.py"
        self.hook_command = f"{shlex.quote(sys.executable)} {shlex.quote(self.capture_path)}"

    def capture_script(self) -> str:
        """``CAPTURE_SCRIPT`` reading this changelog's config."""
        return CAPTURE_SCRIPT.replace(_DEFAULT_CONFIG_LINE, f"CFG = {json.dumps(self.config_path)}")

    async def enable(self, request: NamespaceChangelogEnableRequest) -> NamespaceChangelogEnableResponse:
        """Write the capture script and config, baseline the incremental-tar snapshot so each increment is a pure
        delta, and install the hook. Idempotent."""
        # Omitted roots take the defaults; an explicit empty list would capture nothing.
        roots = request.roots
        if roots is None:
            roots = list(self.default_roots)
        elif not roots:
            raise HTTPException(status_code=400, detail="roots must be a non-empty list of strings (omit to use defaults)")
        await asyncio.to_thread(self._reset, roots, request.write_namespace.model_dump(mode="json"))
        self.install_hook(self.hook_command)
        logger.info(f"agent-changelog configured: object-transfer grant roots={roots}")
        return NamespaceChangelogEnableResponse(roots=roots)

    def _reset(self, roots: list[str], write_namespace: dict) -> None:
        config = {"dir": self.directory, "snar": self.snar, "roots": roots, "epoch": uuid.uuid4().hex,
                  "write_namespace": write_namespace}
        # The config carries the grant, so the directory is the agent user's alone.
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        # Reset under both capture locks, so a hook still running for an earlier capture can't write into this one:
        # it finds the new epoch in the config and stops.
        with open(os.path.join(self.directory, "lock"), "a") as lock, open(
            os.path.join(self.directory, "seq.lock"), "a"
        ) as seq_lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fcntl.flock(seq_lock, fcntl.LOCK_EX)
            Path(self.capture_path).write_text(self.capture_script())
            Path(self.config_path).write_text(json.dumps(config))
            for stale in (
                os.path.join(self.directory, "seq"),
                os.path.join(self.directory, "turn"),
                os.path.join(self.directory, "namespace.json"),
                self.snar,
                *(str(p) for p in Path(self.directory).glob("held.*")),
                *(str(p) for p in Path(self.snar).parent.glob(Path(self.snar).name + ".*")),
            ):
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
            present = [r for r in roots if os.path.exists(r)]
            subprocess.run(
                ["tar", f"--listed-incremental={self.snar}", f"--exclude={self.directory}", "-cf", "/dev/null",
                 *present],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    async def apply(self, request: ObjectChangelogApplyRequest) -> ObjectChangelogApplyResponse:
        """Reconstruct the writable layer by extracting the granted increments in sequence order; agent-env sends
        them already cut at the requested tool call, and none is the valid pre-first-tool baseline.

        Meant for a FRESH agent of the same image: extraction restores absolute paths under /, so files the
        changelog removed are deleted too. As root it restores system directories as well; as another user the
        writable roots still restore, and tar's exit status for each increment is logged. An increment that isn't a
        readable archive fails the apply (422) before it's extracted.
        """
        applied = await self._apply_increments(request.increments)
        logger.info(f"agent-changelog applied {len(applied)} granted increments: {applied}")
        if self.after_apply is not None:
            self.after_apply()
        context_id = await self.resume(request.target_context_id) if request.resume_conversation else None
        return ObjectChangelogApplyResponse(count=len(applied), context_id=context_id)

    async def _apply_increments(self, increments: list[ChangelogIncrement]) -> list[dict]:
        apply_dir = Path(self.directory) / "apply"
        apply_dir.mkdir(parents=True, exist_ok=True)
        applied: list[dict] = []
        for increment in increments:
            local = apply_dir / f"{increment.sequence:06d}.tgz"
            try:
                await download(increment.object, local)
                # Read through before extracting, so a corrupt or truncated capture fails the apply instead of
                # counting as restored.
                listing = await asyncio.to_thread(
                    subprocess.run, ["tar", "--incremental", "-tzf", str(local)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                if listing.returncode != 0:
                    raise HTTPException(
                        status_code=422,
                        detail=f"changelog increment {increment.sequence:06d} is not a readable archive: "
                        f"{_stderr_tail(listing.stderr)}",
                    )
                # Extracted at / so absolute member paths restore; --incremental honours deletions.
                proc = await asyncio.to_thread(
                    subprocess.run, ["tar", "--incremental", "-xzf", str(local), "-C", "/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            finally:
                local.unlink(missing_ok=True)
            if proc.returncode != 0:
                # Expected for members another user can't restore (e.g. "etc: Cannot utime: Operation not
                # permitted"); the writable roots still restore, so it counts.
                logger.warning(f"agent-changelog increment {increment.sequence:06d}: tar exited {proc.returncode}: "
                               f"{_stderr_tail(proc.stderr)}")
            applied.append({"seq": increment.sequence, "tar_rc": proc.returncode})
        return applied
