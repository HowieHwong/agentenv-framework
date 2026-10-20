"""The stand-in agent (``agentenv_protocol.a2a_agent.stand_in``) as the Episode API deploys it, in the place of an agent
a loop plays: ``@local/agentenv-stand-in``, an image registered by its build context alone and built where each deploy
runs. Its context is this folder, the installed protocol package, and that package's requirements for serving an
agent, pinned to the versions installed here."""

from __future__ import annotations

import shutil
import tempfile
import threading
from importlib.metadata import requires, version
from pathlib import Path

import agentenv_protocol
from packaging.requirements import Requirement

from agent_env.a2a_agent.a2a_agent import TAKES_CREDENTIALS, A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.store.base import NotFoundError

STAND_IN_AGENT_ID = "@local/agentenv-stand-in"
_IMAGE_ID = f"{STAND_IN_AGENT_ID}/image"
_PROTOCOL_DISTRIBUTION = "agentenv-framework-protocol"
_HERE = Path(__file__).parent
_writing = threading.Lock()


def stand_in_agent() -> A2AAgent:
    """The stand-in, written to the store first when it isn't there or its build context has changed."""
    with _writing, tempfile.TemporaryDirectory() as staged:
        context = _stage(Path(staged))
        dockerfile = str(context / "Dockerfile")
        digest, _ = DockerImageArtifact.context_identity(_IMAGE_ID, str(context), dockerfile, platform=None)
        try:
            agent = A2AAgent.get(STAND_IN_AGENT_ID)
            if agent.docker_image_artifact.source_digest == digest:
                return agent
        except NotFoundError:
            pass
        image = DockerImageArtifact.put_context(
            _IMAGE_ID, description="The stand-in agent: a driver outside its sandbox answers its turns.",
            context_path=str(context), dockerfile_path=dockerfile, platform=None,
        )
        return A2AAgent.put(id=STAND_IN_AGENT_ID, docker_image_artifact=image, metadata={TAKES_CREDENTIALS: False})


def _stage(context: Path) -> Path:
    for name in ("Dockerfile", "a2a_server.py"):
        shutil.copy2(_HERE / name, context / name)
    shutil.copytree(Path(agentenv_protocol.__file__).parent, context / "agentenv_protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (context / "requirements.txt").write_text("".join(f"{line}\n" for line in _requirements()))
    return context


def _requirements() -> list[str]:
    pinned = []
    for spec in requires(_PROTOCOL_DISTRIBUTION) or []:
        requirement = Requirement(spec)
        if requirement.marker is None or requirement.marker.evaluate({"extra": "agent"}):
            extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
            pinned.append(f"{requirement.name}{extras}=={version(requirement.name)}")
    return sorted(pinned)
