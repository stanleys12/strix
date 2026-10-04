"""Connect to the Docker daemon the way the ``docker`` CLI does, and say why it failed.

``docker.from_env()`` only looks at ``DOCKER_HOST`` and otherwise assumes
``/var/run/docker.sock`` (a named pipe on Windows). The CLI also honours the
current docker context, which is where Docker Desktop on macOS (without the
"default socket" option), OrbStack, Colima and Rancher Desktop register their
sockets. Resolving the endpoint the same way means ``docker ps`` working
implies strix works.
"""

from __future__ import annotations

import errno
import os
import sys
from dataclasses import dataclass
from typing import Any

import docker  # type: ignore[import-untyped, unused-ignore]
from docker.context import ContextAPI  # type: ignore[import-untyped, unused-ignore]
from docker.context.config import (  # type: ignore[import-untyped, unused-ignore]
    get_current_context_name,
)
from docker.errors import DockerException  # type: ignore[import-untyped, unused-ignore]


DEFAULT_CONTEXT = "default"
DEFAULT_SOURCE = "default socket"

SOCKET_MISSING = "socket_missing"
PERMISSION_DENIED = "permission_denied"
CONNECTION_REFUSED = "connection_refused"
UNKNOWN = "unknown"

_ERRNO_REASONS = {
    errno.EACCES: PERMISSION_DENIED,
    errno.EPERM: PERMISSION_DENIED,
    errno.ENOENT: SOCKET_MISSING,
    errno.ECONNREFUSED: CONNECTION_REFUSED,
}
# pywintypes.error from the npipe transport: ERROR_FILE_NOT_FOUND, ERROR_ACCESS_DENIED
_WINERROR_REASONS = {2: SOCKET_MISSING, 5: PERMISSION_DENIED}
_TEXT_REASONS = (
    (("permission denied", "operation not permitted", "access is denied"), PERMISSION_DENIED),
    (("no such file or directory", "cannot find the file specified"), SOCKET_MISSING),
    (("connection refused", "max retries exceeded"), CONNECTION_REFUSED),
)


@dataclass(frozen=True)
class DockerEndpoint:
    """Where strix will talk to Docker and how that address was chosen."""

    host: str | None
    source: str
    tls: Any = None

    @property
    def label(self) -> str:
        return f"{self.host or DEFAULT_SOURCE} ({self.source})"


class DockerConnectionError(RuntimeError):
    """The daemon at ``endpoint`` could not be reached; ``reason`` says why."""

    def __init__(self, endpoint: DockerEndpoint, reason: str, cause: BaseException) -> None:
        super().__init__(f"Cannot connect to Docker at {endpoint.label}: {root_cause_line(cause)}")
        self.endpoint = endpoint
        self.reason = reason
        self.cause = cause

    @property
    def detail(self) -> str:
        return root_cause_line(self.cause)


def resolve_docker_endpoint(environ: dict[str, str] | None = None) -> DockerEndpoint:
    """``DOCKER_HOST``, else the current docker context, else the SDK default."""
    env = os.environ if environ is None else environ
    host = env.get("DOCKER_HOST", "").strip()
    if host:
        return DockerEndpoint(host, "DOCKER_HOST")

    name = env.get("DOCKER_CONTEXT", "").strip() or get_current_context_name()
    if name != DEFAULT_CONTEXT:
        try:
            context = ContextAPI.get_context(name)
        except Exception:  # noqa: BLE001 - a broken context file must not hide Docker itself
            context = None
        if context is not None and context.Host:
            return DockerEndpoint(context.Host, f"docker context '{name}'", context.TLSConfig)
    return DockerEndpoint(None, DEFAULT_SOURCE)


def connect_docker() -> Any:
    """Return a ``docker.DockerClient`` for the resolved endpoint or raise DockerConnectionError."""
    endpoint = resolve_docker_endpoint()
    try:
        if endpoint.host is None:
            return docker.from_env()
        return docker.DockerClient(base_url=endpoint.host, tls=endpoint.tls or False)
    except DockerException as exc:
        raise DockerConnectionError(endpoint, classify_failure(exc), exc) from exc


def classify_failure(exc: BaseException) -> str:
    """Map the SDK's wrapped exception onto one of the reasons above."""
    for inner in walk_exceptions(exc):
        code = getattr(inner, "errno", None)
        if isinstance(code, int) and code in _ERRNO_REASONS:
            return _ERRNO_REASONS[code]
        winerror = getattr(inner, "winerror", None)
        if isinstance(winerror, int) and winerror in _WINERROR_REASONS:
            return _WINERROR_REASONS[winerror]
    text = str(exc).lower()
    return next(
        (reason for needles, reason in _TEXT_REASONS if any(n in text for n in needles)),
        UNKNOWN,
    )


def explain_failure(error: DockerConnectionError, platform: str | None = None) -> tuple[str, str]:
    """Plain-language cause and the fix for this platform."""
    platform = platform or sys.platform
    host = error.endpoint.host
    source = error.endpoint.source
    reason = error.reason

    if reason == PERMISSION_DENIED:
        cause = "Your user is not allowed to use the Docker socket."
        fix = (
            "sudo usermod -aG docker $USER, then log out and back in (or run: newgrp docker)."
            if platform == "linux"
            else "Run strix as the user who installed Docker, or check the socket permissions."
        )
    elif reason == CONNECTION_REFUSED:
        cause = f"Nothing is listening at {host}."
        fix = f"Start the Docker daemon there, or fix {source}."
    elif reason == SOCKET_MISSING and source != DEFAULT_SOURCE:
        cause = f"{source} points at a socket that does not exist."
        fix = "Start that Docker daemon, or run: docker context ls and pick one that is running."
    elif reason == SOCKET_MISSING:
        cause, fix = _NOT_RUNNING[platform if platform in _NOT_RUNNING else "linux"]
    else:
        cause = "Cannot connect to the Docker daemon."
        fix = "Run: docker info in this shell. If it works, set DOCKER_HOST to the host it prints."
    return cause, f"Fix: {fix}"


_NOT_RUNNING = {
    "darwin": (
        "Docker is not running, or it only listens on its own socket.",
        "Start Docker Desktop, OrbStack or Colima. If it is already running, select its "
        "context: docker context use desktop-linux (or orbstack, colima).",
    ),
    "win32": (
        "Docker Desktop is not running.",
        "Start Docker Desktop and wait until it reports running. From WSL, enable "
        "Settings > Resources > WSL integration for this distro.",
    ),
    "linux": (
        "The Docker daemon is not running.",
        "sudo systemctl start docker (Docker Desktop for Linux: "
        "systemctl --user start docker-desktop).",
    ),
}


def walk_exceptions(exc: BaseException) -> list[BaseException]:
    """Every exception reachable through causes, contexts and args, outermost first."""
    seen: list[BaseException] = []
    stack = [exc]
    while stack:
        current = stack.pop(0)
        if any(current is s for s in seen):
            continue
        seen.append(current)
        stack.extend(link for link in (current.__cause__, current.__context__) if link is not None)
        stack.extend(arg for arg in current.args if isinstance(arg, BaseException))
    return seen


def root_cause_line(exc: BaseException) -> str:
    """The innermost exception as one line, e.g. ``FileNotFoundError: [Errno 2] ...``."""
    root = walk_exceptions(exc)[-1]
    text = str(root).strip() or type(root).__name__
    return text if type(root).__name__ in text else f"{type(root).__name__}: {text}"
