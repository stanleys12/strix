"""Connect to the Docker daemon the way the ``docker`` CLI does.

``docker.from_env()`` only looks at ``DOCKER_HOST`` and otherwise assumes
``/var/run/docker.sock`` (a named pipe on Windows). The CLI also honours the
current docker context, which is where Docker Desktop on macOS (without the
"default socket" option), OrbStack and Colima register their sockets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import docker  # type: ignore[import-untyped, unused-ignore]
from docker.context import ContextAPI  # type: ignore[import-untyped, unused-ignore]
from docker.context.config import (  # type: ignore[import-untyped, unused-ignore]
    get_current_context_name,
)
from docker.errors import DockerException  # type: ignore[import-untyped, unused-ignore]


DEFAULT_CONTEXT = "default"


@dataclass(frozen=True)
class DockerEndpoint:
    """Where strix talks to Docker and how that address was chosen."""

    host: str | None
    source: str
    tls: Any = None

    @property
    def label(self) -> str:
        return f"{self.host or 'default socket'} ({self.source})"


class DockerConnectionError(RuntimeError):
    """The daemon at ``endpoint`` could not be reached."""

    def __init__(self, endpoint: DockerEndpoint, cause: BaseException) -> None:
        self.endpoint = endpoint
        self.cause = root_cause(cause)
        super().__init__(f"Cannot connect to Docker at {endpoint.label}: {self.detail}")

    @property
    def detail(self) -> str:
        text = str(self.cause).strip()
        name = type(self.cause).__name__
        return text if name in text else f"{name}: {text}" if text else name


def resolve_docker_endpoint(environ: dict[str, str] | None = None) -> DockerEndpoint:
    """``DOCKER_HOST``, else the current docker context, else the SDK default."""
    env = os.environ if environ is None else environ
    host = env.get("DOCKER_HOST", "").strip()
    if host:
        return DockerEndpoint(host, "DOCKER_HOST")

    name = get_current_context_name()
    if name != DEFAULT_CONTEXT:
        try:
            context = ContextAPI.get_context(name)
        except Exception:  # noqa: BLE001 - a broken context file must not hide Docker itself
            context = None
        if context is not None and context.Host:
            return DockerEndpoint(context.Host, f"docker context '{name}'", context.TLSConfig)
    return DockerEndpoint(None, "default socket")


def connect_docker() -> Any:
    """Return a ``docker.DockerClient`` for the resolved endpoint or raise DockerConnectionError."""
    endpoint = resolve_docker_endpoint()
    try:
        if endpoint.host is None:
            return docker.from_env()
        return docker.DockerClient(base_url=endpoint.host, tls=endpoint.tls or False)
    except DockerException as exc:
        raise DockerConnectionError(endpoint, exc) from exc


def root_cause(exc: BaseException) -> BaseException:
    """The innermost exception: the SDK wraps the OSError in requests/urllib3 errors."""
    current = exc
    seen: list[BaseException] = []
    while not any(current is s for s in seen):
        seen.append(current)
        nested = [arg for arg in current.args if isinstance(arg, BaseException)]
        following = current.__cause__ or current.__context__ or (nested[-1] if nested else None)
        if following is None:
            break
        current = following
    return current
