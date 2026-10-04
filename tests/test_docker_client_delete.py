"""StrixDockerSandboxClient.delete() best-effort teardown.

delete() kills the sandbox container before delegating to the SDK's delete().
The kill is meant to be best-effort, but the ``contextlib.suppress`` around it
must cover the case where the docker daemon socket is already gone: then
``containers.get()`` -> ``inspect_container`` raises requests'
``ConnectionError``, which is a *sibling* of ``docker.errors.APIError`` under
``requests.RequestException`` (not a subclass), so an APIError-only suppress
would let it escape and surface a traceback on every teardown.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents.sandbox.sandboxes.docker import DockerSandboxClient
from docker import errors as docker_errors
from requests.exceptions import ConnectionError as RequestsConnectionError

from strix.runtime.docker_client import StrixDockerSandboxClient


if TYPE_CHECKING:
    from agents.sandbox.session.sandbox_session import SandboxSession


def _client_with_kill_error(exc: Exception) -> StrixDockerSandboxClient:
    """A StrixDockerSandboxClient whose containers.get(...).kill() raises ``exc``."""
    client = StrixDockerSandboxClient.__new__(StrixDockerSandboxClient)
    docker_client = MagicMock()
    docker_client.containers.get.side_effect = exc
    client.docker_client = docker_client
    return client


def _session(
    container_id: str | None = "abc123", pty_terminate_all: AsyncMock | None = None
) -> SandboxSession:
    # delete() reads the inner state's container_id and awaits the inner
    # session's PTY teardown.
    fake = SimpleNamespace(
        _inner=SimpleNamespace(
            state=SimpleNamespace(container_id=container_id),
            pty_terminate_all=pty_terminate_all or AsyncMock(),
        )
    )
    return cast("SandboxSession", fake)


@pytest.mark.parametrize(
    "exc",
    [
        RequestsConnectionError("Connection aborted", FileNotFoundError(2, "No such file")),
        docker_errors.NotFound("gone"),
        docker_errors.APIError("unhappy"),
    ],
)
@pytest.mark.asyncio
async def test_delete_swallows_best_effort_kill_errors(exc: Exception) -> None:
    """A torn-down socket (ConnectionError) or a gone/unhappy container
    (NotFound/APIError) during the kill must not propagate; delete() still
    delegates to the SDK's delete()."""
    client = _client_with_kill_error(exc)
    session = _session()

    with patch.object(
        DockerSandboxClient, "delete", new=AsyncMock(return_value=session)
    ) as super_delete:
        result = await client.delete(session)

    assert result is session
    super_delete.assert_awaited_once()  # teardown proceeded despite the kill error


@pytest.mark.asyncio
async def test_delete_does_not_swallow_unrelated_errors() -> None:
    """A programming error (e.g. ValueError) is not part of best-effort kill and
    must still propagate."""
    client = _client_with_kill_error(ValueError("boom"))
    with pytest.raises(ValueError):
        await client.delete(_session())


@pytest.mark.asyncio
async def test_delete_noop_without_container_id() -> None:
    """No container_id -> no kill attempt, just delegate."""
    client = StrixDockerSandboxClient.__new__(StrixDockerSandboxClient)
    client.docker_client = MagicMock()
    session = _session(container_id=None)

    with patch.object(
        DockerSandboxClient, "delete", new=AsyncMock(return_value=session)
    ) as super_delete:
        await client.delete(session)

    client.docker_client.containers.get.assert_not_called()
    super_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_terminates_pty_streams_even_when_the_container_is_gone() -> None:
    """The SDK's delete() skips shutdown() (and with it PTY teardown) when the
    container no longer exists, which leaves the agent's exec sockets to the
    garbage collector. delete() must terminate them itself, before anything
    else, whatever the container's state."""
    client = _client_with_kill_error(docker_errors.NotFound("gone"))
    order: list[str] = []
    session = _session(
        pty_terminate_all=AsyncMock(side_effect=lambda: order.append("pty_terminate_all"))
    )

    async def _super_delete(_self: object, _session: object) -> SandboxSession:
        order.append("super.delete")
        return session

    with patch.object(DockerSandboxClient, "delete", new=_super_delete):
        await client.delete(session)

    assert order == ["pty_terminate_all", "super.delete"]


@pytest.mark.asyncio
async def test_delete_survives_pty_termination_errors() -> None:
    client = StrixDockerSandboxClient.__new__(StrixDockerSandboxClient)
    client.docker_client = MagicMock()
    session = _session(pty_terminate_all=AsyncMock(side_effect=RuntimeError("daemon gone")))

    with patch.object(
        DockerSandboxClient, "delete", new=AsyncMock(return_value=session)
    ) as super_delete:
        await client.delete(session)

    super_delete.assert_awaited_once()
