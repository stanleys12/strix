"""Docker endpoint resolution and the diagnostics printed when the daemon is unreachable."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from docker.errors import DockerException
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError

from strix.runtime import backends, docker_connection
from strix.runtime.docker_connection import (
    CONNECTION_REFUSED,
    PERMISSION_DENIED,
    SOCKET_MISSING,
    UNKNOWN,
    DockerConnectionError,
    DockerEndpoint,
    classify_failure,
    explain_failure,
    resolve_docker_endpoint,
)


cli_utils: Any = importlib.import_module("strix.interface.utils")


def _sdk_error(inner: BaseException) -> DockerException:
    """Build the exception exactly as docker-py raises it for a dead daemon."""
    protocol = ProtocolError("Connection aborted.", inner)
    requests_exc = RequestsConnectionError(protocol)
    requests_exc.__cause__ = protocol
    sdk_exc = DockerException(f"Error while fetching server API version: {requests_exc}")
    sdk_exc.__cause__ = requests_exc
    return sdk_exc


def test_docker_host_wins_over_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_connection, "get_current_context_name", lambda: "desktop-linux")
    endpoint = resolve_docker_endpoint({"DOCKER_HOST": "tcp://10.0.0.5:2375"})
    assert endpoint == DockerEndpoint("tcp://10.0.0.5:2375", "DOCKER_HOST")


def test_current_context_is_used_like_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_connection, "get_current_context_name", lambda: "desktop-linux")
    monkeypatch.setattr(
        "strix.runtime.docker_connection.ContextAPI.get_context",
        lambda _name: SimpleNamespace(
            Host="unix:///Users/me/.docker/run/docker.sock", TLSConfig=None
        ),
    )
    endpoint = resolve_docker_endpoint({})
    assert endpoint.host == "unix:///Users/me/.docker/run/docker.sock"
    assert endpoint.source == "docker context 'desktop-linux'"


def test_docker_context_env_overrides_the_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_connection, "get_current_context_name", lambda: "default")
    seen: list[str] = []

    def get_context(name: str) -> SimpleNamespace:
        seen.append(name)
        return SimpleNamespace(Host="unix:///run/user/1000/orbstack.sock", TLSConfig=None)

    monkeypatch.setattr("strix.runtime.docker_connection.ContextAPI.get_context", get_context)
    endpoint = resolve_docker_endpoint({"DOCKER_CONTEXT": "orbstack"})
    assert seen == ["orbstack"]
    assert endpoint.source == "docker context 'orbstack'"


def test_default_context_and_broken_context_fall_back_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_connection, "get_current_context_name", lambda: "default")
    assert resolve_docker_endpoint({}) == DockerEndpoint(None, "default socket")

    monkeypatch.setattr(docker_connection, "get_current_context_name", lambda: "gone")

    def boom(_name: str) -> SimpleNamespace:
        raise ValueError("bad meta.json")

    monkeypatch.setattr("strix.runtime.docker_connection.ContextAPI.get_context", boom)
    assert resolve_docker_endpoint({}) == DockerEndpoint(None, "default socket")


@pytest.mark.parametrize(
    ("inner", "reason"),
    [
        (FileNotFoundError(2, "No such file or directory"), SOCKET_MISSING),
        (PermissionError(13, "Permission denied"), PERMISSION_DENIED),
        (ConnectionRefusedError(111, "Connection refused"), CONNECTION_REFUSED),
    ],
)
def test_classifies_the_wrapped_os_error(inner: OSError, reason: str) -> None:
    exc = _sdk_error(inner)
    assert classify_failure(exc) == reason
    assert docker_connection.root_cause_line(exc).startswith(type(inner).__name__)


def test_classifies_windows_named_pipe_errors() -> None:
    class PyWinError(Exception):
        def __init__(self, winerror: int) -> None:
            super().__init__(winerror, "CreateFile", "The system cannot find the file specified.")
            self.winerror = winerror

    assert classify_failure(_sdk_error(PyWinError(2))) == SOCKET_MISSING
    assert classify_failure(_sdk_error(PyWinError(5))) == PERMISSION_DENIED


def test_falls_back_to_the_message_text_then_unknown() -> None:
    assert (
        classify_failure(
            DockerException("Error while fetching server API version: Permission denied")
        )
        == PERMISSION_DENIED
    )
    assert classify_failure(DockerException("something else entirely")) == UNKNOWN


@pytest.mark.parametrize(
    ("platform", "needle"),
    [
        ("darwin", "docker context use desktop-linux"),
        ("win32", "Start Docker Desktop"),
        ("linux", "systemctl start docker"),
    ],
)
def test_not_running_fix_is_per_platform(platform: str, needle: str) -> None:
    error = DockerConnectionError(
        DockerEndpoint(None, "default socket"), SOCKET_MISSING, FileNotFoundError(2, "x")
    )
    cause, fix = explain_failure(error, platform=platform)
    assert "not running" in cause
    assert needle in fix


def test_permission_fix_names_the_docker_group_on_linux() -> None:
    error = DockerConnectionError(
        DockerEndpoint(None, "default socket"), PERMISSION_DENIED, PermissionError(13, "x")
    )
    assert "usermod -aG docker" in explain_failure(error, platform="linux")[1]
    assert "usermod" not in explain_failure(error, platform="darwin")[1]


def test_explicit_endpoint_failures_name_their_source() -> None:
    missing = DockerConnectionError(
        DockerEndpoint("unix:///x.sock", "docker context 'colima'"),
        SOCKET_MISSING,
        FileNotFoundError(2, "x"),
    )
    assert explain_failure(missing, platform="darwin")[0].startswith("docker context 'colima'")

    refused = DockerConnectionError(
        DockerEndpoint("tcp://127.0.0.1:1", "DOCKER_HOST"),
        CONNECTION_REFUSED,
        ConnectionRefusedError(111, "x"),
    )
    cause, fix = explain_failure(refused, platform="linux")
    assert "tcp://127.0.0.1:1" in cause
    assert "DOCKER_HOST" in fix


def test_connect_docker_raises_a_classified_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        docker_connection,
        "resolve_docker_endpoint",
        lambda _environ=None: DockerEndpoint("unix:///nope.sock", "DOCKER_HOST"),
    )

    def dead_client(**_kwargs: Any) -> None:
        raise _sdk_error(FileNotFoundError(2, "No such file or directory"))

    monkeypatch.setattr("strix.runtime.docker_connection.docker.DockerClient", dead_client)

    with pytest.raises(DockerConnectionError) as info:
        docker_connection.connect_docker()

    assert info.value.reason == SOCKET_MISSING
    assert info.value.endpoint.host == "unix:///nope.sock"
    assert "FileNotFoundError" in info.value.detail


def test_check_docker_connection_prints_the_fix_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reported: list[tuple[str, type | None]] = []
    monkeypatch.setattr(
        cli_utils, "report_error", lambda name, exc=None: reported.append((name, type(exc)))
    )
    error = DockerConnectionError(
        DockerEndpoint(None, "default socket"),
        PERMISSION_DENIED,
        _sdk_error(PermissionError(13, "Permission denied")),
    )

    def failing_connect() -> None:
        raise error

    monkeypatch.setattr(docker_connection, "connect_docker", failing_connect)
    monkeypatch.setattr("strix.runtime.docker_connection.sys.platform", "linux")

    with pytest.raises(SystemExit) as exit_info:
        cli_utils.check_docker_connection()

    out = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert reported == [("docker_unavailable_permission_denied", DockerException)]
    assert "DOCKER NOT AVAILABLE" in out
    assert "usermod -aG docker" in out
    assert "Tried: default socket (default socket)" in out
    assert "PermissionError" in out


def test_check_docker_connection_returns_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object()
    monkeypatch.setattr(docker_connection, "connect_docker", lambda: client)
    assert cli_utils.check_docker_connection() is client


@pytest.mark.asyncio
async def test_sandbox_backend_uses_the_same_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = object()
    monkeypatch.setattr(docker_connection, "connect_docker", lambda: resolved)
    captured: dict[str, Any] = {}

    class FakeSession:
        async def start(self) -> None:
            captured["started"] = True

    class FakeClient:
        def __init__(self, docker_client: Any) -> None:
            captured["docker_client"] = docker_client

        async def create(self, *, options: Any, **_kwargs: Any) -> FakeSession:
            captured["image"] = options.image
            return FakeSession()

    monkeypatch.setattr("strix.runtime.docker_client.StrixDockerSandboxClient", FakeClient)

    client, session = await backends._docker_backend(
        image="img:1", manifest=cast("Any", SimpleNamespace()), exposed_ports=(8080,)
    )

    assert captured["docker_client"] is resolved
    assert captured["image"] == "img:1"
    assert captured["started"] is True
    assert isinstance(client, FakeClient)
    assert isinstance(session, FakeSession)
