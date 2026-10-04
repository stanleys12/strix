import http.client
import http.cookiejar
import socket
import sys

import pytest
import urllib3.response

from strix.telemetry import logging as tlog
from strix.telemetry.logging import _is_urllib3_closed_file_noise


class _Args:
    def __init__(
        self, exc_value: BaseException | None, obj: object, err_msg: str | None = None
    ) -> None:
        self.exc_type = type(exc_value) if exc_value is not None else None
        self.exc_value = exc_value
        self.exc_traceback = None
        self.err_msg = err_msg
        self.object = obj


def _urllib3_response() -> urllib3.response.HTTPResponse:
    return urllib3.response.HTTPResponse(body=b"")


def test_filters_urllib3_closed_file_noise() -> None:
    args = _Args(ValueError("I/O operation on closed file."), _urllib3_response())
    assert _is_urllib3_closed_file_noise(args)  # type: ignore[arg-type]


def test_ignores_closed_file_errors_from_other_http_modules() -> None:
    args = _Args(ValueError("I/O operation on closed file."), http.cookiejar.CookieJar())
    assert not _is_urllib3_closed_file_noise(args)  # type: ignore[arg-type]


def test_filters_http_client_closed_file_noise() -> None:
    sock = socket.socket()
    try:
        response = http.client.HTTPResponse(sock)
    finally:
        sock.close()
    args = _Args(ValueError("I/O operation on closed file."), response)
    assert _is_urllib3_closed_file_noise(args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "repr_text",
    [
        "<urllib3.response.HTTPResponse object at 0x7f2d3c1f0b10>",
        "<http.client.HTTPResponse object at 0x7f2d3c1f0b10>",
    ],
)
def test_filters_python314_finalizer_shape(repr_text: str) -> None:
    """3.14 reports IOBase finalizer failures with object=None and the repr in err_msg."""
    args = _Args(
        ValueError("I/O operation on closed file."),
        None,
        err_msg=f"Exception ignored while finalizing file {repr_text}",
    )
    assert _is_urllib3_closed_file_noise(args)  # type: ignore[arg-type]


def test_python314_shape_passes_through_other_files() -> None:
    args = _Args(
        ValueError("I/O operation on closed file."),
        None,
        err_msg="Exception ignored while finalizing file <_io.TextIOWrapper name='x' mode='w'>",
    )
    assert not _is_urllib3_closed_file_noise(args)  # type: ignore[arg-type]
    assert not _is_urllib3_closed_file_noise(
        _Args(ValueError("I/O operation on closed file."), None)  # type: ignore[arg-type]
    )


def test_passes_through_other_unraisables() -> None:
    assert not _is_urllib3_closed_file_noise(
        _Args(ValueError("I/O operation on closed file."), object())  # type: ignore[arg-type]
    )
    assert not _is_urllib3_closed_file_noise(
        _Args(RuntimeError("boom"), _urllib3_response())  # type: ignore[arg-type]
    )
    assert not _is_urllib3_closed_file_noise(
        _Args(ValueError("something else"), _urllib3_response())  # type: ignore[arg-type]
    )


def test_installed_hook_filters_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", calls.append)
    monkeypatch.setattr(tlog, "_unraisable_hook_installed", False)
    tlog._silence_urllib3_finalizer_noise()
    hook = sys.unraisablehook
    assert hook is not calls.append

    hook(_Args(ValueError("I/O operation on closed file."), _urllib3_response()))  # type: ignore[arg-type]
    hook(
        _Args(  # type: ignore[arg-type]
            ValueError("I/O operation on closed file."),
            None,
            err_msg=(
                "Exception ignored while finalizing file "
                "<urllib3.response.HTTPResponse object at 0x1>"
            ),
        )
    )
    assert calls == []

    other = _Args(RuntimeError("boom"), object())
    hook(other)  # type: ignore[arg-type]
    assert calls == [other]
