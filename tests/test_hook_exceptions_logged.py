"""Nothing from sys.unraisablehook or threading.excepthook reaches the terminal.

Python prints "Exception ignored in ..." reports (finalizers, ``__del__``, GC
and weakref callbacks, files closed at interpreter exit) and uncaught thread
exceptions straight to stderr. Strix routes both to its own logger instead,
so they end up in strix.log (and on stderr only under STRIX_DEBUG=1).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from strix.telemetry import logging as tlog


if TYPE_CHECKING:
    from collections.abc import Iterator


def _unraisable(exc: BaseException, obj: object, err_msg: str | None = None) -> object:
    return SimpleNamespace(
        exc_type=type(exc),
        exc_value=exc,
        exc_traceback=None,
        err_msg=err_msg,
        object=obj,
    )


@pytest.fixture
def strix_records() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect()
    root = logging.getLogger("strix")
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


@pytest.fixture
def hooks(monkeypatch: pytest.MonkeyPatch) -> tuple[list[object], list[object]]:
    """Install strix's hooks over recording stand-ins for Python's defaults."""
    unraisable_calls: list[object] = []
    thread_calls: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable_calls.append)
    monkeypatch.setattr(threading, "excepthook", thread_calls.append)
    monkeypatch.setattr(tlog, "_hooks_installed", False)
    tlog.configure_dependency_logging()
    return unraisable_calls, thread_calls


@pytest.mark.parametrize(
    "args",
    [
        _unraisable(ValueError("I/O operation on closed file."), object()),
        _unraisable(
            ValueError("I/O operation on closed file."),
            None,
            err_msg=(
                "Exception ignored while finalizing file "
                "<urllib3.response.HTTPResponse object at 0x1>"
            ),
        ),
        _unraisable(RuntimeError("boom"), object()),
        _unraisable(KeyError("x"), None, err_msg="Exception ignored in: <function f>"),
    ],
)
def test_every_unraisable_is_logged_not_printed(
    args: object,
    hooks: tuple[list[object], list[object]],
    strix_records: list[logging.LogRecord],
    capsys: pytest.CaptureFixture[str],
) -> None:
    unraisable_calls, _ = hooks

    sys.unraisablehook(args)  # type: ignore[arg-type]

    assert unraisable_calls == []
    assert capsys.readouterr().err == ""
    [record] = strix_records
    assert record.levelno == logging.WARNING
    assert record.name == "strix.telemetry"
    message = record.getMessage()
    if args.err_msg:  # type: ignore[attr-defined]
        assert message.startswith(args.err_msg)  # type: ignore[attr-defined]
    else:
        assert message.startswith("Exception ignored in <object object")
    assert str(args.exc_value) in message  # type: ignore[attr-defined]


@pytest.mark.usefixtures("hooks")
def test_unraisable_is_dropped_when_strix_has_no_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After scan teardown the strix logger tree has no handlers; the report
    must be dropped rather than handed to logging.lastResort (stderr)."""
    root = logging.getLogger("strix")
    saved, root.handlers = root.handlers, []
    try:
        sys.unraisablehook(_unraisable(RuntimeError("late"), object()))  # type: ignore[arg-type]
    finally:
        root.handlers = saved
    assert capsys.readouterr().err == ""


def test_thread_exceptions_are_logged_not_printed(
    hooks: tuple[list[object], list[object]],
    strix_records: list[logging.LogRecord],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, thread_calls = hooks

    def _boom() -> None:
        raise RuntimeError("worker failed")

    thread = threading.Thread(target=_boom, name="strix-worker")
    thread.start()
    thread.join()

    assert thread_calls == []
    assert capsys.readouterr().err == ""
    [record] = strix_records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith("Exception in thread strix-worker")
    assert "RuntimeError: worker failed" in message


@pytest.mark.usefixtures("hooks")
def test_thread_system_exit_is_ignored(strix_records: list[logging.LogRecord]) -> None:
    thread = threading.Thread(target=sys.exit, args=(3,))
    thread.start()
    thread.join()
    assert strix_records == []


def test_hooks_install_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tlog, "_hooks_installed", False)
    tlog.configure_dependency_logging()
    installed = (sys.unraisablehook, threading.excepthook)
    tlog.configure_dependency_logging()
    assert (sys.unraisablehook, threading.excepthook) == installed


_EXIT_SCRIPT = r"""
import logging, sys, threading
from strix.telemetry.logging import setup_console_logging
setup_console_logging()

class Leaky:
    def __del__(self):
        raise ValueError("I/O operation on closed file.")

Leaky()                      # collected right away
keep = Leaky()               # collected at interpreter shutdown

def worker():
    raise RuntimeError("worker failed")
t = threading.Thread(target=worker); t.start(); t.join()

import urllib3.response, io
resp = urllib3.response.HTTPResponse(body=io.BytesIO(b""), preload_content=False)
resp._fp.close()             # socket file closed before the response: the 3.14 exit noise
del resp
print("done")
"""


def test_process_exit_prints_nothing_to_stderr() -> None:
    """End to end in a fresh interpreter: finalizer errors (including one at
    interpreter shutdown), a thread traceback and the closed-file response
    case produce no stderr output, with the default (non-debug) handlers."""
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _EXIT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        env={**_safe_env(), **env, "STRIX_DEBUG": ""},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "done"
    assert proc.stderr == ""


def _safe_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "SYSTEMROOT", "TEMP", "TMP"}
    }
