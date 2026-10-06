"""Tests for the model-stream idle watchdog.

A turn that streams a few tokens and then goes silent is not covered by the
request timeout: the read timeout resets on every byte, keepalives included.
The watchdog bounds the gap between events so the turn fails and can be
retried instead of parking the agent forever.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

import pytest
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from strix.config import loader
from strix.config.loader import load_settings
from strix.config.models import StrixProvider, _TurnGuardModel, _with_timeouts
from strix.llm import request_log


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


_STALL_SECONDS = 30.0


def _chunk(text: str) -> bytes:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gw-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


class _StallingHandler(BaseHTTPRequestHandler):
    """Streams a couple of tokens, then stops producing anything."""

    stop = threading.Event()

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(_chunk("Now"))
        self.wfile.write(_chunk(" spawning"))
        self.wfile.flush()
        self.stop.wait(_STALL_SECONDS)


@pytest.fixture
def stalling_gateway() -> Iterator[str]:
    _StallingHandler.stop.clear()
    server = HTTPServer(("127.0.0.1", 0), _StallingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        _StallingHandler.stop.set()
        server.shutdown()
        server.server_close()


def _stream(base_url: str, *, idle_timeout: float) -> AsyncIterator[Any]:
    client = AsyncOpenAI(api_key="tok", base_url=base_url, max_retries=0, timeout=_STALL_SECONDS)
    inner: Model = request_log.RequestLoggingModel(
        OpenAIChatCompletionsModel(model="gw-model", openai_client=client),
        model_name="gw-model",
        provider="openai",
        base_url=base_url,
    )
    guarded = _TurnGuardModel(inner, stream_idle_timeout=idle_timeout)
    return _call(guarded)


def _call(model: Model) -> AsyncIterator[Any]:
    return model.stream_response(
        None,
        "go",
        ModelSettings(),
        [],
        None,
        [],
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )


async def _drain(base_url: str, *, idle_timeout: float) -> list[Any]:
    return [event async for event in _stream(base_url, idle_timeout=idle_timeout)]


@pytest.mark.asyncio
async def test_stalled_stream_hangs_without_the_watchdog(stalling_gateway: str) -> None:
    # Repro: tokens arrive, then nothing. Un-watched, the turn just sits there;
    # the request timeout is far away and would reset on any keepalive byte.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_drain(stalling_gateway, idle_timeout=0), timeout=2)


@pytest.mark.asyncio
async def test_stalled_stream_is_abandoned_by_the_watchdog(stalling_gateway: str) -> None:
    logged: list[request_log.LlmRequestEvent] = []
    request_log.register_sink(logged.append)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="stream_idle_timeout"):
            await _drain(stalling_gateway, idle_timeout=1)
    finally:
        request_log.unregister_sink(logged.append)

    assert time.monotonic() - started < _STALL_SECONDS
    # The request log can tell our timeout from a shutdown.
    assert [event.error_message for event in logged] == [
        "attempt cancelled before the reply was consumed: strix:stream_idle_timeout"
    ]


@pytest.mark.asyncio
async def test_events_keep_flowing_while_the_stream_is_alive() -> None:
    async def _live() -> AsyncIterator[Any]:
        for i in range(5):
            await asyncio.sleep(0.05)
            yield f"event-{i}"

    seen: list[Any] = [event async for event in _with_timeouts(_live(), idle=1.0)]

    assert seen == [f"event-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_first_event_and_whole_stream_are_bounded() -> None:
    async def _slow_start() -> AsyncIterator[Any]:
        await asyncio.sleep(_STALL_SECONDS)
        yield "late"

    async def _endless() -> AsyncIterator[Any]:
        while True:
            await asyncio.sleep(0.05)
            yield "event"

    with pytest.raises(TimeoutError, match="stream_first_event_timeout"):
        [event async for event in _with_timeouts(_slow_start(), idle=5, first_event=0.2)]
    with pytest.raises(TimeoutError, match="stream_total_timeout"):
        [event async for event in _with_timeouts(_endless(), idle=5, total=0.3)]


class _UpstreamInner(Model):
    """Records the upstream provider like the OpenRouter chunk parser, then fails."""

    def __init__(self, provider: str | None, error_type: str | None = None) -> None:
        self._provider = provider
        self._error_type = error_type

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def stream_response(self, *_: Any, **__: Any) -> AsyncIterator[Any]:
        if self._error_type:
            request_log.record_upstream_provider(self._provider, self._error_type)
            raise RuntimeError("upstream error chunk")
        await asyncio.sleep(0.05)
        request_log.record_upstream_provider(self._provider)
        yield "event"
        await asyncio.sleep(_STALL_SECONDS)


@pytest.fixture
def bans(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "strix.llm.provider_bans.ban_provider", lambda *args: calls.append(args) or True
    )
    return calls


async def _drain_guarded(inner: Model, **timeouts: float) -> None:
    logging_model = request_log.RequestLoggingModel(
        inner, model_name="openrouter/z-ai/glm-5.3", provider="openrouter", base_url=None
    )
    guarded = _TurnGuardModel(logging_model, model_name="openrouter/z-ai/glm-5.3", **timeouts)
    [event async for event in _call(guarded)]


@pytest.mark.asyncio
async def test_stream_timeout_bans_the_provider_that_served_it(
    bans: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(TimeoutError):
        await _drain_guarded(_UpstreamInner("InferenceNet"), stream_idle_timeout=0.2)

    assert bans == [("openrouter/z-ai/glm-5.3", "InferenceNet", "stream_idle_timeout")]


@pytest.mark.asyncio
async def test_openrouter_upstream_timeout_bans_the_provider(
    bans: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(RuntimeError):
        await _drain_guarded(_UpstreamInner("InferenceNet", "timeout"))
    with pytest.raises(RuntimeError):
        await _drain_guarded(_UpstreamInner("InferenceNet", "provider_unavailable"))

    assert bans == [("openrouter/z-ai/glm-5.3", "InferenceNet", "upstream_timeout")]


@pytest.mark.asyncio
async def test_timeout_without_a_known_provider_bans_nothing(
    bans: list[tuple[str, str, str]],
) -> None:
    with pytest.raises(TimeoutError, match="stream_first_event_timeout"):
        await _drain_guarded(_UpstreamInner(None), first_event_timeout=0.01)

    assert bans == []


@pytest.fixture
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in (
        "STRIX_LLM",
        "LLM_DISABLE_STREAMING",
        "LLM_STREAM_IDLE_TIMEOUT",
        "LLM_STREAM_FIRST_EVENT_TIMEOUT",
        "LLM_STREAM_TOTAL_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(loader, "_cached", None)
    monkeypatch.setattr(loader, "_override", None)
    yield


class _DummyModel(Model):
    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_idle_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setattr("strix.config.models.MultiProvider.get_model", lambda *_: _DummyModel())
    monkeypatch.setenv("LLM_STREAM_IDLE_TIMEOUT", "45")
    monkeypatch.setenv("LLM_STREAM_FIRST_EVENT_TIMEOUT", "30")
    monkeypatch.setenv("LLM_STREAM_TOTAL_TIMEOUT", "400")
    load_settings()

    model = StrixProvider().get_model("openai/gpt-4o-mini")
    assert isinstance(model, _TurnGuardModel)
    assert model._stream_idle_timeout == 45
    assert model._first_event_timeout == 30
    assert model._total_timeout == 400


def test_idle_timeout_is_off_without_streaming(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    # LLM_DISABLE_STREAMING turns the whole request into one event, so an idle
    # gap would just be the request duration — the request timeout bounds that.
    monkeypatch.setattr("strix.config.models.MultiProvider.get_model", lambda *_: _DummyModel())
    monkeypatch.setenv("LLM_STREAM_IDLE_TIMEOUT", "45")
    monkeypatch.setenv("LLM_DISABLE_STREAMING", "true")
    load_settings()

    model = StrixProvider().get_model("openai/gpt-4o-mini")
    assert isinstance(model, _TurnGuardModel)
    assert model._stream_idle_timeout == model._first_event_timeout == model._total_timeout == 0
