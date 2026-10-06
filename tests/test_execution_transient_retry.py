from __future__ import annotations

from collections import Counter
from typing import Any

import httpx
import pytest
from agents import RunConfig, Runner
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from strix.config import codex
from strix.core import execution
from strix.core.agents import AgentCoordinator
from strix.llm import provider_bans, request_log


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def _midstream_api_error() -> APIError:
    return APIError("An error occurred while processing the request.", _request(), body=None)


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        f"status {status}",
        response=httpx.Response(status_code=status, request=_request()),
        body=None,
    )


def test_midstream_api_error_is_transient() -> None:
    assert execution._is_transient_model_error(_midstream_api_error()) is True


def test_network_errors_are_transient() -> None:
    assert execution._is_transient_model_error(APITimeoutError(_request())) is True
    assert execution._is_transient_model_error(APIConnectionError(request=_request())) is True


def test_server_errors_are_transient() -> None:
    assert (
        execution._is_transient_model_error(
            InternalServerError("boom", response=httpx.Response(500, request=_request()), body=None)
        )
        is True
    )
    for status in (502, 503, 504, 408):
        assert execution._is_transient_model_error(_status_error(status)) is True


def test_rate_limit_is_retried() -> None:
    rate_limited = RateLimitError(
        "slow down", response=httpx.Response(429, request=_request()), body=None
    )
    assert execution._is_transient_model_error(rate_limited) is True


def test_dns_and_connection_errors_are_transient() -> None:
    assert execution._is_transient_model_error(OSError("nodename nor servname provided")) is True
    assert execution._is_transient_model_error(ConnectionError("reset")) is True
    assert execution._is_transient_model_error(TimeoutError("timed out")) is True


def test_content_guardrail_is_not_retried() -> None:
    guardrail = APIError(
        "This content was flagged for possible cybersecurity risk",
        _request(),
        body=None,
    )
    assert codex.is_content_guardrail_error(guardrail) is True
    assert execution._is_transient_model_error(guardrail) is False


def _upstream_error(
    status: int, provider: str, *, error_type: str | None = None, rejected: bool = False
) -> APIStatusError:
    exc = _status_error(status)
    reply = request_log.HttpReply(
        upstream_provider=provider, upstream_error_type=error_type, upstream_rejected=rejected
    )
    setattr(exc, request_log._FAILED_REPLY_ATTR, reply)
    return exc


def test_client_errors_are_transient() -> None:
    bad_request = BadRequestError(
        "bad", response=httpx.Response(400, request=_request()), body=None
    )
    assert execution._is_transient_model_error(bad_request) is True
    assert execution._is_transient_model_error(_status_error(404)) is True
    assert execution._is_transient_model_error(ValueError("nope")) is False


def test_gateway_auth_credit_and_policy_errors_are_not_transient() -> None:
    for status in (401, 402, 403):
        assert execution._is_transient_model_error(_status_error(status)) is False
    assert execution._is_transient_model_error(_upstream_error(401, "Relace")) is True


@pytest.fixture
def bans(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []

    def _ban(model: str, provider: str, reason: str) -> None:
        calls.append((model, provider, reason))

    monkeypatch.setattr(provider_bans, "ban_provider", _ban)
    monkeypatch.setattr(execution, "_provider_4xx_counts", Counter())
    return calls


_OPENROUTER_RUN_CONFIG = RunConfig(model="openrouter/z-ai/glm-5.3")


@pytest.mark.asyncio
async def test_bans_upstream_right_away_on_provider_unavailable_or_rejection(
    bans: list[tuple[str, str, str]],
) -> None:
    unavailable = _upstream_error(502, "Relace", error_type="provider_unavailable")
    await execution._maybe_ban_provider(unavailable, _OPENROUTER_RUN_CONFIG)
    rejected = _upstream_error(400, "InferenceNet", rejected=True)
    await execution._maybe_ban_provider(rejected, _OPENROUTER_RUN_CONFIG)
    assert bans == [
        ("openrouter/z-ai/glm-5.3", "Relace", "provider_unavailable"),
        ("openrouter/z-ai/glm-5.3", "InferenceNet", "upstream_rejected_400"),
    ]


@pytest.mark.asyncio
async def test_bans_upstream_after_repeated_mid_stream_4xx(
    bans: list[tuple[str, str, str]],
) -> None:
    for _ in range(2):
        await execution._maybe_ban_provider(_upstream_error(400, "Relace"), _OPENROUTER_RUN_CONFIG)
        await execution._maybe_ban_provider(
            _upstream_error(504, "Together"), _OPENROUTER_RUN_CONFIG
        )
    await execution._maybe_ban_provider(_status_error(400), _OPENROUTER_RUN_CONFIG)
    assert bans == [("openrouter/z-ai/glm-5.3", "Relace", "repeated_4xx")]


class _FakeStream:
    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc
        self._events: list[Any] = []
        self.run_loop_exception: BaseException | None = None

    async def stream_events(self) -> Any:
        if self._exc is not None:
            raise self._exc
        for event in self._events:
            yield event


def _patch_fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_MAX_DELAY_S", 0.0)


async def _run_once(
    monkeypatch: pytest.MonkeyPatch,
    streams: list[_FakeStream],
) -> Any:
    _patch_fast_backoff(monkeypatch)
    calls = {"n": 0}

    def _fake_run_streamed(*_args: Any, **_kwargs: Any) -> _FakeStream:
        stream = streams[calls["n"]]
        calls["n"] += 1
        return stream

    monkeypatch.setattr(Runner, "run_streamed", _fake_run_streamed)

    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)

    result = await execution._run_cycle(
        object(),
        coordinator,
        "root",
        input_data="task",
        run_config=RunConfig(),
        context={},
        max_turns=5,
        session=None,
        interactive=False,
        event_sink=None,
        hooks=None,
    )
    return result, calls["n"], coordinator


@pytest.mark.asyncio
async def test_run_cycle_retries_transient_midstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = [_FakeStream(exc=_midstream_api_error()), _FakeStream()]
    result, attempts, _coordinator = await _run_once(monkeypatch, streams)

    assert result is streams[1]
    assert attempts == 2


@pytest.mark.asyncio
async def test_run_cycle_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = [
        _FakeStream(exc=_midstream_api_error())
        for _ in range(execution._MAX_TRANSIENT_MODEL_RETRIES + 1)
    ]
    with pytest.raises(APIError):
        await _run_once(monkeypatch, streams)


@pytest.mark.asyncio
async def test_run_cycle_does_not_retry_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = [_FakeStream(exc=_status_error(401)), _FakeStream()]
    with pytest.raises(APIStatusError):
        await _run_once(monkeypatch, streams)
