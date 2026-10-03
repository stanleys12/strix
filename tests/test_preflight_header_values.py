from __future__ import annotations

import asyncio

import pytest

from strix.config import Settings
from strix.interface.scan_setup import check_header_safe_credentials, preflight_model_connection


_ENV_KEYS = ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_EXTRA_HEADERS", "LLM_API_BASE", "STRIX_LLM")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_ascii_credentials_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_API_KEY="sk-plain-ascii",
        LLM_EXTRA_HEADERS='{"X-Team": "security"}',
    )

    check_header_safe_credentials(settings)


def test_smart_quote_in_api_key_is_named_without_leaking_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, LLM_API_KEY="sk-abc\u201d")

    with pytest.raises(ValueError, match="LLM_API_KEY") as excinfo:
        check_header_safe_credentials(settings)

    message = str(excinfo.value)
    assert "U+201D" in message
    assert "RIGHT DOUBLE QUOTATION MARK" in message
    assert "position 7 of 7" in message
    assert "sk-abc" not in message


@pytest.mark.parametrize("char", ["\u00a0", "\ufeff", "\u200b"])
def test_invisible_characters_are_reported(monkeypatch: pytest.MonkeyPatch, char: str) -> None:
    settings = _settings(monkeypatch, LLM_API_KEY=f"{char}sk-abc")

    with pytest.raises(ValueError, match=f"U\\+{ord(char):04X}"):
        check_header_safe_credentials(settings)


def test_extra_header_value_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_API_KEY="sk-plain",
        LLM_EXTRA_HEADERS='{"X-Team": "s\u00e9curit\u00e9"}',
    )

    with pytest.raises(ValueError, match=r"LLM_EXTRA_HEADERS value for 'X-Team'.*U\+00E9"):
        check_header_safe_credentials(settings)


def test_preflight_rejects_the_key_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        LLM_API_KEY="sk-abc\u201d",
        LLM_API_BASE="http://127.0.0.1:9/v1",
    )

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        asyncio.run(preflight_model_connection("openai/gpt-4o-mini", settings=settings))
