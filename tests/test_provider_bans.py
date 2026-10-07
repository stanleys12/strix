"""Tests for per-run OpenRouter provider bans."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx
import pytest

from strix.llm import provider_bans


if TYPE_CHECKING:
    from collections.abc import Iterator


MODEL = "openrouter/z-ai/glm-5.3"
ENDPOINTS = [
    {"provider_name": "Relace", "tag": "relace"},
    {"provider_name": "InferenceNet", "tag": "inference-net/fp4"},
    {"provider_name": "Google", "tag": "google-vertex/us-east5"},
    {"provider_name": "Google", "tag": "google-vertex"},
]


@pytest.fixture(autouse=True)
def endpoints() -> Iterator[MagicMock]:
    for state in (provider_bans._SLUGS, provider_bans._BANNED):
        state.clear()
    response = httpx.Response(
        200,
        json={"data": {"endpoints": ENDPOINTS}},
        request=httpx.Request("GET", "https://openrouter.ai"),
    )
    with patch("strix.llm.provider_bans.httpx.get", return_value=response) as get:
        yield get


def test_ban_maps_provider_names_to_slugs(endpoints: MagicMock) -> None:
    provider_bans.ban_provider(MODEL, "InferenceNet", "context_length")
    provider_bans.ban_provider("z-ai/glm-5.3", "google", "provider_unavailable")
    provider_bans.ban_provider(MODEL, "InferenceNet", "context_length")

    assert provider_bans.banned_providers(MODEL) == ["google-vertex", "inference-net"]
    assert provider_bans.banned_providers("openrouter/other/model") == []
    endpoints.assert_called_once_with(
        "https://openrouter.ai/api/v1/models/z-ai/glm-5.3/endpoints", timeout=5
    )


def test_banning_every_provider_unbans_them_all() -> None:
    provider_bans.ban_provider(MODEL, "Relace", "incomplete_tool_call")
    provider_bans.ban_provider(MODEL, "InferenceNet", "context_length")
    assert provider_bans.banned_providers(MODEL) == ["inference-net", "relace"]
    provider_bans.ban_provider(MODEL, "Google", "provider_unavailable")

    assert provider_bans.banned_providers(MODEL) == []


def test_ban_is_skipped_for_an_unknown_provider_or_failed_lookup(endpoints: MagicMock) -> None:
    provider_bans.ban_provider(MODEL, "Nobody", "provider_unavailable")
    endpoints.side_effect = httpx.ConnectError("offline")
    provider_bans.ban_provider("openrouter/other/model", "Relace", "x")

    assert provider_bans.banned_providers(MODEL) == []


def test_bans_can_be_turned_off(endpoints: MagicMock) -> None:
    with patch("strix.llm.provider_bans.load_settings") as settings:
        settings.return_value.llm.openrouter_provider_bans = False
        provider_bans.ban_provider(MODEL, "Relace", "incomplete_tool_call")

    assert provider_bans.banned_providers(MODEL) == []
    endpoints.assert_not_called()
