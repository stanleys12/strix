"""Per-run bans on the OpenRouter upstream providers behind a model."""

from __future__ import annotations

import logging
import threading

import httpx

from strix.config.loader import load_settings
from strix.telemetry.logging import current_scan_id


logger = logging.getLogger(__name__)

_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"

_LOCK = threading.Lock()
# OpenRouter reports display names ("InferenceNet"); provider.ignore takes slugs ("inference-net").
_SLUGS: dict[str, dict[str, str]] = {}
_BANNED: dict[str, set[str]] = {}
_BAN_COUNTS: dict[str, int] = {}


def _model_id(model: str) -> str:
    model = model.strip()
    for prefix in ("litellm/", "openrouter/"):
        model = model.removeprefix(prefix)
    return model


def _provider_slugs(model: str) -> dict[str, str]:
    if model not in _SLUGS:
        response = httpx.get(_ENDPOINTS_URL.format(model=model), timeout=5)
        response.raise_for_status()
        slugs: dict[str, str] = {}
        for endpoint in response.json()["data"]["endpoints"]:
            if not (tag := endpoint.get("tag")):
                continue
            # A tag is the provider slug plus an optional variant, e.g. "inference-net/fp4".
            slug = str(tag).split("/")[0]
            slugs[slug] = slug
            slugs[str(endpoint.get("provider_name") or slug).lower()] = slug
        _SLUGS[model] = slugs
    return _SLUGS[model]


def ban_provider(model: str, provider: str, reason: str) -> bool:
    """Ban for the rest of the run; ``provider`` may be a display name or slug."""
    if not load_settings().llm.openrouter_provider_bans:
        return False
    model = _model_id(model)
    try:
        slugs = _provider_slugs(model)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "provider_ban_skipped upstream=%s model=%s reason=%s scan_id=%s: "
            "could not list the model's providers: %s",
            provider,
            model,
            reason,
            current_scan_id(),
            exc,
        )
        return False
    slug = slugs.get(provider.strip().lower())
    if slug is None:
        logger.warning(
            "provider_ban_skipped upstream=%s model=%s reason=%s scan_id=%s: unknown provider",
            provider,
            model,
            reason,
            current_scan_id(),
        )
        return False
    with _LOCK:
        banned = _BANNED.setdefault(model, set())
        if slug in banned:
            return True
        banned.add(slug)
        _BAN_COUNTS[model] = _BAN_COUNTS.get(model, 0) + 1
        logger.warning(
            "provider_banned upstream=%s slug=%s model=%s reason=%s scan_id=%s",
            provider,
            slug,
            model,
            reason,
            current_scan_id(),
        )
        if banned >= set(slugs.values()):
            banned.clear()
            logger.warning(
                "provider_bans_reset model=%s scan_id=%s: every provider was banned",
                model,
                current_scan_id(),
            )
            return False
        return True


def banned_providers(model: str) -> list[str]:
    with _LOCK:
        return sorted(_BANNED.get(_model_id(model), ()))


def ban_count(model: str) -> int:
    with _LOCK:
        return _BAN_COUNTS.get(_model_id(model), 0)
