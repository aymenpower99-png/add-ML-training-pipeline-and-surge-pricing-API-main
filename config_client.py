"""
config_client.py — Dynamic config loader for the Pricing API

Fetches pricing configuration from the Config API (Flask) with:
  • Startup fetch (blocking, so first request has config)
  • Background refresh every 60 seconds (async task)
  • Fallback to static config.py if Config API is unavailable

Usage in pricing/engine.py:
    from config_client import get_config
    cfg = get_config()
    base_fare = cfg["BASE_FARE"]

Environment variable:
    ML_CONFIG_API_URL — base URL of the Config API (default: http://localhost:5000)
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Any, Dict

import requests

_static_config = None

def _load_static_config() -> Dict[str, Any]:
    global _static_config
    if _static_config is not None:
        return _static_config

    import config as static_cfg
    _static_config = {
        k: getattr(static_cfg, k)
        for k in dir(static_cfg)
        if not k.startswith("_")
        and k.isupper()
        and not callable(getattr(static_cfg, k, None))
    }
    return _static_config


logger = logging.getLogger(__name__)

CONFIG_API_URL = os.getenv("ML_CONFIG_API_URL", "http://localhost:5000")
REFRESH_INTERVAL_S = 60

_cached_config: Dict[str, Any] = {}
_config_fetched_at: float = 0.0
_config_source: str = "fallback"


def _fetch_config_from_api() -> Dict[str, Any]:
    url = f"{CONFIG_API_URL.rstrip('/')}/api/config"
    print(f"🔄 Fetching config from: {url}", flush=True)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ Config fetched from API ({url}) — {len(data)} keys", flush=True)
        return data
    except Exception as exc:
        print(f"❌ Config API unreachable ({url}): {exc}", flush=True)
        raise


def refresh_config(force: bool = False) -> Dict[str, Any]:
    global _cached_config, _config_fetched_at, _config_source

    now = time.time()
    if not force and _cached_config and (now - _config_fetched_at) < REFRESH_INTERVAL_S:
        return _cached_config

    try:
        _cached_config = _fetch_config_from_api()
        _config_fetched_at = now
        _config_source = "api"
    except Exception:
        if not _cached_config:
            _cached_config = _load_static_config()
            _config_fetched_at = now
            _config_source = "fallback"
            print(f"⚠️ Using static config.py fallback. CONFIG_API_URL={CONFIG_API_URL}", flush=True)
        else:
            print(f"⚠️ Config API still unreachable — using stale cached config", flush=True)

    return _cached_config


def get_config() -> Dict[str, Any]:
    return refresh_config()


def get_config_value(key: str, default: Any = None) -> Any:
    cfg = get_config()
    return cfg.get(key, default)


async def refresh_config_async() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, refresh_config)


async def config_refresh_loop() -> None:
    print(f"🔁 Config refresh loop started (interval={REFRESH_INTERVAL_S}s)", flush=True)
    while True:
        try:
            await refresh_config_async()
        except Exception as exc:
            print(f"❌ Config refresh loop error: {exc}", flush=True)
        await asyncio.sleep(REFRESH_INTERVAL_S)


print(f"🚀 config_client.py loaded — CONFIG_API_URL={CONFIG_API_URL}", flush=True)
refresh_config()