"""Open-Meteo weather and air-quality tools."""
from __future__ import annotations
import difflib
import re
import unicodedata
from typing import Any
import requests
from .. import config
from ..config import TIMEOUT, USER_AGENT, short

def _json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch JSON with a timeout."""
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def _geocode(city: str, count: int = 5) -> list[dict[str, Any]]:
    """Query Open-Meteo geocoding with several candidates."""
    data = _json("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": count, "language": "en", "format": "json"})
    return data.get("results") or []


def _normal_name(value: Any) -> str:
    """Compare place names without case or accent differences."""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in text if not unicodedata.combining(char)).casefold().strip()


def _place_label(place: dict[str, Any]) -> str:
    """Build a full name, region, and country label."""
    parts: list[str] = []
    for key in ("name", "admin1", "country"):
        value = str(place.get(key) or "").strip()
        if value and _normal_name(value) not in {_normal_name(part) for part in parts}:
            parts.append(value)
    return ", ".join(parts) or "Unknown location"


def _choose(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose an exact common city, then the highest population."""
    if not candidates:
        return None
    common = {_normal_name(name) for name in config.COMMON_CITIES}
    exact = [item for item in candidates if _normal_name(item.get("name")) in common]
    pool = exact or candidates
    return max(pool, key=lambda item: float(item.get("population") or 0))


def _suggestion(city: str, chosen: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None] | None:
    """Find a close common city that differs from the chosen result."""
    cleaned = _normal_name(city)
    matches = difflib.get_close_matches(cleaned, [_normal_name(name) for name in config.COMMON_CITIES], n=1, cutoff=0.75)
    if not matches:
        return None
    common_name = next(name for name in config.COMMON_CITIES if _normal_name(name) == matches[0])
    if chosen and _normal_name(chosen.get("name")) == _normal_name(common_name):
        return None
    return common_name, _choose(_geocode(common_name, 5))


def _place(city: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Find a place and an optional confirmation suggestion."""
    cleaned = re.sub(r"[^\w\s-]", "", str(city)).strip()
    chosen = _choose(_geocode(cleaned, 5))
    suggestion = _suggestion(cleaned, chosen)
    if chosen is None and suggestion:
        return suggestion[1], suggestion[1], suggestion[0]
    if chosen is None:
        return None, None, None
    if suggestion:
        return chosen, suggestion[1] or chosen, suggestion[0]
    return chosen, None, None

def get_weather(city: str) -> str:
    """Return current weather data."""
    try:
        place, suggestion, suggested_city = _place(city)
        if not place: return "I couldn't find that place. Which city did you mean?"
        current = _json("https://api.open-meteo.com/v1/forecast", {"latitude": place["latitude"], "longitude": place["longitude"], "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,rain,weather_code", "timezone": "auto"}).get("current", {})
        codes = {0:"clear sky",1:"mainly clear",2:"partly cloudy",3:"overcast",45:"fog",48:"rime fog",51:"light drizzle",53:"drizzle",55:"dense drizzle",61:"light rain",63:"rain",65:"heavy rain",71:"light snow",73:"snow",75:"heavy snow",80:"rain showers",81:"rain showers",82:"heavy rain showers",95:"thunderstorm",96:"thunderstorm with hail",99:"thunderstorm with heavy hail"}
        result = {"location": _place_label(place), "time": current.get("time"), "temperature_c": current.get("temperature_2m"), "feels_like_c": current.get("apparent_temperature"), "humidity_percent": current.get("relative_humidity_2m"), "wind_kmh": current.get("wind_speed_10m"), "rain_mm": current.get("rain"), "conditions": codes.get(current.get("weather_code"), "unknown"), "source": "Open-Meteo"}
        if suggestion and suggested_city:
            result["needs_confirmation"] = True
            result["suggested_city"] = suggested_city
            result["suggested_location"] = _place_label(suggestion)
        return short(result)
    except Exception as error: return f"Weather lookup failed: {error}"

def get_air_quality(city: str) -> str:
    """Return current air-quality data."""
    try:
        place, suggestion, suggested_city = _place(city)
        if not place: return "I couldn't find that place. Which city did you mean?"
        current = _json("https://air-quality-api.open-meteo.com/v1/air-quality", {"latitude": place["latitude"], "longitude": place["longitude"], "current": "pm2_5,pm10,us_aqi,european_aqi", "timezone": "auto"}).get("current", {})
        result = {"location": _place_label(place), "time": current.get("time"), "pm2_5_ug_m3": current.get("pm2_5"), "pm10_ug_m3": current.get("pm10"), "us_aqi": current.get("us_aqi"), "european_aqi": current.get("european_aqi"), "source": "Open-Meteo"}
        if suggestion and suggested_city:
            result["needs_confirmation"] = True
            result["suggested_city"] = suggested_city
            result["suggested_location"] = _place_label(suggestion)
        return short(result)
    except Exception as error: return f"Air-quality lookup failed: {error}"
