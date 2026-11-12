"""Open-Meteo weather and air-quality tools."""
from __future__ import annotations
from typing import Any
import requests
from ..config import TIMEOUT, USER_AGENT, short

def _json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch JSON with a timeout."""
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

def _place(city: str) -> dict[str, Any] | None:
    """Find the first Open-Meteo city match."""
    return (_json("https://geocoding-api.open-meteo.com/v1/search", {"name": str(city), "count": 1, "language": "en", "format": "json"}).get("results") or [None])[0]

def get_weather(city: str) -> str:
    """Return current weather data."""
    try:
        place = _place(city)
        if not place: return f"No location was found for {city}."
        current = _json("https://api.open-meteo.com/v1/forecast", {"latitude": place["latitude"], "longitude": place["longitude"], "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,rain,weather_code", "timezone": "auto"}).get("current", {})
        codes = {0:"clear sky",1:"mainly clear",2:"partly cloudy",3:"overcast",45:"fog",48:"rime fog",51:"light drizzle",53:"drizzle",55:"dense drizzle",61:"light rain",63:"rain",65:"heavy rain",71:"light snow",73:"snow",75:"heavy snow",80:"rain showers",81:"rain showers",82:"heavy rain showers",95:"thunderstorm",96:"thunderstorm with hail",99:"thunderstorm with heavy hail"}
        return short({"location": f"{place.get('name')}, {place.get('country')}", "time": current.get("time"), "temperature_c": current.get("temperature_2m"), "feels_like_c": current.get("apparent_temperature"), "humidity_percent": current.get("relative_humidity_2m"), "wind_kmh": current.get("wind_speed_10m"), "rain_mm": current.get("rain"), "conditions": codes.get(current.get("weather_code"), "unknown"), "source": "Open-Meteo"})
    except Exception as error: return f"Weather lookup failed: {error}"

def get_air_quality(city: str) -> str:
    """Return current air-quality data."""
    try:
        place = _place(city)
        if not place: return f"No location was found for {city}."
        current = _json("https://air-quality-api.open-meteo.com/v1/air-quality", {"latitude": place["latitude"], "longitude": place["longitude"], "current": "pm2_5,pm10,us_aqi,european_aqi", "timezone": "auto"}).get("current", {})
        return short({"location": f"{place.get('name')}, {place.get('country')}", "time": current.get("time"), "pm2_5_ug_m3": current.get("pm2_5"), "pm10_ug_m3": current.get("pm10"), "us_aqi": current.get("us_aqi"), "european_aqi": current.get("european_aqi"), "source": "Open-Meteo"})
    except Exception as error: return f"Air-quality lookup failed: {error}"
