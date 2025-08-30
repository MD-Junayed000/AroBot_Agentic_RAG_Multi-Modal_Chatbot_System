# utils/weather_service.py
import requests
from dataclasses import dataclass
from typing import Optional, Tuple

OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "freezing drizzle (dense)",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    66: "freezing rain (light)", 67: "freezing rain (heavy)",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "rain showers (light)", 81: "rain showers (moderate)", 82: "rain showers (violent)",
    85: "snow showers (light)", 86: "snow showers (heavy)",
    95: "thunderstorm", 96: "thunderstorm with small hail", 99: "thunderstorm with heavy hail"
}

def _label_temp_c(t: float) -> str:
    if t < 5: return "very cold"
    if t < 12: return "chilly"
    if t < 18: return "cool"
    if t < 26: return "warm"
    if t < 32: return "hot"
    return "very hot"

def geocode(place: str) -> Optional[Tuple[float, float, str]]:
    r = requests.get(NOMINATIM, params={"q": place, "format": "json", "limit": 1}, headers={"User-Agent": "arobot"})
    r.raise_for_status()
    j = r.json()
    if not j: return None
    lat = float(j[0]["lat"]); lon = float(j[0]["lon"]); disp = j[0]["display_name"]
    return lat, lon, disp

@dataclass
class WeatherNow:
    place: str
    temp_c: float
    condition: str
    label: str
    wind_kph: float
    humidity: Optional[int] = None

def get_weather(place_or_coords: str) -> Optional[WeatherNow]:
    # Accept "City" or "lat,lon"
    try:
        if "," in place_or_coords and all(p.strip().replace(".", "", 1).replace("-", "", 1).isdigit()
                                          for p in place_or_coords.split(",", 1)):
            lat, lon = [float(x) for x in place_or_coords.split(",", 1)]
            label_place = place_or_coords
        else:
            g = geocode(place_or_coords)
            if not g: return None
            lat, lon, label_place = g

        params = {
            "latitude": lat, "longitude": lon,
            "current_weather": True, "hourly": "relativehumidity_2m"
        }
        r = requests.get(OPENMETEO_FORECAST, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        cw = j.get("current_weather", {})
        temp = float(cw.get("temperature"))
        code = int(cw.get("weathercode", 0))
        cond = WEATHER_CODES.get(code, "unknown")
        lbl = _label_temp_c(temp)
        wind = float(cw.get("windspeed", 0.0))
        return WeatherNow(place=label_place, temp_c=temp, condition=cond, label=lbl, wind_kph=wind)
    except Exception:
        return None

def format_weather(now: WeatherNow) -> str:
    return (f"{now.place}\n"
            f"• {now.condition.capitalize()} and {now.label}\n"
            f"• {now.temp_c:.1f}°C, wind {now.wind_kph:.0f} km/h")
