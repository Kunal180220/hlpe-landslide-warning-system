"""
3-Day Forecast Engine
Fetches rainfall forecast for next 3 days from Open-Meteo
and estimates risk levels for each day.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import urllib.request


def fetch_3day_forecast(lat: float, lon: float) -> Dict:
    """
    Fetch 3-day rainfall + weather forecast from Open-Meteo.
    Returns dict with dates, rainfall, temp, weather codes.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,precipitation_probability_max,"
            f"weathercode,temperature_2m_max,temperature_2m_min"
            f"&forecast_days=4"
            f"&timezone=Asia%2FKolkata"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        rain = daily.get("precipitation_sum", [])
        prob = daily.get("precipitation_probability_max", [])
        wcode = daily.get("weathercode", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])

        # Skip today (index 0), take next 3 days
        result = []
        for i in range(1, min(4, len(dates))):
            r = float(rain[i]) if rain[i] is not None else 0.0
            p = float(prob[i]) if prob and prob[i] is not None else 0.0
            result.append({
                "date": dates[i],
                "rainfall_mm": round(r, 1),
                "rain_probability_pct": round(p, 0),
                "weathercode": wcode[i] if wcode else 0,
                "temp_max": round(float(tmax[i]), 1) if tmax and tmax[i] else None,
                "temp_min": round(float(tmin[i]), 1) if tmin and tmin[i] else None,
                "weather_icon": _wcode_to_icon(wcode[i] if wcode else 0),
                "weather_label": _wcode_to_label(wcode[i] if wcode else 0),
            })
        return {"success": True, "forecast": result, "lat": lat, "lon": lon}

    except Exception as e:
        # Synthetic fallback
        result = []
        np.random.seed(int(abs(lat * 100 + lon * 10)))
        for i in range(1, 4):
            d = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            r = round(float(np.random.exponential(6)), 1)
            result.append({
                "date": d,
                "rainfall_mm": r,
                "rain_probability_pct": min(round(r * 4 + np.random.uniform(10, 30)), 100),
                "weathercode": 61 if r > 10 else 51 if r > 2 else 1,
                "temp_max": round(22 + np.random.normal(0, 3), 1),
                "temp_min": round(15 + np.random.normal(0, 2), 1),
                "weather_icon": "🌧️" if r > 10 else "🌦️" if r > 2 else "⛅",
                "weather_label": "Rain" if r > 10 else "Drizzle" if r > 2 else "Partly Cloudy",
            })
        return {"success": False, "forecast": result, "source": "Synthetic"}


def _wcode_to_icon(code: int) -> str:
    if code == 0: return "☀️"
    if code in [1, 2]: return "⛅"
    if code == 3: return "☁️"
    if code in [45, 48]: return "🌫️"
    if code in [51, 53, 55]: return "🌦️"
    if code in [61, 63, 65]: return "🌧️"
    if code in [71, 73, 75]: return "🌨️"
    if code in [80, 81, 82]: return "⛈️"
    if code in [95, 96, 99]: return "⛈️"
    return "🌡️"


def _wcode_to_label(code: int) -> str:
    if code == 0: return "Clear"
    if code in [1, 2]: return "Partly Cloudy"
    if code == 3: return "Overcast"
    if code in [45, 48]: return "Foggy"
    if code in [51, 53, 55]: return "Drizzle"
    if code in [61, 63, 65]: return "Rain"
    if code in [71, 73, 75]: return "Snow"
    if code in [80, 81, 82]: return "Heavy Showers"
    if code in [95, 96, 99]: return "Thunderstorm"
    return "Variable"


def estimate_forecast_risk(current_fs: float, current_swi: float,
                           forecast_rain_mm: float, soil_type: str = "Laterite") -> Dict:
    """
    Estimate risk for a future day given forecast rainfall.
    Uses simplified SWI projection + FS recalculation.
    """
    # Project SWI forward
    T = 5.0
    projected_swi = current_swi + (forecast_rain_mm - current_swi) / T
    m_ratio = min(projected_swi / max(current_swi * 2, 20.0), 1.0)

    # FS degrades with more rain
    fs_factor = max(1.0 - (forecast_rain_mm / 80.0) * (1 - m_ratio), 0.5)
    projected_fs = round(current_fs * fs_factor, 3)

    # Risk level
    if projected_fs < 1.0 or forecast_rain_mm > 50:
        risk = "EXTREME"
        color = "#cc0000"
    elif projected_fs < 1.2 or forecast_rain_mm > 30:
        risk = "HIGH"
        color = "#ff4444"
    elif projected_fs < 1.5 or forecast_rain_mm > 15:
        risk = "ELEVATED"
        color = "#ff8c00"
    elif projected_fs < 2.0 or forecast_rain_mm > 5:
        risk = "MODERATE"
        color = "#ffd700"
    else:
        risk = "LOW"
        color = "#2ecc71"

    return {
        "projected_fs": projected_fs,
        "projected_risk": risk,
        "projected_color": color,
        "projected_swi": round(projected_swi, 2),
    }


def get_area_forecast(state: str, district: Optional[str], taluk: Optional[str],
                      slope_df, use_api: bool = True) -> Dict:
    """
    Get 3-day forecast for the selected area.
    Returns area-level summary + per-day risk estimates.
    """
    from engine.admin_hierarchy import get_center
    center = get_center(state, district, taluk)

    # Fetch forecast
    forecast_data = fetch_3day_forecast(center[0], center[1])
    forecasts = forecast_data["forecast"]

    # Build per-district/taluk forecast summary
    group_col = "taluk" if district else "district"
    area_forecasts = []

    if slope_df is not None and not slope_df.empty:
        groups = slope_df.groupby(group_col)
        for area_name, grp in groups:
            avg_fs = grp["FS"].mean() if "FS" in grp.columns else 1.5
            avg_swi = grp["SWI"].mean() if "SWI" in grp.columns else 5.0
            current_risk = grp["alert_level"].mode()[0] if "alert_level" in grp.columns else "MODERATE"
            n_slopes = len(grp)

            day_risks = []
            for fc in forecasts:
                proj = estimate_forecast_risk(avg_fs, avg_swi, fc["rainfall_mm"])
                day_risks.append({
                    "date": fc["date"],
                    "rainfall_mm": fc["rainfall_mm"],
                    "rain_prob": fc["rain_probability_pct"],
                    "weather_icon": fc["weather_icon"],
                    "risk": proj["projected_risk"],
                    "color": proj["projected_color"],
                    "fs": proj["projected_fs"],
                })

            area_forecasts.append({
                "area": area_name,
                "level": group_col,
                "n_slopes": n_slopes,
                "current_risk": current_risk,
                "avg_fs": round(avg_fs, 3),
                "day1": day_risks[0] if len(day_risks) > 0 else {},
                "day2": day_risks[1] if len(day_risks) > 1 else {},
                "day3": day_risks[2] if len(day_risks) > 2 else {},
            })

    return {
        "area_forecasts": area_forecasts,
        "raw_forecast": forecasts,
        "source": "Live API" if forecast_data["success"] else "Synthetic",
        "center": center,
    }
