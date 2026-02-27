"""
Rainfall Fetcher — Multi-source rainfall data with SWI and antecedent index
Sources: Open-Meteo (free, no key) → NASA GPM → Manual entry
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os


def fetch_openmeteo(lat: float, lon: float, days_back: int = 15) -> Optional[Dict]:
    """Fetch rainfall from Open-Meteo API (free, no key required)"""
    try:
        import urllib.request
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum,precipitation_hours"
            f"&hourly=precipitation"
            f"&start_date={start_date}&end_date={end_date}"
            f"&timezone=Asia%2FKolkata"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        return data
    except Exception:
        return None


def compute_swi(rainfall_series: List[float], T: float = 5.0) -> List[float]:
    """
    Soil Water Index using exponential filter.
    T = characteristic time length (days). Larger T = slower drain.
    SWI(t) = SWI(t-1) + (R(t) - SWI(t-1)) / T
    """
    swi = [0.0]
    for r in rainfall_series[1:]:
        prev = swi[-1]
        swi.append(prev + (r - prev) / T)
    return swi


def compute_antecedent_index(rainfall_series: List[float]) -> Dict:
    """
    Compute antecedent rainfall indices with exponential weights.
    Returns R_3day, R_7day, R_15day, API
    """
    n = len(rainfall_series)
    r = list(reversed(rainfall_series))  # most recent first
    
    def weighted_sum(series, window, decay):
        total = 0.0
        for i, val in enumerate(series[:window]):
            total += val * (decay ** i)
        return round(total, 2)
    
    return {
        "R_24hr": round(r[0] if r else 0.0, 1),
        "R_3day": round(sum(r[:3]), 1),
        "R_7day": round(sum(r[:7]), 1),
        "R_15day": round(sum(r[:15]), 1),
        "API": weighted_sum(r, 15, 0.85),  # antecedent precipitation index
        "SWI_current": 0.0,  # filled after SWI calc
    }


def fetch_live_rainfall_for_location(lat: float, lon: float, days_back: int = 15) -> Dict:
    """
    Fetch live rainfall for a single lat/lon from Open-Meteo.
    Returns dict with dates, values, source label.
    """
    data = fetch_openmeteo(lat, lon, days_back=days_back)
    if data and "daily" in data:
        dates = data["daily"].get("time", [])
        values = data["daily"].get("precipitation_sum", [])
        values = [float(v) if v is not None else 0.0 for v in values]
        return {
            "dates": dates,
            "values": values,
            "source": "Open-Meteo (Live)",
            "lat": lat,
            "lon": lon,
            "success": True,
        }
    return {
        "dates": [],
        "values": [],
        "source": "API Failed",
        "lat": lat,
        "lon": lon,
        "success": False,
    }


def get_rainfall_for_slope_units(
    slope_df,
    manual_rainfall: Optional[Dict] = None,
    use_api: bool = True,
) -> pd.DataFrame:
    """
    Get rainfall data for all slope units.
    Returns df with rainfall columns added.
    Clearly labels data source: 'Live API', 'Manual', or 'Synthetic'.
    """
    df = slope_df.copy()
    df["lat_r"] = df["lat"].round(1)
    df["lon_r"] = df["lon"].round(1)

    cache = {}

    for idx, row in df.iterrows():
        key = (row["lat_r"], row["lon_r"])

        if key not in cache:
            data = None
            source = "Synthetic"

            if use_api:
                data = fetch_openmeteo(key[0], key[1], days_back=15)

            if data and "daily" in data:
                rain_list = data["daily"].get("precipitation_sum", [])
                rain_list = [float(r) if r is not None else 0.0 for r in rain_list]
                dates = data["daily"].get("time", [])
                source = "Live API (Open-Meteo)"
            else:
                # Synthetic fallback — clearly labelled
                np.random.seed(int(abs(key[0] * 100 + key[1] * 10)))
                rain_list = list(np.random.exponential(5, 15))
                rain_list[-1] = np.random.exponential(8)
                today = datetime.now()
                dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d")
                         for i in range(14, -1, -1)]
                source = "Synthetic (API OFF)"

            # Manual override
            if manual_rainfall and manual_rainfall.get("today_mm", 0) > 0:
                rain_list[-1] = manual_rainfall["today_mm"]
                source = "Manual Entry"

            swi = compute_swi(rain_list)
            api = compute_antecedent_index(rain_list)
            api["SWI_current"] = round(swi[-1], 3)
            api["SWI_max"] = round(max(swi) if swi else 20.0, 3)
            api["m_ratio"] = round(min(swi[-1] / max(max(swi), 1.0), 1.0), 3)
            api["rainfall_series"] = rain_list
            api["rainfall_dates"] = dates
            api["data_source"] = source

            cache[key] = api

        entry = cache[key]
        df.at[idx, "R_24hr"] = entry["R_24hr"]
        df.at[idx, "R_3day"] = entry["R_3day"]
        df.at[idx, "R_7day"] = entry["R_7day"]
        df.at[idx, "R_15day"] = entry["R_15day"]
        df.at[idx, "API"] = entry["API"]
        df.at[idx, "SWI"] = entry["SWI_current"]
        df.at[idx, "m_ratio"] = entry["m_ratio"]
        df.at[idx, "data_source"] = entry["data_source"]
        # Store serialized series for chart
        df.at[idx, "rainfall_series"] = json.dumps(entry["rainfall_series"])
        df.at[idx, "rainfall_dates"] = json.dumps(entry["rainfall_dates"])

    return df
