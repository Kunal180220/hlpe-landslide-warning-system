"""
rainfall_fetcher.py
===================
Dynamic rainfall data fetcher for HLPE.
Sources: NASA GPM IMERG, Open-Meteo, Manual Entry
Computes: Antecedent Rainfall Index, Soil Water Index (SWI)
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    OPENMETEO_AVAILABLE = True
except ImportError:
    OPENMETEO_AVAILABLE = False


class RainfallFetcher:
    """
    Fetches and manages rainfall data from multiple sources.
    Priority: Manual > GPM IMERG > Open-Meteo > Climatology
    """

    def __init__(self, config: dict, data_dir: str = "data/dynamic/rainfall"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bbox = config['study_area']['bbox']
        self.antecedent_days = config['rainfall'].get('antecedent_days', 15)
        self.swi_T = config['rainfall'].get('swi_timescale_days', 5)
        self._manual_records = []
        self._cache_file = self.data_dir / "rainfall_cache.json"
        self._load_cache()
        logger.info(f"RainfallFetcher initialized for {config['study_area']['name']}")

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def get_current_rainfall(self, lat: float, lon: float) -> Dict:
        """
        Main entry point. Returns complete rainfall dict for a point.
        """
        df = self._get_time_series(lat, lon)
        if df is None or df.empty:
            return self._fallback_data()

        result = self._compute_rainfall_indices(df, lat, lon)
        return result

    def add_manual_reading(self, station_id: str, date: str,
                           rainfall_mm: float, lat: float, lon: float):
        """
        Add a manual rain gauge reading (field entry).
        date format: YYYY-MM-DD
        """
        record = {
            'station_id': station_id,
            'date': date,
            'rainfall_mm': float(rainfall_mm),
            'lat': lat,
            'lon': lon,
            'timestamp': datetime.now().isoformat(),
            'source': 'manual'
        }
        self._manual_records.append(record)
        self._save_cache()
        logger.info(f"Manual reading added: {station_id} | {date} | {rainfall_mm}mm")
        return record

    def get_manual_readings(self, station_id: Optional[str] = None) -> pd.DataFrame:
        """Return manual readings as DataFrame."""
        if not self._manual_records:
            return pd.DataFrame()
        df = pd.DataFrame(self._manual_records)
        if station_id:
            df = df[df['station_id'] == station_id]
        return df.sort_values('date', ascending=False)

    def get_rainfall_grid(self, resolution_deg: float = 0.1) -> pd.DataFrame:
        """
        Get rainfall for a grid covering the study area.
        Returns DataFrame with lat, lon, and all rainfall indices.
        """
        lat_range = np.arange(
            self.bbox['min_lat'],
            self.bbox['max_lat'],
            resolution_deg
        )
        lon_range = np.arange(
            self.bbox['min_lon'],
            self.bbox['max_lon'],
            resolution_deg
        )

        records = []
        for lat in lat_range:
            for lon in lon_range:
                data = self.get_current_rainfall(lat, lon)
                data['lat'] = round(lat, 4)
                data['lon'] = round(lon, 4)
                records.append(data)

        df = pd.DataFrame(records)
        logger.info(f"Rainfall grid computed: {len(df)} points")
        return df

    def get_time_series_for_plot(self, lat: float, lon: float,
                                  days: int = 30) -> pd.DataFrame:
        """Get rainfall time series for chart display."""
        return self._get_time_series(lat, lon, days=days)

    # ─────────────────────────────────────────────
    # OPEN-METEO (Primary Free Source)
    # ─────────────────────────────────────────────

    def _fetch_openmeteo(self, lat: float, lon: float, days: int = 20) -> Optional[pd.DataFrame]:
        """
        Fetch rainfall from Open-Meteo (free, no API key needed).
        Returns daily rainfall DataFrame.
        """
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "precipitation_sum,rain_sum",
                "timezone": "auto"
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'rainfall_mm': data['daily']['precipitation_sum']
                })
                df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
                df = df.sort_values('date').reset_index(drop=True)
                df['source'] = 'open_meteo'
                logger.debug(f"Open-Meteo: fetched {len(df)} days for ({lat:.2f}, {lon:.2f})")
                return df
            else:
                logger.warning(f"Open-Meteo returned status {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed: {e}")
            return None

    def _fetch_openmeteo_forecast(self, lat: float, lon: float, days: int = 7) -> Optional[pd.DataFrame]:
        """Fetch 7-day rainfall forecast from Open-Meteo."""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,precipitation_probability_max",
                "forecast_days": days,
                "timezone": "auto"
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'rainfall_mm': data['daily']['precipitation_sum'],
                    'rain_probability': data['daily']['precipitation_probability_max']
                })
                df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
                df['forecast'] = True
                return df
            return None
        except Exception as e:
            logger.warning(f"Forecast fetch failed: {e}")
            return None

    # ─────────────────────────────────────────────
    # GPM IMERG (NASA Satellite)
    # ─────────────────────────────────────────────

    def _fetch_gpm_imerg(self, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """
        Fetch NASA GPM IMERG Late Run data.
        Requires NASA Earthdata account (free).
        Falls back to Open-Meteo if not configured.
        """
        api_key = self.config.get('api_keys', {}).get('nasa_earthdata', '')
        if not api_key:
            logger.debug("No NASA Earthdata key — using Open-Meteo")
            return None

        try:
            # GPM IMERG Late Run API endpoint
            end_date = datetime.now() - timedelta(hours=4)
            start_date = end_date - timedelta(days=self.antecedent_days)

            base_url = "https://gpm.nasa.gov/api/v1/imerg"
            params = {
                "start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "lat": lat,
                "lon": lon,
                "type": "late",
                "format": "json"
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(base_url, params=params,
                                    headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['time'])
                df['rainfall_mm'] = df['precipitationCal']
                df['source'] = 'gpm_imerg'
                return df[['date', 'rainfall_mm', 'source']]

        except Exception as e:
            logger.warning(f"GPM IMERG failed: {e}")

        return None

    # ─────────────────────────────────────────────
    # CORE COMPUTATION
    # ─────────────────────────────────────────────

    def _get_time_series(self, lat: float, lon: float,
                          days: int = 20) -> Optional[pd.DataFrame]:
        """Get best available rainfall time series."""
        # Check manual records first
        manual_df = self._get_manual_for_location(lat, lon)

        # Try GPM first
        df = self._fetch_gpm_imerg(lat, lon)

        # Fallback to Open-Meteo
        if df is None:
            df = self._fetch_openmeteo(lat, lon, days=days)

        if df is None:
            return None

        # Merge manual readings (override satellite data with gauge readings)
        if manual_df is not None and not manual_df.empty:
            df = self._merge_manual(df, manual_df)

        return df

    def _compute_rainfall_indices(self, df: pd.DataFrame,
                                   lat: float, lon: float) -> Dict:
        """
        Compute all rainfall indices from time series.
        """
        rain = df['rainfall_mm'].values
        dates = df['date'].values

        today_rain = float(rain[-1]) if len(rain) > 0 else 0.0
        r_24h = today_rain
        r_3day = float(np.sum(rain[-3:])) if len(rain) >= 3 else np.sum(rain)
        r_7day = float(np.sum(rain[-7:])) if len(rain) >= 7 else np.sum(rain)
        r_15day = float(np.sum(rain[-15:])) if len(rain) >= 15 else np.sum(rain)

        # Weighted Antecedent Precipitation Index (API)
        # API = 0.5*R3 + 0.3*R7 + 0.2*R15
        api = 0.5 * r_3day + 0.3 * r_7day + 0.2 * r_15day

        # Soil Water Index (SWI) — exponential filter
        # SWI(t) = SWI(t-1) + (rain(t) - SWI(t-1)) / T
        swi = self._compute_swi(rain, T=self.swi_T)

        # Peak intensity (max 1-day in last 7 days)
        peak_intensity = float(np.max(rain[-7:])) if len(rain) >= 7 else float(np.max(rain))

        # Source quality
        source = df['source'].iloc[-1] if 'source' in df.columns else 'unknown'

        # Fetch forecast
        forecast = self._fetch_openmeteo_forecast(lat, lon, days=3)
        forecast_3day = 0.0
        forecast_rain_tomorrow = 0.0
        if forecast is not None and not forecast.empty:
            forecast_3day = float(forecast['rainfall_mm'].sum())
            forecast_rain_tomorrow = float(forecast['rainfall_mm'].iloc[0])

        return {
            'r_today_mm': round(r_24h, 2),
            'r_3day_mm': round(r_3day, 2),
            'r_7day_mm': round(r_7day, 2),
            'r_15day_mm': round(r_15day, 2),
            'api': round(api, 2),
            'swi': round(swi, 4),
            'swi_normalized': round(min(swi / 50.0, 1.0), 4),  # normalize to 0-1
            'peak_intensity_mm_day': round(peak_intensity, 2),
            'forecast_tomorrow_mm': round(forecast_rain_tomorrow, 2),
            'forecast_3day_mm': round(forecast_3day, 2),
            'data_source': source,
            'last_updated': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(source, df)
        }

    def _compute_swi(self, rain_series: np.ndarray, T: float = 5.0) -> float:
        """
        Soil Water Index using exponential filter (Brocca et al., 2012).
        SWI(t) = SWI(t-1) * exp(-1/T) + rain(t)
        T = timescale parameter (days), controls memory
        """
        if len(rain_series) == 0:
            return 0.0

        swi = rain_series[0]
        decay = np.exp(-1.0 / T)

        for r in rain_series[1:]:
            swi = swi * decay + r

        return float(swi)

    def _get_manual_for_location(self, lat: float, lon: float,
                                  radius_deg: float = 0.1) -> Optional[pd.DataFrame]:
        """Find manual records near a location."""
        if not self._manual_records:
            return None
        df = pd.DataFrame(self._manual_records)
        dist = np.sqrt((df['lat'] - lat)**2 + (df['lon'] - lon)**2)
        nearby = df[dist <= radius_deg].copy()
        if nearby.empty:
            return None
        nearby['date'] = pd.to_datetime(nearby['date'])
        return nearby[['date', 'rainfall_mm', 'source']].rename(
            columns={'rainfall_mm': 'rainfall_mm_manual'})

    def _merge_manual(self, satellite_df: pd.DataFrame,
                       manual_df: pd.DataFrame) -> pd.DataFrame:
        """Override satellite data with manual gauge readings where available."""
        merged = satellite_df.copy()
        for _, row in manual_df.iterrows():
            mask = merged['date'].dt.date == row['date'].date()
            if mask.any():
                merged.loc[mask, 'rainfall_mm'] = row['rainfall_mm_manual']
                merged.loc[mask, 'source'] = 'manual_override'
        return merged

    def _assess_data_quality(self, source: str, df: pd.DataFrame) -> str:
        """Return data quality assessment."""
        if source == 'manual_override':
            return 'HIGH'
        elif source == 'gpm_imerg':
            return 'HIGH'
        elif source == 'open_meteo':
            return 'MEDIUM'
        else:
            return 'LOW'

    def _fallback_data(self) -> Dict:
        """Return zero-filled fallback when all sources fail."""
        logger.warning("All rainfall sources failed — returning zeros")
        return {
            'r_today_mm': 0.0,
            'r_3day_mm': 0.0,
            'r_7day_mm': 0.0,
            'r_15day_mm': 0.0,
            'api': 0.0,
            'swi': 0.0,
            'swi_normalized': 0.0,
            'peak_intensity_mm_day': 0.0,
            'forecast_tomorrow_mm': 0.0,
            'forecast_3day_mm': 0.0,
            'data_source': 'fallback',
            'last_updated': datetime.now().isoformat(),
            'data_quality': 'NONE'
        }

    def _load_cache(self):
        """Load cached manual records."""
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                data = json.load(f)
                self._manual_records = data.get('manual_records', [])

    def _save_cache(self):
        """Save manual records to disk."""
        with open(self._cache_file, 'w') as f:
            json.dump({'manual_records': self._manual_records}, f, indent=2)
