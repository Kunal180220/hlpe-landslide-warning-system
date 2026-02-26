"""
id_threshold.py
================
Intensity-Duration (ID) Threshold Engine for HLPE.

Implements:
- Power-law ID threshold: I = α · D^(-β)
- Probabilistic exceedance levels
- Real-time threshold monitoring
- Auto-calibration from landslide inventory
- Storm progression tracking
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from scipy.optimize import curve_fit
from scipy.stats import norm


class IDThresholdEngine:
    """
    Intensity-Duration threshold analysis for landslide triggering.
    Auto-calibrates from local landslide inventory when available.
    """

    def __init__(self, config: dict, models_dir: str = "models"):
        self.config = config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        id_cfg = config.get('id_thresholds', {})
        self.auto_calibrate = id_cfg.get('auto_calibrate', True)
        self.min_events = id_cfg.get('min_events_for_calibration', 10)

        # Default global thresholds (Caine, 1980 global)
        self.alpha = id_cfg.get('default_alpha', 14.82)
        self.beta  = id_cfg.get('default_beta', 0.39)

        # Probabilistic thresholds at different exceedance levels
        self.thresholds = {
            0.05: {'alpha': self.alpha * 0.5, 'beta': self.beta},  # T2 (very low)
            0.25: {'alpha': self.alpha * 0.75, 'beta': self.beta}, # T5
            0.50: {'alpha': self.alpha, 'beta': self.beta},         # T10 (medium)
        }

        self._load_calibrated_thresholds()
        logger.info(f"IDThresholdEngine: α={self.alpha:.2f}, β={self.beta:.3f}")

    # ─────────────────────────────────────────────
    # CORE THRESHOLD COMPUTATION
    # ─────────────────────────────────────────────

    def compute_threshold_intensity(self, duration_hours: float,
                                     level: float = 0.50) -> float:
        """
        Compute threshold intensity for given duration.
        I = α * D^(-β)

        Args:
            duration_hours: Storm duration in hours
            level: Exceedance probability (0.05=conservative, 0.50=medium)

        Returns:
            Threshold intensity in mm/hr
        """
        params = self.thresholds.get(level, self.thresholds[0.50])
        alpha = params['alpha']
        beta  = params['beta']
        I_threshold = alpha * (duration_hours ** (-beta))
        return float(I_threshold)

    def check_exceedance(self, rainfall_timeseries: List[float],
                          time_step_hours: float = 24.0) -> Dict:
        """
        Check if current storm exceeds ID thresholds.

        Args:
            rainfall_timeseries: List of rainfall values (mm) per timestep
            time_step_hours: Duration of each timestep in hours

        Returns:
            Dict with exceedance status and position on ID curve
        """
        if not rainfall_timeseries:
            return {'exceeds': False, 'level': None}

        # Compute intensity for different durations
        results = []
        cumulative = 0.0

        for i, rain in enumerate(rainfall_timeseries):
            cumulative += rain
            duration_h = (i + 1) * time_step_hours
            intensity   = cumulative / duration_h  # mm/hr

            # Check against each threshold level
            exceedances = {}
            for level, params in self.thresholds.items():
                I_thresh = self.compute_threshold_intensity(duration_h, level)
                exceedances[f'level_{int(level*100)}'] = intensity > I_thresh
                exceedances[f'ratio_{int(level*100)}'] = round(intensity / max(I_thresh, 0.001), 3)

            results.append({
                'duration_h': round(duration_h, 1),
                'cumulative_mm': round(cumulative, 2),
                'intensity_mm_hr': round(intensity, 3),
                **exceedances
            })

        df = pd.DataFrame(results)
        # Find worst exceedance
        exceeds_50 = any(df.get('level_50', [False]))
        exceeds_25 = any(df.get('level_25', [False]))
        exceeds_5  = any(df.get('level_5', [False]))

        if exceeds_5:   level_str = 'HIGH_PROBABILITY'
        elif exceeds_25: level_str = 'MEDIUM_PROBABILITY'
        elif exceeds_50: level_str = 'LOW_PROBABILITY'
        else:            level_str = 'BELOW_THRESHOLD'

        # Max ratio (how far above threshold)
        max_ratio = df['ratio_50'].max() if 'ratio_50' in df.columns else 0.0

        return {
            'exceeds': exceeds_50 or exceeds_25 or exceeds_5,
            'exceedance_level': level_str,
            'threshold_ratio': round(float(max_ratio), 3),
            'peak_duration_h': float(df.loc[df['ratio_50'].idxmax(), 'duration_h'])
                               if 'ratio_50' in df.columns else 0.0,
            'details': df.to_dict('records')
        }

    def get_id_curve_data(self, duration_range: Optional[List] = None) -> pd.DataFrame:
        """
        Generate ID curve data for plotting.
        Returns DataFrame with duration, and threshold intensity at each level.
        """
        if duration_range is None:
            duration_range = np.logspace(0, 3, 50)  # 1 to 1000 hours

        rows = []
        for d in duration_range:
            row = {'duration_h': d}
            for level in self.thresholds:
                row[f'I_thresh_{int(level*100)}'] = self.compute_threshold_intensity(d, level)
            rows.append(row)

        return pd.DataFrame(rows)

    def get_current_storm_position(self,
                                    rainfall_data: Dict) -> Dict:
        """
        Get current storm's position on the ID curve.
        Uses antecedent rainfall to build storm series.
        """
        # Build approximate storm series from antecedent data
        r_today = rainfall_data.get('r_today_mm', 0)
        r_3day  = rainfall_data.get('r_3day_mm', 0)
        r_7day  = rainfall_data.get('r_7day_mm', 0)

        # Construct approximate daily series (last 7 days)
        daily_series = [
            max(0, r_7day - r_3day) / 4,  # ~days 4-7
            max(0, r_3day - r_today) / 2,  # day 3
            max(0, r_3day - r_today) / 2,  # day 2
            r_today                          # today
        ]

        exceedance = self.check_exceedance(daily_series, time_step_hours=24)

        # Current position for plot (today's storm)
        current_duration = 24.0  # assume 24h storm
        current_intensity = r_today / 24.0  # mm/hr

        thresh_50 = self.compute_threshold_intensity(current_duration, 0.50)
        thresh_25 = self.compute_threshold_intensity(current_duration, 0.25)
        thresh_5  = self.compute_threshold_intensity(current_duration, 0.05)

        position_ratio = current_intensity / max(thresh_50, 0.001)

        return {
            'current_duration_h': current_duration,
            'current_intensity_mm_hr': round(current_intensity, 3),
            'threshold_50_mm_hr': round(thresh_50, 3),
            'threshold_25_mm_hr': round(thresh_25, 3),
            'threshold_5_mm_hr': round(thresh_5, 3),
            'position_ratio': round(position_ratio, 3),
            'exceedance': exceedance,
            'status': self._status_from_ratio(position_ratio)
        }

    def _status_from_ratio(self, ratio: float) -> str:
        if ratio >= 2.0:   return 'FAR_ABOVE'
        elif ratio >= 1.0: return 'ABOVE'
        elif ratio >= 0.7: return 'APPROACHING'
        else:              return 'BELOW'

    # ─────────────────────────────────────────────
    # AUTO-CALIBRATION FROM INVENTORY
    # ─────────────────────────────────────────────

    def calibrate_from_inventory(self,
                                  inventory_df: pd.DataFrame,
                                  rainfall_df: pd.DataFrame) -> Dict:
        """
        Calibrate ID thresholds from local landslide inventory.

        inventory_df columns: date, lat, lon (landslide events)
        rainfall_df columns: date, rainfall_mm, lat, lon (historical rain)

        Returns calibrated alpha, beta parameters.
        """
        if len(inventory_df) < self.min_events:
            logger.warning(f"Only {len(inventory_df)} events — need {self.min_events} for calibration")
            return {'calibrated': False, 'reason': 'insufficient_data'}

        intensities = []
        durations   = []

        for _, event in inventory_df.iterrows():
            event_date = pd.to_datetime(event['date'])
            # Get rainfall for 3 days before event
            nearby_rain = rainfall_df[
                (rainfall_df['date'] >= event_date - timedelta(days=3)) &
                (rainfall_df['date'] <= event_date)
            ]

            if nearby_rain.empty:
                continue

            for d in [1, 2, 3]:
                subset = nearby_rain[
                    nearby_rain['date'] >= event_date - timedelta(days=d)
                ]
                if not subset.empty:
                    total = subset['rainfall_mm'].sum()
                    intensity = total / (d * 24)
                    if intensity > 0:
                        intensities.append(intensity)
                        durations.append(d * 24)
                        break

        if len(intensities) < self.min_events:
            return {'calibrated': False, 'reason': 'insufficient_rain_data'}

        # Fit power law: I = alpha * D^(-beta)
        try:
            def power_law(D, alpha, beta):
                return alpha * np.power(D, -beta)

            popt, _ = curve_fit(
                power_law,
                durations,
                intensities,
                p0=[self.alpha, self.beta],
                bounds=([0, 0], [1000, 2])
            )

            new_alpha, new_beta = popt
            self.alpha = float(new_alpha)
            self.beta  = float(new_beta)

            # Update probabilistic thresholds
            self.thresholds = {
                0.05: {'alpha': self.alpha * 0.45, 'beta': self.beta},
                0.25: {'alpha': self.alpha * 0.70, 'beta': self.beta},
                0.50: {'alpha': self.alpha, 'beta': self.beta},
            }

            self._save_calibrated_thresholds()
            logger.success(f"ID calibration: α={self.alpha:.2f}, β={self.beta:.3f} "
                          f"from {len(intensities)} events")

            return {
                'calibrated': True,
                'alpha': self.alpha,
                'beta': self.beta,
                'n_events': len(intensities),
                'r_squared': self._compute_r2(durations, intensities, power_law, popt)
            }

        except Exception as e:
            logger.error(f"ID calibration failed: {e}")
            return {'calibrated': False, 'reason': str(e)}

    def _compute_r2(self, x, y, func, params) -> float:
        y_pred = func(np.array(x), *params)
        ss_res = np.sum((np.array(y) - y_pred)**2)
        ss_tot = np.sum((np.array(y) - np.mean(y))**2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _load_calibrated_thresholds(self):
        path = self.models_dir / "id_thresholds.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.alpha = data.get('alpha', self.alpha)
            self.beta  = data.get('beta', self.beta)
            self.thresholds = data.get('thresholds', self.thresholds)
            logger.info(f"Loaded calibrated ID thresholds: α={self.alpha:.2f}")

    def _save_calibrated_thresholds(self):
        path = self.models_dir / "id_thresholds.json"
        with open(path, 'w') as f:
            json.dump({
                'alpha': self.alpha,
                'beta': self.beta,
                'thresholds': self.thresholds,
                'calibrated_at': datetime.now().isoformat()
            }, f, indent=2)
