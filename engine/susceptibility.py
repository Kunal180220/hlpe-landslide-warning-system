"""
susceptibility.py
==================
Static Susceptibility Scoring Engine for HLPE.

Computes slope-unit susceptibility from:
- DEM derivatives (slope, aspect, curvature, TWI)
- Geology / lithology
- NDVI (vegetation)
- Past landslide density
- Distance to streams, faults, roads
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger

try:
    import rasterio
    from rasterio.transform import rowcol
    import richdem as rd
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio/richdem not installed — DEM analysis limited")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


class SusceptibilityEngine:
    """
    Computes static susceptibility score for slope units.
    Runs once and updates monthly (or when new data is uploaded).
    """

    def __init__(self, config: dict, data_dir: str = "data/static"):
        self.config = config
        self.data_dir = Path(data_dir)
        logger.info("SusceptibilityEngine initialized")

    # ─────────────────────────────────────────────
    # DEM ANALYSIS
    # ─────────────────────────────────────────────

    def compute_dem_derivatives(self, dem_path: str) -> Dict[str, np.ndarray]:
        """
        Compute slope, aspect, curvature, TWI from DEM.
        Returns dict of 2D arrays.
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available — returning synthetic derivatives")
            return self._synthetic_dem_derivatives()

        try:
            dem = rd.LoadGDAL(dem_path)
            slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
            aspect = rd.TerrainAttribute(dem, attrib='aspect')
            profile_curve = rd.TerrainAttribute(dem, attrib='profile_curvature')
            plan_curve = rd.TerrainAttribute(dem, attrib='planform_curvature')

            # Topographic Wetness Index
            accum = rd.FlowAccumulation(dem, method='D8')
            slope_rad = np.array(slope) * np.pi / 180
            slope_safe = np.where(slope_rad < 0.001, 0.001, slope_rad)
            twi = np.log((np.array(accum) + 1) / np.tan(slope_safe))

            logger.info(f"DEM derivatives computed from {dem_path}")
            return {
                'slope_deg': np.array(slope),
                'aspect_deg': np.array(aspect),
                'profile_curvature': np.array(profile_curve),
                'plan_curvature': np.array(plan_curve),
                'twi': twi,
                'elevation': np.array(dem),
            }

        except Exception as e:
            logger.error(f"DEM analysis failed: {e}")
            return self._synthetic_dem_derivatives()

    def _synthetic_dem_derivatives(self, n: int = 100) -> Dict[str, np.ndarray]:
        """Return synthetic derivatives for testing without DEM."""
        np.random.seed(42)
        return {
            'slope_deg': np.random.uniform(5, 50, (n, n)),
            'aspect_deg': np.random.uniform(0, 360, (n, n)),
            'profile_curvature': np.random.normal(0, 0.5, (n, n)),
            'plan_curvature': np.random.normal(0, 0.5, (n, n)),
            'twi': np.random.uniform(3, 12, (n, n)),
            'elevation': np.random.uniform(500, 2000, (n, n)),
        }

    # ─────────────────────────────────────────────
    # SUSCEPTIBILITY SCORE
    # ─────────────────────────────────────────────

    def compute_susceptibility_score(self, slope_data: Dict) -> float:
        """
        Compute 0-1 susceptibility score for one slope unit.
        Higher = more susceptible to landsliding.
        """
        scores = []
        weights = []

        # ── Slope angle (most important) ──
        slope = slope_data.get('slope_angle_deg', 20)
        s_slope = self._slope_score(slope)
        scores.append(s_slope)
        weights.append(0.30)

        # ── TWI (wetness) ──
        twi = slope_data.get('twi', 6.0)
        s_twi = min(twi / 12.0, 1.0)
        scores.append(s_twi)
        weights.append(0.15)

        # ── Geology ──
        geology = slope_data.get('geology', 'default')
        s_geo = self._geology_score(geology)
        scores.append(s_geo)
        weights.append(0.20)

        # ── NDVI (vegetation protects) ──
        ndvi = slope_data.get('ndvi', 0.4)
        s_ndvi = 1.0 - np.clip(ndvi, 0, 1)  # High NDVI = lower susceptibility
        scores.append(s_ndvi)
        weights.append(0.10)

        # ── Past landslide ──
        prev = slope_data.get('previous_landslide', False)
        react = slope_data.get('reactivation_count', 0)
        s_history = min((0.3 + react * 0.1) if prev else 0.0, 1.0)
        scores.append(s_history)
        weights.append(0.15)

        # ── Distance to stream (closer = more susceptible) ──
        dist_m = slope_data.get('distance_to_stream_m', 500)
        s_stream = max(0, 1.0 - dist_m / 1000.0)
        scores.append(s_stream)
        weights.append(0.10)

        # Weighted sum
        weights = np.array(weights)
        weights = weights / weights.sum()
        score = float(np.dot(scores, weights))
        return round(float(np.clip(score, 0, 1)), 4)

    def compute_batch_susceptibility(self, slope_df: pd.DataFrame) -> pd.Series:
        """Compute susceptibility scores for all slope units."""
        return slope_df.apply(
            lambda row: self.compute_susceptibility_score(row.to_dict()), axis=1
        )

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _slope_score(self, slope_deg: float) -> float:
        """Nonlinear slope susceptibility."""
        if slope_deg < 10:   return 0.05
        elif slope_deg < 20: return 0.2 + (slope_deg - 10) / 10 * 0.2
        elif slope_deg < 35: return 0.4 + (slope_deg - 20) / 15 * 0.4
        elif slope_deg < 50: return 0.8 + (slope_deg - 35) / 15 * 0.15
        else:                return 0.95

    def _geology_score(self, geology: str) -> float:
        """Susceptibility based on geology/parent material."""
        scores = {
            'shale':      0.90, 'colluvium':   0.85, 'alluvium':    0.80,
            'schist':     0.75, 'phyllite':    0.70, 'weathered':   0.70,
            'sandstone':  0.55, 'limestone':   0.50, 'basalt':      0.40,
            'granite':    0.35, 'quartzite':   0.30, 'gneiss':      0.35,
            'default':    0.50,
        }
        return scores.get(geology.lower(), 0.50)

    def generate_synthetic_slope_units(self, n_units: int = 50) -> pd.DataFrame:
        """
        Generate synthetic slope unit data for demo/testing.
        Used when no real data is uploaded.
        """
        np.random.seed(42)
        bbox = self.config['study_area']['bbox']

        lats = np.random.uniform(bbox['min_lat'], bbox['max_lat'], n_units)
        lons = np.random.uniform(bbox['min_lon'], bbox['max_lon'], n_units)

        geology_types = ['granite', 'schist', 'colluvium', 'alluvium',
                         'sandstone', 'shale', 'basalt']
        texture_types = ['loam', 'sandy_loam', 'clay_loam', 'silt_loam',
                         'silty_clay', 'sandy_clay_loam']

        df = pd.DataFrame({
            'slope_id': [f'SU_{i:04d}' for i in range(n_units)],
            'lat': lats,
            'lon': lons,
            'name': [f'Slope Unit {i+1}' for i in range(n_units)],
            'slope_angle_deg': np.random.uniform(10, 55, n_units),
            'slope_aspect': np.random.uniform(0, 360, n_units),
            'curvature': np.random.normal(0, 0.5, n_units),
            'twi': np.random.uniform(4, 11, n_units),
            'elevation_m': np.random.uniform(300, 1800, n_units),
            'ndvi': np.random.uniform(0.1, 0.8, n_units),
            'geology': np.random.choice(geology_types, n_units),
            'texture_class': np.random.choice(texture_types, n_units),
            'soil_depth_m': np.random.uniform(0.5, 3.0, n_units),
            'cohesion_kpa': np.random.uniform(2, 12, n_units),
            'friction_angle_deg': np.random.uniform(22, 38, n_units),
            'unit_weight': np.random.uniform(16, 20, n_units),
            'distance_to_stream_m': np.random.uniform(20, 1000, n_units),
            'previous_landslide': np.random.choice([True, False], n_units, p=[0.2, 0.8]),
            'reactivation_count': np.random.choice([0, 1, 2, 3], n_units, p=[0.7, 0.15, 0.1, 0.05]),
        })

        # Compute susceptibility
        engine = SusceptibilityEngine(self.config)
        df['susceptibility'] = engine.compute_batch_susceptibility(df)

        return df.set_index('slope_id')
