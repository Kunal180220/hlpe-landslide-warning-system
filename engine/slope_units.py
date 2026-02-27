"""
Slope Unit Generator
Generates synthetic slope units within admin boundaries,
or loads user-uploaded slope unit data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
import os

from engine.admin_hierarchy import get_bbox, get_all_taluks_in_district, get_taluks


SOIL_TYPES = {
    "Laterite": {"cohesion": 8.0, "friction": 28.0, "unit_weight": 18.0, "hydraulic_k": 1e-5},
    "Sandy Loam": {"cohesion": 5.0, "friction": 32.0, "unit_weight": 17.0, "hydraulic_k": 5e-5},
    "Clay": {"cohesion": 15.0, "friction": 20.0, "unit_weight": 19.5, "hydraulic_k": 1e-7},
    "Silty Clay": {"cohesion": 12.0, "friction": 22.0, "unit_weight": 19.0, "hydraulic_k": 5e-7},
    "Sandy Clay Loam": {"cohesion": 7.0, "friction": 26.0, "unit_weight": 18.0, "hydraulic_k": 2e-6},
    "Loam": {"cohesion": 10.0, "friction": 25.0, "unit_weight": 17.5, "hydraulic_k": 8e-6},
}

GEOLOGY_TYPES = ["Charnockite", "Gneiss", "Schist", "Granite", "Basalt", "Limestone", "Sandstone"]

LANDUSE_TYPES = ["Dense Forest", "Degraded Forest", "Plantation", "Agriculture", "Barren", "Settlement"]


def generate_slope_units(
    state: str,
    district: Optional[str] = None,
    taluk: Optional[str] = None,
    n_units: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic slope units within the specified admin boundary.
    If taluk is given: generate within that taluk.
    If only district: distribute across taluks in district.
    If only state: distribute across districts.
    """
    np.random.seed(seed)
    
    bbox = get_bbox(state, district, taluk)
    min_lon, max_lon, min_lat, max_lat = bbox
    
    # Determine taluk assignments
    taluk_labels = []
    district_labels = []
    
    if taluk and district:
        taluk_labels = [taluk] * n_units
        district_labels = [district] * n_units
    elif district:
        taluks = get_taluks(state, district)
        if taluks:
            taluk_labels = np.random.choice(taluks, n_units).tolist()
            district_labels = [district] * n_units
        else:
            taluk_labels = ["Unknown"] * n_units
            district_labels = [district] * n_units
    else:
        from engine.admin_hierarchy import get_districts
        districts = get_districts(state)
        if districts:
            district_labels = np.random.choice(districts, n_units).tolist()
            taluk_labels = []
            for d in district_labels:
                t_list = get_taluks(state, d)
                taluk_labels.append(np.random.choice(t_list) if t_list else "Unknown")
        else:
            district_labels = ["Unknown"] * n_units
            taluk_labels = ["Unknown"] * n_units

    # Generate spatial coordinates (on land only — basic check)
    lats = np.random.uniform(min_lat, max_lat, n_units)
    lons = np.random.uniform(min_lon, max_lon, n_units)
    
    # Generate terrain parameters
    slope_angles = np.random.beta(2, 3, n_units) * 60 + 5  # 5-65°
    aspects = np.random.uniform(0, 360, n_units)
    elevations = np.random.uniform(200, 2500, n_units)
    soil_depths = np.random.uniform(0.5, 3.0, n_units)
    
    # Soil and geology
    soil_list = list(SOIL_TYPES.keys())
    geology_list = GEOLOGY_TYPES
    landuse_list = LANDUSE_TYPES
    
    soils = np.random.choice(soil_list, n_units)
    geology = np.random.choice(geology_list, n_units)
    landuse = np.random.choice(landuse_list, n_units)
    
    # Extract soil properties
    cohesion = np.array([SOIL_TYPES[s]["cohesion"] for s in soils])
    friction = np.array([SOIL_TYPES[s]["friction"] for s in soils])
    unit_weight = np.array([SOIL_TYPES[s]["unit_weight"] for s in soils])
    
    # NDVI proxy from landuse
    ndvi_map = {
        "Dense Forest": 0.75, "Degraded Forest": 0.45, "Plantation": 0.55,
        "Agriculture": 0.35, "Barren": 0.10, "Settlement": 0.15
    }
    ndvi = np.array([ndvi_map[lu] + np.random.normal(0, 0.05) for lu in landuse])
    ndvi = np.clip(ndvi, 0.05, 0.95)
    
    # TWI (Topographic Wetness Index) — proxy
    catchment_area = np.random.uniform(1000, 50000, n_units)
    twi = np.log(catchment_area / np.tan(np.radians(np.clip(slope_angles, 1, 80))))
    
    # Past landslide flag
    past_landslide = np.random.choice([0, 1], n_units, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        "slope_id": [f"SU_{i:04d}" for i in range(n_units)],
        "state": state,
        "district": district_labels,
        "taluk": taluk_labels,
        "lat": np.round(lats, 5),
        "lon": np.round(lons, 5),
        "elevation_m": np.round(elevations, 1),
        "slope_angle_deg": np.round(slope_angles, 2),
        "aspect_deg": np.round(aspects, 1),
        "soil_depth_m": np.round(soil_depths, 2),
        "soil_type": soils,
        "geology": geology,
        "landuse": landuse,
        "cohesion_kpa": np.round(cohesion + np.random.normal(0, 1, n_units), 1),
        "friction_angle_deg": np.round(friction + np.random.normal(0, 2, n_units), 1),
        "unit_weight_kn": np.round(unit_weight, 1),
        "ndvi": np.round(ndvi, 3),
        "twi": np.round(twi, 3),
        "past_landslide": past_landslide,
        "catchment_area_m2": np.round(catchment_area, 0),
    })
    
    return df


def load_or_generate(
    state: str,
    district: Optional[str] = None,
    taluk: Optional[str] = None,
    uploaded_df: Optional[pd.DataFrame] = None,
    n_units: int = 60,
) -> pd.DataFrame:
    """Load uploaded data or generate synthetic slope units"""
    if uploaded_df is not None and not uploaded_df.empty:
        # Filter uploaded data to admin level
        df = uploaded_df.copy()
        if "state" in df.columns:
            df = df[df["state"] == state]
        if district and "district" in df.columns:
            df = df[df["district"] == district]
        if taluk and "taluk" in df.columns:
            df = df[df["taluk"] == taluk]
        if not df.empty:
            return df
    
    # Use n_units scaled to level
    n_map = {"state": 120, "district": 80, "taluk": 40}
    level = "taluk" if taluk else "district" if district else "state"
    n = n_units if n_units else n_map[level]
    
    return generate_slope_units(state, district, taluk, n)
