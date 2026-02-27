"""
Factor of Safety Calculator
Infinite slope model with partial saturation from SWI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calc_factor_of_safety(
    slope_angle_deg: float,
    soil_depth_m: float,
    cohesion_kpa: float,
    friction_angle_deg: float,
    unit_weight_kn: float = 18.0,
    m_ratio: float = 0.0,  # saturation ratio 0-1 from SWI
) -> Dict:
    """
    Infinite slope stability: FS = (C' + (γs - γw·m)·Z·cos²β·tanφ') / (γs·Z·sinβ·cosβ)
    """
    gamma_s = unit_weight_kn
    gamma_w = 9.81
    Z = soil_depth_m
    beta = np.radians(slope_angle_deg)
    phi = np.radians(friction_angle_deg)
    C = cohesion_kpa
    m = np.clip(m_ratio, 0.0, 1.0)
    
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    
    if sin_b < 1e-6:
        return {"FS": 99.0, "status": "STABLE", "driving_stress": 0.0, "resisting_stress": 0.0}
    
    driving = gamma_s * Z * sin_b * cos_b
    effective_normal = (gamma_s - gamma_w * m) * Z * cos_b ** 2
    resisting = C + effective_normal * np.tan(phi)
    
    FS = resisting / (driving + 1e-10)
    FS = round(float(np.clip(FS, 0.1, 20.0)), 3)
    
    # Newmark displacement proxy (mm)
    if FS >= 1.0:
        newmark_d = 0.0
    else:
        ky = FS  # yield acceleration proxy
        newmark_d = round(10 ** (1.5 - 5.0 * ky), 1)  # simplified
    
    if FS < 1.0:
        status = "FAIL"
    elif FS < 1.2:
        status = "CRITICAL"
    elif FS < 1.5:
        status = "WATCH"
    else:
        status = "STABLE"
    
    return {
        "FS": FS,
        "status": status,
        "driving_stress": round(float(driving), 2),
        "resisting_stress": round(float(resisting), 2),
        "newmark_displacement_mm": newmark_d,
        "pore_pressure_kpa": round(float(gamma_w * m * Z * cos_b ** 2), 2),
    }


def compute_fs_for_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply FS calculation to entire slope unit dataframe"""
    results = []
    for _, row in df.iterrows():
        m = row.get("m_ratio", 0.3)
        fs_out = calc_factor_of_safety(
            slope_angle_deg=row.get("slope_angle_deg", 25),
            soil_depth_m=row.get("soil_depth_m", 1.5),
            cohesion_kpa=row.get("cohesion_kpa", 8.0),
            friction_angle_deg=row.get("friction_angle_deg", 28.0),
            unit_weight_kn=row.get("unit_weight_kn", 18.0),
            m_ratio=m,
        )
        results.append(fs_out)
    
    fs_df = pd.DataFrame(results)
    for col in fs_df.columns:
        df[col] = fs_df[col].values
    return df


def id_threshold_check(R_24hr: float, R_15day: float, geology: str = "default") -> Dict:
    """
    Rainfall Intensity-Duration threshold check.
    Returns exceedance flag and position on curve.
    """
    # Regional ID parameters (alpha, beta for I = alpha * D^-beta)
    params = {
        "Charnockite": (12.0, 0.45),
        "Gneiss": (10.0, 0.40),
        "Schist": (8.0, 0.38),
        "Granite": (14.0, 0.42),
        "default": (10.0, 0.40),
    }
    alpha, beta = params.get(geology, params["default"])
    
    # Duration = 1 day, intensity = R_24hr
    threshold = alpha * (1.0 ** (-beta))
    
    # Position on curve (> 1.0 means exceeded)
    position = R_24hr / (threshold + 1e-6)
    
    # Antecedent factor
    antecedent_factor = 1.0 + (R_15day / 150.0)  # wet antecedent lowers threshold
    adjusted_threshold = threshold / antecedent_factor
    adjusted_position = R_24hr / (adjusted_threshold + 1e-6)
    
    return {
        "id_threshold_mm": round(threshold, 1),
        "id_adjusted_threshold_mm": round(adjusted_threshold, 1),
        "id_exceedance_ratio": round(adjusted_position, 3),
        "id_exceeded": adjusted_position >= 1.0,
    }


def compute_consensus_score(row) -> Dict:
    """
    Compute final consensus risk score from FS, ID, SWI.
    Risk = w1*(FS danger) + w2*(ID exceeded) + w3*(SWI saturation)
    """
    FS = row.get("FS", 1.5)
    id_ratio = row.get("id_exceedance_ratio", 0.5)
    swi = row.get("SWI", 0.0)
    past_ls = row.get("past_landslide", 0)
    
    # FS component (0-1)
    if FS < 1.0:
        fs_score = 1.0
    elif FS < 1.5:
        fs_score = (1.5 - FS) / 0.5
    else:
        fs_score = 0.0
    
    # ID component (0-1)
    id_score = min(id_ratio, 2.0) / 2.0
    
    # SWI component (0-1)
    swi_score = min(swi / 30.0, 1.0)
    
    # Past landslide reactivation penalty
    reactivation_bonus = 0.15 if past_ls == 1 else 0.0
    
    # Weights
    raw = 0.35 * fs_score + 0.30 * id_score + 0.20 * swi_score + 0.15 * reactivation_bonus
    risk_score = round(min(raw + reactivation_bonus, 1.0) * 100, 1)
    
    if risk_score >= 80:
        alert = "EXTREME"
        color = "#2c0a0a"
        bg = "#ff2929"
    elif risk_score >= 60:
        alert = "HIGH"
        color = "#ff4d4d"
        bg = "#ff6b6b"
    elif risk_score >= 40:
        alert = "ELEVATED"
        color = "#ff8c00"
        bg = "#ffa500"
    elif risk_score >= 20:
        alert = "MODERATE"
        color = "#ffd700"
        bg = "#ffd700"
    else:
        alert = "LOW"
        color = "#2ecc71"
        bg = "#27ae60"
    
    return {
        "risk_score": risk_score,
        "alert_level": alert,
        "risk_color": bg,
    }


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Full computation pipeline: FS → ID → Consensus"""
    # 1. Factor of safety
    df = compute_fs_for_all(df)
    
    # 2. ID threshold
    id_results = []
    for _, row in df.iterrows():
        id_out = id_threshold_check(
            R_24hr=row.get("R_24hr", 0),
            R_15day=row.get("R_15day", 0),
            geology=row.get("geology", "default"),
        )
        id_results.append(id_out)
    id_df = pd.DataFrame(id_results)
    for col in id_df.columns:
        df[col] = id_df[col].values
    
    # 3. Consensus risk
    scores = df.apply(compute_consensus_score, axis=1)
    scores_df = pd.DataFrame(list(scores))
    for col in scores_df.columns:
        df[col] = scores_df[col].values
    
    return df
