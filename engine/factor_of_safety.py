"""
factor_of_safety.py
====================
Physics-based slope stability engine for HLPE.

Implements:
- Infinite Slope Model (core)
- Iverson (2000) pore pressure response
- SWI-based saturation proxy
- Newmark displacement estimate
- Pedotransfer functions for soil parameters
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from loguru import logger


# ─────────────────────────────────────────────────────────
# SOIL PARAMETER LOOKUP (from texture class)
# Based on Saxton & Rawls (2006) pedotransfer functions
# ─────────────────────────────────────────────────────────

SOIL_TEXTURE_PARAMS = {
    # texture_class: (cohesion_kPa, friction_angle_deg, unit_weight_kNm3, Ks_ms)
    'clay':         (10.0, 20.0, 18.5, 1e-8),
    'silty_clay':   (8.0,  22.0, 18.0, 5e-8),
    'sandy_clay':   (6.0,  25.0, 17.5, 1e-7),
    'clay_loam':    (5.0,  27.0, 17.8, 5e-7),
    'silty_clay_loam': (4.5, 26.0, 17.5, 3e-7),
    'sandy_clay_loam': (3.5, 28.0, 17.0, 8e-7),
    'loam':         (3.0,  30.0, 17.0, 2e-6),
    'silt_loam':    (2.5,  28.0, 16.5, 8e-7),
    'silt':         (2.0,  26.0, 16.0, 5e-7),
    'sandy_loam':   (2.0,  32.0, 16.5, 5e-6),
    'loamy_sand':   (1.0,  34.0, 16.0, 2e-5),
    'sand':         (0.5,  36.0, 15.5, 5e-5),
    'default':      (5.0,  32.0, 18.0, 1e-6),
}

GEOLOGY_COHESION_FACTOR = {
    # Multiplier on cohesion based on parent material
    'granite':      1.5,
    'basalt':       1.3,
    'schist':       0.8,
    'shale':        0.6,
    'limestone':    1.2,
    'sandstone':    1.0,
    'alluvium':     0.7,
    'colluvium':    0.6,
    'default':      1.0,
}


@dataclass
class SlopeParameters:
    """All parameters for a slope unit stability analysis."""
    slope_angle_deg: float          # β (degrees)
    soil_depth_m: float = 1.5       # Z (m)
    cohesion_kpa: float = 5.0       # C' (kPa)
    friction_angle_deg: float = 32.0  # φ' (degrees)
    soil_unit_weight: float = 18.0  # γs (kN/m³)
    water_unit_weight: float = 9.81  # γw (kN/m³)
    texture_class: str = 'default'
    geology: str = 'default'
    soil_depth_method: str = 'default'  # how depth was estimated


@dataclass
class StabilityResult:
    """Complete output of slope stability analysis."""
    factor_of_safety: float
    fs_category: str               # EXTREME / HIGH / ELEVATED / MODERATE / LOW
    saturation_ratio_m: float      # m (0-1)
    pore_pressure_head_m: float
    normal_stress_kpa: float
    shear_strength_kpa: float
    driving_stress_kpa: float
    newmark_displacement_cm: float
    failure_probability: float     # P(FS<1) from uncertainty
    confidence: str                # HIGH / MEDIUM / LOW
    warnings: list = field(default_factory=list)


class FactorOfSafetyEngine:
    """
    Computes slope stability using the infinite slope model
    with dynamic pore pressure from SWI and antecedent rainfall.
    """

    # FS thresholds
    FS_EXTREME  = 0.9
    FS_HIGH     = 1.1
    FS_ELEVATED = 1.3
    FS_MODERATE = 1.5

    def __init__(self, config: dict):
        self.config = config
        phys = config.get('physics', {})
        self.fs_thresholds = phys.get('fs_thresholds', {
            'extreme': 0.9, 'high': 1.1, 'elevated': 1.3, 'moderate': 1.5
        })
        logger.info("FactorOfSafetyEngine initialized")

    # ─────────────────────────────────────────────
    # MAIN COMPUTATION
    # ─────────────────────────────────────────────

    def compute(self,
                params: SlopeParameters,
                rainfall_data: Dict) -> StabilityResult:
        """
        Full stability analysis for one slope unit.

        Args:
            params: Slope geometric and soil parameters
            rainfall_data: Output dict from RainfallFetcher.get_current_rainfall()

        Returns:
            StabilityResult with FS and all derived quantities
        """
        warnings_list = []

        # Apply geology modifier to cohesion
        geo_factor = GEOLOGY_COHESION_FACTOR.get(
            params.geology.lower(), GEOLOGY_COHESION_FACTOR['default'])
        C = params.cohesion_kpa * geo_factor

        # Get soil params from texture if not manually set
        if params.texture_class != 'default':
            tx_params = SOIL_TEXTURE_PARAMS.get(params.texture_class,
                                                 SOIL_TEXTURE_PARAMS['default'])
            C = max(C, tx_params[0] * geo_factor)
            phi = tx_params[1]
            gamma_s = tx_params[2]
        else:
            phi = params.friction_angle_deg
            gamma_s = params.soil_unit_weight

        gamma_w = params.water_unit_weight
        Z = params.soil_depth_m
        beta_deg = params.slope_angle_deg

        # Validate slope angle
        if beta_deg >= 90:
            warnings_list.append("Slope angle ≥ 90°: vertical/overhanging — clamped to 89°")
            beta_deg = 89.0
        if beta_deg <= 0:
            return self._flat_slope_result(warnings_list)

        beta_rad = np.radians(beta_deg)
        phi_rad  = np.radians(phi)

        # ── Saturation ratio m from SWI ──
        swi_norm = rainfall_data.get('swi_normalized', 0.0)
        m = self._compute_saturation_ratio(swi_norm, rainfall_data)

        # ── Pore pressure head ──
        psi = m * Z * np.cos(beta_rad)**2  # pressure head at failure plane (m)

        # ── Effective normal stress ──
        sigma_n = (gamma_s * Z * np.cos(beta_rad)**2 -
                   gamma_w * psi)  # kPa

        # ── Shear strength (Mohr-Coulomb) ──
        tau_strength = C + sigma_n * np.tan(phi_rad)

        # ── Driving stress ──
        tau_drive = gamma_s * Z * np.sin(beta_rad) * np.cos(beta_rad)

        # ── Factor of Safety ──
        if tau_drive <= 0:
            fs = 999.0
        else:
            fs = tau_strength / tau_drive

        # Clamp FS to sensible range
        fs = max(0.01, min(fs, 10.0))

        # ── Newmark Displacement (simplified) ──
        disp = self._newmark_displacement(fs, beta_deg)

        # ── Failure probability from uncertainty ──
        p_fail = self._failure_probability(fs, params)

        # ── Category ──
        category = self._categorize_fs(fs)

        # ── Confidence ──
        confidence = self._assess_confidence(
            params, rainfall_data.get('data_quality', 'MEDIUM'))

        # ── Additional warnings ──
        if swi_norm > 0.8:
            warnings_list.append("⚠️ Very high soil water index — near saturation")
        if rainfall_data.get('r_today_mm', 0) > 50:
            warnings_list.append("⚠️ Extreme daily rainfall (>50mm)")
        if m > 0.9:
            warnings_list.append("⚠️ Slope near full saturation")
        if beta_deg > 45:
            warnings_list.append("⚠️ Very steep slope — model uncertainty higher")

        return StabilityResult(
            factor_of_safety=round(fs, 3),
            fs_category=category,
            saturation_ratio_m=round(m, 3),
            pore_pressure_head_m=round(psi, 3),
            normal_stress_kpa=round(sigma_n, 2),
            shear_strength_kpa=round(tau_strength, 2),
            driving_stress_kpa=round(tau_drive, 2),
            newmark_displacement_cm=round(disp, 2),
            failure_probability=round(p_fail, 3),
            confidence=confidence,
            warnings=warnings_list
        )

    def compute_batch(self,
                      slope_df: pd.DataFrame,
                      rainfall_data: Dict) -> pd.DataFrame:
        """
        Compute FS for all slope units in a DataFrame.

        Expected columns in slope_df:
            slope_angle_deg, soil_depth_m, cohesion_kpa,
            friction_angle_deg, texture_class, geology
        """
        results = []
        for idx, row in slope_df.iterrows():
            params = SlopeParameters(
                slope_angle_deg=row.get('slope_angle_deg', 30),
                soil_depth_m=row.get('soil_depth_m', 1.5),
                cohesion_kpa=row.get('cohesion_kpa', 5.0),
                friction_angle_deg=row.get('friction_angle_deg', 32.0),
                soil_unit_weight=row.get('unit_weight', 18.0),
                texture_class=row.get('texture_class', 'default'),
                geology=row.get('geology', 'default'),
            )

            # Use location-specific rainfall if available
            loc_rain = rainfall_data
            if 'lat' in row and 'lon' in row:
                # Could fetch point-specific rainfall here
                pass

            result = self.compute(params, loc_rain)
            results.append({
                'slope_id': idx,
                'factor_of_safety': result.factor_of_safety,
                'fs_category': result.fs_category,
                'saturation_m': result.saturation_ratio_m,
                'pore_pressure_m': result.pore_pressure_head_m,
                'newmark_disp_cm': result.newmark_displacement_cm,
                'failure_prob': result.failure_probability,
                'confidence': result.confidence,
                'warnings': '; '.join(result.warnings)
            })

        return pd.DataFrame(results)

    # ─────────────────────────────────────────────
    # SATURATION RATIO
    # ─────────────────────────────────────────────

    def _compute_saturation_ratio(self, swi_norm: float,
                                   rainfall_data: Dict) -> float:
        """
        Compute saturation ratio m (0-1) using SWI + rainfall combination.
        Based on SLIP model approach (Brocca et al., 2012).

        m = 0 → fully dry slope
        m = 1 → fully saturated (water table at surface)
        """
        # Base from SWI (long-term soil moisture memory)
        m_swi = swi_norm

        # Boost from today's rain (Iverson-style rapid response)
        r_today = rainfall_data.get('r_today_mm', 0.0)
        rapid_boost = min(r_today / 100.0, 0.3)  # max 30% boost from today

        # Combine: 70% from SWI (antecedent), 30% from rapid response
        m = 0.70 * m_swi + rapid_boost

        return float(np.clip(m, 0.0, 1.0))

    # ─────────────────────────────────────────────
    # NEWMARK DISPLACEMENT
    # ─────────────────────────────────────────────

    def _newmark_displacement(self, fs: float, slope_deg: float) -> float:
        """
        Simplified Newmark displacement estimate.
        Jibson (2007) empirical relationship.
        Returns displacement in cm.
        """
        if fs >= 1.5:
            return 0.0

        # Critical acceleration ratio
        Ia = max(0.0, 1.0 - (1.0 / max(fs, 0.01)))  # simplified

        if Ia <= 0:
            return 0.0

        # Jibson (2007): ln(D) = 0.215 + log(Ia/(1-Ia)*Ia^0.5)
        # Simplified version for pre-failure displacement
        slope_factor = np.sin(np.radians(slope_deg)) / np.sin(np.radians(max(slope_deg - 5, 1)))
        D = 0.5 * (1.0 - fs)**2 * slope_factor * 100  # cm

        return float(max(0.0, D))

    # ─────────────────────────────────────────────
    # FAILURE PROBABILITY (UNCERTAINTY)
    # ─────────────────────────────────────────────

    def _failure_probability(self, fs: float, params: SlopeParameters) -> float:
        """
        Estimate P(failure) using simplified reliability analysis.
        Assumes FS follows lognormal distribution with CoV based on data quality.
        """
        # Coefficient of variation based on data quality
        cov_fs = 0.15  # 15% default uncertainty in FS

        if params.texture_class == 'default':
            cov_fs += 0.05  # more uncertainty without texture data

        # Log-normal reliability
        # β_reliability = ln(FS) / (CoV_FS) — simplified
        if fs <= 0:
            return 1.0

        ln_fs = np.log(fs)
        sigma_ln = cov_fs

        # P(failure) = P(FS < 1) = Φ(-β)
        from scipy.stats import norm
        beta_r = ln_fs / sigma_ln
        p_fail = float(norm.cdf(-beta_r))

        return np.clip(p_fail, 0.0, 1.0)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _categorize_fs(self, fs: float) -> str:
        t = self.fs_thresholds
        if fs < t['extreme']:   return 'EXTREME'
        elif fs < t['high']:    return 'HIGH'
        elif fs < t['elevated']: return 'ELEVATED'
        elif fs < t['moderate']: return 'MODERATE'
        else:                   return 'LOW'

    def _assess_confidence(self, params: SlopeParameters,
                            data_quality: str) -> str:
        score = 0
        if params.texture_class != 'default': score += 1
        if params.geology != 'default':       score += 1
        if data_quality in ['HIGH']:          score += 2
        elif data_quality in ['MEDIUM']:      score += 1
        if params.soil_depth_method != 'default': score += 1
        if score >= 4:   return 'HIGH'
        elif score >= 2: return 'MEDIUM'
        else:            return 'LOW'

    def _flat_slope_result(self, warnings: list) -> StabilityResult:
        return StabilityResult(
            factor_of_safety=9.99,
            fs_category='LOW',
            saturation_ratio_m=0.0,
            pore_pressure_head_m=0.0,
            normal_stress_kpa=0.0,
            shear_strength_kpa=999.0,
            driving_stress_kpa=0.0,
            newmark_displacement_cm=0.0,
            failure_probability=0.0,
            confidence='HIGH',
            warnings=warnings + ['Flat slope — no landslide risk']
        )

    def get_soil_params_from_texture(self, texture_class: str) -> Dict:
        """Public method to get soil params from texture class."""
        params = SOIL_TEXTURE_PARAMS.get(texture_class.lower(),
                                          SOIL_TEXTURE_PARAMS['default'])
        return {
            'cohesion_kpa': params[0],
            'friction_angle_deg': params[1],
            'unit_weight_kNm3': params[2],
            'hydraulic_conductivity_ms': params[3]
        }
