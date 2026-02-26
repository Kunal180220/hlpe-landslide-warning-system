"""
test_engines.py
================
Unit tests for HLPE engine modules.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

# Minimal test config
CONFIG = {
    'study_area': {
        'name': 'Test Region',
        'bbox': {'min_lon': 75.0, 'max_lon': 77.0, 'min_lat': 10.0, 'max_lat': 12.0},
        'crs': 'EPSG:4326'
    },
    'rainfall': {'antecedent_days': 15, 'swi_timescale_days': 5},
    'soil': {'default_params': {'cohesion_kpa': 5, 'friction_angle_deg': 32,
                                 'unit_weight_kNm3': 18, 'soil_depth_m': 1.5}},
    'physics': {'fs_thresholds': {'extreme': 0.9, 'high': 1.1,
                                   'elevated': 1.3, 'moderate': 1.5}},
    'id_thresholds': {'auto_calibrate': False, 'default_alpha': 14.82, 'default_beta': 0.39,
                      'min_events_for_calibration': 10},
    'ml': {'enabled': True, 'n_estimators': 10, 'min_training_samples': 5},
    'consensus': {'weight_fs': 0.30, 'weight_ml': 0.35,
                  'weight_id': 0.20, 'weight_swi': 0.15},
    'api_keys': {'nasa_earthdata': ''}
}

RAIN_DATA = {
    'r_today_mm': 25.0,
    'r_3day_mm': 60.0,
    'r_7day_mm': 90.0,
    'r_15day_mm': 120.0,
    'api': 75.0,
    'swi': 30.0,
    'swi_normalized': 0.6,
    'peak_intensity_mm_day': 35.0,
    'forecast_tomorrow_mm': 15.0,
    'forecast_3day_mm': 40.0,
    'data_source': 'test',
    'data_quality': 'HIGH',
    'last_updated': '2024-01-01T00:00:00'
}


# ─────────────────────────────────────────────────────────────
# FACTOR OF SAFETY TESTS
# ─────────────────────────────────────────────────────────────

class TestFactorOfSafety:
    def setup_method(self):
        from engine.factor_of_safety import FactorOfSafetyEngine, SlopeParameters
        self.engine = FactorOfSafetyEngine(CONFIG)
        self.Params = SlopeParameters

    def test_stable_slope(self):
        """Gentle dry slope should be stable."""
        params = self.Params(slope_angle_deg=15.0, soil_depth_m=1.0,
                              cohesion_kpa=8.0, friction_angle_deg=35.0)
        dry_rain = {**RAIN_DATA, 'swi_normalized': 0.0, 'r_today_mm': 0}
        result = self.engine.compute(params, dry_rain)
        assert result.factor_of_safety > 1.5, "Gentle dry slope should be stable"
        assert result.fs_category == 'LOW'

    def test_saturated_steep_slope_fails(self):
        """Very steep fully saturated slope should fail (FS < 1)."""
        params = self.Params(slope_angle_deg=42.0, soil_depth_m=2.0,
                              cohesion_kpa=2.0, friction_angle_deg=25.0)
        wet_rain = {**RAIN_DATA, 'swi_normalized': 0.99, 'r_today_mm': 150}
        result = self.engine.compute(params, wet_rain)
        assert result.factor_of_safety < 1.3, "Steep saturated slope should be high risk"

    def test_fs_decreases_with_saturation(self):
        """FS should decrease as saturation increases."""
        params = self.Params(slope_angle_deg=30.0)
        fs_dry = self.engine.compute(params, {**RAIN_DATA, 'swi_normalized': 0.0, 'r_today_mm': 0})
        fs_wet = self.engine.compute(params, {**RAIN_DATA, 'swi_normalized': 0.9, 'r_today_mm': 100})
        assert fs_dry.factor_of_safety > fs_wet.factor_of_safety

    def test_flat_slope_result(self):
        """Zero slope should return safe result."""
        params = self.Params(slope_angle_deg=0.0)
        result = self.engine.compute(params, RAIN_DATA)
        assert result.fs_category == 'LOW'

    def test_failure_probability_increases_with_lower_fs(self):
        params_steep = self.Params(slope_angle_deg=40.0, cohesion_kpa=1.0)
        params_gentle = self.Params(slope_angle_deg=15.0, cohesion_kpa=10.0)
        r_steep  = self.engine.compute(params_steep, {**RAIN_DATA, 'swi_normalized': 0.8})
        r_gentle = self.engine.compute(params_gentle, {**RAIN_DATA, 'swi_normalized': 0.1})
        assert r_steep.failure_probability >= r_gentle.failure_probability


# ─────────────────────────────────────────────────────────────
# RAINFALL FETCHER TESTS
# ─────────────────────────────────────────────────────────────

class TestRainfallFetcher:
    def setup_method(self):
        from engine.rainfall_fetcher import RainfallFetcher
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.fetcher = RainfallFetcher(CONFIG, data_dir=self.tmpdir)

    def test_swi_computation(self):
        """SWI should be non-negative and respond to rain."""
        rain_series = np.array([0, 0, 50, 80, 10, 5, 0])
        swi = self.fetcher._compute_swi(rain_series, T=5)
        assert swi > 0, "SWI should be positive after rain"

    def test_swi_zero_on_empty(self):
        swi = self.fetcher._compute_swi(np.array([]), T=5)
        assert swi == 0.0

    def test_manual_reading_saved(self):
        record = self.fetcher.add_manual_reading(
            'STA001', '2024-01-15', 45.5, 11.0, 76.0)
        assert record['rainfall_mm'] == 45.5
        df = self.fetcher.get_manual_readings()
        assert len(df) == 1

    def test_fallback_data_structure(self):
        fallback = self.fetcher._fallback_data()
        for key in ['r_today_mm', 'r_3day_mm', 'swi', 'swi_normalized']:
            assert key in fallback


# ─────────────────────────────────────────────────────────────
# ID THRESHOLD TESTS
# ─────────────────────────────────────────────────────────────

class TestIDThreshold:
    def setup_method(self):
        from engine.id_threshold import IDThresholdEngine
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.engine = IDThresholdEngine(CONFIG, models_dir=self.tmpdir)

    def test_threshold_decreases_with_duration(self):
        """Longer duration → lower intensity threshold (power law)."""
        I_1h  = self.engine.compute_threshold_intensity(1)
        I_24h = self.engine.compute_threshold_intensity(24)
        assert I_1h > I_24h, "Shorter duration should have higher threshold intensity"

    def test_conservative_threshold_lower(self):
        """5% threshold should be below 50% threshold."""
        I_50 = self.engine.compute_threshold_intensity(24, level=0.50)
        I_5  = self.engine.compute_threshold_intensity(24, level=0.05)
        assert I_5 < I_50

    def test_no_rain_below_threshold(self):
        result = self.engine.get_current_storm_position({
            **RAIN_DATA, 'r_today_mm': 0, 'r_3day_mm': 0
        })
        assert result['current_intensity_mm_hr'] == 0.0

    def test_id_curve_data_shape(self):
        df = self.engine.get_id_curve_data()
        assert len(df) > 0
        assert 'duration_h' in df.columns
        assert 'I_thresh_50' in df.columns


# ─────────────────────────────────────────────────────────────
# CONSENSUS SCORER TESTS
# ─────────────────────────────────────────────────────────────

class TestConsensusScorer:
    def setup_method(self):
        from engine.consensus_scorer import ConsensusScorer
        from engine.factor_of_safety import FactorOfSafetyEngine, SlopeParameters
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.scorer = ConsensusScorer(CONFIG, models_dir=self.tmpdir)
        self.fs_engine = FactorOfSafetyEngine(CONFIG)
        self.Params = SlopeParameters

    def test_high_risk_scenario(self):
        """Very unstable slope should get high risk score."""
        params = self.Params(slope_angle_deg=45.0, cohesion_kpa=1.0,
                              friction_angle_deg=22.0, soil_depth_m=2.0)
        wet_rain = {**RAIN_DATA, 'swi_normalized': 0.95, 'r_today_mm': 150}
        fs_result = self.fs_engine.compute(params, wet_rain)
        ml_result = {'ml_probability': 0.85, 'ml_risk_score': 85,
                      'ml_confidence': 'LOW', 'shap_values': {}, 'model_mode': 'proxy', 'trained': False}
        id_result = {'threshold_ratio': 2.5, 'exceedance_level': 'HIGH_PROBABILITY',
                      'status': 'FAR_ABOVE', 'current_duration_h': 24,
                      'current_intensity_mm_hr': 6.0, 'exceedance': {'exceeds': True}}
        result = self.scorer.compute_risk_score(fs_result, ml_result, id_result, wet_rain)
        assert result['risk_score'] > 60, "Should be high risk"

    def test_stable_scenario(self):
        """Very stable dry slope should get low risk score."""
        params = self.Params(slope_angle_deg=10.0, cohesion_kpa=12.0,
                              friction_angle_deg=38.0, soil_depth_m=1.0)
        dry_rain = {**RAIN_DATA, 'swi_normalized': 0.0, 'r_today_mm': 0,
                    'r_3day_mm': 0, 'r_7day_mm': 0, 'swi': 0,
                    'forecast_3day_mm': 0}
        fs_result = self.fs_engine.compute(params, dry_rain)
        ml_result = {'ml_probability': 0.02, 'ml_risk_score': 2,
                      'ml_confidence': 'LOW', 'shap_values': {}, 'model_mode': 'proxy', 'trained': False}
        id_result = {'threshold_ratio': 0.1, 'exceedance_level': 'BELOW_THRESHOLD',
                      'status': 'BELOW', 'current_duration_h': 24,
                      'current_intensity_mm_hr': 0.0, 'exceedance': {'exceeds': False}}
        result = self.scorer.compute_risk_score(fs_result, ml_result, id_result, dry_rain)
        assert result['risk_score'] < 50, "Should be low risk"

    def test_weight_normalization(self):
        weights = self.scorer.weights
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1"

    def test_output_keys(self):
        params = self.Params(slope_angle_deg=30.0)
        fs_result = self.fs_engine.compute(params, RAIN_DATA)
        ml_result = {'ml_probability': 0.3, 'ml_risk_score': 30,
                      'ml_confidence': 'LOW', 'shap_values': {}, 'model_mode': 'proxy', 'trained': False}
        id_result = {'threshold_ratio': 0.5, 'exceedance_level': 'BELOW_THRESHOLD',
                      'status': 'BELOW', 'current_duration_h': 24,
                      'current_intensity_mm_hr': 0.5, 'exceedance': {'exceeds': False}}
        result = self.scorer.compute_risk_score(fs_result, ml_result, id_result, RAIN_DATA)
        for key in ['risk_score', 'risk_level', 'risk_color', 'recommended_action',
                    'component_scores', 'factor_of_safety']:
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────
# SUSCEPTIBILITY TESTS
# ─────────────────────────────────────────────────────────────

class TestSusceptibility:
    def setup_method(self):
        from engine.susceptibility import SusceptibilityEngine
        self.engine = SusceptibilityEngine(CONFIG)

    def test_steep_geology_high_susceptibility(self):
        high_risk = {'slope_angle_deg': 45, 'geology': 'shale',
                     'ndvi': 0.1, 'previous_landslide': True, 'reactivation_count': 2,
                     'twi': 10, 'distance_to_stream_m': 50}
        score = self.engine.compute_susceptibility_score(high_risk)
        assert score > 0.6, f"Expected high susceptibility, got {score}"

    def test_gentle_stable_low_susceptibility(self):
        low_risk = {'slope_angle_deg': 8, 'geology': 'granite',
                    'ndvi': 0.7, 'previous_landslide': False, 'reactivation_count': 0,
                    'twi': 4, 'distance_to_stream_m': 1000}
        score = self.engine.compute_susceptibility_score(low_risk)
        assert score < 0.4, f"Expected low susceptibility, got {score}"

    def test_score_range(self):
        for _ in range(20):
            import random
            data = {
                'slope_angle_deg': random.uniform(0, 70),
                'geology': random.choice(['granite', 'shale', 'alluvium']),
                'ndvi': random.uniform(0, 1),
                'previous_landslide': random.choice([True, False]),
                'reactivation_count': random.randint(0, 3),
                'twi': random.uniform(3, 12),
                'distance_to_stream_m': random.uniform(10, 2000)
            }
            score = self.engine.compute_susceptibility_score(data)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_synthetic_generation(self):
        df = self.engine.generate_synthetic_slope_units(n_units=10)
        assert len(df) == 10
        assert 'slope_angle_deg' in df.columns
        assert 'susceptibility' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
