"""
consensus_scorer.py
====================
Final Risk Score Fusion Engine for HLPE.

Combines all model outputs into a single 0-100 risk score.
Weights are auto-calibrated from validation data.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


RISK_LEVELS = {
    'EXTREME':  {'min': 90, 'color': '#1a1a1a', 'emoji': '⚫', 'action': 'Immediate evacuation'},
    'HIGH':     {'min': 70, 'color': '#dc2626', 'emoji': '🔴', 'action': 'Alert field teams'},
    'ELEVATED': {'min': 50, 'color': '#ea580c', 'emoji': '🟠', 'action': 'Restrict access'},
    'MODERATE': {'min': 30, 'color': '#ca8a04', 'emoji': '🟡', 'action': 'Daily monitoring'},
    'LOW':      {'min': 0,  'color': '#16a34a', 'emoji': '🟢', 'action': 'Normal operations'},
}


class ConsensusScorer:
    """
    Fuses physics, ML, ID threshold, and SWI signals into a final risk score.
    """

    def __init__(self, config: dict, models_dir: str = "models"):
        self.config = config
        self.models_dir = Path(models_dir)

        # Default weights (sum to 1.0)
        consensus_cfg = config.get('consensus', {})
        self.weights = {
            'fs':  consensus_cfg.get('weight_fs',  0.30),
            'ml':  consensus_cfg.get('weight_ml',  0.35),
            'id':  consensus_cfg.get('weight_id',  0.20),
            'swi': consensus_cfg.get('weight_swi', 0.15),
        }
        self._normalize_weights()
        self._load_calibrated_weights()

        logger.info(f"ConsensusScorer weights: {self.weights}")

    # ─────────────────────────────────────────────
    # MAIN SCORING
    # ─────────────────────────────────────────────

    def compute_risk_score(self,
                           stability_result,   # StabilityResult object
                           ml_result: Dict,
                           id_result: Dict,
                           rainfall_data: Dict,
                           slope_metadata: Optional[Dict] = None) -> Dict:
        """
        Compute final consensus risk score (0-100).

        Args:
            stability_result: Output from FactorOfSafetyEngine.compute()
            ml_result: Output from MLRiskPredictor.predict()
            id_result: Output from IDThresholdEngine.get_current_storm_position()
            rainfall_data: Output from RainfallFetcher.get_current_rainfall()
            slope_metadata: Optional dict with slope_id, lat, lon, name

        Returns:
            Complete risk assessment dict
        """
        # ── Component Scores (0-100) ──
        score_fs  = self._fs_to_score(stability_result.factor_of_safety)
        score_ml  = ml_result.get('ml_probability', 0.0) * 100
        score_id  = self._id_to_score(id_result)
        score_swi = rainfall_data.get('swi_normalized', 0.0) * 100

        # ── Weighted Consensus ──
        raw_score = (
            self.weights['fs']  * score_fs  +
            self.weights['ml']  * score_ml  +
            self.weights['id']  * score_id  +
            self.weights['swi'] * score_swi
        )

        # ── Reactivation Multiplier ──
        if slope_metadata:
            prev = slope_metadata.get('previous_landslide', False)
            react_count = slope_metadata.get('reactivation_count', 0)
            if prev:
                reactivation_boost = min(react_count * 3, 15)  # max +15 pts
                raw_score = min(raw_score + reactivation_boost, 100)

        # ── Forecast Penalty ──
        forecast_3day = rainfall_data.get('forecast_3day_mm', 0)
        if forecast_3day > 50:
            raw_score = min(raw_score * 1.10, 100)  # +10% if heavy rain forecast
        elif forecast_3day > 25:
            raw_score = min(raw_score * 1.05, 100)

        final_score = float(np.clip(raw_score, 0.0, 100.0))
        risk_level  = self._score_to_level(final_score)
        risk_info   = RISK_LEVELS[risk_level]

        # ── Data Confidence ──
        confidence = self._compute_confidence(
            stability_result.confidence,
            ml_result.get('ml_confidence', 'LOW'),
            rainfall_data.get('data_quality', 'MEDIUM')
        )

        # ── Driver Analysis ──
        drivers = self._identify_drivers(
            score_fs, score_ml, score_id, score_swi, stability_result, rainfall_data)

        # ── Trend ──
        # (would compare to previous snapshot in production)

        result = {
            # Core output
            'risk_score': round(final_score, 1),
            'risk_level': risk_level,
            'risk_color': risk_info['color'],
            'risk_emoji': risk_info['emoji'],
            'recommended_action': risk_info['action'],

            # Component breakdown
            'component_scores': {
                'physics_fs': round(score_fs, 1),
                'ml_probability': round(score_ml, 1),
                'id_threshold': round(score_id, 1),
                'soil_moisture_swi': round(score_swi, 1),
            },

            # Key physics values
            'factor_of_safety': stability_result.factor_of_safety,
            'fs_category': stability_result.fs_category,
            'saturation_ratio': stability_result.saturation_ratio_m,
            'newmark_displacement_cm': stability_result.newmark_displacement_cm,

            # ML values
            'ml_probability': ml_result.get('ml_probability', 0),
            'ml_model_mode': ml_result.get('model_mode', 'unknown'),
            'shap_values': ml_result.get('shap_values', {}),

            # Rainfall context
            'rainfall_today_mm': rainfall_data.get('r_today_mm', 0),
            'rainfall_3day_mm': rainfall_data.get('r_3day_mm', 0),
            'swi': rainfall_data.get('swi', 0),
            'forecast_3day_mm': rainfall_data.get('forecast_3day_mm', 0),

            # Quality
            'confidence': confidence,
            'data_quality': rainfall_data.get('data_quality', 'UNKNOWN'),
            'physics_warnings': stability_result.warnings,

            # Risk drivers
            'top_drivers': drivers,

            # Metadata
            'computed_at': datetime.now().isoformat(),
            'slope_id': slope_metadata.get('slope_id', 'unknown') if slope_metadata else 'unknown',
        }

        return result

    def compute_batch(self, slope_df: pd.DataFrame,
                      rainfall_data: Dict,
                      fs_results: pd.DataFrame,
                      ml_results: pd.DataFrame,
                      id_result: Dict) -> pd.DataFrame:
        """
        Compute risk scores for all slope units.
        Returns enriched DataFrame with risk scores.
        """
        merged = slope_df.copy()

        # Merge FS results
        if not fs_results.empty:
            merged = merged.join(
                fs_results.set_index('slope_id')[
                    ['factor_of_safety','fs_category','saturation_m','failure_prob']
                ], how='left')

        # Merge ML results
        if not ml_results.empty:
            merged = merged.join(
                ml_results.set_index('slope_id')[
                    ['ml_probability','ml_risk_score']
                ], how='left')

        # Fill defaults
        merged['factor_of_safety'] = merged.get('factor_of_safety', 1.5)
        merged['ml_probability'] = merged.get('ml_probability', 0.0)

        # Compute consensus score for each row
        scores = []
        for _, row in merged.iterrows():
            fs   = row.get('factor_of_safety', 1.5)
            ml_p = row.get('ml_probability', 0.0)
            id_r = id_result.get('threshold_ratio', 0.0)
            swi  = rainfall_data.get('swi_normalized', 0.0)

            s_fs  = self._fs_to_score(fs)
            s_ml  = ml_p * 100
            s_id  = self._id_to_score(id_result)
            s_swi = swi * 100

            raw = (self.weights['fs']  * s_fs +
                   self.weights['ml']  * s_ml +
                   self.weights['id']  * s_id +
                   self.weights['swi'] * s_swi)

            final = float(np.clip(raw, 0, 100))
            level = self._score_to_level(final)

            scores.append({
                'risk_score': round(final, 1),
                'risk_level': level,
                'risk_color': RISK_LEVELS[level]['color'],
                'risk_emoji': RISK_LEVELS[level]['emoji'],
            })

        score_df = pd.DataFrame(scores, index=merged.index)
        return pd.concat([merged, score_df], axis=1)

    # ─────────────────────────────────────────────
    # SCORING FUNCTIONS
    # ─────────────────────────────────────────────

    def _fs_to_score(self, fs: float) -> float:
        """
        Convert Factor of Safety to 0-100 risk score.
        Nonlinear mapping: FS=1.0 → 80pts, FS=1.5 → 20pts, FS<0.9 → 95-100pts
        """
        if fs >= 2.0:   return 5.0
        elif fs >= 1.5: return 5.0 + (1.5 - fs) / 0.5 * 25.0    # 5-30
        elif fs >= 1.3: return 30.0 + (1.3 - fs) / 0.2 * 20.0   # 30-50
        elif fs >= 1.1: return 50.0 + (1.1 - fs) / 0.2 * 20.0   # 50-70
        elif fs >= 0.9: return 70.0 + (0.9 - fs) / 0.2 * 20.0   # 70-90
        else:           return min(90.0 + (0.9 - fs) * 50, 100)  # 90-100

    def _id_to_score(self, id_result: Dict) -> float:
        """Convert ID threshold result to 0-100 score."""
        ratio = id_result.get('threshold_ratio', 0.0)
        level = id_result.get('exceedance_level', 'BELOW_THRESHOLD')

        base = min(ratio * 50, 70)

        bonus = {'FAR_ABOVE': 30, 'ABOVE': 20,
                 'APPROACHING': 10, 'BELOW': 0}.get(
            id_result.get('status', 'BELOW'), 0)

        return float(np.clip(base + bonus, 0, 100))

    def _score_to_level(self, score: float) -> str:
        if score >= 90:   return 'EXTREME'
        elif score >= 70: return 'HIGH'
        elif score >= 50: return 'ELEVATED'
        elif score >= 30: return 'MODERATE'
        else:             return 'LOW'

    def _compute_confidence(self, phys_conf: str,
                             ml_conf: str, data_qual: str) -> str:
        score_map = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0, 'NONE': -1}
        total = (score_map.get(phys_conf, 0) +
                 score_map.get(ml_conf, 0) +
                 score_map.get(data_qual, 0))
        if total >= 4: return 'HIGH'
        elif total >= 2: return 'MEDIUM'
        else: return 'LOW'

    def _identify_drivers(self, s_fs, s_ml, s_id, s_swi,
                           stability_result, rainfall_data) -> List[str]:
        """Identify top risk drivers for field report."""
        drivers = []
        components = [
            (s_fs,  f"Physics: FS={stability_result.factor_of_safety:.2f} ({stability_result.fs_category})"),
            (s_ml,  f"AI model: {round(s_ml, 0):.0f}% failure probability"),
            (s_id,  f"Rainfall intensity above ID threshold"),
            (s_swi, f"High soil moisture (SWI={rainfall_data.get('swi', 0):.1f})"),
        ]
        sorted_comp = sorted(components, key=lambda x: x[0], reverse=True)
        drivers = [c[1] for c in sorted_comp if c[0] > 20]
        return drivers[:3]  # Top 3 drivers

    # ─────────────────────────────────────────────
    # WEIGHT CALIBRATION
    # ─────────────────────────────────────────────

    def calibrate_weights(self, validation_results: pd.DataFrame) -> Dict:
        """
        Auto-calibrate consensus weights from validation data.
        validation_results: DataFrame with actual outcomes and component scores.
        """
        # Simplified calibration via correlation with actual outcomes
        if len(validation_results) < 20:
            return {'calibrated': False, 'reason': 'insufficient_data'}

        try:
            y = validation_results['actual_failure'].values
            correlations = {
                'fs':  abs(np.corrcoef(validation_results['score_fs'], y)[0,1]),
                'ml':  abs(np.corrcoef(validation_results['score_ml'], y)[0,1]),
                'id':  abs(np.corrcoef(validation_results['score_id'], y)[0,1]),
                'swi': abs(np.corrcoef(validation_results['score_swi'], y)[0,1]),
            }

            total = sum(correlations.values())
            if total > 0:
                self.weights = {k: v/total for k, v in correlations.items()}

            self._save_calibrated_weights()
            logger.success(f"Weights calibrated: {self.weights}")
            return {'calibrated': True, 'weights': self.weights}

        except Exception as e:
            return {'calibrated': False, 'reason': str(e)}

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

    def _save_calibrated_weights(self):
        path = self.models_dir / "consensus_weights.json"
        with open(path, 'w') as f:
            json.dump({'weights': self.weights,
                       'calibrated_at': datetime.now().isoformat()}, f, indent=2)

    def _load_calibrated_weights(self):
        path = self.models_dir / "consensus_weights.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.weights = data.get('weights', self.weights)
            self._normalize_weights()
            logger.info(f"Loaded calibrated weights: {self.weights}")



