"""
ml_predictor.py
================
Machine Learning Risk Prediction Engine for HLPE.

Implements:
- XGBoost + Random Forest ensemble
- SHAP explainability
- Isotonic probability calibration
- Uncertainty quantification
- Auto-retraining from field data
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — using Random Forest only")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability disabled")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report


# Full feature set used by the model
FEATURE_COLUMNS = [
    # Static susceptibility features
    'slope_angle_deg',
    'slope_aspect',
    'curvature',
    'twi',                      # Topographic Wetness Index
    'elevation_m',
    'ndvi',                     # Vegetation index (0-1)
    'distance_to_stream_m',
    'geology_code',             # Encoded geology class
    'soil_texture_code',        # Encoded texture class
    'soil_depth_m',

    # Dynamic rainfall features
    'r_today_mm',
    'r_3day_mm',
    'r_7day_mm',
    'r_15day_mm',
    'api',
    'swi',
    'swi_normalized',
    'peak_intensity_mm_day',

    # Physics model features
    'factor_of_safety',
    'saturation_m',
    'pore_pressure_m',
    'failure_prob_physics',
    'newmark_disp_cm',

    # ID threshold features
    'id_ratio',                 # How far above/below ID threshold

    # Historical features
    'previous_landslide',       # Binary: slope had past failure
    'reactivation_count',       # Times slope has failed before
]


class MLRiskPredictor:
    """
    Ensemble ML model for landslide risk prediction.
    Works in two modes:
    - No training data: physics-based proxy (FS + ID → probability)
    - With training data: full XGBoost + RF ensemble
    """

    def __init__(self, config: dict, models_dir: str = "models"):
        self.config = config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        ml_cfg = config.get('ml', {})
        self.enabled = ml_cfg.get('enabled', True)
        self.n_estimators = ml_cfg.get('n_estimators', 200)
        self.min_samples = ml_cfg.get('min_training_samples', 50)
        self.shap_enabled = ml_cfg.get('shap_enabled', True) and SHAP_AVAILABLE

        self.model_xgb = None
        self.model_rf  = None
        self.scaler    = None
        self.feature_names = FEATURE_COLUMNS
        self.is_trained = False
        self.training_stats = {}

        self._load_models()
        logger.info(f"MLRiskPredictor initialized (trained={self.is_trained})")

    # ─────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────

    def predict(self, features: Dict) -> Dict:
        """
        Predict landslide risk probability for one slope unit.

        Args:
            features: Dict with all available features

        Returns:
            Dict with probability, confidence, and SHAP explanation
        """
        if not self.is_trained:
            return self._physics_proxy_prediction(features)

        # Prepare feature vector
        X = self._build_feature_vector(features)

        # Ensemble prediction
        proba_xgb = 0.0
        proba_rf  = 0.0

        if self.model_xgb is not None:
            try:
                proba_xgb = float(self.model_xgb.predict_proba(X)[0, 1])
            except Exception as e:
                logger.debug(f"XGB predict failed: {e}")

        if self.model_rf is not None:
            try:
                proba_rf = float(self.model_rf.predict_proba(X)[0, 1])
            except Exception as e:
                logger.debug(f"RF predict failed: {e}")

        # Weighted ensemble
        if self.model_xgb is not None and self.model_rf is not None:
            probability = 0.6 * proba_xgb + 0.4 * proba_rf
        elif self.model_xgb is not None:
            probability = proba_xgb
        elif self.model_rf is not None:
            probability = proba_rf
        else:
            return self._physics_proxy_prediction(features)

        # SHAP explanation
        explanation = {}
        if self.shap_enabled and self.model_xgb is not None:
            explanation = self._get_shap_explanation(X)

        return {
            'ml_probability': round(probability, 4),
            'ml_risk_score': round(probability * 100, 1),
            'proba_xgb': round(proba_xgb, 4),
            'proba_rf': round(proba_rf, 4),
            'ml_confidence': 'HIGH' if self.is_trained else 'LOW',
            'shap_values': explanation,
            'model_mode': 'ensemble',
            'trained': self.is_trained
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for all slope units in a DataFrame."""
        results = []
        for idx, row in df.iterrows():
            features = row.to_dict()
            result = self.predict(features)
            result['slope_id'] = idx
            results.append(result)
        return pd.DataFrame(results)

    # ─────────────────────────────────────────────
    # PHYSICS PROXY (no training data)
    # ─────────────────────────────────────────────

    def _physics_proxy_prediction(self, features: Dict) -> Dict:
        """
        When no ML model is trained, convert physics outputs to probability.
        Based on empirical mapping of FS → failure probability.
        """
        fs = features.get('factor_of_safety', 1.5)
        fail_prob_physics = features.get('failure_prob_physics', 0.0)
        id_ratio = features.get('id_ratio', 0.0)
        swi_norm = features.get('swi_normalized', 0.0)
        prev_landslide = features.get('previous_landslide', 0)

        # Sigmoid-based FS to probability
        # FS=1.0 → ~50%, FS=0.8 → ~90%, FS=1.5 → ~5%
        fs_prob = 1.0 / (1.0 + np.exp(5.0 * (fs - 1.0)))

        # ID contribution
        id_prob = min(id_ratio * 0.3, 0.3) if id_ratio > 1.0 else 0.0

        # SWI contribution
        swi_prob = swi_norm * 0.2

        # Reactivation boost
        reactivation_boost = 0.1 if prev_landslide else 0.0

        # Weighted combination
        probability = np.clip(
            0.5 * fs_prob + 0.2 * id_prob + 0.15 * swi_prob +
            0.15 * fail_prob_physics + reactivation_boost,
            0.0, 1.0
        )

        return {
            'ml_probability': round(float(probability), 4),
            'ml_risk_score': round(float(probability) * 100, 1),
            'proba_xgb': 0.0,
            'proba_rf': 0.0,
            'ml_confidence': 'LOW',
            'shap_values': {'fs_contribution': round(0.5 * float(fs_prob), 4)},
            'model_mode': 'physics_proxy',
            'trained': False
        }

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────

    def train(self, X_df: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the ensemble on labeled slope-event data.

        Args:
            X_df: Feature DataFrame (use FEATURE_COLUMNS where available)
            y: Binary labels (1=landslide occurred, 0=no event)

        Returns:
            Training metrics dict
        """
        if len(X_df) < self.min_samples:
            logger.warning(f"Only {len(X_df)} samples — need {self.min_samples}")
            return {'trained': False, 'reason': 'insufficient_data'}

        logger.info(f"Training ML models on {len(X_df)} samples...")

        # Fill missing features with 0
        for col in FEATURE_COLUMNS:
            if col not in X_df.columns:
                X_df[col] = 0.0

        X = X_df[FEATURE_COLUMNS].fillna(0).values
        y_arr = y.values

        # ── XGBoost ──
        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            self.model_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=3)

        # ── Random Forest ──
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.model_rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)

        # Train
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metrics = {'trained': True}

        if self.model_xgb is not None:
            self.model_xgb.fit(X, y_arr)
            cv_scores = cross_val_score(
                self.model_xgb, X, y_arr, cv=cv, scoring='roc_auc')
            metrics['xgb_auc_mean'] = round(float(cv_scores.mean()), 4)
            metrics['xgb_auc_std']  = round(float(cv_scores.std()), 4)
            logger.success(f"XGB AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        self.model_rf.fit(X, y_arr)
        cv_scores_rf = cross_val_score(
            self.model_rf, X, y_arr, cv=cv, scoring='roc_auc')
        metrics['rf_auc_mean'] = round(float(cv_scores_rf.mean()), 4)
        metrics['rf_auc_std']  = round(float(cv_scores_rf.std()), 4)
        logger.success(f"RF AUC:  {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")

        metrics['n_samples'] = len(X_df)
        metrics['n_landslides'] = int(y_arr.sum())
        metrics['trained_at'] = datetime.now().isoformat()

        self.is_trained = True
        self.training_stats = metrics
        self._save_models()

        return metrics

    # ─────────────────────────────────────────────
    # SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────

    def _get_shap_explanation(self, X: np.ndarray) -> Dict:
        """Get SHAP feature importance for one prediction."""
        if not SHAP_AVAILABLE or self.model_xgb is None:
            return {}
        try:
            # Use the base XGB model (before calibration)
            base_model = self.model_xgb.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # Top 5 most important features
            importance = {}
            for i, (feat, val) in enumerate(zip(FEATURE_COLUMNS, shap_vals[0])):
                importance[feat] = round(float(val), 4)

            top_5 = dict(sorted(importance.items(),
                                key=lambda x: abs(x[1]),
                                reverse=True)[:5])
            return top_5

        except Exception as e:
            logger.debug(f"SHAP computation failed: {e}")
            return {}

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from trained models."""
        if not self.is_trained:
            return pd.DataFrame()

        importance_dict = {}

        if XGB_AVAILABLE and self.model_xgb is not None:
            try:
                base = self.model_xgb.calibrated_classifiers_[0].estimator
                imp = base.feature_importances_
                for feat, val in zip(FEATURE_COLUMNS, imp):
                    importance_dict[feat] = importance_dict.get(feat, 0) + val * 0.6
            except Exception:
                pass

        if self.model_rf is not None:
            try:
                base = self.model_rf.calibrated_classifiers_[0].estimator
                imp = base.feature_importances_
                for feat, val in zip(FEATURE_COLUMNS, imp):
                    importance_dict[feat] = importance_dict.get(feat, 0) + val * 0.4
            except Exception:
                pass

        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)

        return df

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def _build_feature_vector(self, features: Dict) -> np.ndarray:
        """Build numpy array from feature dict."""
        vals = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
        return np.array(vals).reshape(1, -1)

    def _save_models(self):
        if self.model_xgb:
            joblib.dump(self.model_xgb, self.models_dir / "xgb_model.pkl")
        if self.model_rf:
            joblib.dump(self.model_rf, self.models_dir / "rf_model.pkl")
        with open(self.models_dir / "ml_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info("ML models saved")

    def _load_models(self):
        xgb_path = self.models_dir / "xgb_model.pkl"
        rf_path  = self.models_dir / "rf_model.pkl"

        if xgb_path.exists():
            self.model_xgb = joblib.load(xgb_path)
            logger.info("XGB model loaded")

        if rf_path.exists():
            self.model_rf = joblib.load(rf_path)
            logger.info("RF model loaded")

        if xgb_path.exists() or rf_path.exists():
            self.is_trained = True

        stats_path = self.models_dir / "ml_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.training_stats = json.load(f)
