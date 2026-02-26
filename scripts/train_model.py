"""
train_model.py
==============
CLI script to train the ML model from labeled data.

Usage:
    python scripts/train_model.py --data path/to/training.csv
    python scripts/train_model.py --data data/static/landslide_inventory/labeled_events.csv
"""

import argparse
import sys
import yaml
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.ml_predictor import MLRiskPredictor
from engine.id_threshold import IDThresholdEngine
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train HLPE ML model")
    parser.add_argument("--data", required=True, help="Path to labeled training CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--label", default="failure",
                        help="Name of binary label column (default: 'failure')")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading training data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} samples, {df[args.label].sum():.0f} positive events")

    # Train ML model
    ml = MLRiskPredictor(config)
    y  = df[args.label]
    X  = df.drop(columns=[args.label])

    metrics = ml.train(X, y)
    if metrics.get('trained'):
        logger.success("✅ Model trained and saved!")
        logger.success(f"XGB AUC: {metrics.get('xgb_auc_mean', 'N/A')}")
        logger.success(f"RF  AUC: {metrics.get('rf_auc_mean', 'N/A')}")
    else:
        logger.error(f"Training failed: {metrics.get('reason')}")

    # Calibrate ID thresholds if inventory provided
    if 'date' in df.columns and 'lat' in df.columns:
        logger.info("Attempting ID threshold calibration...")
        idt = IDThresholdEngine(config)
        # Simplified — would need historical rainfall too
        logger.info("ID calibration requires historical rainfall data — skipped for now")


if __name__ == "__main__":
    main()
