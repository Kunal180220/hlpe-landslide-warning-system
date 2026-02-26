"""
setup_data.py
=============
Initial data download script for HLPE.
Run once to set up the project with open-source data.
"""

import os
import sys
import requests
import yaml
from pathlib import Path
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error("config/config.yaml not found. Copy from config_template.yaml first.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_directories():
    dirs = [
        "data/static/dem",
        "data/static/geology",
        "data/static/soil",
        "data/static/landslide_inventory",
        "data/static/slope_units",
        "data/dynamic/rainfall",
        "data/dynamic/soil_moisture",
        "data/dynamic/risk_snapshots",
        "data/user_uploads",
        "models/calibration",
        "reports",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.success(f"Created {len(dirs)} directories")


def test_rainfall_api(config):
    """Test Open-Meteo API connection."""
    bbox = config['study_area']['bbox']
    lat = (bbox['min_lat'] + bbox['max_lat']) / 2
    lon = (bbox['min_lon'] + bbox['max_lon']) / 2

    logger.info(f"Testing Open-Meteo API for ({lat:.2f}, {lon:.2f})...")
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon,
                  "daily": "precipitation_sum", "forecast_days": 1}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            logger.success("✅ Open-Meteo API: Connected")
        else:
            logger.warning(f"⚠️ Open-Meteo returned {r.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Open-Meteo test failed: {e} (may work later)")


def generate_demo_slope_units(config):
    """Generate demo slope unit data if none exists."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.susceptibility import SusceptibilityEngine
    import pandas as pd

    out_path = Path("data/static/slope_units/slope_units_demo.csv")
    if out_path.exists():
        logger.info("Demo slope units already exist")
        return

    logger.info("Generating demo slope units...")
    engine = SusceptibilityEngine(config)
    df = engine.generate_synthetic_slope_units(n_units=50)
    df.to_csv(out_path)
    logger.success(f"✅ Generated {len(df)} demo slope units → {out_path}")


def write_gitignore():
    content = """
# Python
__pycache__/
*.py[cod]
*.pyo
.env
venv/
.venv/

# Data (large files)
data/dynamic/
*.tif
*.tiff
*.nc
*.h5

# Models (regenerable)
models/*.pkl
models/*.json

# Logs
logs/
*.log

# Reports
reports/*.pdf

# Streamlit
.streamlit/

# OS
.DS_Store
Thumbs.db
"""
    Path(".gitignore").write_text(content.strip())
    logger.success("✅ .gitignore created")


def write_streamlit_config():
    Path(".streamlit").mkdir(exist_ok=True)
    config_content = """
[theme]
primaryColor = "#1e3a5f"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1a202c"
font = "sans serif"

[server]
headless = true
port = 8501
"""
    Path(".streamlit/config.toml").write_text(config_content.strip())
    logger.success("✅ Streamlit config created")


def write_engine_init():
    init_content = '''"""HLPE Engine modules."""
from .rainfall_fetcher import RainfallFetcher
from .factor_of_safety import FactorOfSafetyEngine, SlopeParameters
from .id_threshold import IDThresholdEngine
from .ml_predictor import MLRiskPredictor
from .consensus_scorer import ConsensusScorer
from .susceptibility import SusceptibilityEngine

__all__ = [
    "RainfallFetcher",
    "FactorOfSafetyEngine",
    "SlopeParameters",
    "IDThresholdEngine",
    "MLRiskPredictor",
    "ConsensusScorer",
    "SusceptibilityEngine",
]
'''
    Path("engine/__init__.py").write_text(init_content)
    Path("dashboard/__init__.py").write_text("")
    Path("tests/__init__.py").write_text("")
    logger.success("✅ __init__.py files created")


def main():
    logger.info("=" * 60)
    logger.info("HLPE — Initial Setup")
    logger.info("=" * 60)

    config = load_config()
    logger.info(f"Region: {config['study_area']['name']}")

    create_directories()
    write_gitignore()
    write_streamlit_config()
    write_engine_init()
    test_rainfall_api(config)
    generate_demo_slope_units(config)

    logger.success("=" * 60)
    logger.success("✅ Setup complete!")
    logger.success("Run: streamlit run dashboard/app.py")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
