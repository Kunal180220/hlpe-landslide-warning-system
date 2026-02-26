# 🏔️ HLPE — Hybrid Landslide Prediction Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI + Physics Hybrid Landslide Early Warning System — Streamlit dashboard with live rainfall, Factor of Safety, ML ensemble and ID threshold analysis**

---

## 🌟 Key Features

| Feature | Description |
|--------|-------------|
| 🌧️ **Live Rainfall** | Auto-fetches daily from NASA GPM + Open-Meteo. Manual entry supported. |
| 🧮 **Physics Engine** | Infinite slope FS model + Soil Water Index (SWI) + Iverson pore pressure |
| 🤖 **AI/ML Layer** | XGBoost + Random Forest ensemble with SHAP explainability |
| 🗺️ **Interactive Map** | Real-time risk map — zoom to any slope unit |
| 📁 **Data Upload** | Drag & drop shapefiles, CSVs, GeoJSON, rain gauge data |
| 📊 **ID Thresholds** | Intensity-Duration curves auto-calibrated to your region |
| 🔔 **Micro-Alerts** | Slope-unit level alerts, not vague district warnings |
| 📈 **Historical Timeline** | Watch risk evolution across past weeks/months |
| 📄 **PDF Reports** | Auto-generate daily risk summary reports |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              HLPE — 5-Layer Hybrid Engine               │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Static Susceptibility (slope, geology, soil)  │
│  Layer 2: Dynamic Hydrology (SWI, antecedent rain)      │
│  Layer 3: Physics Stability (Factor of Safety)          │
│  Layer 4: ML Fusion (XGBoost + RF + SHAP)               │
│  Layer 5: Consensus Risk Score (0-100)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/hlpe-landslide.git
cd hlpe-landslide
pip install -r requirements.txt
```

### 2. Configure Your Study Area
```bash
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your bounding box, region name, etc.
```

### 3. Download Initial Data
```bash
python scripts/setup_data.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
hlpe-landslide/
├── config/
│   └── config.yaml              # Study area, API keys, parameters
├── data/
│   ├── static/                  # DEM, geology, soil, inventory
│   └── dynamic/                 # Live rainfall, soil moisture, snapshots
├── engine/
│   ├── rainfall_fetcher.py      # NASA GPM + Open-Meteo + manual
│   ├── swi_calculator.py        # Soil Water Index engine
│   ├── factor_of_safety.py      # Infinite slope physics model
│   ├── id_threshold.py          # Intensity-Duration threshold curves
│   ├── susceptibility.py        # Static susceptibility scoring
│   ├── ml_predictor.py          # XGBoost + RF ensemble
│   └── consensus_scorer.py      # Final risk score fusion
├── dashboard/
│   ├── app.py                   # Main Streamlit app
│   ├── map_view.py              # Interactive Folium map
│   ├── upload_panel.py          # Data upload UI
│   ├── alert_engine.py          # Alert generation
│   └── report_generator.py      # PDF report export
├── models/                      # Saved ML models
├── scripts/
│   ├── setup_data.py            # Initial data download
│   └── train_model.py           # ML training script
├── tests/                       # Unit tests
├── requirements.txt
└── README.md
```

---

## 📊 Risk Levels

| Level | Factor of Safety | ML Probability | Action |
|-------|-----------------|----------------|--------|
| ⚫ EXTREME | < 0.9 | > 85% | Immediate evacuation |
| 🔴 HIGH | 0.9 – 1.1 | 65 – 85% | Alert field teams |
| 🟠 ELEVATED | 1.1 – 1.3 | 40 – 65% | Restrict access |
| 🟡 MODERATE | 1.3 – 1.5 | 20 – 40% | Daily monitoring |
| 🟢 LOW | > 1.5 | < 20% | Normal operations |

---

## 🌍 Data Sources (All Free & Open)

| Data | Source | Resolution | Update Frequency |
|------|--------|------------|-----------------|
| Rainfall | NASA GPM IMERG | 0.1° (~10km) | 4-hour lag |
| Rainfall backup | Open-Meteo ERA5 | 0.25° | Daily |
| Soil Moisture | NASA SMAP | 9km | 2-3 days |
| DEM / Elevation | SRTM via OpenTopography | 30m | Static |
| Soil Properties | ISRIC SoilGrids | 250m | Static |
| Geology | USGS / user upload | Variable | Static |

---

## 🤝 Contributing

1. Fork the repo
2. Add your local data to `data/user_uploads/`
3. Log field observations through the dashboard
4. Submit pull requests for engine improvements

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

## 📚 Scientific References

- Iverson, R.M. (2000). Landslide triggering by rain infiltration. *Water Resources Research*.
- Montgomery & Dietrich (1994). SHALSTAB model. *Water Resources Research*.
- Brocca et al. (2012). SLIP model. *Natural Hazards and Earth System Sciences*.
- Baum et al. (2008). TRIGRS v2.0. *USGS Open-File Report*.
- Kirschbaum et al. (2015). NASA LHASA model. *Geomorphology*.
