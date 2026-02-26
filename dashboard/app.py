"""
app.py — HLPE Dashboard
========================
Main Streamlit application for the Hybrid Landslide Prediction Engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# Page config MUST be first
st.set_page_config(
    page_title="HLPE — Landslide Early Warning",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports ──
from engine.rainfall_fetcher import RainfallFetcher
from engine.factor_of_safety import FactorOfSafetyEngine, SlopeParameters
from engine.id_threshold import IDThresholdEngine
from engine.ml_predictor import MLRiskPredictor
from engine.consensus_scorer import ConsensusScorer, RISK_LEVELS
from engine.susceptibility import SusceptibilityEngine

# ── Styling ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a4f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.3rem 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
    }
    .alert-extreme { background: #1a1a1a; color: white; }
    .alert-high    { background: #fee2e2; border-left: 4px solid #dc2626; }
    .alert-elevated{ background: #ffedd5; border-left: 4px solid #ea580c; }
    .alert-moderate{ background: #fef9c3; border-left: 4px solid #ca8a04; }
    .alert-low     { background: #dcfce7; border-left: 4px solid #16a34a; }
    .shap-bar { font-family: monospace; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Default config for demo
    return {
        'study_area': {
            'name': 'Demo Region',
            'bbox': {'min_lon': 75.0, 'max_lon': 77.0,
                     'min_lat': 10.0, 'max_lat': 12.0},
            'crs': 'EPSG:4326'
        },
        'rainfall': {'antecedent_days': 15, 'swi_timescale_days': 5},
        'soil': {'default_params': {'cohesion_kpa': 5, 'friction_angle_deg': 32,
                                    'unit_weight_kNm3': 18, 'soil_depth_m': 1.5}},
        'physics': {'fs_thresholds': {'extreme': 0.9, 'high': 1.1,
                                       'elevated': 1.3, 'moderate': 1.5}},
        'id_thresholds': {'auto_calibrate': True, 'default_alpha': 14.82, 'default_beta': 0.39},
        'ml': {'enabled': True, 'n_estimators': 200, 'min_training_samples': 50},
        'consensus': {'weight_fs': 0.30, 'weight_ml': 0.35,
                      'weight_id': 0.20, 'weight_swi': 0.15},
        'api_keys': {'nasa_earthdata': ''}
    }

@st.cache_resource
def init_engines(config):
    rf    = RainfallFetcher(config)
    fs    = FactorOfSafetyEngine(config)
    idt   = IDThresholdEngine(config)
    ml    = MLRiskPredictor(config)
    cs    = ConsensusScorer(config)
    susc  = SusceptibilityEngine(config)
    return rf, fs, idt, ml, cs, susc

config = load_config()
rain_engine, fs_engine, id_engine, ml_engine, scorer, susc_engine = init_engines(config)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if 'slope_data' not in st.session_state:
    st.session_state.slope_data = susc_engine.generate_synthetic_slope_units(40)
if 'selected_slope' not in st.session_state:
    st.session_state.selected_slope = None
if 'risk_results' not in st.session_state:
    st.session_state.risk_results = {}
if 'rainfall_cache' not in st.session_state:
    st.session_state.rainfall_cache = {}

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="main-header">
    <h1>🏔️ HLPE — Hybrid Landslide Prediction Engine</h1>
    <p style="margin:0; opacity:0.85; font-size:1rem;">
        {config['study_area']['name']} &nbsp;|&nbsp;
        Physics + AI Landslide Early Warning System &nbsp;|&nbsp;
        Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/300x80/1e3a5f/ffffff?text=HLPE+System", width=300)
    st.markdown("---")

    page = st.radio("📍 Navigation", [
        "🗺️ Risk Map",
        "🔬 Slope Analysis",
        "🌧️ Rainfall Monitor",
        "📈 ID Threshold",
        "📁 Upload Data",
        "📊 ML Insights",
        "📄 Generate Report",
        "⚙️ Settings"
    ])

    st.markdown("---")
    st.markdown("### 🔴 Live Alerts")
    alert_box = st.empty()

    st.markdown("---")
    st.caption("HLPE v1.0 | MIT License")
    st.caption("Physics + AI Hybrid Model")

# ─────────────────────────────────────────────────────────────
# HELPER: Compute risk for one slope
# ─────────────────────────────────────────────────────────────

def compute_risk_for_slope(slope_id: str, rain_override: dict = None) -> dict:
    """Full pipeline for one slope unit."""
    row = st.session_state.slope_data.loc[slope_id]

    # 1. Rainfall
    cache_key = f"{row.lat:.2f}_{row.lon:.2f}"
    if cache_key in st.session_state.rainfall_cache and not rain_override:
        rain = st.session_state.rainfall_cache[cache_key]
    else:
        rain = rain_engine.get_current_rainfall(row.lat, row.lon)
        if rain_override:
            rain.update(rain_override)
        st.session_state.rainfall_cache[cache_key] = rain

    # 2. Factor of Safety
    params = SlopeParameters(
        slope_angle_deg=float(row.get('slope_angle_deg', 30)),
        soil_depth_m=float(row.get('soil_depth_m', 1.5)),
        cohesion_kpa=float(row.get('cohesion_kpa', 5.0)),
        friction_angle_deg=float(row.get('friction_angle_deg', 32.0)),
        soil_unit_weight=float(row.get('unit_weight', 18.0)),
        texture_class=str(row.get('texture_class', 'default')),
        geology=str(row.get('geology', 'default')),
    )
    fs_result = fs_engine.compute(params, rain)

    # 3. ID Threshold
    id_result = id_engine.get_current_storm_position(rain)

    # 4. ML Prediction
    features = {
        'slope_angle_deg': params.slope_angle_deg,
        'slope_aspect': float(row.get('slope_aspect', 180)),
        'twi': float(row.get('twi', 6.0)),
        'elevation_m': float(row.get('elevation_m', 800)),
        'ndvi': float(row.get('ndvi', 0.4)),
        'geology_code': hash(params.geology) % 10,
        'soil_texture_code': hash(params.texture_class) % 10,
        'soil_depth_m': params.soil_depth_m,
        'factor_of_safety': fs_result.factor_of_safety,
        'saturation_m': fs_result.saturation_ratio_m,
        'pore_pressure_m': fs_result.pore_pressure_head_m,
        'failure_prob_physics': fs_result.failure_probability,
        'newmark_disp_cm': fs_result.newmark_displacement_cm,
        'id_ratio': id_result.get('threshold_ratio', 0.0),
        'previous_landslide': 1 if row.get('previous_landslide', False) else 0,
        'reactivation_count': int(row.get('reactivation_count', 0)),
        **rain
    }
    ml_result = ml_engine.predict(features)

    # 5. Consensus Score
    slope_meta = {
        'slope_id': slope_id,
        'previous_landslide': row.get('previous_landslide', False),
        'reactivation_count': int(row.get('reactivation_count', 0)),
    }
    final = scorer.compute_risk_score(fs_result, ml_result, id_result, rain, slope_meta)
    final['fs_result'] = fs_result
    final['rain'] = rain
    final['id_result'] = id_result

    return final


def compute_all_risks():
    """Compute risk for all slope units with progress bar."""
    results = {}
    slopes = st.session_state.slope_data
    prog = st.progress(0, text="Computing risk for all slope units...")

    for i, (sid, _) in enumerate(slopes.iterrows()):
        try:
            results[sid] = compute_risk_for_slope(sid)
        except Exception as e:
            results[sid] = {'risk_score': 0, 'risk_level': 'LOW',
                            'risk_color': '#16a34a', 'risk_emoji': '🟢', 'error': str(e)}
        prog.progress((i + 1) / len(slopes),
                      text=f"Processing {i+1}/{len(slopes)}: {sid}")

    prog.empty()
    st.session_state.risk_results = results
    return results

# ─────────────────────────────────────────────────────────────
# PAGE: RISK MAP
# ─────────────────────────────────────────────────────────────

if page == "🗺️ Risk Map":
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    slopes = st.session_state.slope_data

    with col1:
        if st.button("🔄 Compute All Risks", type="primary", use_container_width=True):
            compute_all_risks()

    results = st.session_state.risk_results

    # Summary counts
    if results:
        levels = [r.get('risk_level', 'LOW') for r in results.values()]
        counts = {l: levels.count(l) for l in ['EXTREME','HIGH','ELEVATED','MODERATE','LOW']}

        with col1: st.metric("⚫ Extreme", counts.get('EXTREME', 0))
        with col2: st.metric("🔴 High", counts.get('HIGH', 0))
        with col3: st.metric("🟠 Elevated", counts.get('ELEVATED', 0))
        with col4: st.metric("🟢 Low+Moderate", counts.get('LOW', 0) + counts.get('MODERATE', 0))

    st.markdown("---")

    # Map + Table columns
    map_col, table_col = st.columns([3, 2])

    with map_col:
        st.subheader("🗺️ Interactive Risk Map")

        try:
            import folium
            from streamlit_folium import st_folium

            bbox = config['study_area']['bbox']
            center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
            center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='CartoDB positron'
            )

            # Add tile layers
            folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Relief Map'
            ).add_to(m)

            # Plot slope units
            for sid, row in slopes.iterrows():
                risk_info = results.get(sid, {})
                score = risk_info.get('risk_score', 0)
                level = risk_info.get('risk_level', 'LOW')
                color = risk_info.get('risk_color', '#16a34a')

                # Size based on risk
                radius = 8 + score / 10

                popup_html = f"""
                <b>Slope Unit: {sid}</b><br>
                📍 {row.lat:.4f}°N, {row.lon:.4f}°E<br>
                <b>Risk Score: {score:.0f}/100</b><br>
                Level: {risk_info.get('risk_emoji','')} {level}<br>
                FS: {risk_info.get('factor_of_safety', '—')}<br>
                Rain Today: {risk_info.get('rainfall_today_mm', '—')} mm<br>
                Action: {risk_info.get('recommended_action', '—')}
                """

                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{sid}: {level} ({score:.0f})"
                ).add_to(m)

            # Legend
            legend_html = """
            <div style='position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                        background: white; padding: 10px; border-radius: 8px;
                        border: 2px solid #ccc; font-size: 12px;'>
                <b>Risk Level</b><br>
                <span style='color:#1a1a1a'>⚫ Extreme (≥90)</span><br>
                <span style='color:#dc2626'>🔴 High (70-89)</span><br>
                <span style='color:#ea580c'>🟠 Elevated (50-69)</span><br>
                <span style='color:#ca8a04'>🟡 Moderate (30-49)</span><br>
                <span style='color:#16a34a'>🟢 Low (&lt;30)</span>
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl().add_to(m)

            map_data = st_folium(m, height=480, use_container_width=True)

        except ImportError:
            st.warning("folium / streamlit-folium not installed. Install with: pip install folium streamlit-folium")
            st.dataframe(slopes[['lat','lon','slope_angle_deg','geology']].head(10))

    with table_col:
        st.subheader("📋 Slope Unit Risk Table")

        # Build display table
        display_rows = []
        for sid, row in slopes.iterrows():
            r = results.get(sid, {})
            display_rows.append({
                'ID': sid,
                'Lat': round(row.lat, 3),
                'Lon': round(row.lon, 3),
                'Risk': f"{r.get('risk_emoji','🟢')} {r.get('risk_level','—')}",
                'Score': r.get('risk_score', '—'),
                'FS': r.get('factor_of_safety', '—'),
                'Rain(mm)': r.get('rainfall_today_mm', '—'),
            })

        disp_df = pd.DataFrame(display_rows)

        # Filter by risk level
        filter_level = st.selectbox("Filter by level",
                                     ['All', 'EXTREME', 'HIGH', 'ELEVATED', 'MODERATE', 'LOW'])
        if filter_level != 'All':
            disp_df = disp_df[disp_df['Risk'].str.contains(filter_level)]

        selected = st.dataframe(
            disp_df,
            hide_index=True,
            use_container_width=True,
            height=380,
            on_select='rerun',
            selection_mode='single-row'
        )

        if selected and selected.selection.rows:
            idx = selected.selection.rows[0]
            st.session_state.selected_slope = disp_df.iloc[idx]['ID']
            st.success(f"Selected: {st.session_state.selected_slope} → Go to Slope Analysis")

# ─────────────────────────────────────────────────────────────
# PAGE: SLOPE ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "🔬 Slope Analysis":
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("🔬 Detailed Slope Stability Analysis")

    slopes = st.session_state.slope_data
    slope_ids = list(slopes.index)

    default_idx = 0
    if st.session_state.selected_slope in slope_ids:
        default_idx = slope_ids.index(st.session_state.selected_slope)

    sel = st.selectbox("Select Slope Unit", slope_ids, index=default_idx)
    row = slopes.loc[sel]

    st.markdown("---")
    info_col, param_col = st.columns([1, 1])

    with info_col:
        st.markdown("#### 📍 Slope Info")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Slope ID | `{sel}` |
        | Location | {row.lat:.4f}°N, {row.lon:.4f}°E |
        | Elevation | {row.get('elevation_m', 'N/A'):.0f} m |
        | Slope Angle | **{row.get('slope_angle_deg', 'N/A'):.1f}°** |
        | Geology | {row.get('geology', 'N/A')} |
        | Soil Texture | {row.get('texture_class', 'N/A')} |
        | Soil Depth | {row.get('soil_depth_m', 'N/A'):.1f} m |
        | NDVI | {row.get('ndvi', 'N/A'):.2f} |
        | Previous Failure | {'⚠️ Yes' if row.get('previous_landslide') else '✅ No'} |
        """)

    with param_col:
        st.markdown("#### 🧮 Override Parameters")
        override_slope = st.slider("Slope angle (°)", 5.0, 80.0,
                                    float(row.get('slope_angle_deg', 30)), 0.5)
        override_depth = st.slider("Soil depth (m)", 0.3, 5.0,
                                    float(row.get('soil_depth_m', 1.5)), 0.1)
        override_c = st.slider("Cohesion C' (kPa)", 0.0, 20.0,
                                 float(row.get('cohesion_kpa', 5.0)), 0.5)
        override_phi = st.slider("Friction angle φ' (°)", 15.0, 45.0,
                                   float(row.get('friction_angle_deg', 32.0)), 0.5)
        override_rain = st.slider("Today's rainfall (mm) — Override", 0.0, 200.0,
                                   0.0, 5.0)

    if st.button("🔄 Run Analysis", type="primary"):
        with st.spinner("Running 5-layer hybrid analysis..."):
            # Apply overrides
            slopes.loc[sel, 'slope_angle_deg'] = override_slope
            slopes.loc[sel, 'soil_depth_m'] = override_depth
            slopes.loc[sel, 'cohesion_kpa'] = override_c
            slopes.loc[sel, 'friction_angle_deg'] = override_phi

            rain_override = None
            if override_rain > 0:
                rain_override = {'r_today_mm': override_rain,
                                 'swi_normalized': min(override_rain / 100, 1.0)}

            result = compute_risk_for_slope(sel, rain_override)
            st.session_state.risk_results[sel] = result

    result = st.session_state.risk_results.get(sel)

    if result:
        st.markdown("---")
        # ── Risk Score Banner ──
        level = result['risk_level']
        score = result['risk_score']
        color = result['risk_color']
        emoji = result['risk_emoji']
        action = result['recommended_action']

        alert_cls = f"alert-{level.lower()}"
        st.markdown(f"""
        <div class="risk-card {alert_cls}" style="font-size:1.4rem; border-radius:12px; padding:1.5rem;">
            {emoji} Risk Level: <b>{level}</b> &nbsp;|&nbsp;
            Score: <b>{score:.0f}/100</b> &nbsp;|&nbsp;
            ⚡ Action: <i>{action}</i>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── Metric Cards ──
        m1, m2, m3, m4, m5 = st.columns(5)
        fs = result.get('factor_of_safety', '—')
        fs_val = float(fs) if isinstance(fs, (int, float)) else 1.5
        m1.metric("⚖️ Factor of Safety", f"{fs:.3f}" if isinstance(fs, float) else fs,
                  delta=f"{'FAIL' if fs_val < 1 else 'OK'}")
        m2.metric("💧 Saturation m", f"{result.get('saturation_ratio', 0):.2%}")
        m3.metric("🌧️ Rain Today", f"{result.get('rainfall_today_mm', 0):.1f} mm")
        m4.metric("📐 SWI", f"{result.get('swi', 0):.2f}")
        m5.metric("🤖 ML Prob", f"{result.get('ml_probability', 0):.1%}")

        st.markdown("---")

        # ── Charts ──
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 📊 Component Risk Breakdown")
            comp = result.get('component_scores', {})
            if comp:
                fig = go.Figure(go.Bar(
                    x=list(comp.values()),
                    y=['Physics (FS)', 'AI/ML', 'ID Threshold', 'Soil Moisture (SWI)'],
                    orientation='h',
                    marker_color=['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981'],
                    text=[f"{v:.0f}" for v in comp.values()],
                    textposition='outside'
                ))
                fig.update_layout(
                    xaxis=dict(range=[0, 100], title="Risk Score (0-100)"),
                    height=280, margin=dict(l=20, r=30, t=20, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### 🧠 SHAP — Why This Risk?")
            shap_vals = result.get('shap_values', {})
            if shap_vals:
                features = list(shap_vals.keys())
                values = list(shap_vals.values())
                colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in values],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Red=Increases Risk | Green=Decreases",
                    height=280, margin=dict(l=20, r=30, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP values available after ML model is trained with local data.")

        # ── FS vs Saturation Sensitivity ──
        st.markdown("#### 📉 FS Sensitivity to Rainfall (What-If Analysis)")
        saturation_range = np.linspace(0, 1, 50)
        fs_curve = []
        for m in saturation_range:
            rain_sim = {**result.get('rain', {}), 'swi_normalized': m,
                        'r_today_mm': m * 80}
            params = SlopeParameters(
                slope_angle_deg=override_slope,
                soil_depth_m=override_depth,
                cohesion_kpa=override_c,
                friction_angle_deg=override_phi,
                texture_class=str(row.get('texture_class', 'default')),
                geology=str(row.get('geology', 'default')),
            )
            r = fs_engine.compute(params, rain_sim)
            fs_curve.append(r.factor_of_safety)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=saturation_range * 100, y=fs_curve,
            fill='tozeroy', fillcolor='rgba(59,130,246,0.1)',
            line=dict(color='#3b82f6', width=2),
            name='Factor of Safety'
        ))
        fig2.add_hline(y=1.0, line_dash='dash', line_color='red',
                       annotation_text='Failure Threshold (FS=1.0)')
        fig2.add_hline(y=1.3, line_dash='dot', line_color='orange',
                       annotation_text='FS=1.3 (Elevated)')

        # Current point
        curr_m = result.get('saturation_ratio', 0.3)
        curr_fs = result.get('factor_of_safety', 1.5)
        if isinstance(curr_fs, float):
            fig2.add_trace(go.Scatter(
                x=[curr_m * 100], y=[curr_fs],
                mode='markers', marker=dict(size=14, color='red', symbol='star'),
                name='Current State'
            ))

        fig2.update_layout(
            xaxis_title='Saturation Ratio m (%)',
            yaxis_title='Factor of Safety',
            height=300,
            legend=dict(x=0.7, y=0.9),
            plot_bgcolor='rgba(248,250,252,1)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Warnings ──
        fs_obj = result.get('fs_result')
        if fs_obj and fs_obj.warnings:
            st.markdown("#### ⚠️ Model Warnings")
            for w in fs_obj.warnings:
                st.warning(w)

        # ── Drivers ──
        drivers = result.get('top_drivers', [])
        if drivers:
            st.markdown("#### 🎯 Top Risk Drivers")
            for d in drivers:
                st.markdown(f"• {d}")

    else:
        st.info("👆 Click **Run Analysis** to compute risk for this slope unit.")

# ─────────────────────────────────────────────────────────────
# PAGE: RAINFALL MONITOR
# ─────────────────────────────────────────────────────────────

elif page == "🌧️ Rainfall Monitor":
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("🌧️ Dynamic Rainfall Monitor")

    tab1, tab2, tab3 = st.tabs(["📡 Live Data", "✏️ Manual Entry", "📈 Time Series"])

    with tab1:
        st.markdown("#### Fetch live rainfall for a location")
        col1, col2, col3 = st.columns(3)
        with col1:
            lat_in = st.number_input("Latitude", value=float(
                (config['study_area']['bbox']['min_lat'] +
                 config['study_area']['bbox']['max_lat']) / 2), format="%.4f")
        with col2:
            lon_in = st.number_input("Longitude", value=float(
                (config['study_area']['bbox']['min_lon'] +
                 config['study_area']['bbox']['max_lon']) / 2), format="%.4f")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_btn = st.button("🌐 Fetch Rainfall", type="primary")

        if fetch_btn:
            with st.spinner("Fetching from Open-Meteo / NASA GPM..."):
                rain = rain_engine.get_current_rainfall(lat_in, lon_in)

            qual_color = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴', 'NONE': '⛔'}
            st.markdown(f"**Data Source:** {rain['data_source']} &nbsp; "
                        f"**Quality:** {qual_color.get(rain['data_quality'], '?')} {rain['data_quality']}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Today", f"{rain['r_today_mm']} mm")
            m2.metric("3-Day", f"{rain['r_3day_mm']} mm")
            m3.metric("7-Day", f"{rain['r_7day_mm']} mm")
            m4.metric("15-Day", f"{rain['r_15day_mm']} mm")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("SWI", f"{rain['swi']:.2f}")
            m6.metric("API", f"{rain['api']:.1f}")
            m7.metric("Forecast Tomorrow", f"{rain['forecast_tomorrow_mm']} mm")
            m8.metric("Forecast 3-Day", f"{rain['forecast_3day_mm']} mm")

            # SWI gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rain['swi_normalized'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Soil Water Index (SWI) — Saturation %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 60], 'color': '#fef9c3'},
                        {'range': [60, 80], 'color': '#ffedd5'},
                        {'range': [80, 100], 'color': '#fee2e2'},
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 3},
                        'thickness': 0.75, 'value': 80
                    }
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### ✏️ Add Manual Rain Gauge Reading")
        stations = config.get('rainfall', {}).get('manual_stations', [])

        col1, col2, col3 = st.columns(3)
        with col1:
            sta_options = [s['id'] for s in stations] + ['Custom']
            sta_sel = st.selectbox("Station ID", sta_options)
            if sta_sel == 'Custom':
                sta_id = st.text_input("Custom Station ID")
            else:
                sta_id = sta_sel

        with col2:
            m_date = st.date_input("Date", datetime.now())
            m_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0,
                                      step=0.5)

        with col3:
            if sta_sel != 'Custom':
                sta_info = next((s for s in stations if s['id'] == sta_sel), None)
                m_lat = st.number_input("Station Lat", value=float(sta_info['lat']) if sta_info else 11.0)
                m_lon = st.number_input("Station Lon", value=float(sta_info['lon']) if sta_info else 76.0)
            else:
                m_lat = st.number_input("Station Lat", value=11.0)
                m_lon = st.number_input("Station Lon", value=76.0)

        if st.button("💾 Save Reading", type="primary"):
            record = rain_engine.add_manual_reading(
                sta_id, str(m_date), m_rain, m_lat, m_lon)
            st.success(f"✅ Saved: {sta_id} | {m_date} | {m_rain} mm")

        # Show existing
        manual_df = rain_engine.get_manual_readings()
        if not manual_df.empty:
            st.markdown("#### 📋 Manual Records")
            st.dataframe(manual_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### 📈 Rainfall Time Series")
        ts_lat = st.number_input("Lat for time series", value=float(
            (config['study_area']['bbox']['min_lat'] + config['study_area']['bbox']['max_lat']) / 2))
        ts_lon = st.number_input("Lon for time series", value=float(
            (config['study_area']['bbox']['min_lon'] + config['study_area']['bbox']['max_lon']) / 2))

        if st.button("📊 Plot Time Series"):
            with st.spinner("Fetching 30-day history..."):
                ts = rain_engine.get_time_series_for_plot(ts_lat, ts_lon, days=30)
            if ts is not None and not ts.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=ts['date'], y=ts['rainfall_mm'],
                    name='Daily Rainfall', marker_color='#3b82f6'))
                fig.update_layout(
                    xaxis_title='Date', yaxis_title='Rainfall (mm)',
                    height=350, plot_bgcolor='rgba(248,250,252,1)')
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: ID THRESHOLD
# ─────────────────────────────────────────────────────────────

elif page == "📈 ID Threshold":
    import plotly.graph_objects as go

    st.subheader("📈 Intensity-Duration Threshold Analysis")

    st.markdown(f"""
    **Current Thresholds:** I = **{id_engine.alpha:.2f}** × D^(**-{id_engine.beta:.3f}**)
    *(Based on: {config['study_area']['name']})*
    """)

    # ID Curve Plot
    curve_df = id_engine.get_id_curve_data()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df['duration_h'], y=curve_df['I_thresh_5'],
        fill=None, line=dict(color='rgba(220,38,38,0.3)', dash='dot'),
        name='High Probability Threshold'
    ))
    fig.add_trace(go.Scatter(
        x=curve_df['duration_h'], y=curve_df['I_thresh_25'],
        fill='tonexty', fillcolor='rgba(220,38,38,0.1)',
        line=dict(color='rgba(234,88,12,0.5)', dash='dash'),
        name='Medium Probability Threshold'
    ))
    fig.add_trace(go.Scatter(
        x=curve_df['duration_h'], y=curve_df['I_thresh_50'],
        fill='tonexty', fillcolor='rgba(234,88,12,0.1)',
        line=dict(color='#1e3a5f', width=2.5),
        name='Base Threshold (I = α·D^(-β))'
    ))

    # Current storm position
    bbox = config['study_area']['bbox']
    center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
    center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2
    rain = rain_engine.get_current_rainfall(center_lat, center_lon)
    storm = id_engine.get_current_storm_position(rain)

    fig.add_trace(go.Scatter(
        x=[storm['current_duration_h']],
        y=[storm['current_intensity_mm_hr']],
        mode='markers+text',
        marker=dict(size=16, color='red', symbol='star'),
        text=[f"  Current Storm\n  ({storm['current_intensity_mm_hr']:.3f} mm/hr)"],
        textposition='top right',
        name='Current Storm'
    ))

    fig.update_layout(
        xaxis=dict(type='log', title='Duration (hours)', range=[0, 3]),
        yaxis=dict(type='log', title='Intensity (mm/hr)'),
        height=450,
        legend=dict(x=0.65, y=0.95),
        plot_bgcolor='rgba(248,250,252,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        title='Intensity-Duration Threshold Curves'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Status
    status = storm.get('exceedance', {}).get('exceedance_level', 'BELOW_THRESHOLD')
    ratio  = storm.get('threshold_ratio', 0)
    colors_map = {
        'HIGH_PROBABILITY': '🔴 HIGH PROBABILITY — Storm above threshold!',
        'MEDIUM_PROBABILITY': '🟠 MEDIUM PROBABILITY — Approaching threshold',
        'LOW_PROBABILITY': '🟡 LOW PROBABILITY — Below base threshold',
        'BELOW_THRESHOLD': '🟢 BELOW THRESHOLD — No exceedance'
    }
    st.info(f"**Storm Position:** {colors_map.get(status, status)}  "
            f"| Threshold Ratio: **{ratio:.2f}x**")

    # Calibration
    st.markdown("---")
    st.subheader("🔧 Calibrate Thresholds from Your Data")
    st.markdown("Upload a CSV with past landslide events to auto-calibrate regional thresholds.")
    cal_file = st.file_uploader("Landslide Inventory (CSV: date, lat, lon)",
                                 type=['csv'], key='id_cal')
    if cal_file:
        inv_df = pd.read_csv(cal_file)
        st.success(f"Loaded {len(inv_df)} events")
        st.dataframe(inv_df.head())
        if st.button("🔧 Calibrate"):
            st.warning("Calibration needs historical rainfall data. Coming in next version.")

# ─────────────────────────────────────────────────────────────
# PAGE: UPLOAD DATA
# ─────────────────────────────────────────────────────────────

elif page == "📁 Upload Data":
    st.subheader("📁 Data Upload & Management")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗾 Shapefiles / GeoJSON",
        "📊 Slope Unit CSV",
        "⚠️ Landslide Inventory",
        "🌄 DEM / Raster"
    ])

    with tab1:
        st.markdown("#### Upload Slope Unit Shapefiles or GeoJSON")
        st.markdown("Supported: `.shp` (with sidecars), `.geojson`, `.gpkg`")
        shp_file = st.file_uploader("Upload GeoJSON / Shapefile",
                                     type=['geojson', 'json'], key='shp')
        if shp_file and GEOPANDAS_AVAILABLE:
            try:
                gdf = gpd.read_file(shp_file)
                st.success(f"✅ Loaded {len(gdf)} features")
                st.dataframe(gdf.drop(columns=['geometry']).head(10))
                if st.button("Import as Slope Units"):
                    # Convert GDF to slope unit DataFrame
                    df = gdf.drop(columns=['geometry']).copy()
                    df['lat'] = gdf.geometry.centroid.y
                    df['lon'] = gdf.geometry.centroid.x
                    if 'slope_id' not in df.columns:
                        df['slope_id'] = [f'SU_{i:04d}' for i in range(len(df))]
                    df = df.set_index('slope_id')
                    # Fill missing columns
                    for col in ['slope_angle_deg','soil_depth_m','cohesion_kpa',
                                'friction_angle_deg','unit_weight','ndvi',
                                'geology','texture_class','previous_landslide',
                                'reactivation_count','twi','elevation_m']:
                        if col not in df.columns:
                            df[col] = 0
                    st.session_state.slope_data = df
                    st.success(f"✅ Imported {len(df)} slope units!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        elif not GEOPANDAS_AVAILABLE:
            st.warning("geopandas not installed. Install: pip install geopandas")

    with tab2:
        st.markdown("#### Upload Slope Unit Data as CSV")
        st.markdown("""
        **Required columns:** `slope_id`, `lat`, `lon`, `slope_angle_deg`

        **Optional:** `soil_depth_m`, `cohesion_kpa`, `friction_angle_deg`,
        `geology`, `texture_class`, `ndvi`, `previous_landslide`
        """)

        # Template download
        template = pd.DataFrame({
            'slope_id': ['SU_0001', 'SU_0002'],
            'lat': [11.234, 11.456],
            'lon': [76.123, 76.345],
            'slope_angle_deg': [28.5, 35.2],
            'soil_depth_m': [1.5, 2.0],
            'cohesion_kpa': [5.0, 3.5],
            'friction_angle_deg': [32.0, 28.0],
            'geology': ['granite', 'schist'],
            'texture_class': ['loam', 'clay_loam'],
            'ndvi': [0.45, 0.30],
            'previous_landslide': [False, True],
        })
        st.download_button("📥 Download Template CSV", template.to_csv(index=False),
                           "slope_units_template.csv", "text/csv")

        csv_file = st.file_uploader("Upload Slope Units CSV", type=['csv'], key='csv_upload')
        if csv_file:
            df = pd.read_csv(csv_file)
            st.success(f"✅ Loaded {len(df)} slope units")
            st.dataframe(df.head())
            if 'slope_id' in df.columns:
                df = df.set_index('slope_id')
            if st.button("✅ Use This Data"):
                st.session_state.slope_data = df
                st.session_state.risk_results = {}
                st.success(f"Slope data updated with {len(df)} units!")

    with tab3:
        st.markdown("#### Upload Landslide Inventory")
        st.markdown("Used for ID threshold calibration and ML training.")
        inv_template = pd.DataFrame({
            'date': ['2022-08-15', '2023-06-20'],
            'lat': [11.234, 11.678],
            'lon': [76.123, 76.567],
            'type': ['rotational', 'debris_flow'],
            'damage': ['road_blocked', 'property'],
            'rainfall_24h_mm': [85, 120],
        })
        st.download_button("📥 Download Inventory Template",
                           inv_template.to_csv(index=False),
                           "inventory_template.csv", "text/csv")

        inv_file = st.file_uploader("Upload Inventory CSV", type=['csv'], key='inv')
        if inv_file:
            inv = pd.read_csv(inv_file)
            st.success(f"✅ Loaded {len(inv)} landslide events")
            st.dataframe(inv)

    with tab4:
        st.markdown("#### Upload DEM (Digital Elevation Model)")
        st.markdown("Supported: GeoTIFF (.tif). SRTM 30m recommended.")
        dem_file = st.file_uploader("Upload DEM GeoTIFF", type=['tif', 'tiff'], key='dem')
        if dem_file:
            save_path = Path("data/static/dem") / dem_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(dem_file.read())
            st.success(f"✅ DEM saved to {save_path}")
            st.info("DEM will be used for automatic slope/TWI computation on next run.")

# ─────────────────────────────────────────────────────────────
# PAGE: ML INSIGHTS
# ─────────────────────────────────────────────────────────────

elif page == "📊 ML Insights":
    import plotly.express as px

    st.subheader("📊 Machine Learning Model Insights")

    st.markdown(f"""
    **Model Status:** {'✅ Trained' if ml_engine.is_trained else '⚠️ Not yet trained (using physics proxy)'}
    """)

    if ml_engine.training_stats:
        st.json(ml_engine.training_stats)

    st.markdown("---")
    st.markdown("### 🏋️ Train Model from Labeled Data")
    st.markdown("""
    Upload a CSV with slope features + known landslide outcomes to train the ML model.
    The more field data you add, the better it gets.
    """)

    train_file = st.file_uploader("Training Data CSV (features + label column 'failure')",
                                   type=['csv'], key='train_ml')
    if train_file:
        train_df = pd.read_csv(train_file)
        st.dataframe(train_df.head())

        if 'failure' in train_df.columns:
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training XGBoost + Random Forest ensemble..."):
                    y = train_df['failure']
                    X = train_df.drop(columns=['failure'])
                    metrics = ml_engine.train(X, y)
                if metrics.get('trained'):
                    st.success("✅ Model trained successfully!")
                    st.json(metrics)
                else:
                    st.error(f"Training failed: {metrics.get('reason', 'unknown')}")
        else:
            st.error("CSV must have a 'failure' column (1=landslide, 0=no event)")

    # Feature Importance
    if ml_engine.is_trained:
        st.markdown("---")
        st.markdown("### 🎯 Global Feature Importance")
        fi = ml_engine.get_feature_importance()
        if not fi.empty:
            fig = px.bar(fi.head(15), x='importance', y='feature',
                         orientation='h', color='importance',
                         color_continuous_scale='Blues')
            fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: GENERATE REPORT
# ─────────────────────────────────────────────────────────────

elif page == "📄 Generate Report":
    st.subheader("📄 Daily Risk Report Generator")

    if not st.session_state.risk_results:
        st.warning("No risk results yet. Go to Risk Map and click 'Compute All Risks' first.")
    else:
        results = st.session_state.risk_results
        slopes = st.session_state.slope_data

        # Summary table
        summary = []
        for sid, r in results.items():
            summary.append({
                'Slope ID': sid,
                'Risk Level': f"{r.get('risk_emoji','')} {r.get('risk_level','')}",
                'Score': r.get('risk_score', 0),
                'FS': r.get('factor_of_safety', '—'),
                'Rain (mm)': r.get('rainfall_today_mm', '—'),
                'ML Prob': f"{r.get('ml_probability', 0):.1%}",
                'Action': r.get('recommended_action', '—'),
            })

        summary_df = pd.DataFrame(summary).sort_values('Score', ascending=False)

        st.markdown(f"""
        ## Daily Landslide Risk Report
        **Region:** {config['study_area']['name']}
        **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
        **Total Slope Units Assessed:** {len(results)}
        """)

        # Alert counts
        extreme = sum(1 for r in results.values() if r.get('risk_level') == 'EXTREME')
        high    = sum(1 for r in results.values() if r.get('risk_level') == 'HIGH')
        if extreme > 0:
            st.error(f"⚫ {extreme} slope(s) at EXTREME risk — Immediate action required!")
        if high > 0:
            st.warning(f"🔴 {high} slope(s) at HIGH risk — Field teams should be alerted!")

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # CSV Export
        csv = summary_df.to_csv(index=False)
        st.download_button(
            "📥 Download Risk Report (CSV)",
            csv,
            f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            type="primary"
        )

# ─────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────

elif page == "⚙️ Settings":
    st.subheader("⚙️ System Settings")

    tab1, tab2, tab3 = st.tabs(["🗺️ Study Area", "⚙️ Model Parameters", "🔑 API Keys"])

    with tab1:
        st.markdown("#### Study Area Configuration")
        area_name = st.text_input("Region Name", config['study_area']['name'])
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Min Latitude", value=config['study_area']['bbox']['min_lat'])
            min_lon = st.number_input("Min Longitude", value=config['study_area']['bbox']['min_lon'])
        with col2:
            max_lat = st.number_input("Max Latitude", value=config['study_area']['bbox']['max_lat'])
            max_lon = st.number_input("Max Longitude", value=config['study_area']['bbox']['max_lon'])

        if st.button("💾 Save Study Area"):
            config['study_area']['name'] = area_name
            config['study_area']['bbox'] = {
                'min_lat': min_lat, 'max_lat': max_lat,
                'min_lon': min_lon, 'max_lon': max_lon
            }
            with open("config/config.yaml", 'w') as f:
                yaml.dump(config, f)
            st.success("✅ Config saved! Restart app to apply.")

    with tab2:
        st.markdown("#### Consensus Weights (sum = 1.0)")
        w_fs  = st.slider("Physics FS weight", 0.0, 1.0, config['consensus']['weight_fs'], 0.05)
        w_ml  = st.slider("ML weight", 0.0, 1.0, config['consensus']['weight_ml'], 0.05)
        w_id  = st.slider("ID Threshold weight", 0.0, 1.0, config['consensus']['weight_id'], 0.05)
        w_swi = st.slider("SWI weight", 0.0, 1.0, config['consensus']['weight_swi'], 0.05)
        total = w_fs + w_ml + w_id + w_swi
        st.metric("Total Weight", f"{total:.2f}", delta=f"{'OK ✅' if abs(total-1.0)<0.01 else 'Must = 1.0 ❌'}")

        st.markdown("#### ID Threshold Parameters")
        new_alpha = st.number_input("α (alpha)", value=float(id_engine.alpha), format="%.3f")
        new_beta  = st.number_input("β (beta)", value=float(id_engine.beta), format="%.4f")
        if st.button("Update Parameters"):
            id_engine.alpha = new_alpha
            id_engine.beta  = new_beta
            st.success("Parameters updated for this session.")

    with tab3:
        st.markdown("#### API Keys (Optional)")
        st.info("NASA Earthdata key gives access to GPM IMERG satellite rainfall (free registration at earthdata.nasa.gov)")
        nasa_key = st.text_input("NASA Earthdata Token", type="password",
                                  value=config.get('api_keys', {}).get('nasa_earthdata', ''))
        if st.button("Save API Keys"):
            config['api_keys']['nasa_earthdata'] = nasa_key
            with open("config/config.yaml", 'w') as f:
                yaml.dump(config, f)
            st.success("API keys saved!")

# ─────────────────────────────────────────────────────────────
# SIDEBAR ALERTS UPDATE
# ─────────────────────────────────────────────────────────────

results = st.session_state.risk_results
if results:
    extreme_slopes = [s for s, r in results.items() if r.get('risk_level') == 'EXTREME']
    high_slopes    = [s for s, r in results.items() if r.get('risk_level') == 'HIGH']

    if extreme_slopes:
        alert_box.error(f"⚫ EXTREME RISK\n{', '.join(extreme_slopes[:3])}")
    elif high_slopes:
        alert_box.warning(f"🔴 HIGH RISK\n{', '.join(high_slopes[:3])}")
    else:
        alert_box.success("🟢 No critical alerts")
else:
    alert_box.info("Run analysis to see alerts")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
